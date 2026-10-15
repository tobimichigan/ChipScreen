"""chiptox.py - ChipTox: multimodal toxicity scorer (Model & Algorithm).
Chemistry (Tox21 + ToxCast, RDKit) x morphology (JUMP Cell Painting) -> multi-task toxicity probabilities with uncertainty,
scaffold-grouped 40/15/15/30 splits, tuned classical + attention/CNN deep models, soft-voting ensemble, best model saved."""
import re
import time

import joblib
import numpy as np
import pandas as pd

import chem
import features as FE
import jump
import models as M
from utils import (safe, group_kfold, greedy_ensemble, fast_macro_auc, LOG, Timer, tq, check_memory, force_cleanup, finish_plot, group_split, verify_data_splits, check_domain_relevance,
                   multilabel_metrics, best_thresholds, overfit_report, poor_areas, log_mem, save_json, downcast_df)

CYTO_PAT = re.compile(r"cytotox|viabil|xtt|cell_?viab", re.I)
SCORE_ASSAYS = ["tox21:SR-MMP", "tox21:SR-ARE", "tox21:SR-p53", "tox21:SR-HSE", "tox21:SR-ATAD5", "composite:cytotox"]


# =========================================================================== data
def build_tox_table(res, cfg):
    """Unify Tox21 + ToxCast on the InChIKey connectivity block. -> dict(df, Y, task_names)"""
    cache = cfg.cache / "tox_unified.parquet"
    p21, pTC = res.tox21(), res.toxcast()
    if not p21 and not pTC:
        raise FileNotFoundError("neither tox21.csv nor toxcast_data.csv found under root")
    frames = []
    if p21:
        t = pd.read_csv(p21)
        tcols = [c for c in t.columns if c not in ("mol_id", "smiles")]
        t = t.rename(columns={c: f"tox21:{c}" for c in tcols})
        frames.append(t)
    if pTC:
        c = pd.read_csv(pTC)
        lab = [x for x in c.columns if x != "smiles"]
        pos, neg = c[lab].eq(1).sum(), c[lab].eq(0).sum()
        ok = [x for x in lab if pos[x] >= cfg.toxcast_min_pos and neg[x] >= cfg.toxcast_min_pos]
        ok = sorted(ok, key=lambda x: -min(pos[x], neg[x]))[: cfg.toxcast_max_tasks]
        cy = [x for x in lab if CYTO_PAT.search(x)]
        comp = c[cy].max(axis=1, skipna=True) if cy else None
        if comp is not None:
            comp[c[cy].isna().all(axis=1)] = np.nan
        tc = c[["smiles"] + ok].rename(columns={x: f"toxcast:{x}" for x in ok})
        if comp is not None:
            tc["composite:cytotox"] = comp
        LOG.info(f"[chiptox] ToxCast: {len(ok)} assays kept (>= {cfg.toxcast_min_pos} pos/neg); cytotox composite from {len(cy)} assays")
        frames.append(tc)
    # standardise + join key
    out = []
    for f in frames:
        st = chem.smiles_table(f["smiles"].tolist(), "standardise SMILES")
        f = f.reset_index(drop=True)
        f["ik14"], f["std_smiles"], f["scaffold"] = st["ik14"], st["std_smiles"], st["scaffold"]
        out.append(f.dropna(subset=["ik14"]))
    lab_cols = [c for f in out for c in f.columns if ":" in c]
    allf = pd.concat(out, ignore_index=True)
    agg = {c: "max" for c in lab_cols}               # positive if any source reports an active call (NaN-aware)
    agg.update(std_smiles="first", scaffold="first", smiles="first")
    df = allf.groupby("ik14", as_index=False).agg(agg)
    df["group"] = np.where(df.scaffold.isin(["", "NOSCAF"]), df.ik14, df.scaffold)   # acyclic cpds get their own group
    Y = df[lab_cols].astype(float)
    keep = Y.notna().sum(1) > 0
    df, Y = df[keep].reset_index(drop=True), Y[keep].reset_index(drop=True)
    LOG.info(f"[chiptox] unified table: {len(df):,} compounds x {Y.shape[1]} tasks; {df.group.nunique():,} scaffold groups")
    return dict(df=df, Y=Y.values.astype(np.float32), task_names=list(Y.columns))


FEATURE_VERSION = "v3"


def featurize(std_smiles, cfg):
    """Morgan counts + MACCS + atom-pair counts + topological-torsion counts + RDKit descriptors (+ ratios). Count blocks are log1p-compressed."""
    std_smiles = list(std_smiles)
    blocks, names = [], []
    fp = np.log1p(chem.morgan_fp(std_smiles, cfg.fp_bits, 2, True).astype(np.float32)); blocks.append(fp); names += [f"fp{i}" for i in range(fp.shape[1])]
    mk = chem.maccs(std_smiles); blocks.append(mk); names += [f"maccs{i}" for i in range(mk.shape[1])]
    ap = np.log1p(chem.atompair_counts(std_smiles, 1024)); blocks.append(ap); names += [f"ap{i}" for i in range(ap.shape[1])]
    tt = np.log1p(chem.torsion_counts(std_smiles, 512)); blocks.append(tt); names += [f"tt{i}" for i in range(tt.shape[1])]
    D, dn = chem.descriptors(std_smiles)
    Dd = FE.chem_engineered(pd.DataFrame(D, columns=dn)); blocks.append(Dd.values.astype(np.float32)); names += list(Dd.columns)
    return np.hstack(blocks), names


def chem_features(df, cfg):
    cache = cfg.cache / f"chem_feats_{FEATURE_VERSION}_{cfg.fp_bits}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if len(z["ik"]) == len(df) and (z["ik"] == df.ik14.values).all():
            return z["X"], list(z["names"])
    X, names = featurize(df.std_smiles.tolist(), cfg)
    np.savez_compressed(cache, X=X, names=np.array(names), ik=df.ik14.values)
    return X, names


# =========================================================================== modelling helpers
def _fit_all_classical(Xtr, Ytr, Xva, Yva, Xall_splits, w, tasks, cfg, names=("lgbm", "rf", "logreg"), tag="chem"):
    """Hyper-parameters are tuned on the validation split for up to 3 representative (best-labelled) tasks, then every task is
    fitted on the train split with the winning configuration. Returns {model: {split: (n, T) probabilities}}, params, scores."""
    T = Ytr.shape[1]
    cnt = np.array([(~np.isnan(Ytr[:, j])).sum() * (len(np.unique(Ytr[~np.isnan(Ytr[:, j]), j])) == 2) for j in range(T)])
    rep = [j for j in np.argsort(-cnt)[:3] if cnt[j] > 30]
    preds = {n: {s: np.full((len(X), T), np.nan, np.float32) for s, X in Xall_splits.items()} for n in names}
    best_params, scores = {}, {}
    for name in names:
        cand = []
        for j in tq(rep, f"tune {name} ({tag})", leave=False):
            mt, mv = ~np.isnan(Ytr[:, j]), ~np.isnan(Yva[:, j])
            if len(np.unique(Yva[mv, j])) < 2:
                continue
            try:
                est, bp, sc = M.tune_classical(name, Xtr[mt], Ytr[mt, j].astype(int), Xva[mv], Yva[mv, j].astype(int), w[mt], "binary", cfg.search_iters, cfg.seed)
                cand.append((sc, bp))
            except Exception as e:
                LOG.warning(f"[tune] {name} task {j}: {e!r}")
        if not cand:
            LOG.warning(f"[tune] {name}: no tunable task, skipped")
            continue
        sc, bp = max(cand, key=lambda t: t[0])
        best_params[name], scores[name] = bp, sc
        for j in tq(range(T), f"fit {name} ({tag})", leave=False):
            mt = ~np.isnan(Ytr[:, j])
            if mt.sum() < 20 or len(np.unique(Ytr[mt, j])) < 2:
                continue
            est, _ = M.classical_space(name, "binary", cfg.seed, pos_weight=float((Ytr[mt, j] == 0).sum() / max((Ytr[mt, j] == 1).sum(), 1)))
            est.set_params(**bp)
            try:
                est.fit(Xtr[mt], Ytr[mt, j].astype(int), sample_weight=w[mt])
            except TypeError:
                est.fit(Xtr[mt], Ytr[mt, j].astype(int))
            for s, X in Xall_splits.items():
                preds[name][s][:, j] = est.predict_proba(X)[:, 1]
            check_memory(cfg, f"fit {name}")
        force_cleanup()
    preds = {k: v for k, v in preds.items() if k in best_params}
    return preds, best_params, scores


def _fill(p):
    return np.nan_to_num(p, nan=0.5)


def _evaluate_all(pred_by_split, Y_by_split, thr, task_names):
    return {s: multilabel_metrics(Y_by_split[s], pred_by_split[s], thr, task_names) for s in pred_by_split}


# =========================================================================== deep model zoo (shared by training and the scorer)
DEEP = {
    "attn_mlp": lambda d, T: M.AttnMLP(d, T, (512, 256), drop=0.5, noise=0.10),
    "cnn_attn": lambda d, T: M.HybridCNNAttn(d, T, drop=0.4, noise=0.10),
}


class _Member:
    """Predictions of one model family: out-of-fold (pool), in-sample (pool), test, holdout (each averaged over the fold models)."""

    def __init__(self, n_pool, n_te, n_ho, T):
        self.oof = np.full((n_pool, T), np.nan, np.float32)
        self.pool_in = np.zeros((n_pool, T), np.float32)
        self.test = np.zeros((n_te, T), np.float32)
        self.holdout = np.zeros((n_ho, T), np.float32)
        self.cnt = np.zeros(T)
        self.hold_members = []          # per-fold holdout predictions (-> uncertainty)

    def add(self, j, va, p_va, p_pool, p_te, p_ho):
        self.oof[va, j] = p_va
        self.pool_in[:, j] += p_pool; self.test[:, j] += p_te; self.holdout[:, j] += p_ho; self.cnt[j] += 1

    def finish(self):
        c = np.maximum(self.cnt, 1)
        self.pool_in /= c; self.test /= c; self.holdout /= c
        for nm in ("oof", "pool_in", "test", "holdout"):
            a = getattr(self, nm); a[:] = np.nan_to_num(a, nan=0.5)
        return self


def _cv_lgbm(Xp, Yp, Xte, Xho, w, folds, params, cfg, T):
    import lightgbm as lgb
    mem = _Member(len(Xp), len(Xte), len(Xho), T)
    base = {k: v for k, v in params.items() if k not in ("n_estimators",)}
    for j in tq(range(T), "LGBM CV (tasks)", leave=False):
        lab = ~np.isnan(Yp[:, j])
        for f, (tr, va) in enumerate(folds):
            trl, val = tr[lab[tr]], va[lab[va]]
            if len(np.unique(Yp[trl, j])) < 2 or len(np.unique(Yp[val, j])) < 2:
                continue
            y = Yp[:, j].astype(int)
            spw = float((y[trl] == 0).sum() / max((y[trl] == 1).sum(), 1))
            est = lgb.LGBMClassifier(**base, n_estimators=800, verbose=-1, n_jobs=-1, subsample_freq=1, scale_pos_weight=spw, random_state=cfg.seed + f)
            est.fit(Xp[trl], y[trl], sample_weight=w[trl], eval_set=[(Xp[val], y[val])], eval_metric="auc", callbacks=[lgb.early_stopping(40, verbose=False)])
            pp = est.predict_proba(Xp)[:, 1]
            mem.add(j, va, pp[va], pp, est.predict_proba(Xte)[:, 1], est.predict_proba(Xho)[:, 1])
        check_memory(cfg, "lgbm cv")
    return mem.finish()


def _cv_logreg(Xp, Yp, Xte, Xho, w, folds, C, cfg, T):
    from sklearn.linear_model import LogisticRegression
    mem = _Member(len(Xp), len(Xte), len(Xho), T)
    for j in tq(range(T), "LogReg CV (tasks)", leave=False):
        lab = ~np.isnan(Yp[:, j]); y = Yp[:, j].astype(int)
        for f, (tr, va) in enumerate(folds):
            trl = tr[lab[tr]]
            if len(np.unique(y[trl])) < 2:
                continue
            est = LogisticRegression(C=C, max_iter=300, class_weight="balanced").fit(Xp[trl], y[trl], sample_weight=w[trl])
            pp = est.predict_proba(Xp)[:, 1]
            mem.add(j, va, pp[va], pp, est.predict_proba(Xte)[:, 1], est.predict_proba(Xho)[:, 1])
    return mem.finish()


def _fit_rf_oob(Xp, Yp, Xte, Xho, w, cfg, T):
    """Random forest on the whole pool; its OUT-OF-BAG probabilities are the honest out-of-fold predictions."""
    from sklearn.ensemble import RandomForestClassifier
    mem = _Member(len(Xp), len(Xte), len(Xho), T)
    prior = np.nanmean(Yp, 0)
    for j in tq(range(T), "RF OOB (tasks)", leave=False):
        lab = np.where(~np.isnan(Yp[:, j]))[0]; y = Yp[lab, j].astype(int)
        if len(np.unique(y)) < 2 or y.sum() < 3:
            continue
        rf = RandomForestClassifier(n_estimators=250, max_features="sqrt", min_samples_leaf=2, class_weight="balanced_subsample", oob_score=True,
                                    n_jobs=-1, random_state=cfg.seed).fit(Xp[lab], y, sample_weight=w[lab])
        oob = rf.oob_decision_function_[:, 1]
        oob = np.where(np.isfinite(oob), oob, prior[j])
        mem.oof[lab, j] = oob
        mem.pool_in[:, j] = rf.predict_proba(Xp)[:, 1]; mem.test[:, j] = rf.predict_proba(Xte)[:, 1]; mem.holdout[:, j] = rf.predict_proba(Xho)[:, 1]; mem.cnt[j] = 1
        check_memory(cfg, "rf")
    m = mem.finish()
    return m


def _cv_deep(name, Xp, Yp, Xte, Xho, w, folds, cfg, T):
    mem = _Member(len(Xp), len(Xte), len(Xho), T)
    d_in = Xp.shape[1]
    states, nets, hists = [], [], []
    for f, (tr, va) in enumerate(folds):
        model, h, best = M.fit_torch(DEEP[name](d_in, T), Xp[tr], Yp[tr], Xp[va], Yp[va], w=w[tr], task="multilabel", epochs=cfg.deep_epochs, patience=cfg.deep_patience,
                                     batch=cfg.deep_batch, wd=5e-3, label_smoothing=0.05, mixup=0.2, ckpt=str(cfg.models / f"chiptox_{name}_fold{f}.pt"), desc=f"{name} fold {f+1}/{len(folds)}", seed=cfg.seed + f)
        pp, _ = M.predict(model, Xp); pt, _ = M.predict(model, Xte); ph, _ = M.predict(model, Xho)
        for j in range(T):
            mem.add(j, va, pp[va, j], pp[:, j], pt[:, j], ph[:, j])
        mem.hold_members.append(ph)
        LOG.info(f"[chiptox] {name} fold {f+1}: val macro-AUROC {best:.3f}")
        states.append({k: v.detach().cpu() for k, v in model.state_dict().items()}); nets.append(model.cpu()); hists.append(h)
        force_cleanup()
    mem.finish()
    return mem, states, nets, hists


def _blend(members, w):
    out = {}
    for part in ("oof", "pool_in", "test", "holdout"):
        out[part] = sum(w[k] * getattr(members[k], part) for k in w)
    return out


def _split_metrics(P, Y_pool, Y_te, Y_ho, is_val, thr, tasks):
    """P: dict(oof, pool_in, test, holdout). 'train' = in-sample predictions of the train rows, 'val' = out-of-fold predictions of the val rows."""
    tr = ~is_val
    pieces = {"train": (Y_pool[tr], P["pool_in"][tr]), "val": (Y_pool[is_val], P["oof"][is_val]), "cv_oof": (Y_pool, P["oof"]), "test": (Y_te, P["test"]), "holdout": (Y_ho, P["holdout"])}
    return {s: multilabel_metrics(y, p, thr, tasks) for s, (y, p) in pieces.items()}


# =========================================================================== main ChipTox pipeline
def run_chiptox(res, cfg, morph=None):
    out = {}
    with Timer("ChipTox: data"):
        tb = build_tox_table(res, cfg)
        df, Y, tasks = tb["df"], tb["Y"], tb["task_names"]
        Xc_raw, cnames = chem_features(df, cfg)
        log_mem("chem features", cfg)
    # ---- strict 4-way scaffold-group split. holdout (30%) and test (15%) are NEVER used for fitting / tuning / thresholds / blending.
    splits = group_split(df.group.values, cfg.split_fracs, seed=cfg.seed)
    out["split_report"] = verify_data_splits(splits, df.group.values, len(df))
    pool = np.sort(np.r_[splits["train"], splits["val"]]); te, ho = splits["test"], splits["holdout"]
    is_val = np.isin(pool, splits["val"])
    safe(lambda: FE.eda_plots(Xc_raw[:, cfg.fp_bits:], Y, cnames[cfg.fp_bits:], "chiptox_chem", cfg, tasks, splits), "chiptox EDA plots")
    prep = FE.TabularPreprocessor(corr_thresh=0.98, min_freq=cfg.min_bit_freq, seed=cfg.seed).fit(Xc_raw[pool], cnames)
    Xp, Xte, Xho = (prep.transform(Xc_raw[i]) for i in (pool, te, ho))
    Yp, Yte, Yho = Y[pool], Y[te], Y[ho]
    n_tr = int((~is_val).sum())
    dom_idx = {"train": np.where(~is_val)[0], "val": np.where(is_val)[0], "test": len(pool) + np.arange(len(te)), "holdout": len(pool) + len(te) + np.arange(len(ho))}
    out["domain_shift"] = safe(lambda: check_domain_relevance(np.vstack([Xp, Xte, Xho]), dom_idx, y=Y[np.r_[pool, te, ho], tasks.index("composite:cytotox")] if "composite:cytotox" in tasks else None), "domain shift", {})
    w, outl = FE.outlier_weights(Xp, seed=cfg.seed)
    out["n_outliers"] = int(outl.sum())
    LOG.info(f"[chiptox] pool {len(pool)} (train {n_tr} + val {int(is_val.sum())}) | test {len(te)} | holdout {len(ho)} | {Xp.shape[1]} features | outliers down-weighted {int(outl.sum())}")
    T = len(tasks)
    folds = group_kfold(df.group.values[pool], cfg.cv_folds, cfg.seed)
    members, deep_states, deep_nets = {}, {}, {}
    cls_params = {}
    models = set(cfg.chiptox_models)
    # ---- classical members
    with Timer("ChipTox: classical members (CV bagging)"):
        if models & {"lgbm", "logreg"}:
            tr0, va0 = folds[0]
            cnt = np.array([(~np.isnan(Yp[tr0, j])).sum() * (len(np.unique(Yp[tr0][~np.isnan(Yp[tr0, j]), j])) == 2) for j in range(T)])
            rep = [j for j in np.argsort(-cnt)[:3] if cnt[j] > 30]
            for nm in [m for m in ("lgbm", "logreg") if m in models]:
                cand = []
                for j in tq(rep, f"tune {nm}", leave=False):
                    mt, mv = tr0[~np.isnan(Yp[tr0, j])], va0[~np.isnan(Yp[va0, j])]
                    if len(np.unique(Yp[mv, j])) < 2:
                        continue
                    try:
                        _, bp, sc = M.tune_classical(nm, Xp[mt], Yp[mt, j].astype(int), Xp[mv], Yp[mv, j].astype(int), w[mt], "binary", cfg.search_iters, cfg.seed)
                        cand.append((sc, bp))
                    except Exception as e:
                        LOG.warning(f"[tune] {nm} task {j}: {e!r}")
                if cand:
                    cls_params[nm] = max(cand, key=lambda t: t[0])[1]
                    LOG.info(f"[chiptox] {nm} tuned params {cls_params[nm]}")
            if "lgbm" in cls_params:
                members["lgbm"] = _cv_lgbm(Xp, Yp, Xte, Xho, w, folds, cls_params["lgbm"], cfg, T)
            if "logreg" in cls_params:
                members["logreg"] = _cv_logreg(Xp, Yp, Xte, Xho, w, folds, cls_params["logreg"].get("C", 0.05), cfg, T)
        if "rf" in models:
            members["rf"] = _fit_rf_oob(Xp, Yp, Xte, Xho, w, cfg, T)
        force_cleanup()
    # ---- deep members
    hists = {}
    with Timer("ChipTox: deep members (CV bagging)"):
        for nm in [m for m in ("attn_mlp", "cnn_attn") if m in models]:
            mem, st, nets, h = _cv_deep(nm, Xp, Yp, Xte, Xho, w, folds, cfg, T)
            members[nm], deep_states[nm], deep_nets[nm], hists[nm] = mem, st, nets, h
            safe(lambda: M.plot_history(h[0], f"chiptox_{nm}_fold1", cfg, "val macro-AUROC"), f"history {nm}")
    if not members:
        raise RuntimeError("no ChipTox member trained successfully")
    # ---- OOF-driven blending (Caruana) + evaluation
    wts = greedy_ensemble({k: v.oof for k, v in members.items()}, Yp)
    LOG.info(f"[chiptox] OOF blend weights: { {k: round(v, 3) for k, v in wts.items()} }")
    allP = {k: dict(oof=v.oof, pool_in=v.pool_in, test=v.test, holdout=v.holdout) for k, v in members.items()}
    allP["ensemble"] = _blend(members, wts)
    if len(members) > 2:   # an equal-weight blend is the lowest-variance option: kept as a candidate
        allP["equal_blend"] = _blend(members, {k: 1 / len(members) for k in members})
    metrics, thr_by = {}, {}
    for k, P in allP.items():
        thr_by[k] = best_thresholds(Yp, P["oof"])            # thresholds from OUT-OF-FOLD predictions only
        metrics[k] = _split_metrics(P, Yp, Yte, Yho, is_val, thr_by[k], tasks)
    # ---- selection on OOF + TEST with an explicit overfitting penalty (holdout stays untouched until the final report)
    sel = {}
    for k, m in metrics.items():
        gap_fit = m["train"]["macro_auroc"] - m["cv_oof"]["macro_auroc"]
        # score = mean of the two honest estimates (out-of-fold on 55% of the data, test on 15%) minus a penalty for in-sample over-fitting
        sel[k] = 0.5 * (m["test"]["macro_auroc"] + m["cv_oof"]["macro_auroc"]) - cfg.select_gap_penalty * max(0.0, gap_fit - 0.10)
    best_name = max(sel, key=sel.get)
    bm = metrics[best_name]
    LOG.info(f"[chiptox] selected '{best_name}': train {bm['train']['macro_auroc']:.3f} | val(OOF) {bm['val']['macro_auroc']:.3f} | test {bm['test']['macro_auroc']:.3f} | HOLDOUT {bm['holdout']['macro_auroc']:.3f}")
    rows = {}
    for k, m in metrics.items():
        r = {}
        for s in m:
            r[f"{s}_auroc"] = m[s]["macro_auroc"]; r[f"{s}_balacc"] = m[s]["bal_acc"]; r[f"{s}_acc"] = m[s]["acc"]; r[f"{s}_ap"] = m[s]["macro_ap"]
        r["selection_score"] = sel[k]
        rows[k] = r
    summary = pd.DataFrame(rows).T
    summary.to_csv(cfg.reports / "chiptox_model_comparison.csv")
    gap = {k: overfit_report({s: dict(auroc=m[s]["macro_auroc"]) for s in ("train", "val", "test", "holdout")}, "auroc", cfg.overfit_gap_warn) for k, m in metrics.items()}
    out.update(summary=summary, gap=gap, best_name=best_name, ensemble_weights=wts, cls_params=cls_params, selection_scores=sel)
    LOG.info("\n" + summary[["train_auroc", "val_auroc", "test_auroc", "holdout_auroc", "holdout_balacc", "selection_score"]].round(3).to_string())
    # ---- CV stability (per-fold OOF macro-AUROC of the selected model)
    fold_id = np.zeros(len(pool), int)
    for f, (_, va) in enumerate(folds):
        fold_id[va] = f
    fa = [fast_macro_auc(Yp[fold_id == f], allP[best_name]["oof"][fold_id == f]) for f in range(len(folds))]
    out["cv"] = dict(fold_auroc=[round(x, 4) for x in fa], mean=float(np.nanmean(fa)), std=float(np.nanstd(fa)), folds=len(folds))
    # ---- uncertainty: disagreement between all fold-level members (+ deep-only spread)
    hm = [p for m_ in members.values() for p in (m_.hold_members or [m_.holdout])]
    member_std = np.std(np.stack(hm), 0)
    out["uncertainty_holdout"] = dict(member_std_mean=float(member_std.mean()), n_members=len(hm))
    safe(lambda: _tox_plots(metrics, best_name, allP, Yp, Yte, Yho, is_val, tasks, gap, member_std, thr_by, cfg), "chiptox plots")
    ptt = bm["holdout"]["per_task"]
    out["poor_tasks"] = poor_areas(ptt, "auroc", 0.65)
    out["holdout_per_task"] = ptt
    ptt.to_csv(cfg.reports / "chiptox_holdout_per_task.csv", index=False)
    # ---- persist: preprocessing, thresholds, blend weights, all deep fold models (scorer = deep ensemble)
    joblib.dump(dict(prep=prep, task_names=tasks, best=best_name, thresholds=thr_by[best_name], weights=wts, cls_params=cls_params, feature_names=cnames, fp_bits=cfg.fp_bits,
                     deep_models=list(deep_states)), cfg.models / "chiptox_best_bundle.joblib")
    import torch
    torch.save(deep_states, cfg.models / "chiptox_deep_best.pt")
    P = allP[best_name]["holdout"]
    hold = df.iloc[ho][["ik14", "std_smiles", "scaffold"]].reset_index(drop=True).copy()
    hold["tox_score"] = _composite(P, tasks)
    hold["tox_uncertainty"] = member_std.mean(1)
    for a in SCORE_ASSAYS:
        if a in tasks:
            hold[f"p:{a}"] = P[:, tasks.index(a)]
    hold["mc_std"] = member_std.mean(1)
    for j, a in enumerate(tasks):
        hold[f"y:{a}"] = Yho[:, j]
    hold.to_csv(cfg.reports / "chiptox_holdout_scores.csv", index=False)
    deep_first = {k: v[0].to(M.DEV) if False else v[0] for k, v in deep_nets.items()}
    out.update(holdout_scores=hold, tasks=tasks, splits=splits, df=df, Y=Y, Xc=dict(pool=Xp, test=Xte, holdout=Xho), prep=prep, deep=deep_first, allp=allP, cnames=cnames)
    # ---- multimodal (chemistry x morphology): only meaningful if every split has enough overlapping compounds
    if morph is not None:
        try:
            out["multimodal"] = _multimodal(df, Y, tasks, Xc_raw, cnames, splits, morph, cfg, out)
        except Exception as e:
            LOG.warning(f"[chiptox] multimodal stage failed: {e!r}")
            out["multimodal"] = dict(status=f"failed: {e!r}")
    else:
        out["multimodal"] = dict(status="skipped: JUMP profiles unavailable")
    force_cleanup()
    return out


def _composite(P, tasks):
    """Composite toxicity score = mean of ranks of the mechanistic stress-response / cytotoxicity heads (0..1)."""
    cols = [tasks.index(a) for a in SCORE_ASSAYS if a in tasks] or list(range(P.shape[1]))
    from scipy.stats import rankdata
    R = np.stack([rankdata(P[:, j]) / len(P) for j in cols], 1)
    return R.mean(1)


def _tox_plots(metrics, best, allP, Yp, Yte, Yho, is_val, tasks, gap, member_std, thr_by, cfg):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
    names = list(metrics)
    sp = ["train", "val", "test", "holdout"]
    fig, ax = plt.subplots(1, 3, figsize=(21, 5))
    x = np.arange(len(names))
    for i, s in enumerate(sp):
        ax[0].bar(x + i * 0.2 - 0.3, [metrics[k][s]["macro_auroc"] for k in names], 0.2, label=s)
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=35, ha="right"); ax[0].set_ylim(0.4, 1.02); ax[0].legend(); ax[0].set_title("ChipTox macro-AUROC by split (val = out-of-fold)")
    for i, s in enumerate(sp):
        ax[1].bar(x + i * 0.2 - 0.3, [metrics[k][s]["bal_acc"] for k in names], 0.2, label=s)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=35, ha="right"); ax[1].set_ylim(0.4, 1.02); ax[1].set_title("balanced accuracy (thresholds from out-of-fold predictions)")
    g1 = [gap[k]["gap_train_holdout"] for k in names]; g2 = [metrics[k]["cv_oof"]["macro_auroc"] - metrics[k]["holdout"]["macro_auroc"] for k in names]
    ax[2].bar(x - 0.2, g1, 0.4, label="train - holdout", color=["crimson" if gap[k]["overfit"] else "seagreen" for k in names]); ax[2].bar(x + 0.2, g2, 0.4, label="OOF - holdout", color="slategray")
    ax[2].axhline(cfg.overfit_gap_warn, ls="--", c="k"); ax[2].set_xticks(x); ax[2].set_xticklabels(names, rotation=35, ha="right"); ax[2].legend(); ax[2].set_title("generalisation gaps (red = overfit flag)")
    finish_plot(fig, "chiptox_model_comparison_gap", cfg)
    pt = {s: metrics[best][s]["per_task"].set_index("task")["auroc"] for s in ("val", "test", "holdout")}
    tk = list(pt["holdout"].index)
    fig, ax = plt.subplots(figsize=(max(12, len(tasks) * 0.28), 5))
    for i, s in enumerate(("val", "test", "holdout")):
        ax.bar(np.arange(len(tk)) + i * 0.27 - 0.27, [pt[s].get(t, np.nan) for t in tk], 0.27, label=s)
    ax.axhline(0.65, c="r", ls=":"); ax.set_xticks(range(len(tk))); ax.set_xticklabels([t.split(":")[-1] for t in tk], rotation=90, fontsize=7); ax.legend()
    ax.set_title(f"{best}: per-task AUROC (red line = weak-area threshold)")
    finish_plot(fig, "chiptox_per_task_auroc", cfg)
    keyt = [a for a in SCORE_ASSAYS if a in tasks][:6] or tasks[:6]
    P = allP[best]["holdout"]
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    for a in keyt:
        j = tasks.index(a); m = ~np.isnan(Yho[:, j])
        if m.sum() < 10 or len(np.unique(Yho[m, j])) < 2:
            continue
        fpr, tpr, _ = roc_curve(Yho[m, j], P[m, j]); pr, rc, _ = precision_recall_curve(Yho[m, j], P[m, j])
        ax[0].plot(fpr, tpr, label=a); ax[1].plot(rc, pr, label=a)
    ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set_title("ROC - unseen HOLDOUT (30%)"); ax[0].legend(fontsize=7); ax[1].set_title("Precision-Recall - unseen HOLDOUT"); ax[1].set_xlabel("recall")
    pp, yy = [], []
    for j in range(len(tasks)):
        m = ~np.isnan(Yho[:, j]); pp.append(P[m, j]); yy.append(Yho[m, j])
    pp, yy = np.concatenate(pp), np.concatenate(yy)
    ix = np.digitize(pp, np.linspace(0, 1, 11)) - 1
    ok = [b for b in range(10) if (ix == b).sum() > 20]
    ax[2].plot([0, 1], [0, 1], "k:"); ax[2].plot([pp[ix == b].mean() for b in ok], [yy[ix == b].mean() for b in ok], "o-")
    ax[2].set_title("reliability on holdout (pooled tasks)"); ax[2].set_xlabel("predicted"); ax[2].set_ylabel("observed")
    finish_plot(fig, "chiptox_holdout_roc_pr_calibration", cfg)
    thr = thr_by[best]; yt, yh = [], []
    for j in range(len(tasks)):
        m = ~np.isnan(Yho[:, j]); yt.append(Yho[m, j]); yh.append((P[m, j] >= thr[j]).astype(int))
    cm = confusion_matrix(np.concatenate(yt).astype(int), np.concatenate(yh))
    fig, ax = plt.subplots(figsize=(4.5, 4)); ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["pred 0", "pred 1"]); ax.set_yticklabels(["true 0", "true 1"]); ax.set_title("pooled confusion matrix, HOLDOUT")
    finish_plot(fig, "chiptox_holdout_confusion", cfg)
    err = np.abs(np.nan_to_num(Yho, nan=0.0) - P); mk = ~np.isnan(Yho)
    fig, ax = plt.subplots(figsize=(5.5, 4.5)); ax.hexbin(member_std[mk], err[mk], gridsize=40, bins="log", cmap="viridis")
    ax.set_xlabel("member disagreement (std)"); ax.set_ylabel("|y - p|"); ax.set_title("uncertainty vs error (holdout)")
    finish_plot(fig, "chiptox_uncertainty_vs_error", cfg)


# =========================================================================== multimodal (chemistry x morphology)
def _multimodal(df, Y, tasks, Xc_raw, cnames, splits, morph, cfg, out):
    mdf = morph["morph"]
    ov = df.reset_index().merge(mdf, on="ik14", how="inner")
    if len(ov) == 0:
        return dict(status="no overlap between toxicity compounds and JUMP compounds")
    ridx = ov["index"].values
    fnames = morph["feat_names"]
    Xm_raw = ov[fnames].values.astype(np.float32)
    sp_of = np.full(len(df), "", object)
    for s, i in splits.items():
        sp_of[i] = s
    sub_sp = {s: np.where(sp_of[ridx] == s)[0] for s in splits}
    info = dict(n_overlap=int(len(ov)), per_split={s: int(len(i)) for s, i in sub_sp.items()})
    LOG.info(f"[multimodal] overlapping compounds {info}")
    # tasks with enough labelled positives across the subset
    Ys = Y[ridx]
    keep = [j for j in range(len(tasks)) if all(np.nansum(Ys[sub_sp[s], j]) >= 3 and (Ys[sub_sp[s], j] == 0).sum() >= 3 for s in ("train", "val", "test", "holdout"))]
    if len(keep) == 0 or min(len(i) for i in sub_sp.values()) < 15:
        if len(ov) >= 40:
            LOG.info(f"[multimodal] overlap too small for a 4-way split ({info['per_split']}) -> repeated scaffold-grouped CV on the {len(ov)} overlapping compounds")
            return _multimodal_cv(df, Y, tasks, Xc_raw, cnames, ridx, Xm_raw, fnames, cfg, info)
        info["status"] = "insufficient overlap for any leak-free multimodal evaluation (<40 compounds)"
        return info
    mt = [tasks[j] for j in keep]
    Ys = Ys[:, keep]
    prep_c = FE.TabularPreprocessor(corr_thresh=0.98).fit(Xc_raw[ridx][sub_sp["train"]], cnames)
    prep_m = FE.TabularPreprocessor(corr_thresh=0.95, max_features=cfg.max_morph_feats).fit(Xm_raw[sub_sp["train"]], fnames)
    Xc = {s: prep_c.transform(Xc_raw[ridx][i]) for s, i in sub_sp.items()}
    Xm = {s: prep_m.transform(Xm_raw[i]) for s, i in sub_sp.items()}
    Yv = {s: Ys[i] for s, i in sub_sp.items()}
    w, _ = FE.outlier_weights(Xm["train"], seed=cfg.seed)
    res_ = {}
    # morph-only + chem-only classical (LGBM) and deep fusion
    for tag, X in (("chem_only", Xc), ("morph_only", Xm), ("concat", {s: np.hstack([Xc[s], Xm[s]]) for s in Xc})):
        p, bp, sc = _fit_all_classical(X["train"], Yv["train"], X["val"], Yv["val"], X, w, mt, cfg, names=("lgbm", "logreg"), tag=tag)
        for k, v in p.items():
            res_[f"{tag}:{k}"] = v
    m, h, best = M.fit_torch(M.FusionNet(Xc["train"].shape[1], Xm["train"].shape[1], len(mt)), [Xc["train"], Xm["train"]], Yv["train"], [Xc["val"], Xm["val"]], Yv["val"],
                             w=w, task="multilabel", epochs=cfg.deep_epochs, patience=cfg.deep_patience, batch=64, ckpt=str(cfg.models / "chiptox_fusion.pt"), desc="fusion", seed=cfg.seed, mixup=0.1)
    M.plot_history(h, "chiptox_fusion", cfg, "val macro-AUROC")
    res_["fusion:deep"] = {s: M.predict(m, [Xc[s], Xm[s]])[0] for s in Xc}
    val = {k: multilabel_metrics(Yv["val"], _fill(v["val"]), None, mt)["macro_auroc"] for k, v in res_.items()}
    res_["fusion:ensemble"] = {s: M.soft_vote({k: _fill(v[s]) for k, v in res_.items()}, {k: max(a - 0.5, 1e-3) ** 2 for k, a in val.items()}) for s in Xc}
    met = {k: {s: multilabel_metrics(Yv[s], _fill(v[s]), best_thresholds(Yv["val"], _fill(v["val"])), mt) for s in v} for k, v in res_.items()}
    table = pd.DataFrame({k: {f"{s}_auroc": m_[s]["macro_auroc"] for s in m_} for k, m_ in met.items()}).T
    table.to_csv(cfg.reports / "chiptox_multimodal_comparison.csv")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(9, 0.6 * len(table)), 4.5))
    for i, s in enumerate(("train", "val", "test", "holdout")):
        ax.bar(np.arange(len(table)) + i * 0.2 - 0.3, table[f"{s}_auroc"], 0.2, label=s)
    ax.set_xticks(range(len(table))); ax.set_xticklabels(table.index, rotation=40, ha="right", fontsize=8); ax.set_ylim(0.3, 1.02)
    ax.legend(); ax.set_title(f"Multimodal ablation (n={len(ov)} overlapping compounds): chemistry vs morphology vs fusion")
    finish_plot(fig, "chiptox_multimodal_ablation", cfg)
    info.update(status="ok", tasks=mt, table=table, best_holdout=table["holdout_auroc"].idxmax(), fusion_gap=overfit_report(
        {s: dict(auroc=met["fusion:deep"][s]["macro_auroc"]) for s in met["fusion:deep"]}, "auroc", cfg.overfit_gap_warn))
    return info


def _multimodal_cv(df, Y, tasks, Xc_raw, cnames, ridx, Xm_raw, fnames, cfg, info, reps=3, min_class=8):
    """Does morphology add information beyond chemistry? Repeated scaffold-grouped K-fold CV on the overlapping compounds.
    Leakage control: folds never split a scaffold; scaling + PCA are fitted INSIDE each training fold (sklearn Pipeline); label-free column
    filtering is done once. Reported with a paired bootstrap over tasks. With ~100 compounds this has LOW statistical power -> read the CI."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    groups = df.group.values[ridx]
    Ys = Y[ridx]
    use = [j for j in range(len(tasks)) if np.nansum(Ys[:, j]) >= min_class and (Ys[:, j] == 0).sum() >= min_class]
    if not use:
        return {**info, "status": f"grouped CV not possible: no task has >= {min_class} positives and negatives among the {len(ridx)} overlapping compounds"}
    Xc = FE.TabularPreprocessor(corr_thresh=0.98, min_freq=0.05).fit_transform(Xc_raw[ridx], cnames)      # label-free filtering / imputation
    Xm = FE.TabularPreprocessor(corr_thresh=0.95, max_features=cfg.max_morph_feats).fit_transform(Xm_raw, fnames)
    sets = {"chem_only": Xc, "morph_only": Xm, "chem+morph": np.hstack([Xc, Xm])}
    mk = lambda: make_pipeline(StandardScaler(), PCA(n_components=min(20, len(ridx) // 5), random_state=0), LogisticRegression(C=0.1, max_iter=500, class_weight="balanced"))
    auc = {k: np.full((reps, len(use)), np.nan) for k in sets}
    for r in range(reps):
        folds = group_kfold(groups, 5, cfg.seed + r)
        for ti, j in enumerate(use):
            lab = ~np.isnan(Ys[:, j]); y = np.nan_to_num(Ys[:, j]).astype(int)
            oof = {k: np.full(len(ridx), np.nan) for k in sets}
            for tr, va in folds:
                trl, val = tr[lab[tr]], va[lab[va]]
                if len(np.unique(y[trl])) < 2 or len(val) == 0:
                    continue
                for k, X in sets.items():
                    oof[k][val] = mk().fit(X[trl], y[trl]).predict_proba(X[val])[:, 1]
            for k in sets:
                m = ~np.isnan(oof[k]) & lab
                if m.sum() > 10 and len(np.unique(y[m])) == 2:
                    auc[k][r, ti] = fast_macro_auc(y[m].astype(float), oof[k][m])
    mean = {k: np.nanmean(v, 0) for k, v in auc.items()}               # per-task AUROC averaged over repeats
    rng = np.random.default_rng(cfg.seed)
    d = mean["chem+morph"] - mean["chem_only"]; d = d[np.isfinite(d)]
    boot = [rng.choice(d, len(d)).mean() for _ in range(2000)] if len(d) else [np.nan]
    table = pd.DataFrame({k: dict(mean_auroc=float(np.nanmean(v)), tasks=int(np.isfinite(v).sum())) for k, v in mean.items()}).T
    table.to_csv(cfg.reports / "chiptox_multimodal_cv.csv")
    lo, hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    verdict = "morphology adds signal" if lo > 0 else ("morphology hurts" if hi < 0 else "no detectable benefit (CI includes 0)")
    LOG.info(f"[multimodal] grouped CV on {len(ridx)} compounds, {len(use)} tasks: " + ", ".join(f"{k}={table.loc[k, 'mean_auroc']:.3f}" for k in table.index) + f" | chem+morph minus chem: {float(np.mean(boot)):+.3f} [{lo:+.3f}, {hi:+.3f}] -> {verdict}")
    return {**info, "status": "ok (repeated scaffold-grouped CV; too few overlapping compounds for a 4-way split)", "tasks_used": len(use), "table": table,
            "delta_auroc_concat_minus_chem": float(np.mean(boot)), "delta_ci95": [lo, hi], "verdict": verdict}


# =========================================================================== scoring API for the agent
class ToxScorer:
    """Score arbitrary SMILES with the saved deep fold-ensemble (mean prediction + disagreement-based uncertainty)."""

    def __init__(self, cfg, out=None):
        import torch
        self.cfg = cfg
        b = joblib.load(cfg.models / "chiptox_best_bundle.joblib")
        self.b = b
        states = torch.load(cfg.models / "chiptox_deep_best.pt", map_location="cpu")
        d_in, T = len(b["prep"].idx_), len(b["task_names"])
        self.nets = []
        for nm, sds in states.items():
            for sd in sds:
                net = DEEP[nm](d_in, T); net.load_state_dict(sd); net.eval(); self.nets.append(net)

    def score(self, smiles):
        st = chem.smiles_table(smiles, "score: standardise")
        ok = st.std_smiles.notna().values
        if not ok.any():
            LOG.warning("[scorer] none of the supplied SMILES could be parsed")
            return pd.DataFrame(columns=["smiles", "tox_score", "uncertainty"])
        Xraw, _ = featurize(st.std_smiles[ok].tolist(), self.cfg)
        X = self.b["prep"].transform(Xraw)
        preds = np.stack([M.predict(n, X)[0] for n in self.nets])
        mean, std = preds.mean(0), preds.std(0)
        res = pd.DataFrame(mean, columns=self.b["task_names"])
        res.insert(0, "smiles", st.std_smiles[ok].values)
        res["tox_score"] = _composite(mean, self.b["task_names"])
        res["uncertainty"] = std.mean(1)
        return res
