"""utils.py - memory safety, logging, plotting, strict 4-way group splits, leakage checks, metrics, overfit detection."""
import ctypes
import gc
import json
import logging
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
import psutil
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
LOG = logging.getLogger("chipscreen")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    LOG.addHandler(h)
    LOG.setLevel(logging.INFO)
    LOG.propagate = False            # notebooks attach a root handler -> every line was printed twice

PLOT_REGISTRY = []   # (name, path) of every figure written - consumed by the report


# --------------------------------------------------------------------------- reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------- memory
class MemoryLimitExceeded(RuntimeError):
    pass


def mem_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def sys_mem_frac():
    vm = psutil.virtual_memory()
    return 1.0 - vm.available / vm.total


def force_cleanup(tag=""):
    """Aggressive cleanup: python GC, matplotlib figures, CUDA cache, glibc arena trim."""
    import matplotlib.pyplot as plt
    plt.close("all")
    for _ in range(3):
        gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def log_mem(tag, cfg=None):
    msg = f"[mem] {tag}: process {mem_gb():.2f} GB | system {sys_mem_frac()*100:.0f}% used"
    try:
        import torch
        if torch.cuda.is_available():
            msg += f" | gpu alloc {torch.cuda.memory_allocated()/1e9:.2f} GB"
    except Exception:
        pass
    LOG.info(msg)


def check_memory(cfg, tag=""):
    """Raise MemoryLimitExceeded before the kernel is killed."""
    if sys_mem_frac() > cfg.mem_limit_frac:
        force_cleanup(tag)
        if sys_mem_frac() > cfg.mem_limit_frac:
            raise MemoryLimitExceeded(f"system memory {sys_mem_frac()*100:.0f}% > limit at '{tag}'")


def tq(it, desc, total=None, **kw):
    return tqdm(it, desc=desc, total=total, mininterval=0.5, dynamic_ncols=True, **kw)


def downcast_df(df):
    """Memory-efficient dtypes."""
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_float_dtype(s):
            df[c] = s.astype(np.float32)
        elif pd.api.types.is_integer_dtype(s):
            df[c] = pd.to_numeric(s, downcast="integer")
        elif s.dtype == object and s.nunique(dropna=True) < 0.5 * max(len(s), 1):
            df[c] = s.astype("category")
    return df


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        LOG.info(f"=== {self.name} ===")
        return self

    def __exit__(self, *a):
        LOG.info(f"=== {self.name} done in {time.time()-self.t:.1f}s ===")


# --------------------------------------------------------------------------- plotting
def finish_plot(fig, name, cfg):
    """plt.savefig (always) + plt.show (configurable)."""
    import matplotlib.pyplot as plt
    path = cfg.plots / f"{name}.png"
    fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    PLOT_REGISTRY.append((name, str(path)))
    if cfg.show_plots:
        plt.show()
    plt.close(fig)
    return str(path)


def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, tuple)):
            return list(o)
        return str(o)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)


# --------------------------------------------------------------------------- strict group-aware 4-way split
SPLIT_NAMES = ("train", "val", "test", "holdout")


def group_split(groups, fracs=(0.40, 0.15, 0.15, 0.30), y=None, seed=42):
    """Assign whole groups (scaffolds / volumes / plates / images) to train/val/test/holdout.
    Big groups first, greedy by remaining deficit; optional single-label class balancing.
    Returns dict name -> sorted index array."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    n = len(groups)
    ug, inv = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inv)
    order = np.lexsort((rng.random(len(ug)), -sizes))
    fr = np.asarray(fracs, float)
    target = fr * n
    cur = np.zeros(4)
    if y is not None:
        y = np.asarray(y)
        classes, yi = np.unique(y, return_inverse=True)
        K = len(classes)
        g_cls = np.zeros((len(ug), K))
        np.add.at(g_cls, (inv, yi), 1)
        tgt_cls = fr[:, None] * g_cls.sum(0)[None, :]
        cur_cls = np.zeros((4, K))
    assign = np.zeros(len(ug), int)
    for gi in order:
        deficit = (target - cur) / np.maximum(target, 1)
        score = deficit.copy()
        if y is not None:
            cdef = (tgt_cls - cur_cls) / np.maximum(tgt_cls, 1)
            score = score + (cdef * g_cls[gi][None, :]).sum(1) / max(g_cls[gi].sum(), 1)
        s = int(np.argmax(score + 1e-9 * rng.random(4)))
        assign[gi] = s
        cur[s] += sizes[gi]
        if y is not None:
            cur_cls[s] += g_cls[gi]
    # local refinement: move whole groups between splits while the total deviation from the target sizes (and class targets) shrinks
    sz = sizes.astype(float)
    cur = np.bincount(assign, weights=sz, minlength=4)
    for _ in range(4):
        moved = False
        for gi in order:
            a = assign[gi]
            best_b, best_gain = a, 0.0
            for b in range(4):
                if b == a:
                    continue
                new_cur = cur.copy(); new_cur[a] -= sz[gi]; new_cur[b] += sz[gi]
                gain = np.abs(cur - target).sum() - np.abs(new_cur - target).sum()
                if y is not None:
                    n_cls = cur_cls.copy(); n_cls[a] -= g_cls[gi]; n_cls[b] += g_cls[gi]
                    gain += 0.5 * (np.abs(cur_cls - tgt_cls).sum() - np.abs(n_cls - tgt_cls).sum())
                if gain > best_gain + 1e-9:
                    best_b, best_gain = b, gain
            if best_b != a:
                cur[a] -= sz[gi]; cur[best_b] += sz[gi]
                if y is not None:
                    cur_cls[a] -= g_cls[gi]; cur_cls[best_b] += g_cls[gi]
                assign[gi] = best_b; moved = True
        if not moved:
            break
    row_split = assign[inv]
    return {nm: np.where(row_split == i)[0] for i, nm in enumerate(SPLIT_NAMES)}


def safe(fn, name, default=None):
    """Run a non-essential sub-step (plot, EDA, CV, ...); log and continue instead of killing the whole stage."""
    try:
        return fn()
    except Exception as e:
        import traceback
        LOG.warning(f"[safe] sub-step '{name}' failed and was skipped: {e!r}\n{traceback.format_exc(limit=3)}")
        try:
            import matplotlib.pyplot as plt; plt.close('all')
        except Exception:
            pass
        return default


def verify_data_splits(splits, groups=None, n_total=None, verbose=True):
    """Hard checks: no index overlap, full coverage, no group shared across splits. Raises on leakage."""
    names = list(splits)
    all_idx = np.concatenate([splits[k] for k in names])
    if len(np.unique(all_idx)) != len(all_idx):
        raise AssertionError("DATA LEAK: an index appears in more than one split")
    if n_total is not None and len(all_idx) != n_total:
        raise AssertionError(f"split coverage mismatch: {len(all_idx)} != {n_total}")
    rep = {"sizes": {k: int(len(v)) for k, v in splits.items()}}
    tot = max(len(all_idx), 1)
    rep["fractions"] = {k: round(len(v) / tot, 4) for k, v in splits.items()}
    if groups is not None:
        groups = np.asarray(groups)
        gsets = {k: set(groups[v].tolist()) for k, v in splits.items()}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                inter = gsets[a] & gsets[b]
                if inter:
                    raise AssertionError(f"GROUP LEAK between {a} and {b}: {len(inter)} shared groups")
        rep["groups"] = {k: len(v) for k, v in gsets.items()}
    rep["holdout_isolated"] = True
    if verbose:
        LOG.info(f"[splits] verified: sizes={rep['sizes']} fractions={rep['fractions']} (no index/group leakage)")
    return rep


def check_domain_relevance(X, splits, y=None, max_feats=40, seed=0):
    """Distribution-shift screen between train and every other split (KS test per feature + label prevalence)."""
    from scipy.stats import ks_2samp
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    cols = rng.choice(X.shape[1], size=min(max_feats, X.shape[1]), replace=False)
    out = {}
    for k in ("val", "test", "holdout"):
        if k not in splits or len(splits[k]) < 5 or len(splits["train"]) < 5:
            continue
        a = X[splits["train"]][:, cols]
        b = X[splits[k]][:, cols]
        pv = np.array([ks_2samp(a[:, j], b[:, j]).pvalue for j in range(len(cols))])
        out[k] = {"frac_features_shifted_p<0.01": float((pv < 0.01).mean())}
        if y is not None:
            yy = np.asarray(y, float)
            out[k]["label_prevalence"] = float(np.nanmean(yy[splits[k]]))
            out[k]["train_prevalence"] = float(np.nanmean(yy[splits["train"]]))
    for k, v in out.items():
        if v["frac_features_shifted_p<0.01"] > 0.5:
            LOG.warning(f"[domain] {k} differs strongly from train ({v['frac_features_shifted_p<0.01']:.0%} features shifted)")
    return out


# --------------------------------------------------------------------------- metrics
def best_thresholds(Y, P):
    """Per-task Youden-J threshold chosen on VALIDATION data only."""
    from sklearn.metrics import roc_curve
    thr = np.full(Y.shape[1], 0.5)
    for j in range(Y.shape[1]):
        m = ~np.isnan(Y[:, j])
        if m.sum() < 10 or len(np.unique(Y[m, j])) < 2:
            continue
        fpr, tpr, t = roc_curve(Y[m, j], P[m, j])
        thr[j] = float(t[np.argmax(tpr - fpr)])
        if not np.isfinite(thr[j]):
            thr[j] = 0.5
    return thr


def multilabel_metrics(Y, P, thr=None, task_names=None):
    """Macro/pooled metrics for (n, T) labels with NaN = unlabeled. Accuracy is reported but AUROC/PR-AUC/balanced
    accuracy are the headline numbers because tox labels are heavily imbalanced (raw accuracy is inflated)."""
    from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score, accuracy_score,
                                 f1_score, matthews_corrcoef)
    Y = np.asarray(Y, float)
    P = np.asarray(P, float)
    if Y.ndim == 1:
        Y, P = Y[:, None], P[:, None]
    thr = np.full(Y.shape[1], 0.5) if thr is None else thr
    rows = []
    for j in range(Y.shape[1]):
        m = ~np.isnan(Y[:, j])
        if m.sum() < 5 or len(np.unique(Y[m, j])) < 2:
            continue
        yt, pj = Y[m, j].astype(int), P[m, j]
        yh = (pj >= thr[j]).astype(int)
        rows.append(dict(task=(task_names[j] if task_names else j), n=int(m.sum()), pos=int(yt.sum()),
                         auroc=roc_auc_score(yt, pj), ap=average_precision_score(yt, pj),
                         bal_acc=balanced_accuracy_score(yt, yh), acc=accuracy_score(yt, yh),
                         f1=f1_score(yt, yh, zero_division=0), mcc=matthews_corrcoef(yt, yh)))
    df = pd.DataFrame(rows)
    if df.empty:
        return dict(macro_auroc=np.nan, macro_ap=np.nan, bal_acc=np.nan, acc=np.nan, f1=np.nan, mcc=np.nan, n_tasks=0, per_task=df)
    return dict(macro_auroc=df.auroc.mean(), macro_ap=df.ap.mean(), bal_acc=df.bal_acc.mean(), acc=df.acc.mean(),
                f1=df.f1.mean(), mcc=df.mcc.mean(), n_tasks=len(df), per_task=df)


def multiclass_metrics(y, proba, classes=None):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, top_k_accuracy_score
    y = np.asarray(y)
    pred = proba.argmax(1)
    out = dict(acc=accuracy_score(y, pred), bal_acc=balanced_accuracy_score(y, pred),
               macro_f1=f1_score(y, pred, average="macro", zero_division=0))
    try:
        if proba.shape[1] >= 3:
            out["top3"] = top_k_accuracy_score(y, proba, k=3, labels=np.arange(proba.shape[1]))
    except Exception:
        pass
    return out


def overfit_report(metrics_by_split, key, warn=0.05):
    """Generalisation-gap analysis. metrics_by_split: {'train': {...}, 'val': ..., 'test': ..., 'holdout': ...}."""
    g = lambda s: metrics_by_split.get(s, {}).get(key, np.nan)
    rep = dict(metric=key, train=g("train"), val=g("val"), test=g("test"), holdout=g("holdout"))
    rep["gap_train_val"] = rep["train"] - rep["val"]
    rep["gap_train_holdout"] = rep["train"] - rep["holdout"]
    rep["gap_val_holdout"] = rep["val"] - rep["holdout"]
    rep["gap_test_holdout"] = rep["test"] - rep["holdout"]
    recs = []
    if rep["gap_train_holdout"] > warn:
        rep["overfit"] = True
        recs += ["Raise Dropout / weight decay / label smoothing; add augmentation or mixup",
                 "Reduce model capacity or tighten early-stopping patience",
                 "Prefer simpler ensemble members (bagging) and verify scaffold/group split is respected"]
    else:
        rep["overfit"] = False
    if abs(rep["gap_val_holdout"]) > warn:
        recs.append("Validation optimism: val-tuned thresholds/hyper-parameters do not transfer to holdout; widen search regularisation")
    if np.isfinite(rep["train"]) and rep["train"] < 0.6:
        recs.append("Under-fitting: add informative features, capacity or training epochs")
    rep["recommendations"] = recs or ["No significant generalisation gap detected"]
    return rep


def poor_areas(per_task_df, col="auroc", thresh=0.65, label="task"):
    """Return the tasks/classes the model performs poorly on, for the 'identify weak areas' requirement."""
    if per_task_df is None or len(per_task_df) == 0:
        return []
    bad = per_task_df[per_task_df[col] < thresh]
    return [dict(item=str(r["task"]), **{col: float(r[col]), "n": int(r.get("n", 0)), "pos": int(r.get("pos", 0))})
            for _, r in bad.iterrows()]


# --------------------------------------------------------------------------- parallel I/O (network drives such as Google Drive are latency-bound)
def parallel_map(fn, items, workers=16, desc=""):
    """Thread-pool map with progress bar. Exceptions are RETURNED (as exception objects) instead of raised, so one bad file never kills a stage."""
    from concurrent.futures import ThreadPoolExecutor
    items = list(items)

    def wrap(x):
        try:
            return fn(x)
        except Exception as e:  # noqa: BLE001
            return e
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return list(tqdm(ex.map(wrap, items), total=len(items), desc=desc, mininterval=0.5, dynamic_ncols=True))


# --------------------------------------------------------------------------- CV folds / ensembling
def group_kfold(groups, k=5, seed=0):
    """Group-aware K-fold (whole scaffolds / volumes stay together). Greedy balancing by group size. -> list of (train_idx, valid_idx)."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    ug, inv = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inv)
    order = np.lexsort((rng.random(len(ug)), -sizes))
    fold_of = np.zeros(len(ug), int)
    load = np.zeros(k)
    for g in order:
        f = int(np.argmin(load + 1e-6 * rng.random(k)))
        fold_of[g] = f
        load[f] += sizes[g]
    row_fold = fold_of[inv]
    return [(np.where(row_fold != f)[0], np.where(row_fold == f)[0]) for f in range(k)]


def fast_macro_auc(Y, P):
    """Mean per-task AUROC (Mann-Whitney) ignoring NaN labels; tasks with a single class are skipped."""
    from scipy.stats import rankdata
    Y, P = np.asarray(Y, float), np.asarray(P, float)
    if Y.ndim == 1:
        Y, P = Y[:, None], P[:, None]
    out = []
    for j in range(Y.shape[1]):
        m = ~np.isnan(Y[:, j])
        y = Y[m, j]
        npos = y.sum()
        nneg = len(y) - npos
        if npos < 2 or nneg < 2:
            continue
        r = rankdata(P[m, j])
        out.append((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    return float(np.mean(out)) if out else float("nan")


def greedy_ensemble(preds, Y, iters=40, seed=0):
    """Caruana-style forward selection with replacement on OUT-OF-FOLD predictions (maximises macro-AUROC). -> dict name -> weight."""
    names = list(preds)
    cnt = {n: 0 for n in names}
    cur, k = None, 0
    best_score = -1
    for _ in range(iters):
        cand = []
        for n in names:
            trial = preds[n] if cur is None else (cur * k + preds[n]) / (k + 1)
            cand.append((fast_macro_auc(Y, trial), n))
        sc, n = max(cand)
        if sc <= best_score - 1e-6 and k >= 5:
            break
        best_score = max(best_score, sc)
        cur = preds[n] if cur is None else (cur * k + preds[n]) / (k + 1)
        k += 1
        cnt[n] += 1
    tot = sum(cnt.values())
    return {n: c / tot for n, c in cnt.items() if c > 0}


# --------------------------------------------------------------------------- secrets (never printed or written to disk)
def get_secret(name):
    """Environment variable -> Kaggle Secrets (UserSecretsClient) -> Colab userdata. Returns None when not found."""
    v = os.environ.get(name)
    if v:
        return v.strip()
    try:
        from kaggle_secrets import UserSecretsClient
        v = UserSecretsClient().get_secret(name)
        if v:
            return v.strip()
    except Exception:
        pass
    try:
        from google.colab import userdata
        v = userdata.get(name)
        if v:
            return v.strip()
    except Exception:
        pass
    return None
