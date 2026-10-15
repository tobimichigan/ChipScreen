"""dosecurve.py - DoseCurve QC Workbench (Tool & Platform).
 (1) plate QC on JUMP well activity: Z'-factor / SSMD from real negative/positive controls, edge effects, row/column drift
 (2) dose-response on BBBC021 (nuclei counted from the real DAPI images): 4-parameter logistic fit, bootstrap CI,
     curve-quality flags, replicate CV, and a learned curve-quality classifier checked on held-out compounds.
 Dose tables with metadata only (IDR idr0035/0036 etc.) are reported as 'no readout' instead of being invented."""
import os

import numpy as np
import pandas as pd

from utils import LOG, Timer, tq, finish_plot, group_split, verify_data_splits, check_memory, force_cleanup, save_json


# =========================================================================== plate QC
def zprime(neg, pos):
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    if len(neg) < 3 or len(pos) < 3:
        return np.nan
    return 1 - 3 * (neg.std(ddof=1) + pos.std(ddof=1)) / (abs(pos.mean() - neg.mean()) + 1e-9)


def ssmd(neg, pos):
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    if len(neg) < 3 or len(pos) < 3:
        return np.nan
    return (pos.mean() - neg.mean()) / np.sqrt(pos.var(ddof=1) + neg.var(ddof=1) + 1e-12)


def _mad_sd(x):
    x = np.asarray(x, float)
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def plate_qc(well):
    """Plate QC on the per-well phenotypic activity (RMS robust-z of the whole Cell Painting profile).
    Design notes (why not one pooled Z'): JUMP plates hold SEVERAL different positive-control compounds with different phenotype
    strengths, so pooling them into one distribution makes Z' meaningless (it gave median -3 on the real plates). Instead:
      * separation is evaluated PER positive-control compound (robust median / MAD) and summarised by the median and the best value
      * detectability = fraction of positive-control wells above the negative-control 95th percentile
      * edge / drift tests use the NEGATIVE-control wells (spread over the plate by design); compound wells carry layout structure
        and would produce false 'drift'.
    A plate FAILS only for: no detectable positive control, noisy negative controls, strong edge effect, or strong drift."""
    from scipy.stats import mannwhitneyu, spearmanr
    rows = []
    gmed = well[well.ctrl == "negcon"].groupby("plate").activity.median()
    g_mu, g_sd = gmed.median(), _mad_sd(gmed.values)
    for plate, g in well.groupby("plate"):
        neg, pos, comp = g[g.ctrl == "negcon"], g[g.ctrl == "poscon"], g[g.ctrl == "compound"]
        mu_n, sd_n = float(np.median(neg.activity)) if len(neg) else np.nan, _mad_sd(neg.activity) if len(neg) > 2 else np.nan
        sd_floor = max(sd_n if np.isfinite(sd_n) else 0.0, 0.02 * abs(mu_n) if np.isfinite(mu_n) else 0.0, 1e-6)
        zs, ss, per_pos = [], [], {}
        for jcp, gp in pos.groupby("jcp"):
            if len(gp) < 2 or not np.isfinite(mu_n):
                continue
            mu_p, sd_p = float(np.median(gp.activity)), max(_mad_sd(gp.activity), 0.02 * abs(mu_n))
            zs.append(1 - 3 * (sd_floor + sd_p) / (abs(mu_p - mu_n) + 1e-9)); ss.append(abs(mu_p - mu_n) / np.sqrt(sd_floor ** 2 + sd_p ** 2))
            per_pos[jcp] = zs[-1]
        detect = float((pos.activity > (np.percentile(neg.activity, 95) if len(neg) > 5 else mu_n + 3 * sd_floor)).mean()) if len(pos) and len(neg) else np.nan
        # edge / drift on negative controls (fall back to compound wells only when no/few negatives)
        ref = neg if len(neg) >= 12 else comp
        edge = ref[(ref.row.isin([0, g.row.max()])) | (ref.col.isin([1, g.col.max()]))]
        inner = ref.drop(edge.index)
        edge_p = mannwhitneyu(edge.activity, inner.activity).pvalue if len(edge) > 4 and len(inner) > 4 else np.nan
        edge_rel = (edge.activity.median() - inner.activity.median()) / (abs(inner.activity.median()) + 1e-9) if len(edge) > 4 and len(inner) > 4 else np.nan
        rc = spearmanr(ref.col, ref.activity) if len(ref) > 12 else (np.nan, np.nan)
        rr = spearmanr(ref.row, ref.activity) if len(ref) > 12 else (np.nan, np.nan)
        plate_shift = (mu_n - g_mu) / (g_sd + 1e-9) if np.isfinite(mu_n) and g_sd > 0 else np.nan
        neg_cv = sd_n / (abs(mu_n) + 1e-9) if np.isfinite(sd_n) else np.nan
        fails, notes = [], []
        if len(pos) >= 3 and np.isfinite(detect) and detect < 0.3:
            fails.append("positive controls not detectable")
        elif len(pos) < 3:
            notes.append("few/no positive controls")
        if np.isfinite(neg_cv) and neg_cv > 0.35:
            fails.append("noisy negative controls")
        if np.isfinite(edge_p) and edge_p < 0.001 and abs(edge_rel) > 0.15:
            fails.append("edge effect")
        if np.isfinite(rc[1]) and rc[1] < 0.001 and abs(rc[0]) > 0.35:
            fails.append("column drift")
        if np.isfinite(rr[1]) and rr[1] < 0.001 and abs(rr[0]) > 0.35:
            fails.append("row drift")
        if np.isfinite(plate_shift) and abs(plate_shift) > 4:
            notes.append("plate-level shift vs other plates (batch effect)")
        zp = float(np.median(zs)) if zs else np.nan
        if np.isfinite(zp) and zp < 0 and not fails:
            notes.append("weak median Z' (heterogeneous positive controls)")
        rows.append(dict(plate=plate, n_neg=len(neg), n_pos=len(pos), n_comp=len(comp), zprime=zp, zprime_best=(max(zs) if zs else np.nan), ssmd=(float(np.median(ss)) if ss else np.nan),
                         pos_detect=detect, neg_cv=neg_cv, edge_p=edge_p, edge_rel=edge_rel, col_trend=rc[0], row_trend=rr[0], plate_shift_z=plate_shift,
                         qc_flags="; ".join(fails + notes) or "PASS", fail_reasons="; ".join(fails), passed=len(fails) == 0))
    return pd.DataFrame(rows)


def plate_heatmaps(well, cfg, qc):
    import matplotlib.pyplot as plt
    plates = qc.sort_values("pos_detect").plate.tolist()
    pick = (plates[:2] + plates[-2:]) if len(plates) >= 4 else plates
    fig, ax = plt.subplots(1, len(pick), figsize=(5.2 * len(pick), 4))
    ax = np.atleast_1d(ax)
    for a, p in zip(ax, pick):
        g = well[well.plate == p]
        H = np.full((g.row.max() + 1, g.col.max()), np.nan)
        H[g.row.values, g.col.values - 1] = g.activity.values
        im = a.imshow(H, cmap="magma", vmin=np.nanpercentile(H, 2), vmax=np.nanpercentile(H, 98))
        z = float(qc[qc.plate == p].pos_detect.iloc[0])
        a.set_title(f"{p}  pos-detect={z:.2f}", fontsize=9); plt.colorbar(im, ax=a, fraction=0.04)
    fig.suptitle("Plate activity maps (worst-2 and best-2 positive-control detectability)")
    finish_plot(fig, "dosecurve_plate_heatmaps", cfg)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].hist(qc.pos_detect.dropna(), bins=20); ax[0].axvline(0.3, c="r", ls="--"); ax[0].set_title("positive-control detectability across plates")
    ax[1].scatter(qc.col_trend, qc.row_trend, c=qc.passed.map({True: "tab:green", False: "tab:red"})); ax[1].set_xlabel("column trend"); ax[1].set_ylabel("row trend"); ax[1].set_title("drift (Spearman)")
    qc.qc_flags.str.split("; ").explode().value_counts().plot.barh(ax=ax[2]); ax[2].set_title("QC flags / notes")
    finish_plot(fig, "dosecurve_plate_qc_summary", cfg)


# =========================================================================== 4PL dose response
def four_pl(x, bottom, top, logec50, hill):
    return bottom + (top - bottom) / (1 + 10 ** ((logec50 - np.log10(x)) * hill))


def fit_4pl(dose, resp, n_boot=60, seed=0):
    """Robust 4PL with multi-start + bootstrap CI. Returns dict with parameters, R2, CI, and quality flags."""
    from scipy.optimize import curve_fit
    d, r = np.asarray(dose, float), np.asarray(resp, float)
    m = (d > 0) & np.isfinite(r)
    d, r = d[m], r[m]
    out = dict(n_points=len(d), n_doses=len(np.unique(d)))
    if out["n_doses"] < 4:
        return {**out, "fit_ok": False, "qc_flags": "too few doses"}
    lo, hi = np.percentile(r, 5), np.percentile(r, 95)
    ld = np.log10(d)
    best = None
    for hill0 in (1.0, -1.0, 2.0, -2.0):
        for ec0 in np.linspace(ld.min(), ld.max(), 4):
            p0 = [lo, hi, ec0, hill0]
            try:
                p, cov = curve_fit(lambda x, b, t, e, h: four_pl(x, b, t, e, h), d, r, p0=p0,
                                   bounds=([r.min() - abs(np.ptp(r)), r.min() - abs(np.ptp(r)), ld.min() - 2, -6], [r.max() + abs(np.ptp(r)), r.max() + abs(np.ptp(r)), ld.max() + 2, 6]), maxfev=4000)
                sse = float(((four_pl(d, *p) - r) ** 2).sum())
                if best is None or sse < best[0]:
                    best = (sse, p, cov)
            except Exception:
                continue
    if best is None:
        return {**out, "fit_ok": False, "qc_flags": "no convergence"}
    sse, p, cov = best
    sst = float(((r - r.mean()) ** 2).sum()) + 1e-12
    r2 = 1 - sse / sst
    rng = np.random.default_rng(seed)
    ecs, hs = [], []
    pred = four_pl(d, *p)
    resid = r - pred
    for _ in range(n_boot):
        rb = pred + rng.choice(resid, len(resid))
        try:
            pb, _ = curve_fit(lambda x, b, t, e, h: four_pl(x, b, t, e, h), d, rb, p0=p, maxfev=2000)
            ecs.append(pb[2]); hs.append(pb[3])
        except Exception:
            pass
    ci = (np.percentile(ecs, [2.5, 97.5]) if len(ecs) > 10 else [np.nan, np.nan])
    span = p[1] - p[0]
    flags = []
    if r2 < 0.8: flags.append("poor fit R2<0.8")
    if p[2] < ld.min() - 0.5 or p[2] > ld.max() + 0.5: flags.append("EC50 outside tested range (extrapolated)")
    if abs(span) < 2 * (resid.std() + 1e-9): flags.append("span within noise (no effect)")
    if abs(p[3]) > 4: flags.append("implausible Hill slope")
    if np.isfinite(ci[0]) and (ci[1] - ci[0]) > 1.5: flags.append("wide EC50 CI (>1.5 log)")
    return {**out, "fit_ok": True, "bottom": p[0], "top": p[1], "logEC50": p[2], "hill": p[3], "span": span, "r2": r2,
            "logEC50_lo": ci[0], "logEC50_hi": ci[1], "ci_width": (ci[1] - ci[0]) if np.isfinite(ci[0]) else np.nan,
            "resid_sd": resid.std(), "qc_flags": "; ".join(flags) or "PASS"}


def _count_nuclei(path):
    """Real image analysis: Gaussian smooth -> Otsu -> distance-transform watershed -> object count."""
    import tifffile
    from scipy import ndimage as ndi
    from skimage import filters, measure, morphology, segmentation, feature
    im = tifffile.imread(path).astype(np.float32)
    if im.ndim > 2:
        im = im.reshape(-1, *im.shape[-2:])[0]
    im = filters.gaussian(im, 1.0)
    # subtract large-scale background
    bg = filters.gaussian(im, 25)
    im = np.clip(im - bg, 0, None)
    if im.max() <= 0:
        return 0, 0.0
    t = filters.threshold_otsu(im)
    mask = morphology.remove_small_objects(im > t, 20)
    mask = ndi.binary_fill_holes(mask)
    dist = ndi.distance_transform_edt(mask)
    pk = feature.peak_local_max(dist, min_distance=5, labels=measure.label(mask))
    mk = np.zeros_like(mask, int); mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    lab = segmentation.watershed(-dist, mk, mask=mask)
    return int(lab.max()), float(mask.mean())


def build_bbbc021_curves(res, cfg, max_images=2400):
    imgcsv = (res.bbbc("BBBC021", r"image\.csv$") or [None])[0]
    if imgcsv is None:
        return None
    meta = pd.read_csv(imgcsv)
    meta["Image_Metadata_Concentration"] = pd.to_numeric(meta["Image_Metadata_Concentration"], errors="coerce")
    avail = {os.path.basename(p): p for p in res.cat[(res.cat.rel.str.contains("BBBC021")) & (res.cat.ext.isin([".tif", ".tiff"]))].path}
    if not avail:
        LOG.warning("[dosecurve] BBBC021 image CSV found but raw DAPI TIFFs were not downloaded (BBBC_INCLUDE_IMAGES=False) -> dose-response from images unavailable")
        return dict(status="no images", meta_rows=len(meta))
    meta["path"] = meta["Image_FileName_DAPI"].map(avail)
    meta = meta.dropna(subset=["path"])
    if len(meta) > max_images:   # balanced subsample over compound x dose, keep both replicates
        meta = meta.groupby(["Image_Metadata_Compound", "Image_Metadata_Concentration"], group_keys=False).apply(lambda g: g.sample(min(len(g), 4), random_state=cfg.seed)).head(max_images)
    cache = cfg.cache / "bbbc021_counts.parquet"
    if cache.exists():
        cnt = pd.read_parquet(cache)
    else:
        rows = []
        for r in tq(list(meta.itertuples()), "BBBC021: counting nuclei"):
            try:
                n, fa = _count_nuclei(r.path)
            except Exception:
                continue
            rows.append((r.ImageNumber, r.Image_Metadata_Compound, r.Image_Metadata_Concentration, r.Image_Metadata_Plate_DAPI, r.Image_Metadata_Well_DAPI, r.Replicate, n, fa))
            if len(rows) % 200 == 0:
                check_memory(cfg, "bbbc021")
        cnt = pd.DataFrame(rows, columns=["image", "compound", "conc", "plate", "well", "replicate", "nuclei", "foreground"])
        cnt.to_parquet(cache)
    return dict(status="ok", counts=cnt)


def run_dosecurve(res, cfg, morph=None):
    out = {}
    with Timer("DoseCurve QC Workbench"):
        # ---- A) plate QC from JUMP controls
        if morph is not None and len(morph["well"]):
            qc = plate_qc(morph["well"])
            qc.to_csv(cfg.reports / "dosecurve_plate_qc.csv", index=False)
            plate_heatmaps(morph["well"], cfg, qc)
            out["plate_qc"] = qc
            LOG.info(f"[dosecurve] {len(qc)} plates: {int(qc.passed.sum())} pass | median per-control Z'={qc.zprime.median():.2f}, best-control Z'={qc.zprime_best.median():.2f} | median pos-detect={qc.pos_detect.median():.2f}\n{qc.qc_flags.str.split('; ').explode().value_counts().to_string()}")
        else:
            out["plate_qc"] = None
        # ---- B) dose-response
        bb = build_bbbc021_curves(res, cfg)
        out["bbbc021_status"] = (bb or {}).get("status", "missing")
        if bb and bb.get("status") == "ok":
            cnt = bb["counts"]
            # normalise to the plate DMSO median (% of control)
            ctrl = cnt[cnt.compound.str.upper() == "DMSO"].groupby("plate").nuclei.median()
            gmed = float(cnt[cnt.compound.str.upper() == "DMSO"].nuclei.median() or cnt.nuclei.median())
            cnt["pct_ctrl"] = 100 * cnt.nuclei / cnt.plate.map(ctrl).fillna(gmed).clip(lower=1)
            fits, curves = [], {}
            comps = [c for c in cnt.compound.unique() if str(c).upper() != "DMSO"]
            for c in tq(comps, "fitting 4PL curves"):
                g = cnt[cnt.compound == c]
                f = fit_4pl(g.conc.values, g.pct_ctrl.values, seed=cfg.seed)
                rep_cv = g.groupby("conc").pct_ctrl.agg(lambda s: s.std() / (s.mean() + 1e-9)).mean()
                fits.append(dict(compound=c, rep_cv=rep_cv, **f)); curves[c] = g
            fit_df = pd.DataFrame(fits)
            fit_df.to_csv(cfg.reports / "dosecurve_fits.csv", index=False)
            out["fits"] = fit_df
            out["curve_flags"] = fit_df["qc_flags"].str.split("; ").explode().value_counts().to_dict()
            _dose_plots(fit_df, curves, cfg)
            out["quality_model"] = _curve_quality_model(fit_df, cfg)
        elif bb:
            out["note"] = "BBBC021 raw images absent: only the metadata table was downloaded. Set BBBC_INCLUDE_IMAGES=True in the downloader to enable image-derived curves."
        # ---- C) dose tables that carry no readout (reported, not fabricated)
        from catalog import discover_dose_tables
        found, meta_only = discover_dose_tables(res.cat, 120)
        out["dose_tables_with_readout"] = [f["rel"] for f in found]
        out["dose_tables_metadata_only"] = [m["rel"] for m in meta_only]
        save_json(dict(with_readout=found[:40], metadata_only=meta_only[:60]), cfg.reports / "dosecurve_dose_table_inventory.json")
    force_cleanup()
    return out


def _dose_plots(fit_df, curves, cfg):
    import matplotlib.pyplot as plt
    ok = fit_df[fit_df.fit_ok == True].sort_values("r2", ascending=False)
    pick = list(ok.head(4).compound) + list(ok.tail(2).compound)
    if not pick:
        return
    fig, ax = plt.subplots(2, 3, figsize=(16, 8)); ax = ax.ravel()
    for a, c in zip(ax, pick):
        g = curves[c]; r = fit_df[fit_df.compound == c].iloc[0]
        a.scatter(g.conc, g.pct_ctrl, s=14, alpha=0.7)
        xs = np.logspace(np.log10(g.conc.min()), np.log10(g.conc.max()), 120)
        a.plot(xs, four_pl(xs, r.bottom, r.top, r.logEC50, r.hill), c="r")
        a.axvspan(10 ** r.logEC50_lo, 10 ** r.logEC50_hi, color="r", alpha=0.12) if np.isfinite(r.logEC50_lo) else None
        a.set_xscale("log"); a.set_title(f"{c}  R2={r.r2:.2f}  {str(r.qc_flags)[:38]}", fontsize=8); a.set_xlabel("conc (uM)"); a.set_ylabel("% DMSO nuclei")
    fig.suptitle("DoseCurve: best/worst 4PL fits (band = bootstrap EC50 95% CI)")
    finish_plot(fig, "dosecurve_example_fits", cfg)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].hist(ok.r2, bins=20); ax[0].set_title("R2 distribution")
    ax[1].scatter(ok.logEC50, ok.hill, c=ok.r2, cmap="viridis"); ax[1].set_xlabel("logEC50"); ax[1].set_ylabel("Hill"); ax[1].set_title("EC50 vs Hill")
    ax[2].scatter(ok.rep_cv, ok.ci_width); ax[2].set_xlabel("replicate CV"); ax[2].set_ylabel("EC50 CI width (log10)"); ax[2].set_title("replicate noise -> EC50 uncertainty")
    finish_plot(fig, "dosecurve_fit_summary", cfg)


def _curve_quality_model(fit_df, cfg):
    """Learn to flag unreliable curves from curve statistics, evaluated on a strictly held-out 40/15/15/30 compound split.
    Label = curve passed all rule-based checks. This is a consistency model for triage, not a biological ground truth."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score
    d = fit_df[fit_df.fit_ok == True].copy()
    y = (d["qc_flags"] == "PASS").astype(int).values
    if len(d) < 40 or y.sum() < 6 or (1 - y).sum() < 6:
        return dict(status=f"insufficient curves for a 4-way split (n={len(d)}, pass={int(y.sum())})")
    X = d[["rep_cv", "n_points", "n_doses", "resid_sd", "span", "hill"]].fillna(0).values
    sp = group_split(d.compound.values, cfg.split_fracs, y=y, seed=cfg.seed)
    verify_data_splits(sp, d.compound.values, len(d))
    m = GradientBoostingClassifier(random_state=cfg.seed, max_depth=2, n_estimators=120, subsample=0.8).fit(X[sp["train"]], y[sp["train"]])
    res = {}
    for s, i in sp.items():
        if len(np.unique(y[i])) < 2:
            continue
        p = m.predict_proba(X[i])[:, 1]
        res[s] = dict(auroc=roc_auc_score(y[i], p), acc=accuracy_score(y[i], p > 0.5), n=len(i))
    return dict(status="ok", metrics=res)
