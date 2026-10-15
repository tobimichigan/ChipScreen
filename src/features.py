"""features.py - leak-free preprocessing (fit on TRAIN only), outlier detection + sample-weight adjustment,
feature engineering, EDA plots."""
import numpy as np
import pandas as pd

from utils import LOG, finish_plot, check_memory


class TabularPreprocessor:
    """NaN-fraction + variance + correlation filtering, median imputation, winsorising and robust/standard scaling.
    Everything is fitted on the training split only, then applied unchanged to val/test/holdout."""

    def __init__(self, max_nan=0.4, var_eps=1e-8, corr_thresh=0.97, clip_q=(0.5, 99.5), scale="robust", max_features=None,
                 corr_sample=5000, seed=0, min_freq=None):
        self.min_freq = min_freq
        self.max_nan, self.var_eps, self.corr_thresh, self.clip_q = max_nan, var_eps, corr_thresh, clip_q
        self.scale, self.max_features, self.corr_sample, self.seed = scale, max_features, corr_sample, seed

    def fit(self, X, names=None):
        X = np.asarray(X, np.float32)
        X[~np.isfinite(X)] = np.nan
        names = np.array(names if names is not None else [f"f{i}" for i in range(X.shape[1])])
        keep = np.isnan(X).mean(0) <= self.max_nan
        med = np.nanmedian(np.where(keep[None, :], X, 0), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xi = np.where(np.isnan(X), med[None, :], X)
        sd = Xi.std(0)
        keep &= sd > self.var_eps
        if self.min_freq:                       # drop very rare substructure bits (pure noise at this sample size)
            keep &= (np.abs(Xi) > 1e-9).mean(0) >= self.min_freq
        idx = np.where(keep)[0]
        if self.corr_thresh < 1 and len(idx) > 1:
            rng = np.random.default_rng(self.seed)
            rows = rng.choice(len(Xi), min(self.corr_sample, len(Xi)), replace=False)
            Z = Xi[rows][:, idx]
            Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
            C = np.abs(Z.T @ Z) / len(Z)
            np.fill_diagonal(C, 0)
            drop = set()
            order = np.argsort(-Xi[:, idx].var(0))   # keep the higher-variance twin
            for a in order:
                if a in drop:
                    continue
                for b in np.where(C[a] > self.corr_thresh)[0]:
                    if b not in drop and b != a:
                        drop.add(b)
            idx = np.array([k for j, k in enumerate(idx) if j not in drop])
        if self.max_features and len(idx) > self.max_features:
            idx = idx[np.argsort(-Xi[:, idx].var(0))[: self.max_features]]
        idx = np.sort(idx)
        self.idx_, self.names_, self.med_ = idx, names[idx], med[idx]
        Z = Xi[:, idx]
        self.lo_, self.hi_ = np.percentile(Z, self.clip_q[0], axis=0), np.percentile(Z, self.clip_q[1], axis=0)
        Z = np.clip(Z, self.lo_, self.hi_)
        if self.scale == "robust":
            self.c_ = np.median(Z, 0)
            q1, q3 = np.percentile(Z, [25, 75], axis=0)
            self.s_ = np.where((q3 - q1) > 1e-8, q3 - q1, Z.std(0) + 1e-8)
        else:
            self.c_, self.s_ = Z.mean(0), Z.std(0) + 1e-8
        LOG.info(f"[prep] {X.shape[1]} -> {len(idx)} features after NaN/variance/correlation filtering")
        return self

    def transform(self, X):
        X = np.asarray(X, np.float32)[:, self.idx_]
        X = np.where(np.isfinite(X), X, self.med_[None, :])
        X = np.clip(X, self.lo_, self.hi_)
        return ((X - self.c_) / self.s_).astype(np.float32)

    def fit_transform(self, X, names=None):
        return self.fit(X, names).transform(X)


def outlier_weights(Xtr, contamination=0.03, min_w=0.3, seed=0):
    """IsolationForest + robust-z screening on TRAIN rows. Outliers are down-weighted (not dropped):
    w = 1 for inliers, decays to min_w for the most anomalous rows. Returns (weights, outlier_mask)."""
    from sklearn.ensemble import IsolationForest
    n = len(Xtr)
    if n < 50:
        return np.ones(n, np.float32), np.zeros(n, bool)
    iso = IsolationForest(n_estimators=150, contamination=contamination, random_state=seed, n_jobs=-1).fit(Xtr)
    s = -iso.score_samples(Xtr)                      # higher = more anomalous
    thr = np.quantile(s, 1 - contamination)
    z = np.clip((s - thr) / (s.max() - thr + 1e-9), 0, 1)
    w = 1.0 - (1.0 - min_w) * z
    w = np.where(s > thr, w, 1.0).astype(np.float32)
    med = np.median(Xtr, 0)
    mad = np.median(np.abs(Xtr - med), 0)
    valid = mad > 1e-3                                  # sparse / binary columns have MAD=0 and would flag every molecule
    if valid.sum() >= 5:
        rz = np.abs(Xtr[:, valid] - med[valid]) / (1.4826 * mad[valid])
        extreme = (rz > 8).mean(1) > 0.10               # >10% of continuous features beyond 8 robust-sigma
        w = np.where(extreme, np.minimum(w, min_w + 0.1), w).astype(np.float32)
    return w, (w < 1.0)


def chem_engineered(desc_df):
    """Interaction/ratio features on RDKit descriptors (applied row-wise => no leakage)."""
    d = desc_df.copy()
    g = lambda c: d[c] if c in d else pd.Series(np.nan, index=d.index)
    d["eng_logP_per_TPSA"] = g("MolLogP") / (g("TPSA") + 1.0)
    d["eng_hetero_frac"] = (g("NumHAcceptors") + g("NumHDonors")) / (g("HeavyAtomCount") + 1.0)
    d["eng_arom_frac"] = g("NumAromaticRings") / (g("RingCount") + 1.0)
    d["eng_rotb_per_heavy"] = g("NumRotatableBonds") / (g("HeavyAtomCount") + 1.0)
    d["eng_mw_logP"] = g("MolWt") * g("MolLogP")
    d["eng_lipinski_viol"] = ((g("MolWt") > 500).astype(float) + (g("MolLogP") > 5) + (g("NumHDonors") > 5) + (g("NumHAcceptors") > 10))
    d["eng_halogen_heavy"] = (g("fr_halogen")) / (g("HeavyAtomCount") + 1.0)
    return d


def morph_group_stats(X, names):
    """Per compartment x measurement-family mean/std summary features (compact multi-scale view of a CellProfiler profile)."""
    import re
    fam = {}
    for j, n in enumerate(names):
        p = n.split("_")
        key = "_".join(p[:2]) if len(p) > 2 else p[0]
        fam.setdefault(key, []).append(j)
    out, cols = [], []
    for k, idx in fam.items():
        if len(idx) < 3:
            continue
        out += [X[:, idx].mean(1), X[:, idx].std(1)]
        cols += [f"grp_{k}_mean", f"grp_{k}_std"]
    return (np.stack(out, 1).astype(np.float32), cols) if out else (np.zeros((len(X), 0), np.float32), [])


# --------------------------------------------------------------------------- EDA
def eda_plots(X, y, names, tag, cfg, task_names=None, split_idx=None, max_pts=4000):
    """Label balance, missingness, feature distributions, PCA + t-SNE embedding coloured by label/split."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    rng = np.random.default_rng(cfg.seed)
    Y = np.asarray(y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    prev = np.nanmean(Y, 0)
    lab = np.sum(~np.isnan(Y), 0)
    order = np.argsort(-prev)[:40]
    ax[0].bar(range(len(order)), prev[order])
    ax[0].set_title(f"{tag}: positive rate per task (top {len(order)})")
    ax[0].set_xlabel("task index"); ax[0].set_ylabel("prevalence")
    ax[1].hist(lab, bins=30); ax[1].set_title("labelled compounds per task"); ax[1].set_xlabel("n labelled")
    nanfrac = np.isnan(np.asarray(X[:3000], float)).mean(0)
    ax[2].hist(nanfrac, bins=30); ax[2].set_title("feature missingness (before cleaning)"); ax[2].set_xlabel("NaN fraction")
    finish_plot(fig, f"eda_{tag}_overview", cfg)

    Xs = np.asarray(X, np.float64)
    Xs[~np.isfinite(Xs)] = np.nan
    med = np.nanmedian(Xs, 0); med = np.where(np.isfinite(med), med, 0.0)
    Xs = np.where(np.isnan(Xs), med[None, :], Xs)
    lo, hi = np.percentile(Xs, 1, axis=0), np.percentile(Xs, 99, axis=0)
    Xs = np.clip(Xs, lo, hi)
    Xs = ((Xs - Xs.mean(0)) / (Xs.std(0) + 1e-6)).astype(np.float32)           # standardised -> PCA/t-SNE can never overflow
    sel = rng.choice(len(Xs), min(max_pts, len(Xs)), replace=False)
    Z = PCA(n_components=min(30, Xs.shape[1], len(sel) - 1), random_state=0).fit_transform(Xs[sel])
    T = TSNE(n_components=2, perplexity=min(30, max(5, len(sel) // 100)), init="pca", random_state=0).fit_transform(Z)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].scatter(Z[:, 0], Z[:, 1], s=4, c=np.nan_to_num(Y[sel, 0], nan=0.5), cmap="coolwarm")
    ax[0].set_title(f"PCA (first task) - {tag}")
    ax[1].scatter(T[:, 0], T[:, 1], s=4, c=np.nan_to_num(Y[sel, 0], nan=0.5), cmap="coolwarm"); ax[1].set_title("t-SNE (first task)")
    if split_idx is not None:
        sp = np.empty(len(Xs), object)
        for k, v in split_idx.items():
            sp[v] = k
        for k in ("train", "val", "test", "holdout"):
            m = sp[sel] == k
            ax[2].scatter(T[m, 0], T[m, 1], s=4, label=k, alpha=0.6)
        ax[2].legend(); ax[2].set_title("t-SNE coloured by split (domain-shift check)")
    finish_plot(fig, f"eda_{tag}_embedding", cfg)
