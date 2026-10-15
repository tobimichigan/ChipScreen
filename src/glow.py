"""glow.py - GlowFromLight: bright-field -> fluorescence translation with TRUST MAPS (Model & Algorithm).
Data: Allen hiPSC single-cell OME-TIFF volumes (bright-field + DNA / membrane / structure fluorescence channels).
Model: attention U-Net (residual blocks + BatchNorm + SE/spatial attention + attention gates) with a heteroscedastic head (mean + log-variance).
Trust map = 1 - normalised( aleatoric sigma  (+)  MC-dropout epistemic std ), validated against real error on holdout volumes.
Splits are by VOLUME (all slices of a cell stay in one split) 40/15/15/30."""
import os
import re
import time

import cv2
import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F

import models as M
from utils import (parallel_map, safe, LOG, Timer, tq, finish_plot, group_split, verify_data_splits, force_cleanup, check_memory, mem_gb, overfit_report, save_json)

DEV = M.DEV
TARGET_PATS = [("dna", r"dna|hoechst|dapi|nuc"), ("membrane", r"mem|cellmask|cell_?mask"), ("structure", r"struct|tubulin|lamin|tom20|fibrillarin|sec61|alpha|gfp")]
BF_PAT = r"^bf$|bright|trans|\bbf\b|dic|phase"


def _channel_names(path):
    try:
        with tifffile.TiffFile(path) as tf:
            if tf.ome_metadata:
                names = re.findall(r'<Channel[^>]*?Name="([^"]+)"', tf.ome_metadata)
                return names
    except Exception:
        pass
    return []


def _load_volume(path):
    """-> array (C,Z,Y,X) float32 and channel names. Handles CZYX / ZCYX / TCZYX axes from the OME metadata."""
    with tifffile.TiffFile(path) as tf:
        s = tf.series[0]
        arr, axes = s.asarray(), s.axes
    names = _channel_names(path)
    while arr.ndim > 4 or ("T" in axes and arr.ndim == len(axes) and axes.index("T") is not None and False):
        arr = arr[0]; axes = axes[1:]
    if "T" in axes:
        i = axes.index("T"); arr = np.take(arr, 0, axis=i); axes = axes.replace("T", "")
    if arr.ndim == 3:
        arr = arr[None]; axes = "C" + axes
    if axes[:4] != "CZYX":
        order = [axes.index(a) for a in "CZYX" if a in axes]
        arr = np.transpose(arr, order)
    return arr.astype(np.float32), names


def channel_stats(arr):
    """Per-channel statistics used to tell transmitted light from fluorescence: fluorescence = dark background + sparse bright
    structures (low median/p99.8, high skew, small foreground fraction); bright-field = dense, low-contrast, high median/p99.8."""
    from skimage.filters import threshold_otsu
    from scipy.stats import skew
    out = []
    zmid = arr.shape[1] // 2
    for c in range(arr.shape[0]):
        v = arr[c]
        sl = v[zmid]
        p998 = np.percentile(v, 99.8) + 1e-6
        try:
            fg = float((sl > threshold_otsu(sl)).mean())
        except Exception:
            fg = np.nan
        out.append(dict(ch=c, med_over_p998=float(np.median(v) / p998), skew=float(skew(v[:: max(1, v.shape[0] // 8)].ravel()[::7])), fg_frac=fg,
                        cv=float(v.std() / (v.mean() + 1e-6))))
    return pd.DataFrame(out)


def infer_channel_roles(stats, names):
    """-> dict(input=int, targets={name: int}, mode=str, note=str). Uses OME names when informative, otherwise image statistics."""
    low = [n.lower() for n in names]
    named = bool(low) and not all(re.fullmatch(r"c:?\d+|channel\s*\d+|\d+", n) for n in low)
    C = len(stats)
    if named:
        bf = next((i for i, n in enumerate(low) if re.search(BF_PAT, n)), None)
        tg = {}
        for key, pat in TARGET_PATS:
            j = next((i for i, n in enumerate(low) if re.search(pat, n) and i != bf), None)
            if j is not None:
                tg[key] = j
        if bf is not None and tg:
            return dict(input=bf, targets=tg, mode="bright-field -> fluorescence", note=f"roles from OME channel names {names}")
    # statistics: a transmitted-light channel has the highest median/p99.8 AND the lowest skew, clearly separated from the others
    sc = stats.med_over_p998.values - 0.05 * stats['skew'].values
    bf = int(np.argmax(sc))
    others = np.delete(sc, bf)
    is_bf = C >= 2 and stats.med_over_p998.iloc[bf] > 0.30 and (sc[bf] - others.max()) > 0.15
    rest = [i for i in range(C) if i != bf] if is_bf else list(range(C))
    # among fluorescence channels: smallest foreground fraction = compact nuclear stain, largest = cell-wide membrane stain
    order = sorted(rest, key=lambda i: (stats.fg_frac.iloc[i] if np.isfinite(stats.fg_frac.iloc[i]) else 0.5))
    tg = {}
    if is_bf:
        for k, i in zip(("dna", "membrane", "structure"), order):
            tg[k] = i
        return dict(input=bf, targets=tg, mode="bright-field -> fluorescence", note=f"channel {bf} identified as transmitted light from image statistics")
    # No transmitted-light channel in the data: honest fallback = translate from the nuclear stain (cheapest label) to the other stains
    dna = order[0]
    tg = {f"ch{j}": j for j in order if j != dna}                  # channel identity unknown -> neutral names
    return dict(input=dna, targets=tg, mode="nuclear-stain -> other fluorescence (NO bright-field channel found)",
                note=f"statistics show no transmitted-light channel (med/p99.8={stats.med_over_p998.round(3).tolist()}); using channel {dna} as the input")


def _norm(a):
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.8)
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def _norm_vol(v):
    """Per-VOLUME percentile normalisation of a fluorescence channel (Z,Y,X). Per-slice normalisation would stretch dim out-of-focus
    planes to full range and turn them into noise, destroying the z-structure the model has to learn."""
    sub = v[:: max(1, v.shape[0] // 12)]
    lo, hi = np.percentile(sub, 1), np.percentile(sub, 99.8)
    return np.clip((v - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def _hp_bf(v):
    """Bright-field has very low contrast on a large smooth illumination background (CV ~ 3%). High-pass + per-volume robust scaling
    exposes the cell texture: (plane - Gaussian background) / (6 * robust sigma) + 0.5."""
    sig = max(4.0, 0.08 * min(v.shape[-2:]))
    hp = np.stack([p - cv2.GaussianBlur(p.astype(np.float32), (0, 0), sig) for p in v])
    s = 1.4826 * np.median(np.abs(hp - np.median(hp))) + 1e-6
    return np.clip(hp / (6 * s) + 0.5, 0, 1).astype(np.float32)


def _slice_volume(p, roles, targets_ref, S, nz):
    arr, _ = _load_volume(p)
    if arr.shape[0] <= max(max(roles["targets"].values()), roles["input"]):
        return None
    Z = arr.shape[1]
    bf = _hp_bf(arr[roles["input"]]) if roles["mode"].startswith("bright") else _norm_vol(arr[roles["input"]])
    tg = {k: _norm_vol(arr[roles["targets"][k]]) for k in targets_ref}
    lo, hi = int(Z * 0.15), max(int(Z * 0.85), int(Z * 0.15) + 1)
    zs = np.unique(np.linspace(lo, hi - 1, nz).astype(int))
    step = max(1, Z // 14)                                   # z-context offset: out-of-focus planes carry depth information
    X, Y, meta = [], [], []
    for z in zs:
        ctx = [cv2.resize(bf[int(np.clip(z + o, 0, Z - 1))], (S, S), interpolation=cv2.INTER_AREA) for o in (-step, 0, step)]
        ys = [cv2.resize(tg[k][z], (S, S), interpolation=cv2.INTER_AREA) for k in targets_ref]
        X.append(np.stack(ctx).astype(np.float16)); Y.append(np.stack(ys).astype(np.float16)); meta.append(f"{os.path.basename(p)}|z{z}")
    return X, Y, meta


def build_glow_dataset(res, cfg):
    cache = cfg.cache / f"glow_v3_{cfg.glow_size}_{cfg.glow_z_per_volume}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["Y"], z["vol"], list(z["targets"]), list(z["meta"]), str(z["mode"])
    vols = res.allen_volumes()
    if not vols:
        raise FileNotFoundError("no Allen *.ome.tif volumes found")
    S, nz = cfg.glow_size, cfg.glow_z_per_volume
    arr, names = _load_volume(vols[0])
    st = channel_stats(arr)
    LOG.info("[glow] channel statistics (volume 0):\n" + st.round(3).to_string(index=False))
    roles = infer_channel_roles(st, names)
    roles["mode"] = roles["mode"]
    targets_ref = list(roles["targets"])
    mode = roles["mode"]
    LOG.info(f"[glow] MODE = {mode} | input channel {roles['input']} | targets {roles['targets']} | {roles['note']}")
    del arr
    results = parallel_map(lambda p: _slice_volume(p, roles, targets_ref, S, nz), vols, workers=min(cfg.io_workers, 6), desc="GlowFromLight: reading volumes")
    X, Y, V, meta = [], [], [], []
    for p, r in zip(vols, results):
        if r is None or isinstance(r, Exception):
            if isinstance(r, Exception):
                LOG.warning(f"[glow] unreadable {os.path.basename(p)}: {r!r}")
            continue
        X += r[0]; Y += r[1]; meta += r[2]; V += [p] * len(r[0])
    if not X:
        raise RuntimeError("no usable Allen volumes (channel layout not understood)")
    X, Y, V = np.stack(X), np.stack(Y), np.array(V)
    np.savez_compressed(cache, X=X, Y=Y, vol=V, targets=np.array(targets_ref), meta=np.array(meta), mode=np.array(mode))
    LOG.info(f"[glow] dataset: {len(X)} slices from {len(set(V))} volumes, targets {targets_ref}, {(X.nbytes + Y.nbytes)/1e6:.0f} MB")
    return X, Y, V, targets_ref, meta, mode


# =========================================================================== losses / metrics
def gauss_nll(mu, logvar, y):
    logvar = logvar.clamp(-8, 3)
    return (0.5 * torch.exp(-logvar) * (y - mu) ** 2 + 0.5 * logvar).mean()


def dtw_dist(a, b, band=8):
    """Sakoe-Chiba banded DTW between two 1-D profiles (shape-similarity metric)."""
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            c = abs(a[i - 1] - b[j - 1])
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m] / (n + m)


def image_metrics(pred, gt, tol=0.10):
    """pred/gt: (T,H,W) in [0,1]. Returns per-image dict of Pearson, SSIM, PSNR, SNR(dB), tolerance %, DTW."""
    from skimage.metrics import structural_similarity as ssim
    out = {}
    pr, g = pred.reshape(len(pred), -1), gt.reshape(len(gt), -1)
    out["pearson"] = float(np.mean([np.corrcoef(a, b)[0, 1] if a.std() > 1e-6 and b.std() > 1e-6 else 0 for a, b in zip(pr, g)]))
    fgm = gt > 0.2                                      # structure-only correlation (background pixels flatter the global Pearson)
    out["pearson_fg"] = float(np.mean([np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 50 and a[m].std() > 1e-6 and b[m].std() > 1e-6 else 0 for a, b, m in zip(pred, gt, fgm)]))
    out["ssim"] = float(np.mean([ssim(a, b, data_range=1.0) for a, b in zip(pred, gt)]))
    mse = ((pred - gt) ** 2).mean((1, 2))
    out["psnr"] = float(np.mean(10 * np.log10(1.0 / (mse + 1e-8))))
    out["snr_db"] = float(np.mean(10 * np.log10((gt ** 2).mean((1, 2)) / (mse + 1e-8))))
    out["tol_pct"] = float(100 * (np.abs(pred - gt) <= tol).mean())
    prof = lambda im: cv2.resize(im[im.shape[0] // 2][None], (64, 1), interpolation=cv2.INTER_AREA)[0]
    out["dtw"] = float(np.mean([dtw_dist(prof(a), prof(b)) for a, b in zip(pred, gt)]))
    return out


def _aug(x, y):
    if np.random.rand() < 0.5: x, y = x.flip(-1), y.flip(-1)
    if np.random.rand() < 0.5: x, y = x.flip(-2), y.flip(-2)
    k = np.random.randint(4); x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
    if np.random.rand() < 0.5:                       # random zoom (identical for input and target)
        s = np.random.uniform(0.8, 1.0); H = x.shape[-1]; c = int(H * s)
        i, j = np.random.randint(0, H - c + 1, 2)
        x = F.interpolate(x[..., i:i + c, j:j + c], size=(H, H), mode="bilinear", align_corners=False)
        y = F.interpolate(y[..., i:i + c, j:j + c], size=(H, H), mode="bilinear", align_corners=False)
    x = x * float(np.random.uniform(0.9, 1.1)) + float(np.random.uniform(-0.03, 0.03))   # input-only photometric jitter
    return x.clamp(0, 1), y


@torch.no_grad()
def predict_glow(net, X, batch=8, mc=0):
    """-> mean (N,T,H,W), aleatoric sigma (N,T,H,W), epistemic std (N,T,H,W) (zeros if mc=0)."""
    T = net.out.out_channels // 2
    net = net.to(DEV)
    mus, sig, epi = [], [], []
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(np.asarray(X[i:i + batch], np.float32)).to(DEV)
        net.eval()
        outs = []
        for dims in ([], [-1], [-2], [-1, -2]):           # 4-way flip test-time augmentation
            xi = torch.flip(xb, dims) if dims else xb
            oi = net(xi)
            outs.append(torch.flip(oi, dims) if dims else oi)
        o = torch.stack(outs).mean(0)
        mu, lv = o[:, :T], o[:, T:].clamp(-8, 3)
        a = torch.exp(0.5 * lv)
        if mc > 0:
            for m_ in net.modules():
                if isinstance(m_, (nn.Dropout, nn.Dropout2d)):
                    m_.train()
            sims = torch.stack([net(xb)[:, :T] for _ in range(mc)])
            net.eval()
            e = sims.std(0); mu = sims.mean(0) * 0.5 + mu * 0.5
        else:
            e = torch.zeros_like(mu)
        mus.append(mu.clamp(0, 1).cpu().numpy()); sig.append(a.cpu().numpy()); epi.append(e.cpu().numpy())
    return np.concatenate(mus), np.concatenate(sig), np.concatenate(epi)


def train_glow(Xtr, Ytr, Xva, Yva, cfg, T):
    torch.manual_seed(cfg.seed)
    net = M.AttnUNet(3, 2 * T, base=cfg.glow_base, depth=4, drop_max=0.3, noise=0.02).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3, min_lr=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.type == "cuda")
    ema = M.EMA(net, 0.97)
    best, bad, state, hist = -1, 0, None, []
    n = len(Xtr)
    bar = tq(range(cfg.glow_epochs), "GlowFromLight training", leave=False)
    for ep in bar:
        net.train(); perm = np.random.permutation(n); tl = 0
        for i in range(0, n, cfg.glow_batch):
            idx = perm[i:i + cfg.glow_batch]
            if len(idx) < 2: continue
            x = torch.as_tensor(Xtr[idx].astype(np.float32)).to(DEV); y = torch.as_tensor(Ytr[idx].astype(np.float32)).to(DEV)
            x, y = _aug(x, y)
            with torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
                o = net(x).float()
            mu, lv = o[:, :T], o[:, T:]
            loss = gauss_nll(torch.sigmoid(mu), lv, y) + 0.5 * F.l1_loss(torch.sigmoid(mu), y) + 0.3 * (1 - M.ssim_t(torch.sigmoid(mu), y)) + 0.2 * M.grad_l1(torch.sigmoid(mu), y) + 0.2 * M.pearson_loss(torch.sigmoid(mu), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); scaler.step(opt); scaler.update(); ema.update(net); tl += float(loss) * len(idx)
        mu, _, _ = predict_glow(_Sig(ema.m, T), Xva)
        vm = image_metrics_fast(mu, Yva)
        sched.step(vm)
        sub = np.random.default_rng(0).choice(n, min(n, 24), replace=False)
        mt, _, _ = predict_glow(_Sig(ema.m, T), Xtr[sub]); tm = image_metrics_fast(mt, Ytr[sub])
        hist.append(dict(epoch=ep + 1, loss=tl / n, train_metric=tm, val_metric=vm, lr=opt.param_groups[0]["lr"]))
        bar.set_postfix(loss=f"{tl/n:.3f}", val_pearson=f"{vm:.3f}", mem=f"{mem_gb():.1f}G")
        if vm > best + 1e-4:
            best, bad = vm, 0; state = {k: v.detach().cpu().clone() for k, v in ema.m.state_dict().items()}
            torch.save(state, cfg.models / "glow_best.pt")
        else:
            bad += 1
            if bad >= cfg.glow_patience: break
    ema.m.load_state_dict(state); ema.m.eval(); force_cleanup()
    return ema.m, pd.DataFrame(hist), best


class _Sig(nn.Module):
    """Wrapper that applies sigmoid to the mean channels so predict_glow returns intensities in [0,1]."""
    def __init__(self, net, T):
        super().__init__(); self.net, self.T = net, T; self.out = net.out
    def forward(self, x):
        o = self.net(x); return torch.cat([torch.sigmoid(o[:, :self.T]), o[:, self.T:]], 1)
    def modules(self): return self.net.modules()
    def eval(self): self.net.eval(); return self
    def to(self, d): self.net.to(d); return self


def image_metrics_fast(pred, gt):
    pr, g = pred.reshape(len(pred), pred.shape[1], -1), np.asarray(gt, np.float32).reshape(len(gt), gt.shape[1], -1)
    a, b = pr - pr.mean(-1, keepdims=True), g - g.mean(-1, keepdims=True)
    return float(((a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)).mean())


def run_glow(res, cfg):
    out = {}
    with Timer("GlowFromLight"):
        X, Y, vol, targets, meta, mode = build_glow_dataset(res, cfg)
        out['mode'] = mode
        T = len(targets)
        sp = group_split(vol, cfg.split_fracs, seed=cfg.seed)
        out["split_report"] = verify_data_splits(sp, vol, len(vol))
        if min(len(v) for v in sp.values()) == 0:
            raise RuntimeError("a split is empty - too few volumes for a 40/15/15/30 split")
        net, hist, best = train_glow(X[sp["train"]], Y[sp["train"], :], X[sp["val"]], Y[sp["val"]], cfg, T)
        M.plot_history(hist, "glow", cfg, "val Pearson")
        wrap = _Sig(net, T)
        met, store = {}, {}
        for s, i in sp.items():
            mu, al, ep = predict_glow(wrap, X[i], mc=cfg.glow_mc if s in ("test", "holdout") else 0)
            store[s] = (mu, al, ep, Y[i].astype(np.float32), X[i].astype(np.float32))
            met[s] = {t: image_metrics(mu[:, k], store[s][3][:, k], cfg.glow_tol) for k, t in enumerate(targets)}
        tab = pd.DataFrame({(s, t): m for s, d in met.items() for t, m in d.items()}).T
        tab.index.names = ["split", "target"]
        # naive baseline: the mean training target image ("a cell sits in the middle"). The model must beat it clearly to be worth anything.
        base = {t: Y[sp["train"], k].astype(np.float32).mean(0) for k, t in enumerate(targets)}
        gt_h = Y[sp["holdout"]].astype(np.float32)
        out["baseline_holdout"] = {t: image_metrics(np.repeat(base[t][None], len(gt_h), 0), gt_h[:, k], cfg.glow_tol) for k, t in enumerate(targets)}
        LOG.info("[glow] HOLDOUT Pearson  model vs mean-image baseline: " + "; ".join(f"{t}: {met['holdout'][t]['pearson']:.3f} vs {out['baseline_holdout'][t]['pearson']:.3f}" for t in targets))
        tab.to_csv(cfg.reports / "glow_metrics.csv")
        out["metrics"] = tab
        out["gap"] = {t: overfit_report({s: dict(pearson=met[s][t]["pearson"]) for s in met}, "pearson", cfg.overfit_gap_warn) for t in targets}
        # ---- trust-map validation on unseen volumes
        trust = _trust_validation(store, targets, cfg)
        out["trust"] = trust
        safe(lambda: _glow_plots(store, tab, hist, targets, trust, cfg), "glow plots")
        out.update(net=net, targets=targets, weak=[dict(target=t, pearson=met["holdout"][t]["pearson"]) for t in targets if met["holdout"][t]["pearson"] < 0.6])
        out["best_val_pearson"] = best
    del X, Y; force_cleanup()
    return out


def trust_map(al, ep):
    """Trust in [0,1]: 1 = reliable. Combines predicted aleatoric sigma and MC-dropout epistemic std (rank-normalised)."""
    u = al + ep
    return 1.0 - np.clip((u - np.percentile(u, 1)) / (np.percentile(u, 99) - np.percentile(u, 1) + 1e-8), 0, 1)


def _trust_validation(store, targets, cfg):
    from scipy.stats import spearmanr
    res = {}
    for s in ("test", "holdout"):
        mu, al, ep, y, _ = store[s]
        err = np.abs(mu - y)
        tr = trust_map(al, ep)
        sub = np.random.default_rng(0).choice(err.size, min(err.size, 400000), replace=False)
        rho = spearmanr(tr.ravel()[sub], err.ravel()[sub])[0]
        q = np.quantile(tr, [0.25, 0.75])
        lo, hi = err[tr <= q[0]].mean(), err[tr >= q[1]].mean()
        # coverage: fraction of pixels with |err| <= 2*sigma_total
        cover = float((err <= 2 * (al + ep + 1e-6)).mean())
        res[s] = dict(spearman_trust_vs_error=float(rho), err_low_trust=float(lo), err_high_trust=float(hi), ratio=float(lo / (hi + 1e-9)), coverage_2sigma=cover)
    LOG.info(f"[glow] trust-map validation: {res}")
    return res


def _glow_plots(store, tab, hist, targets, trust, cfg):
    import matplotlib.pyplot as plt
    mu, al, ep, y, x = store["holdout"]
    n = min(4, len(mu)); T = len(targets)
    pick = np.linspace(0, len(mu) - 1, n).astype(int)
    fig, ax = plt.subplots(n * T, 5, figsize=(17, 3.2 * n * T))
    ax = np.atleast_2d(ax)
    r = 0
    for i in pick:
        for k, t in enumerate(targets):
            tr = trust_map(al[i, k], ep[i, k]); err = np.abs(mu[i, k] - y[i, k])
            for a, (im, ti, cm) in zip(ax[r], [(x[i, 1], "input (centre plane)", "gray"), (y[i, k], f"GT {t}", "magma"), (mu[i, k], "prediction", "magma"), (tr, "trust map", "viridis"), (err, "abs error", "hot")]):
                a.imshow(im, cmap=cm, vmin=0, vmax=1); a.set_title(ti, fontsize=8); a.axis("off")
            r += 1
    fig.suptitle("GlowFromLight on unseen HOLDOUT volumes"); finish_plot(fig, "glow_holdout_examples", cfg)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for s, c in zip(("val", "test", "holdout"), ("tab:blue", "tab:orange", "tab:green")):
        m_, a_, e_, y_, _ = store[s]
        err = np.abs(m_ - y_).ravel(); u = (a_ + e_).ravel()
        sub = np.random.default_rng(1).choice(err.size, min(err.size, 60000), replace=False)
        order = np.argsort(u[sub]); cum = np.cumsum(err[sub][order]) / np.arange(1, len(sub) + 1)
        ax[0].plot(np.linspace(0, 100, len(cum)), cum, c=c, label=s)
    ax[0].set_xlabel("% pixels kept (most trusted first)"); ax[0].set_ylabel("mean abs error"); ax[0].set_title("selective prediction: error falls when low-trust pixels are dropped"); ax[0].legend()
    m_, a_, e_, y_, _ = store["holdout"]; err = np.abs(m_ - y_); tr = trust_map(a_, e_)
    sub = np.random.default_rng(2).choice(err.size, min(err.size, 80000), replace=False)
    ax[1].hexbin(tr.ravel()[sub], err.ravel()[sub], gridsize=40, bins="log", cmap="viridis"); ax[1].set_xlabel("trust"); ax[1].set_ylabel("|error|"); ax[1].set_title(f"holdout trust vs error (rho={trust['holdout']['spearman_trust_vs_error']:.2f})")
    for t in targets:
        ax[2].plot(["train", "val", "test", "holdout"], [tab.loc[(s, t), "pearson"] for s in ("train", "val", "test", "holdout")], "o-", label=t)
    ax[2].set_title("Pearson by split (generalisation gap)"); ax[2].legend(); ax[2].set_ylim(0, 1)
    finish_plot(fig, "glow_trust_and_generalisation", cfg)
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for a, m_name in zip(ax, ("pearson", "ssim", "psnr", "tol_pct")):
        for i, s in enumerate(("train", "val", "test", "holdout")):
            a.bar(np.arange(len(targets)) + i * 0.2 - 0.3, [tab.loc[(s, t), m_name] for t in targets], 0.2, label=s)
        a.set_xticks(range(len(targets))); a.set_xticklabels(targets); a.set_title(m_name)
    ax[0].legend(); finish_plot(fig, "glow_metrics_by_split", cfg)
