"""chipcell.py - ChipCell Atlas: segmentation + phenotype platform (Tool & Platform).
 A) Instance segmentation: attention U-Net (foreground + boundary heads) trained on BBBC038 (DSB2018) and Cell Tracking Challenge
    ground-truth masks, multi-scale/CLAHE/bilateral preprocessing, distance-transform watershed to split touching cells.
    Splits by image / by sequence (no frame leakage), 40/15/15/30.
 B) Phenotype atlas: per-cell shape/intensity/texture features from the segmentations (unsupervised cell-state map) and
    compound-level mechanism-of-action classification from JUMP Cell Painting profiles joined to Drug-Repurposing-Hub MoA labels."""
import os

import cv2
import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F

import chem
import features as FE
import models as M
from utils import (parallel_map, safe, LOG, Timer, tq, finish_plot, group_split, verify_data_splits, force_cleanup, check_memory, mem_gb, overfit_report,
                   multiclass_metrics, check_domain_relevance, poor_areas)

DEV = M.DEV


# =========================================================================== robust IO
_IO_WARNED = set()


def read_image(path):
    """tifffile (needs `imagecodecs` for LZW/zstd TIFFs) -> OpenCV -> PIL. Raises with the first failure reason."""
    errs = []
    if str(path).lower().endswith((".tif", ".tiff")):
        try:
            return tifffile.imread(path)
        except Exception as e:
            errs.append(f"tifffile: {e!r}")
    try:
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if im is not None:
            return im
        errs.append("cv2: returned None")
    except Exception as e:
        errs.append(f"cv2: {e!r}")
    try:
        from PIL import Image
        return np.array(Image.open(path))
    except Exception as e:
        errs.append(f"PIL: {e!r}")
    raise IOError(" | ".join(errs))


def _warn_once(tag, msg):
    if tag not in _IO_WARNED:
        _IO_WARNED.add(tag)
        LOG.warning(f"[chipcell] {tag}: {msg}")


# =========================================================================== preprocessing
def _gray(im):
    im = np.asarray(im)
    if im.ndim == 3:
        im = im[..., :3].mean(-1) if im.shape[-1] in (3, 4) else im[0]
    return im.astype(np.float32)


def preprocess(im, S):
    """-> (3,S,S) float16: [CLAHE+bilateral normalised, Gaussian-DoG fine scale, DoG coarse scale]. Polarity-normalised so cells are bright."""
    g = _gray(im)
    lo, hi = np.percentile(g, 0.5), np.percentile(g, 99.5)
    g = np.clip((g - lo) / (hi - lo + 1e-6), 0, 1)
    if np.median(g) > 0.5:                      # bright-background modality (histology / phase): invert
        g = 1 - g
    g = cv2.resize(g, (S, S), interpolation=cv2.INTER_AREA)
    u8 = (g * 255).astype(np.uint8)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8)
    cl = cv2.bilateralFilter(cl, 5, 40, 5).astype(np.float32) / 255
    d1 = cv2.GaussianBlur(g, (0, 0), 1.0) - cv2.GaussianBlur(g, (0, 0), 2.5)
    d2 = cv2.GaussianBlur(g, (0, 0), 3.0) - cv2.GaussianBlur(g, (0, 0), 8.0)
    return np.stack([cl, np.tanh(4 * d1), np.tanh(3 * d2)]).astype(np.float16)


def _boundary(lab):
    k = np.ones((3, 3), np.uint8)
    b = np.zeros(lab.shape, np.uint8)
    for i in np.unique(lab):
        if i == 0:
            continue
        m = (lab == i).astype(np.uint8)
        b |= (cv2.dilate(m, k) - cv2.erode(m, k)).astype(np.uint8)
    return b


def _load_b038(item, S):
    img_p, mask_ps = item
    im = read_image(img_p)
    lab = np.zeros(im.shape[:2], np.int32)
    for k, mp in enumerate(mask_ps, 1):
        m = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        if m is not None:
            m = m if m.ndim == 2 else m[..., 0]
            lab[m > 0] = k
    return preprocess(im, S), cv2.resize(lab.astype(np.float32), (S, S), interpolation=cv2.INTER_NEAREST).astype(np.int32), "b038:" + os.path.basename(img_p), "bbbc038"


def _load_ctc(d, S):
    raw, gt = np.squeeze(read_image(d["raw"])), np.squeeze(read_image(d["gt"]))
    if raw.ndim == 3 and gt.ndim == 3 and raw.shape[0] == gt.shape[0]:      # 3-D stack: take the mid slice that carries labels
        zs = np.where((gt > 0).reshape(gt.shape[0], -1).any(1))[0]
        z = int(zs[len(zs) // 2]) if len(zs) else gt.shape[0] // 2
        raw, gt = raw[z], gt[z]
    if raw.ndim != 2 or gt.ndim != 2 or raw.shape != gt.shape:
        raise ValueError(f"unsupported/mismatched shapes raw{raw.shape} gt{gt.shape}")
    if gt.max() == 0:
        raise ValueError("empty ground-truth mask")
    return preprocess(raw, S), cv2.resize(gt.astype(np.float32), (S, S), interpolation=cv2.INTER_NEAREST).astype(np.int32), f"ctc:{d['ds']}:{d['seq']}", f"ctc:{d['ds']}"


def seg_targets(lab):
    """Auxiliary training targets from an instance label map: boundary band (uint8) and per-instance normalised distance-to-edge (float16)."""
    from scipy import ndimage as ndi
    bd = _boundary(lab)
    d = np.zeros(lab.shape, np.float32)
    for i in np.unique(lab):
        if i == 0:
            continue
        m = lab == i
        e = ndi.distance_transform_edt(m)
        d[m] = e[m] / (e[m].max() + 1e-6)
    return bd, d.astype(np.float16)


def build_seg_dataset(res, cfg):
    """Images are read with a thread pool: on Google Drive each small file costs ~0.3 s latency, so serial reading of ~20k BBBC038 mask files took >1 h."""
    cache = cfg.cache / f"seg_{cfg.seg_size}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["L"], z["group"], list(z["src"])
    S = cfg.seg_size
    X, L, G, src = [], [], [], []
    fails = {}
    for name, items, fn in (("BBBC038: images+masks", res.bbbc038_train(), _load_b038), ("CTC: raw+GT pairs", res.ctc_pairs(cfg.max_ctc_per_seq, cfg.seed), _load_ctc)):
        ok = 0
        for r in parallel_map(lambda it: fn(it, S), items, workers=cfg.io_workers, desc=name):
            if isinstance(r, Exception):
                key = f"{name.split(':')[0]} {type(r).__name__}"
                fails[key] = fails.get(key, 0) + 1
                _warn_once(key, str(r)[:200])
                continue
            X.append(r[0]); L.append(r[1]); G.append(r[2]); src.append(r[3]); ok += 1
        LOG.info(f"[chipcell] {name.split(':')[0]}: {ok} loaded, {len(items) - ok} skipped")
        check_memory(cfg, "seg load")
    if not X:
        raise FileNotFoundError("no segmentation image/mask pairs found (BBBC038 stage1_train or CTC training)")
    X, L, G = np.stack(X), np.stack(L), np.array(G)
    np.savez_compressed(cache, X=X, L=L, group=G, src=np.array(src))
    LOG.info(f"[chipcell] segmentation set: {len(X)} images ({pd.Series(src).value_counts().to_dict()}), {X.nbytes/1e6:.0f} MB")
    return X, L, G, src


# =========================================================================== instance post-processing & metrics
def instances_from_probs(fg, bd, thr=0.5, bd_w=0.6, min_area=15, dist=None):
    """Foreground = fg > thr (so the semantic mask never depends on the boundary head); touching cells are separated by a
    marker-controlled watershed whose markers come from the distance transform of the boundary-suppressed core."""
    from scipy import ndimage as ndi
    from skimage import measure, morphology, segmentation, feature
    mask = morphology.remove_small_objects(fg > thr, min_area)
    mask = ndi.binary_fill_holes(mask)
    if mask.sum() == 0:
        return np.zeros(fg.shape, np.int32)
    if dist is not None:                           # learned per-cell distance map (StarDist/Cellpose-like): best markers for touching cells
        dmap = ndi.gaussian_filter(dist.astype(np.float32) * (1 - 0.5 * bd), 1.0) * mask
        pk = feature.peak_local_max(dmap, min_distance=4, threshold_abs=0.15, labels=measure.label(mask), exclude_border=False)
        dist = dmap
    else:
        core = mask & ((fg - bd_w * bd) > 0.7 * thr)
        dist = ndi.gaussian_filter(ndi.distance_transform_edt(mask), 1.0)
        lab_src = measure.label(core if core.sum() >= min_area else mask)
        pk = feature.peak_local_max(dist, min_distance=4, labels=lab_src, exclude_border=False)
    if len(pk) == 0:
        return measure.label(mask).astype(np.int32)
    mk = np.zeros(mask.shape, np.int32); mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    out = segmentation.watershed(-dist, mk, mask=mask).astype(np.int32)
    return out


def inst_from_pred(p, thr=0.5):
    return instances_from_probs(p[0], p[1], thr=thr, dist=(p[2] if len(p) > 2 else None))


def tune_threshold(P, L, grid=(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65), n=80):
    """Pick the foreground threshold on VALIDATION predictions only (maximises mean instance F1)."""
    best, bt = -1, 0.5
    for t in grid:
        f = np.mean([instance_scores(inst_from_pred(p, t), l)["f1"] for p, l in zip(P[:n], L[:n])])
        if f > best:
            best, bt = f, t
    return bt


def _contingency(a, b):
    na, nb = int(a.max()) + 1, int(b.max()) + 1
    return np.bincount(a.ravel().astype(np.int64) * nb + b.ravel(), minlength=na * nb).reshape(na, nb)


def instance_scores(pred, gt, iou_thr=0.5):
    """Dice (semantic), IoU, instance F1@0.5, mean matched IoU (AJI-like), count error."""
    p, g = pred > 0, gt > 0
    inter = (p & g).sum()
    dice = 2 * inter / (p.sum() + g.sum() + 1e-9) if (p.sum() + g.sum()) else 1.0
    iou = inter / ((p | g).sum() + 1e-9) if (p | g).sum() else 1.0
    if pred.max() == 0 or gt.max() == 0:
        return dict(dice=dice, iou=iou, f1=float(pred.max() == 0 and gt.max() == 0), mean_iou=0.0, count_err=abs(int(pred.max() > 0) * len(np.unique(pred)) - (len(np.unique(gt)) - 1)))
    C = _contingency(gt, pred)
    ag, ap = C.sum(1, keepdims=True), C.sum(0, keepdims=True)
    IoU = C / (ag + ap - C + 1e-9)
    IoU[0, :] = 0; IoU[:, 0] = 0
    m = IoU > iou_thr
    tp = int(m.sum()); fp = pred.max() - tp; fn = gt.max() - tp
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    mean_iou = float(IoU.max(1)[1:].mean()) if gt.max() > 0 else 0.0
    return dict(dice=dice, iou=iou, f1=f1, mean_iou=mean_iou, count_err=abs(len(np.unique(pred)) - len(np.unique(gt))))


def cell_features(img, lab):
    """Per-cell shape/intensity/texture descriptors from a segmentation (regionprops + GLCM)."""
    from skimage import measure
    from skimage.feature import graycomatrix, graycoprops
    rows = []
    g = img.astype(np.float32)
    for r in measure.regionprops(lab, intensity_image=g):
        if r.area < 15:
            continue
        sl = (g[r.slice] * 255).clip(0, 255).astype(np.uint8)
        msk = r.image
        sl = np.where(msk, sl, 0)
        try:
            glcm = graycomatrix(sl // 8, [1], [0, np.pi / 2], levels=32, symmetric=True, normed=True)
            con, hom, en = graycoprops(glcm, "contrast").mean(), graycoprops(glcm, "homogeneity").mean(), graycoprops(glcm, "energy").mean()
        except Exception:
            con = hom = en = np.nan
        rows.append(dict(area=r.area, perimeter=r.perimeter, eccentricity=r.eccentricity, solidity=r.solidity, extent=r.extent,
                         major=r.major_axis_length, minor=r.minor_axis_length, aspect=r.major_axis_length / (r.minor_axis_length + 1e-6),
                         circularity=4 * np.pi * r.area / (r.perimeter ** 2 + 1e-6), mean_int=r.mean_intensity, max_int=r.max_intensity,
                         std_int=float(g[r.slice][r.image].std()), glcm_contrast=con, glcm_homog=hom, glcm_energy=en))
    return pd.DataFrame(rows)


# =========================================================================== training
class SegNet(nn.Module):
    """Attention U-Net with 3 heads: foreground logit, boundary logit, per-cell distance map logit."""
    def __init__(self, base):
        super().__init__(); self.u = M.AttnUNet(3, 3, base=base, depth=4, drop_max=0.3, noise=0.02); self.out = self.u.out
    def forward(self, x): return self.u(x)


def _seg_aug(x, fg, bd, dist):
    ts = [x, fg, bd, dist]
    if np.random.rand() < 0.5: ts = [t.flip(-1) for t in ts]
    if np.random.rand() < 0.5: ts = [t.flip(-2) for t in ts]
    k = np.random.randint(4); ts = [torch.rot90(t, k, (-2, -1)) for t in ts]
    if np.random.rand() < 0.5:
        s = np.random.uniform(0.75, 1.0); H = ts[0].shape[-1]; c = int(H * s); i, j = np.random.randint(0, H - c + 1, 2)
        up = lambda t, mode: F.interpolate(t[..., i:i + c, j:j + c], size=(H, H), mode=mode, **({"align_corners": False} if mode == "bilinear" else {}))
        ts = [up(ts[0], "bilinear"), up(ts[1], "nearest"), up(ts[2], "nearest"), up(ts[3], "bilinear")]
    x = ts[0] * float(np.random.uniform(0.85, 1.15))
    return x, ts[1], ts[2], ts[3]


@torch.no_grad()
def predict_seg(net, X, batch=8):
    """-> (N,3,H,W) probabilities [fg, boundary, distance], 4-way flip TTA."""
    net = net.to(DEV).eval(); out = []
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(np.asarray(X[i:i + batch], np.float32)).to(DEV)
        acc = 0
        for dims in ([], [-1], [-2], [-1, -2]):
            xi = torch.flip(xb, dims) if dims else xb
            oi = torch.sigmoid(net(xi))
            acc = acc + (torch.flip(oi, dims) if dims else oi)
        out.append((acc / 4).cpu().numpy())
    return np.concatenate(out)


def _dice_np(p, l, t=0.5):
    return 2 * (((p > t) & (l > 0)).sum()) / (((p > t).sum() + (l > 0).sum()) + 1e-9)


def train_seg(Xtr, Ltr, Xva, Lva, cfg):
    torch.manual_seed(cfg.seed)
    net = SegNet(cfg.seg_base).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3, min_lr=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.type == "cuda")
    ema = M.EMA(net, 0.97)
    tg = parallel_map(seg_targets, list(Ltr), workers=cfg.io_workers, desc="segmentation targets")
    BDtr = np.stack([t[0] for t in tg]); Dtr = np.stack([t[1] for t in tg])
    n = len(Xtr); best, bad, state, hist = -1, 0, None, []
    fgfrac = float((Ltr > 0).mean()); pw = torch.tensor(min(10.0, (1 - fgfrac) / max(fgfrac, 1e-3)) ** 0.5, device=DEV)
    for ep in tq(range(cfg.seg_epochs), "ChipCell segmentation training", leave=False):
        net.train(); perm = np.random.permutation(n); tl = 0
        for i in range(0, n, cfg.seg_batch):
            idx = perm[i:i + cfg.seg_batch]
            if len(idx) < 2: continue
            x = torch.as_tensor(Xtr[idx].astype(np.float32)).to(DEV)
            fg = torch.as_tensor((Ltr[idx] > 0).astype(np.float32))[:, None].to(DEV)
            bd = torch.as_tensor(BDtr[idx].astype(np.float32))[:, None].to(DEV); dd = torch.as_tensor(Dtr[idx].astype(np.float32))[:, None].to(DEV)
            x, fg, bd, dd = _seg_aug(x, fg, bd, dd)
            with torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
                o = net(x).float()
            lf, lb, ld = o[:, :1], o[:, 1:2], o[:, 2:3]
            l_dist = (F.l1_loss(torch.sigmoid(ld), dd, reduction="none") * (0.2 + 4.0 * fg)).mean()
            loss = F.binary_cross_entropy_with_logits(lf, fg, pos_weight=pw) + M.dice_loss(lf, fg) + 0.5 * F.binary_cross_entropy_with_logits(lb, bd, pos_weight=torch.tensor(4.0, device=DEV)) + l_dist
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); scaler.step(opt); scaler.update(); ema.update(net); tl += float(loss) * len(idx)
        pv = predict_seg(ema.m, Xva)
        vd = float(np.mean([instance_scores(inst_from_pred(p), l)["f1"] for p, l in zip(pv[:60], Lva[:60])]))
        vdice = float(np.mean([_dice_np(p[0], l) for p, l in zip(pv, Lva)]))
        sub = np.random.default_rng(0).choice(n, min(n, 40), replace=False); pt = predict_seg(ema.m, Xtr[sub])
        td = float(np.mean([_dice_np(p[0], l) for p, l in zip(pt, Ltr[sub])]))
        score = 0.5 * vd + 0.5 * vdice
        sched.step(score)
        hist.append(dict(epoch=ep + 1, loss=tl / n, train_metric=td, val_metric=vdice, val_f1=vd, lr=opt.param_groups[0]["lr"]))
        if score > best + 1e-4:
            best, bad = score, 0; state = {k: v.detach().cpu().clone() for k, v in ema.m.state_dict().items()}
            torch.save(state, cfg.models / "chipcell_seg_best.pt")
        else:
            bad += 1
            if bad >= cfg.seg_patience: break
    ema.m.load_state_dict(state); ema.m.eval(); force_cleanup()
    return ema.m, pd.DataFrame(hist), best


def evaluate_seg(net, X, L, src, tag, thr=0.5):
    P = predict_seg(net, X)
    rows = []
    for p, l, s in zip(P, L, src):
        inst = inst_from_pred(p, thr)
        rows.append(dict(source=s, **instance_scores(inst, l)))
    return pd.DataFrame(rows), P


# =========================================================================== phenotype: MoA from JUMP profiles
def moa_atlas(res, cfg, morph, max_train_wells=24):
    """Mechanism-of-action classification. Compounds (not wells) are split by scaffold into 40/15/15/30; models are trained on the
    replicate WELL profiles of the training compounds (more observations per class) and evaluated at COMPOUND level by averaging
    the predicted probabilities over each compound's wells."""
    drugs, samples = res.repurposing_drugs(), res.repurposing_samples()
    if morph is None or not drugs or not samples:
        return dict(status="skipped: need JUMP profiles + Drug Repurposing Hub files")
    d = pd.read_csv(drugs, sep="\t", comment="!", low_memory=False)
    s = pd.read_csv(samples, sep="\t", comment="!", low_memory=False)
    d = d[d.moa.notna()].copy(); d["moa1"] = d.moa.str.split("|").str[0].str.strip().str.lower()
    s = s.dropna(subset=["InChIKey", "smiles"]).drop_duplicates("pert_iname")
    lab = s.merge(d[["pert_iname", "moa1"]].drop_duplicates("pert_iname"), on="pert_iname")
    lab["ik14"] = chem.ik14_from_inchikey(lab.InChIKey)
    mm = morph["morph"][["ik14", "n_wells"]].merge(lab[["ik14", "moa1", "smiles"]].drop_duplicates("ik14"), on="ik14")
    vc = mm.moa1.value_counts()
    keep = vc[vc >= cfg.min_moa_compounds].index[: cfg.max_moa_classes]
    mm = mm[mm.moa1.isin(keep)].reset_index(drop=True)
    info = dict(n_compounds=len(mm), classes=len(keep), counts=vc[keep].to_dict(), n_compounds_with_any_moa=int(vc.sum()))
    if len(keep) < 3 or len(mm) < 60:
        return {**info, "status": f"insufficient labelled overlap (compounds={len(mm)}, classes={len(keep)}, min per class={cfg.min_moa_compounds})"}
    st = chem.smiles_table(mm.smiles.tolist(), "MoA scaffolds")
    grp = np.where(st.scaffold.isin(["", "NOSCAF", None]) | st.scaffold.isna(), mm.ik14, st.scaffold)
    classes = sorted(keep); y = mm.moa1.map({c: i for i, c in enumerate(classes)}).values; K = len(classes)
    sp = group_split(grp, cfg.split_fracs, y=y, seed=cfg.seed)
    info["split_report"] = verify_data_splits(sp, grp, len(mm))
    comp_split = {k: set(mm.ik14.values[i]) for k, i in sp.items()}
    comp_y = dict(zip(mm.ik14, y))
    WX, WI, fn = morph["well_X"], np.asarray(morph["well_ik"]), list(morph["feat_names"])
    rng = np.random.default_rng(cfg.seed)
    rows = {k: [] for k in sp}
    for k, cs in comp_split.items():
        for c in cs:
            ii = np.where(WI == c)[0]
            if k in ("train", "val") and len(ii) > max_train_wells:
                ii = rng.choice(ii, max_train_wells, replace=False)
            rows[k] += ii.tolist()
    rows = {k: np.array(v, int) for k, v in rows.items()}
    if any(len(v) == 0 for v in rows.values()):
        return {**info, "status": f"a split has no well profiles ({({k: len(v) for k, v in rows.items()})})"}
    prep = FE.TabularPreprocessor(corr_thresh=0.95, max_features=min(cfg.max_morph_feats, 800)).fit(WX[rows["train"]].astype(np.float32), fn)
    Xs = {k: prep.transform(WX[v].astype(np.float32)) for k, v in rows.items()}
    cid = {k: WI[v] for k, v in rows.items()}
    ys = {k: np.array([comp_y[c] for c in cid[k]]) for k in rows}
    info["wells_per_split"] = {k: int(len(v)) for k, v in rows.items()}
    w, _ = FE.outlier_weights(Xs["train"], seed=cfg.seed)
    proba = {}
    for nm in ("lgbm", "logreg"):
        try:
            est, bp, sc = M.tune_classical(nm, Xs["train"], ys["train"], Xs["val"], ys["val"], w, "multiclass", min(cfg.search_iters, 4), cfg.seed)
            proba[f"clf:{nm}"] = {k: _full_proba(est, Xs[k], K) for k in Xs}
        except Exception as e:
            LOG.warning(f"[moa] {nm} failed: {e!r}")
    net, hist, best = M.fit_torch(M.HybridCNNAttn(Xs["train"].shape[1], K, drop=0.4), Xs["train"], ys["train"], Xs["val"], ys["val"], w=w, task="multiclass",
                                   epochs=cfg.deep_epochs, patience=cfg.deep_patience, batch=128, ckpt=str(cfg.models / "chipcell_moa_deep.pt"), desc="MoA deep", seed=cfg.seed)
    M.plot_history(hist, "chipcell_moa_deep", cfg, "val well-level balanced acc")
    proba["deep:cnn_attn"] = {k: M.predict(net, Xs[k], task="multiclass")[0] for k in Xs}

    def to_compound(P, ids, yy):
        df = pd.DataFrame(P); df["c"] = ids
        g = df.groupby("c").mean()
        cy = np.array([comp_y[c] for c in g.index])
        return g.values.astype(np.float32), cy, g.index.values

    comp_proba, comp_true = {}, {}
    for n_, p in proba.items():
        comp_proba[n_] = {}
        for k in Xs:
            comp_proba[n_][k], comp_true[k], _ = to_compound(p[k], cid[k], ys[k])
    va = {n_: multiclass_metrics(comp_true["val"], p["val"])["bal_acc"] for n_, p in comp_proba.items()}
    comp_proba["ensemble"] = {k: M.soft_vote({n_: p[k] for n_, p in comp_proba.items()}, {n_: max(a, 1e-3) ** 2 for n_, a in va.items()}) for k in Xs}
    met = {n_: {k: multiclass_metrics(comp_true[k], p[k]) for k in p} for n_, p in comp_proba.items()}
    table = pd.DataFrame({n_: {f"{k}_{m}": v for k in m_ for m, v in m_[k].items()} for n_, m_ in met.items()}).T
    table.to_csv(cfg.reports / "chipcell_moa_comparison.csv")
    best_n = max(met, key=lambda n_: met[n_]["test"]["bal_acc"])
    info.update(status="ok", table=table, best=best_n, chance_acc=1.0 / K, level="compound (mean over replicate wells)",
                gap=overfit_report({k: dict(bal_acc=met[best_n][k]["bal_acc"]) for k in met[best_n]}, "bal_acc", cfg.overfit_gap_warn))
    from sklearn.metrics import recall_score
    ph = comp_proba[best_n]["holdout"].argmax(1)
    rec = recall_score(comp_true["holdout"], ph, labels=range(K), average=None, zero_division=0)
    info["weak_classes"] = [dict(moa=classes[i], recall=float(rec[i])) for i in range(K) if rec[i] < 0.3 and (comp_true["holdout"] == i).sum() > 0]
    Xcomp = {k: pd.DataFrame(Xs[k]).assign(c=cid[k]).groupby("c").mean().values for k in Xs}
    safe(lambda: _moa_plots(met, best_n, classes, comp_true, comp_proba, Xcomp, cfg), "MoA plots")
    import joblib; joblib.dump(dict(prep=prep, classes=classes, best=best_n, features=fn), cfg.models / "chipcell_moa_bundle.joblib")
    return info


def _full_proba(est, X, K):
    p = est.predict_proba(X); out = np.zeros((len(X), K), np.float32); out[:, np.asarray(est.classes_).astype(int)] = p; return out


def _moa_plots(met, best, classes, ys, proba, Xs, cfg):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    from sklearn.decomposition import PCA
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.5))
    names = list(met)
    for i, s in enumerate(("train", "val", "test", "holdout")):
        ax[0].bar(np.arange(len(names)) + i * 0.2 - 0.3, [met[n][s]["bal_acc"] for n in names], 0.2, label=s)
    ax[0].axhline(1 / len(classes), c="k", ls=":", label="chance"); ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(names, rotation=30, ha="right"); ax[0].legend(); ax[0].set_title("MoA balanced accuracy by split (compound level, scaffold-grouped)")
    cm = confusion_matrix(ys["holdout"], proba[best]["holdout"].argmax(1), labels=range(len(classes)), normalize="true")
    ax[1].imshow(cm, cmap="Blues", vmin=0, vmax=1); ax[1].set_xticks(range(len(classes))); ax[1].set_yticks(range(len(classes)))
    ax[1].set_xticklabels([c[:14] for c in classes], rotation=90, fontsize=7); ax[1].set_yticklabels([c[:14] for c in classes], fontsize=7); ax[1].set_title(f"{best}: holdout confusion (recall)")
    Z = PCA(2, random_state=0).fit_transform(np.vstack([Xs[k] for k in ("train", "val", "test", "holdout")]))
    yy = np.concatenate([ys[k] for k in ("train", "val", "test", "holdout")])
    ax[2].scatter(Z[:, 0], Z[:, 1], c=yy, cmap="tab20", s=14); ax[2].set_title("morphological phenotype map (PCA of compound profiles), coloured by MoA")
    finish_plot(fig, "chipcell_moa_atlas", cfg)


# =========================================================================== main
def run_chipcell(res, cfg, morph=None):
    out = {}
    with Timer("ChipCell Atlas"):
        X, L, G, src = build_seg_dataset(res, cfg)
        src_code = pd.factorize(np.array(src))[0]
        sp = group_split(G, cfg.split_fracs, y=src_code, seed=cfg.seed)      # imaging sources balanced across splits, groups never shared
        out["split_report"] = verify_data_splits(sp, G, len(G))
        if any(len(v) == 0 for v in sp.values()):
            raise RuntimeError(f"empty split for segmentation: {({k: len(v) for k, v in sp.items()})}")
        net, hist, best = train_seg(X[sp["train"]], L[sp["train"]], X[sp["val"]], L[sp["val"]], cfg)
        M.plot_history(hist.assign(), "chipcell_seg", cfg, "val Dice")
        src = np.array(src); table, store = {}, {}
        thr = tune_threshold(predict_seg(net, X[sp["val"]]), L[sp["val"]])
        out["seg_threshold"] = thr
        LOG.info(f"[chipcell] foreground threshold tuned on validation: {thr}")
        for s, i in sp.items():
            df, P = evaluate_seg(net, X[i], L[i], src[i], s, thr)
            table[s] = df; store[s] = P
        summ = pd.DataFrame({s: d[["dice", "iou", "f1", "mean_iou", "count_err"]].mean() for s, d in table.items()}).T
        summ.to_csv(cfg.reports / "chipcell_segmentation_metrics.csv")
        per_src = pd.concat({s: d.groupby("source")[["dice", "f1"]].mean() for s, d in table.items()})
        per_src.to_csv(cfg.reports / "chipcell_segmentation_by_source.csv")
        out.update(seg_summary=summ, seg_by_source=per_src, seg_gap=overfit_report({s: dict(dice=summ.loc[s, "dice"]) for s in summ.index}, "dice", cfg.overfit_gap_warn), seg_best=best)
        out["weak_sources"] = poor_areas(per_src.loc["holdout"].reset_index().rename(columns={"source": "task"}), "f1", 0.5) if "holdout" in per_src.index.get_level_values(0) else []
        safe(lambda: _seg_plots(X, L, store, sp, table, summ, hist, cfg, thr), "segmentation plots")
        # ---- per-cell phenotype features from PREDICTED instances on holdout images (unsupervised cell-state map)
        rows = []
        Xh, Ph, sh = X[sp["holdout"]], store["holdout"], src[sp["holdout"]]
        for k in tq(range(min(len(Xh), 150)), "cell features (holdout)"):
            inst = inst_from_pred(Ph[k], thr)
            f = cell_features(Xh[k, 0].astype(np.float32), inst)
            if len(f):
                f["source"] = sh[k]; f["image"] = k; rows.append(f.head(cfg.max_cells_per_image))
        cells = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        out["cells"] = cells
        if len(cells) > 60:
            _cell_atlas(cells, cfg)
        out["moa"] = safe(lambda: moa_atlas(res, cfg, morph), "MoA atlas", dict(status="failed (see log)"))
        out["net"] = net
    del X, L; force_cleanup()
    return out


def _seg_plots(X, L, store, sp, table, summ, hist, cfg, thr):
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for i, m in enumerate(("dice", "iou", "f1")):
        ax[i].bar(summ.index, summ[m], color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"]); ax[i].set_ylim(0, 1); ax[i].set_title(f"segmentation {m} by split"); 
        for j, v in enumerate(summ[m]): ax[i].text(j, v + 0.01, f"{v:.2f}", ha="center")
    finish_plot(fig, "chipcell_seg_metrics_by_split", cfg)
    Xh, Lh, Ph = X[sp["holdout"]], L[sp["holdout"]], store["holdout"]
    n = min(6, len(Xh)); pick = np.linspace(0, len(Xh) - 1, n).astype(int)
    fig, ax = plt.subplots(3, n, figsize=(3.2 * n, 9)); ax = np.atleast_2d(ax).reshape(3, n)
    for c, i in enumerate(pick):
        inst = inst_from_pred(Ph[i], thr); im = Xh[i, 0].astype(np.float32)
        ax[0, c].imshow(im, cmap="gray"); ax[0, c].set_title("input (CLAHE)", fontsize=8)
        ov = np.dstack([im] * 3); ov[find_boundaries(Lh[i])] = [0, 1, 0]; ax[1, c].imshow(ov); ax[1, c].set_title(f"ground truth ({int(Lh[i].max())})", fontsize=8)
        ov = np.dstack([im] * 3); ov[find_boundaries(inst)] = [1, 0, 0]; ax[2, c].imshow(ov); ax[2, c].set_title(f"prediction ({int(inst.max())})", fontsize=8)
        for r in range(3): ax[r, c].axis("off")
    finish_plot(fig, "chipcell_holdout_segmentations", cfg)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    h = table["holdout"]; ax[0].hist(h.f1, bins=20); ax[0].set_title("holdout per-image F1@IoU0.5"); ax[1].hist(h.dice, bins=20); ax[1].set_title("holdout per-image Dice")
    finish_plot(fig, "chipcell_holdout_distributions", cfg)


def _cell_atlas(cells, cfg):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    num = cells.drop(columns=["source", "image"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    Z = FE.TabularPreprocessor(corr_thresh=0.95).fit_transform(num.values, num.columns)
    P = PCA(2, random_state=0).fit_transform(Z)
    k = int(np.clip(len(cells) // 80, 3, 7)); lab = KMeans(k, n_init=10, random_state=0).fit_predict(Z)
    cells["phenotype_cluster"] = lab
    cells.to_csv(cfg.reports / "chipcell_cell_features_holdout.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    ax[0].scatter(P[:, 0], P[:, 1], c=lab, cmap="tab10", s=6); ax[0].set_title(f"cell-state map (PCA of {num.shape[1]} features, k={k})")
    prof = num.assign(c=lab).groupby("c").mean(); prof = (prof - prof.mean()) / (prof.std() + 1e-9)
    im = ax[1].imshow(prof.T.values, cmap="coolwarm", aspect="auto"); ax[1].set_yticks(range(len(prof.columns))); ax[1].set_yticklabels(prof.columns, fontsize=7); ax[1].set_xlabel("cluster"); ax[1].set_title("cluster phenotype signatures (z)")
    plt.colorbar(im, ax=ax[1]); finish_plot(fig, "chipcell_cell_atlas", cfg)
