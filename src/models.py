"""models.py - deep (attention MLP, 1D-CNN+Transformer hybrid, multimodal fusion, attention U-Net) and classical
(tuned, bagged/boosted, soft-voting) learners, plus the shared training loop."""
import copy
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils import LOG, finish_plot, force_cleanup, mem_gb

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================== building blocks
class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.05):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        return x + torch.randn_like(x) * self.sigma if self.training and self.sigma > 0 else x


class FeatureGate(nn.Module):
    """Squeeze-and-excite style feature attention for tabular vectors."""

    def __init__(self, d, r=8):
        super().__init__()
        h = max(8, d // r)
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d), nn.Sigmoid())

    def forward(self, x):
        return x * (0.5 + self.net(x))


class AttnMLP(nn.Module):
    """Deeper MLP: input noise -> feature attention -> [Linear-BN-GELU-Dropout] x N with residuals. L2 via AdamW weight decay."""

    def __init__(self, d_in, d_out, hidden=(512, 256, 128), drop=0.4, noise=0.05):
        super().__init__()
        self.noise = GaussianNoise(noise)
        self.gate = FeatureGate(d_in)
        dims = (d_in,) + tuple(hidden)
        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        for a, b in zip(dims[:-1], dims[1:]):
            self.blocks.append(nn.Sequential(nn.Linear(a, b), nn.BatchNorm1d(b), nn.GELU(), nn.Dropout(drop)))
            self.skips.append(nn.Linear(a, b, bias=False) if a != b else nn.Identity())
        self.emb_dim = hidden[-1]
        self.head = nn.Linear(hidden[-1], d_out)

    def embed(self, x):
        x = self.gate(self.noise(x))
        for blk, sk in zip(self.blocks, self.skips):
            x = blk(x) + sk(x)
        return x

    def forward(self, x):
        return self.head(self.embed(x))


class Res1D(nn.Module):
    def __init__(self, c, p):
        super().__init__()
        self.c1, self.b1 = nn.Conv1d(c, c, 5, padding=2), nn.BatchNorm1d(c)
        self.c2, self.b2 = nn.Conv1d(c, c, 5, padding=2), nn.BatchNorm1d(c)
        self.drop = nn.Dropout1d(p)       # spatial (channel-wise) dropout

    def forward(self, x):
        h = self.drop(F.gelu(self.b1(self.c1(x))))
        return F.gelu(x + self.b2(self.c2(h)))


class HybridCNNAttn(nn.Module):
    """Feature-vector -> conv stem + residual 1D-CNN (local feature-neighbourhood structure, e.g. CellProfiler channel
    families) -> Transformer encoder (self-attention across pooled tokens) -> learned-query attention pooling -> head."""

    def __init__(self, d_in, d_out, c=48, d_model=96, tokens=24, heads=4, layers=2, drop=0.3, noise=0.05):
        super().__init__()
        self.d_in, self.tokens = d_in, tokens
        self.noise, self.gate = GaussianNoise(noise), FeatureGate(d_in)
        self.stem = nn.Sequential(nn.Conv1d(1, c, 7, padding=3), nn.BatchNorm1d(c), nn.GELU())
        self.res = nn.Sequential(Res1D(c, drop * 0.5), nn.MaxPool1d(2, ceil_mode=True), Res1D(c, drop))
        self.pool = nn.AdaptiveAvgPool1d(tokens)
        self.proj = nn.Linear(c, d_model)
        self.pos = nn.Parameter(torch.zeros(1, tokens, d_model))
        enc = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=2 * d_model, dropout=drop * 0.6, batch_first=True, activation="gelu", norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.q = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.att = nn.MultiheadAttention(d_model, heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.emb_dim = d_model
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(d_model, d_out))

    def embed(self, x):
        x = self.gate(self.noise(x)).unsqueeze(1)
        h = self.pool(self.res(self.stem(x))).transpose(1, 2)
        h = self.tr(self.proj(h) + self.pos)
        o, _ = self.att(self.q.expand(len(h), -1, -1), h, h)
        return self.norm(o.squeeze(1))

    def forward(self, x):
        return self.head(self.embed(x))


class FusionNet(nn.Module):
    """Multimodal: chemistry branch (AttnMLP) + morphology branch (HybridCNNAttn) -> modality tokens -> attention fusion.
    Modality-dropout makes the net robust to a missing modality at inference."""

    def __init__(self, d_chem, d_morph, d_out, d=96, drop=0.35, p_moddrop=0.15):
        super().__init__()
        self.a = AttnMLP(d_chem, d_out, hidden=(384, 192, d), drop=drop)
        self.b = HybridCNNAttn(d_morph, d_out, d_model=d, drop=drop * 0.8)
        self.p = p_moddrop
        enc = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        self.fuse = nn.TransformerEncoder(enc, 1)
        self.mod_emb = nn.Parameter(torch.zeros(1, 2, d))
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Dropout(drop), nn.Linear(2 * d, d_out))

    def forward(self, xc, xm):
        ea, eb = self.a.embed(xc), self.b.embed(xm)
        if self.training and self.p > 0:
            ma = (torch.rand(len(ea), 1, device=ea.device) > self.p).float()
            mb = (torch.rand(len(eb), 1, device=eb.device) > self.p).float()
            both_off = (ma + mb) == 0
            ma = torch.where(both_off, torch.ones_like(ma), ma)
            ea, eb = ea * ma, eb * mb
        t = self.fuse(torch.stack([ea, eb], 1) + self.mod_emb)
        return self.head(t.flatten(1))


# =========================================================================== training loop
def _as_list(X):
    return list(X) if isinstance(X, (list, tuple)) else [X]


def _to_t(a, dtype=torch.float32):
    return torch.as_tensor(np.asarray(a), dtype=dtype)


def predict(model, Xs, batch=1024, mc=0, task="multilabel"):
    """Returns (prob, std). mc>0 -> Monte-Carlo dropout (dropout on, BatchNorm frozen) for epistemic uncertainty."""
    Xs = _as_list(Xs)
    n = len(Xs[0])
    model = model.to(DEV)

    def run():
        outs = []
        with torch.no_grad():
            for i in range(0, n, batch):
                xb = [_to_t(x[i:i + batch]).to(DEV) for x in Xs]
                lg = model(*xb).float()
                outs.append((torch.sigmoid(lg) if task != "multiclass" else torch.softmax(lg, 1)).cpu().numpy())
        return np.concatenate(outs)

    if mc <= 0:
        model.eval()
        return run(), np.zeros(1)
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)) or isinstance(m, nn.MultiheadAttention):
            m.train()
    sims = np.stack([run() for _ in range(mc)])
    model.eval()
    return sims.mean(0), sims.std(0)


def _val_metric(task, Y, P):
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    if task == "multiclass":
        return balanced_accuracy_score(Y, P.argmax(1))
    vals = []
    for j in range(Y.shape[1]):
        m = ~np.isnan(Y[:, j])
        if m.sum() > 5 and len(np.unique(Y[m, j])) == 2:
            vals.append(roc_auc_score(Y[m, j], P[m, j]))
    return float(np.mean(vals)) if vals else float("nan")


def fit_torch(model, Xtr, Ytr, Xva, Yva, w=None, task="multilabel", epochs=120, lr=2e-3, wd=1e-3, label_smoothing=0.05,
              mixup=0.2, patience=14, min_delta=1e-4, batch=256, clip=1.0, ckpt=None, desc="train", seed=0):
    """Generic trainer. Early stopping + best-state checkpoint on peak VALIDATION metric, ReduceLROnPlateau, AdamW weight
    decay, gradient clipping, label smoothing, class/pos weights, optional mixup, sample weights (outlier down-weighting).
    task: 'multilabel' (NaN = unlabelled, masked) | 'multiclass'. The test/holdout splits are never seen here."""
    torch.manual_seed(seed)
    Xtr, Xva = _as_list(Xtr), _as_list(Xva)
    n = len(Xtr[0])
    model = model.to(DEV)
    Ytr = np.asarray(Ytr)
    if task == "multilabel":
        Ytr = Ytr.astype(np.float32)
        pos = np.nansum(Ytr, 0)
        neg = np.sum(~np.isnan(Ytr), 0) - pos
        pw = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, 20), dtype=torch.float32, device=DEV)
    else:
        classes = np.unique(Ytr)
        cw = len(Ytr) / (len(classes) * np.bincount(Ytr.astype(int), minlength=int(classes.max()) + 1).clip(1))
        cw = torch.tensor(cw, dtype=torch.float32, device=DEV)
    w = np.ones(n, np.float32) if w is None else np.asarray(w, np.float32)
    tensors = [_to_t(x) for x in Xtr] + [_to_t(Ytr, torch.float32 if task == "multilabel" else torch.long), _to_t(w)]
    ds = torch.utils.data.TensorDataset(*tensors)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, drop_last=n > batch * 2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=max(2, patience // 3), min_lr=lr / 100)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.type == "cuda")
    k_in = len(Xtr)
    best, best_state, bad, hist = -1e9, None, 0, []
    sub = np.random.default_rng(seed).choice(n, min(n, 4000), replace=False)
    bar = tqdm(range(epochs), desc=desc, leave=False)
    for ep in bar:
        model.train()
        tl, cnt = 0.0, 0
        for batch_t in dl:
            xs = [b.to(DEV) for b in batch_t[:k_in]]
            yb, wb = batch_t[k_in].to(DEV), batch_t[k_in + 1].to(DEV)
            if len(yb) < 2:
                continue
            lam, perm = 1.0, None
            if mixup > 0 and np.random.rand() < 0.5:
                lam = float(np.random.beta(mixup, mixup))
                perm = torch.randperm(len(yb), device=DEV)
                xs = [lam * x + (1 - lam) * x[perm] for x in xs]
            with torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
                lg = model(*xs).float()
            if task == "multilabel":
                def bce(y):
                    m = ~torch.isnan(y)
                    ys = torch.where(m, y, torch.zeros_like(y))
                    ys = ys * (1 - label_smoothing) + 0.5 * label_smoothing
                    l = F.binary_cross_entropy_with_logits(lg, ys, pos_weight=pw, reduction="none") * m
                    return (l.sum(1) / m.sum(1).clamp(min=1)), m
                if perm is None:
                    l, _ = bce(yb)
                else:
                    ym = lam * torch.nan_to_num(yb) + (1 - lam) * torch.nan_to_num(yb[perm])
                    mm = (~torch.isnan(yb)) & (~torch.isnan(yb[perm]))
                    ys = ym * (1 - label_smoothing) + 0.5 * label_smoothing
                    l = F.binary_cross_entropy_with_logits(lg, ys, pos_weight=pw, reduction="none") * mm
                    l = l.sum(1) / mm.sum(1).clamp(min=1)
            else:
                ce = lambda y: F.cross_entropy(lg, y, weight=cw, label_smoothing=label_smoothing, reduction="none")
                l = ce(yb) if perm is None else lam * ce(yb) + (1 - lam) * ce(yb[perm])
            loss = (l * wb).sum() / wb.sum().clamp(min=1e-6)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()
            tl += float(loss) * len(yb)
            cnt += len(yb)
        ptr, _ = predict(model, [x[sub] for x in Xtr], task=task)
        pva, _ = predict(model, Xva, task=task)
        mtr = _val_metric(task, Ytr[sub], ptr)
        mva = _val_metric(task, np.asarray(Yva), pva)
        sched.step(mva if np.isfinite(mva) else 0)
        hist.append(dict(epoch=ep + 1, loss=tl / max(cnt, 1), train_metric=mtr, val_metric=mva, lr=opt.param_groups[0]["lr"]))
        bar.set_postfix(loss=f"{tl/max(cnt,1):.3f}", val=f"{mva:.3f}", mem=f"{mem_gb():.1f}G")
        if mva > best + min_delta:
            best, bad = mva, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            if ckpt:
                torch.save(best_state, ckpt)          # periodic save on every new peak validation metric
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    force_cleanup()
    return model, pd.DataFrame(hist), best


def plot_history(hist, name, cfg, ylabel="metric"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4))
    ax[0].plot(hist.epoch, hist.loss); ax[0].set_title(f"{name}: training loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(hist.epoch, hist.train_metric, label="train"); ax[1].plot(hist.epoch, hist.val_metric, label="val")
    pk = int(hist.val_metric.idxmax())
    ax[1].scatter([hist.epoch[pk]], [hist.val_metric[pk]], c="red", s=70, zorder=5, label=f"peak {hist.val_metric[pk]:.3f}")
    ax[1].fill_between(hist.epoch, hist.val_metric, hist.train_metric, alpha=0.12, color="orange")
    ax[1].set_title(f"{name}: {ylabel} (shaded = generalisation gap)"); ax[1].legend(); ax[1].set_xlabel("epoch")
    ax[2].semilogy(hist.epoch, hist.lr); ax[2].set_title("learning rate (ReduceLROnPlateau)"); ax[2].set_xlabel("epoch")
    finish_plot(fig, f"history_{name}", cfg)


# =========================================================================== classical learners
def classical_space(name, task, seed, n_jobs=-1, pos_weight=None):
    from scipy.stats import loguniform, randint, uniform
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    import lightgbm as lgb
    if name == "logreg":
        return LogisticRegression(max_iter=500, class_weight="balanced"), {"C": loguniform(1e-3, 3e0)}
    if name == "rf":
        return (RandomForestClassifier(n_jobs=n_jobs, random_state=seed, class_weight="balanced_subsample"),
                {"n_estimators": [150, 250], "max_depth": [None, 14, 24], "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.05, 0.15]})
    if name == "et":
        return (ExtraTreesClassifier(n_jobs=n_jobs, random_state=seed, class_weight="balanced_subsample"),
                {"n_estimators": [200, 300], "max_depth": [None, 18], "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.1]})
    if name == "lgbm":
        kw = dict(verbose=-1, random_state=seed, n_jobs=n_jobs, subsample_freq=1)
        if task != "multiclass":
            kw["scale_pos_weight"] = pos_weight or 1.0
        else:
            kw["class_weight"] = "balanced"
        return (lgb.LGBMClassifier(**kw),
                {"n_estimators": [100, 200] if task == "multiclass" else [150, 300, 500], "learning_rate": loguniform(0.03, 0.2), "num_leaves": randint(8, 32 if task == "multiclass" else 64),
                 "colsample_bytree": uniform(0.3, 0.6), "subsample": uniform(0.6, 0.4), "reg_lambda": loguniform(1e-2, 20),
                 "min_child_samples": randint(5, 40)})
    raise ValueError(name)


def tune_classical(name, Xtr, ytr, Xva, yva, w=None, task="binary", n_iter=10, seed=0):
    """Randomised search whose ONLY scoring data is the validation split (PredefinedSplit). Returns fitted-on-train model,
    best params and validation score."""
    from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
    pw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1)) if task != "multiclass" else None
    est, space = classical_space(name, task, seed, pos_weight=pw)
    Xall = np.vstack([Xtr, Xva])
    yall = np.concatenate([ytr, yva])
    fold = np.r_[-np.ones(len(ytr), int), np.zeros(len(yva), int)]
    wall = np.r_[np.ones(len(ytr)) if w is None else w, np.ones(len(yva))]
    scoring = "balanced_accuracy" if task == "multiclass" else "roc_auc"
    rs = RandomizedSearchCV(est, space, n_iter=n_iter, scoring=scoring, cv=PredefinedSplit(fold), refit=False, random_state=seed, n_jobs=1, error_score=np.nan)
    try:
        rs.fit(Xall, yall, **({"sample_weight": wall} if name != "logreg" or True else {}))
    except TypeError:
        rs.fit(Xall, yall)
    best = rs.best_params_
    est.set_params(**best)
    try:
        est.fit(Xtr, ytr, sample_weight=w)
    except TypeError:
        est.fit(Xtr, ytr)
    return est, best, float(rs.best_score_)


def soft_vote(preds, weights=None):
    """preds: dict name -> array (same shape). weights: dict name -> float (e.g. from validation metric)."""
    names = list(preds)
    w = np.array([1.0 if weights is None else weights.get(k, 1.0) for k in names], float)
    w = w / w.sum()
    return sum(wi * preds[k] for wi, k in zip(w, names))


def cross_validate_group(X, y, groups, make_model, n_splits=5, task="binary", seed=0):
    """K-fold CV (group-aware) on the supplied rows - callers pass train+val rows ONLY."""
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    sg = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    sc = []
    for tr, te in sg.split(X, y, groups):
        if len(np.unique(y[te])) < 2 and task != "multiclass":
            continue
        m = make_model()
        m.fit(X[tr], y[tr])
        if task == "multiclass":
            sc.append(balanced_accuracy_score(y[te], m.predict(X[te])))
        else:
            sc.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(sc)) if sc else float("nan"), float(np.std(sc)) if sc else float("nan"), len(sc)


# =========================================================================== image networks
class SEAttention(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(4, c // r), 1), nn.ReLU(True), nn.Conv2d(max(4, c // r), c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        a = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.conv(a))


class ResAttnBlock(nn.Module):
    """Residual conv block + BatchNorm after every conv + channel & spatial attention + spatial dropout."""

    def __init__(self, cin, cout, p=0.1):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout)
        self.c2, self.b2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout)
        self.skip = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()
        self.se, self.sa, self.drop = SEAttention(cout), SpatialAttention(), nn.Dropout2d(p)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)), inplace=True)
        h = self.drop(h)
        h = self.b2(self.c2(h))
        h = self.sa(self.se(h))
        return F.relu(h + self.skip(x), inplace=True)


class AttnGate(nn.Module):
    """Additive attention gate on U-Net skip connections."""

    def __init__(self, cg, cx, ci):
        super().__init__()
        self.wg, self.wx = nn.Conv2d(cg, ci, 1), nn.Conv2d(cx, ci, 1)
        self.psi = nn.Sequential(nn.ReLU(True), nn.Conv2d(ci, 1, 1), nn.Sigmoid())

    def forward(self, g, x):
        return x * self.psi(self.wg(g) + self.wx(x))


class AttnUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=24, depth=4, drop_max=0.3, noise=0.02, final="linear"):
        super().__init__()
        self.depth = depth
        self.noise = GaussianNoise(noise)
        ch = [base * 2 ** i for i in range(depth + 1)]
        drops = np.linspace(0.05, drop_max, depth + 1)
        self.enc = nn.ModuleList([ResAttnBlock(in_ch if i == 0 else ch[i - 1], ch[i], float(drops[i])) for i in range(depth + 1)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in range(depth)])
        self.gate = nn.ModuleList([AttnGate(ch[i], ch[i], max(ch[i] // 2, 4)) for i in range(depth)])
        self.dec = nn.ModuleList([ResAttnBlock(ch[i] * 2, ch[i], float(drops[i])) for i in range(depth)])
        self.out = nn.Conv2d(ch[0], out_ch, 1)
        self.final = final

    def forward(self, x):
        H, W = x.shape[-2:]
        m = 2 ** self.depth
        ph, pw = (-H) % m, (-W) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        x = self.noise(x)
        skips = []
        for i in range(self.depth):
            x = self.enc[i](x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.enc[self.depth](x)
        for i in reversed(range(self.depth)):
            x = self.up[i](x)
            x = self.dec[i](torch.cat([x, self.gate[i](x, skips[i])], 1))
        x = self.out(x)[..., :H, :W]
        return torch.sigmoid(x) if self.final == "sigmoid" else x


def _gauss_win(win=7, sigma=1.5, c=1):
    g = torch.arange(win, dtype=torch.float32) - win // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(c, 1, win, win).contiguous()


def ssim_t(a, b, win=7):
    c = a.shape[1]
    w = _gauss_win(win, 1.5, c).to(a.device)
    mu_a, mu_b = F.conv2d(a, w, padding=win // 2, groups=c), F.conv2d(b, w, padding=win // 2, groups=c)
    saa = F.conv2d(a * a, w, padding=win // 2, groups=c) - mu_a ** 2
    sbb = F.conv2d(b * b, w, padding=win // 2, groups=c) - mu_b ** 2
    sab = F.conv2d(a * b, w, padding=win // 2, groups=c) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return s.mean()


def grad_l1(a, b):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=a.device).view(1, 1, 3, 3) / 8
    ky = kx.transpose(2, 3)
    c = a.shape[1]
    f = lambda t, k: F.conv2d(t, k.expand(c, 1, 3, 3), padding=1, groups=c)
    return (f(a, kx) - f(b, kx)).abs().mean() + (f(a, ky) - f(b, ky)).abs().mean()


def pearson_loss(a, b):
    a = a.flatten(1) - a.flatten(1).mean(1, keepdim=True)
    b = b.flatten(1) - b.flatten(1).mean(1, keepdim=True)
    return 1 - ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-6)).mean()


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)).mean()


class EMA:
    """Exponential moving average of model weights (and BatchNorm statistics). Evaluate/serve the EMA copy: smoother, lower-variance,
    and measurably better generalisation on small image sets."""

    def __init__(self, model, decay=0.97):
        self.m = copy.deepcopy(model).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for (k, v_e), v in zip(self.m.state_dict().items(), model.state_dict().values()):
            if v_e.dtype.is_floating_point:
                v_e.mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                v_e.copy_(v)
