"""catalog.py - recursive discovery of every file under the root, role resolution (which file feeds which module),
schema probing (feature / metadata / label columns) and dose-response-table discovery."""
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from utils import LOG, tq, save_json

CATEGORY = {
    ".csv": "table", ".tsv": "table", ".txt": "text", ".parquet": "table", ".xlsx": "table", ".xls": "table",
    ".json": "json", ".tif": "image", ".tiff": "image", ".png": "image", ".jpg": "image", ".jpeg": "image",
    ".svg": "image", ".xml": "xml", ".nxml": "xml", ".pdf": "pdf", ".soft": "text", ".md": "text", ".yml": "text",
    ".yaml": "text", ".py": "code", ".sh": "code", ".docx": "doc",
}
SKIP_DIR_PARTS = ("__MACOSX", ".ipynb_checkpoints")
DOSE_PAT = re.compile(r"dose|conc|\bum\b|µm|micromol|molar", re.I)
READOUT_PAT = re.compile(r"viab|count|intens|inhib|respons|signal|growth|surviv|cytotox|activity|area|auc|ic50|%", re.I)


def _scan_dir(d):
    """One directory level -> (sub-directories, file names). os.scandir uses d_type so no per-entry stat is needed here."""
    subs, files = [], []
    try:
        with os.scandir(d) as it:
            for e in it:
                try:
                    (subs if e.is_dir(follow_symlinks=False) else files).append(e.path)
                except OSError:
                    continue
    except OSError:
        pass
    return subs, files


def _walk_parallel(root, workers=32):
    """Breadth-first parallel directory walk (a serial os.walk over ~50k files on Google Drive took 23 minutes)."""
    from concurrent.futures import ThreadPoolExecutor
    files, level = [], [str(root)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        while level:
            nxt = []
            for subs, fl in ex.map(_scan_dir, level):
                nxt += [s for s in subs if not any(p in s for p in SKIP_DIR_PARTS)]
                files += fl
            level = nxt
    return files


def _sizes(paths, workers=32):
    from concurrent.futures import ThreadPoolExecutor

    def sz(p):
        try:
            return os.path.getsize(p)
        except OSError:
            return -1
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(tq(ex.map(sz, paths, chunksize=64), "catalog: file sizes", total=len(paths)))


def build_catalog(root, cfg=None, refresh=False):
    """Recursive discovery of every file under `root` (all folders / sub-folders). The result is cached in cfg.cache so reruns skip the walk."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")
    cache = (cfg.cache / "file_catalog_full.parquet") if cfg is not None else None
    meta = (cfg.cache / "file_catalog_full.root.txt") if cfg is not None else None
    if cache is not None and not refresh and getattr(cfg, "cache_catalog", True) and cache.exists() and meta.exists() and meta.read_text() == str(root):
        df = pd.read_parquet(cache)
        LOG.info(f"[catalog] loaded cached catalog: {len(df):,} files ({df.bytes.sum()/1e9:.2f} GB). Use run_pipeline(refresh_catalog=True) after adding data.")
    else:
        workers = getattr(cfg, "io_workers", 32) if cfg is not None else 32
        paths = [p for p in _walk_parallel(root, workers) if not any(x in p for x in SKIP_DIR_PARTS) and not p.endswith(".part")]
        sizes = _sizes(paths, workers)
        rel = [os.path.relpath(p, root) for p in paths]
        df = pd.DataFrame(dict(rel=rel, name=[os.path.basename(p) for p in paths], ext=[os.path.splitext(p)[1].lower() for p in paths], bytes=sizes,
                               source=[r.split(os.sep)[0] for r in rel], path=paths))
        df = df[df.bytes >= 0].reset_index(drop=True)
        df["category"] = df.ext.map(lambda e: CATEGORY.get(e, "other"))
        if cache is not None:
            df.to_parquet(cache); meta.write_text(str(root))
    LOG.info(f"[catalog] {len(df):,} files, {df.bytes.sum()/1e9:.2f} GB under {root}")
    if cfg is not None:
        df.drop(columns=["path"]).to_csv(cfg.out_p / "file_catalog.csv", index=False)
        summ = df.groupby("source").agg(files=("rel", "size"), gb=("bytes", lambda b: round(b.sum() / 1e9, 3)))
        LOG.info("\n" + summ.to_string())
    return df


class Resolver:
    """Maps pipeline roles -> concrete files/folders found by the recursive walk (nothing is hard-coded to a depth)."""

    def __init__(self, cat, root):
        self.cat = cat
        self.root = Path(root)
        self.paths = dict(zip(cat["rel"], cat["path"]))

    def find(self, name_regex, source=None, ext=None):
        d = self.cat
        if source:
            d = d[d.source == source]
        if ext:
            d = d[d.ext == ext]
        m = d[d["name"].str.contains(name_regex, regex=True, na=False)]
        return sorted(m["path"].tolist())

    def one(self, name_regex, **kw):
        r = self.find(name_regex, **kw)
        return r[0] if r else None

    # ---- chemistry / toxicity
    def tox21(self):
        return self.one(r"^tox21\.csv$")

    def toxcast(self):
        return self.one(r"^toxcast_data\.csv$")

    # ---- JUMP
    def jump_profiles(self):
        d = self.cat[(self.cat.source == "jump_cell_painting") & (self.cat.ext == ".parquet")]
        return sorted(d["path"].tolist())

    def jump_meta(self, stem):
        return self.one(rf"^{stem}\.csv$", source="jump_cell_painting")

    # ---- annotation
    def repurposing_drugs(self):
        return self.one(r"^repurposing_drugs.*\.txt$")

    def repurposing_samples(self):
        return self.one(r"^repurposing_samples.*\.txt$")

    def livertox_master(self):
        return self.one(r"^masterlist.*\.xlsx$")

    # ---- BBBC
    def bbbc(self, bid, name_regex):
        d = self.cat[self.cat.rel.str.startswith(f"bbbc{os.sep}{bid}")]
        m = d[d["name"].str.contains(name_regex, regex=True, na=False)]
        return sorted(m["path"].tolist())

    # ---- images
    def allen_volumes(self):
        d = self.cat[(self.cat.source == "allen_cell_explorer") & self.cat["name"].str.lower().str.endswith((".ome.tif", ".ome.tiff"))]
        return sorted(d["path"].tolist())

    def bbbc038_train(self):
        """-> list of (image_path, [mask paths]) for every stage1_train image id (folder/<id>/images/<id>.png + masks/*.png)."""
        d = self.cat[self.cat.rel.str.contains(r"BBBC038[\\/]stage1_train[\\/]", regex=True) & (self.cat.ext == ".png")]
        imgs, masks = {}, {}
        for rel, path in zip(d["rel"], d["path"]):
            parts = rel.replace("\\", "/").split("/")
            if len(parts) < 5:
                continue
            iid, kind = parts[-3], parts[-2]
            if kind == "images":
                imgs[iid] = path
            elif kind == "masks":
                masks.setdefault(iid, []).append(path)
        return [(imgs[i], sorted(masks.get(i, []))) for i in sorted(imgs) if i in masks]

    def ctc_pairs(self, max_per_seq=30, seed=0):
        """Cell Tracking Challenge: pair every man_seg GT frame with its raw frame. -> list of dict(ds, seq, raw, gt)."""
        d = self.cat[(self.cat.source == "cell_tracking_challenge") & (self.cat.ext.isin([".tif", ".tiff"]))]
        raw, gt = {}, {}
        for rel, path in zip(d["rel"], d["path"]):
            parts = rel.replace("\\", "/").split("/")
            try:
                i = parts.index("training")
            except ValueError:
                continue
            ds = parts[i + 1]
            fn = parts[-1]
            seq_dir = parts[-2]
            if re.match(r"^t\d+\.tif", fn) and re.fullmatch(r"\d+", seq_dir):
                raw[(ds, seq_dir, int(re.findall(r"\d+", fn)[0]))] = path
            elif fn.startswith("man_seg") and parts[-2] == "SEG" and re.fullmatch(r"\d+_GT", parts[-3]):
                gt[(ds, parts[-3].split("_")[0], int(re.findall(r"\d+", fn)[0]))] = path
        keys = sorted(k for k in gt if k in raw)
        rng = np.random.default_rng(seed)
        by_seq = {}
        for k in keys:
            by_seq.setdefault((k[0], k[1]), []).append(k)
        out = []
        for (ds, seq), ks in by_seq.items():
            if len(ks) > max_per_seq:
                ks = [ks[i] for i in sorted(rng.choice(len(ks), max_per_seq, replace=False))]
            out += [dict(ds=ds, seq=seq, frame=k[2], raw=raw[k], gt=gt[k]) for k in ks]
        return out

    # ---- literature
    def pmc_xml(self):
        return sorted(self.cat[(self.cat.source == "pmc_open_access") & (self.cat.ext == ".xml")]["path"].tolist())

    def livertox_nxml(self):
        return sorted(self.cat[(self.cat.source == "livertox") & (self.cat.ext == ".nxml")]["path"].tolist())

    # ---- IDR / HPA
    def idr_annotations(self):
        return self.find(r"annotation\.csv$", source="idr")

    def hpa(self, name):
        return self.one(rf"^{name}\.tsv$", source="hpa")


# --------------------------------------------------------------------------- schema probing
def read_head(path, n=200):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        return pd.DataFrame(columns=pf.schema_arrow.names), pf.metadata.num_rows
    if ext in (".csv", ".tsv", ".txt"):
        sep = "\t" if ext in (".tsv", ".txt") else ","
        df = pd.read_csv(path, sep=sep, nrows=n, comment="!" if "repurposing" in path else None, low_memory=False, on_bad_lines="skip")
        return df, None
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, nrows=n), None
    raise ValueError(ext)


def classify_columns(cols):
    """Split columns into metadata / morphological-feature / label-like groups (JUMP/CellProfiler naming aware)."""
    meta = [c for c in cols if c.startswith("Metadata_") or c.lower().startswith(("image_metadata", "image_filename", "image_pathname"))]
    feat = [c for c in cols if re.match(r"^(Cells|Cytoplasm|Nuclei)_", c)]
    other = [c for c in cols if c not in meta and c not in feat]
    return dict(metadata=meta, morph_features=feat, other=other)


def schema_report(res, cfg):
    """Probe the key tables and persist schema JSON (columns, dtypes, feature groups) for provenance."""
    targets = dict(tox21=res.tox21(), toxcast=res.toxcast(), jump_well=res.jump_meta("well"), jump_plate=res.jump_meta("plate"),
                   jump_compound=res.jump_meta("compound"), jump_ctrl=res.jump_meta("perturbation_control"),
                   repurposing_drugs=res.repurposing_drugs(), repurposing_samples=res.repurposing_samples(),
                   bbbc021_image=(res.bbbc("BBBC021", r"image\.csv$") or [None])[0],
                   bbbc021_moa=(res.bbbc("BBBC021", r"moa\.csv$") or [None])[0],
                   bbbc021_compound=(res.bbbc("BBBC021", r"compound\.csv$") or [None])[0])
    pp = res.jump_profiles()
    if pp:
        targets["jump_profile_example"] = pp[0]
    rep = {}
    for k, p in tq(targets.items(), "schema probe"):
        if not p:
            rep[k] = dict(path=None, status="NOT FOUND")
            continue
        try:
            df, nrows = read_head(p)
            cols = list(df.columns)
            rep[k] = dict(path=os.path.relpath(p, cfg.root), n_columns=len(cols), n_rows=nrows, columns=cols[:60],
                          groups={g: len(v) for g, v in classify_columns(cols).items()},
                          dtypes={c: str(t) for c, t in list(df.dtypes.items())[:40]})
        except Exception as e:
            rep[k] = dict(path=p, status=f"unreadable: {e!r}")
    save_json(rep, cfg.out_p / "schema_report.json")
    return rep


def discover_dose_tables(cat, max_tables=400):
    """Scan table headers across the whole tree for (dose column + numeric readout column) pairs."""
    found, meta_only = [], []
    d = cat[(cat.ext.isin([".csv", ".tsv"])) & (cat.bytes > 0) & (cat.bytes < 400e6)]
    for path, rel in tq(list(zip(d["path"], d["rel"]))[:max_tables * 5], "scan for dose-response tables"):
        try:
            sep = "\t" if path.endswith(".tsv") else ","
            h = pd.read_csv(path, sep=sep, nrows=50, low_memory=False, on_bad_lines="skip")
        except Exception:
            continue
        dose = [c for c in h.columns if DOSE_PAT.search(str(c))]
        if not dose:
            continue
        num = [c for c in h.columns if pd.api.types.is_numeric_dtype(h[c]) and c not in dose and READOUT_PAT.search(str(c))]
        (found if num else meta_only).append(dict(rel=rel, dose_cols=dose, readout_cols=num[:6]))
        if len(found) >= max_tables:
            break
    return found, meta_only
