"""jump.py - memory-safe construction of morphological profiles from the JUMP Cell Painting parquet plates.
Streaming per plate: robust-z against the plate's negative controls. Outputs
  * compound-level mean profiles (ChipTox multimodal, ChipCell MoA baseline)
  * a well-level QC table (DoseCurve plate QC: Z', edge effects, drift)
  * capped per-compound well-level profiles (ChipCell MoA: many replicate observations per compound)
All joins are key-normalised and diagnosed: an unmappable plate is skipped with a clear log line instead of crashing the stage."""
import os
import re

import numpy as np
import pandas as pd

import chem
from catalog import classify_columns
from utils import LOG, tq, check_memory, force_cleanup, MemoryLimitExceeded, downcast_df


def norm_well(w):
    """'A1' / 'a01' / 'A001' -> 'A01'"""
    m = re.match(r"^\s*([A-Za-z]+)\s*0*(\d+)\s*$", str(w))
    return f"{m.group(1).upper()}{int(m.group(2)):02d}" if m else str(w).strip().upper()


def parse_well(w):
    m = re.match(r"^([A-Z]+)(\d+)$", norm_well(w))
    if not m:
        return 0, 0
    return ord(m.group(1)[0]) - 65, int(m.group(2))


def _control_ids(res):
    """-> (negcon ids, poscon ids) from perturbation_control.csv; empty sets if the file is missing/unparseable."""
    p = res.jump_meta("perturbation_control")
    if not p:
        return set(), set()
    c = pd.read_csv(p, dtype=str)
    jc = "Metadata_JCP2022" if "Metadata_JCP2022" in c.columns else c.columns[0]
    tcs = [x for x in c.columns if "type" in x.lower()]
    tc = tcs[0] if tcs else c.columns[-1]
    t = c[tc].fillna("").str.lower()
    return set(c.loc[t.str.contains("neg"), jc]), set(c.loc[t.str.contains("pos"), jc])


def build_morph(res, cfg, max_wells_per_compound=60):
    """Returns dict(morph, well, feat_names, compound_meta, well_X, well_ik) or None when JUMP data are absent."""
    cm, cw, cx, cmeta = (cfg.cache / n for n in ("jump_morph_compound.parquet", "jump_well_activity.parquet", "jump_well_profiles.npz", "jump_compound_meta.parquet"))
    if all(p.exists() for p in (cm, cw, cx, cmeta)):
        m = pd.read_parquet(cm)
        z = np.load(cx, allow_pickle=True)
        LOG.info(f"[jump] loaded cached profiles: compounds {m.shape}, well profiles {z['X'].shape}")
        return dict(morph=m, well=pd.read_parquet(cw), feat_names=list(z["feat"]), compound_meta=pd.read_parquet(cmeta), well_X=z["X"], well_ik=z["ik"])
    plates = res.jump_profiles()
    if cfg.max_jump_plates:
        plates = plates[: cfg.max_jump_plates]
    if not plates or not res.jump_meta("well") or not res.jump_meta("compound"):
        LOG.warning("[jump] profiles or metadata missing - morphology modality unavailable")
        return None
    import pyarrow.parquet as pq
    names0 = pq.ParquetFile(plates[0]).schema_arrow.names
    feats_all = classify_columns(names0)["morph_features"]
    if not feats_all:
        raise RuntimeError(f"no Cells_/Cytoplasm_/Nuclei_ feature columns in {plates[0]} (columns: {names0[:8]})")
    LOG.info(f"[jump] {len(plates)} plates, {len(feats_all)} feature columns")
    # ---------------------------------------------------------------- metadata
    well = pd.read_csv(res.jump_meta("well"), usecols=lambda c: c in ("Metadata_Source", "Metadata_Plate", "Metadata_Well", "Metadata_JCP2022"), dtype=str)
    well["well"] = well.Metadata_Well.map(norm_well)
    plate_names = {os.path.splitext(os.path.basename(p))[0] for p in plates}
    well = well[well.Metadata_Plate.isin(plate_names)].drop_duplicates(["Metadata_Plate", "well"])
    comp = pd.read_csv(res.jump_meta("compound"), dtype=str)
    comp = comp[comp.Metadata_JCP2022.isin(set(well.Metadata_JCP2022))].reset_index(drop=True)
    LOG.info(f"[jump] well table rows for these plates: {len(well):,}; distinct perturbations: {well.Metadata_JCP2022.nunique():,}; compounds with structures: {len(comp):,}")
    if len(well) == 0:
        raise RuntimeError(f"well.csv has no rows for the {len(plate_names)} downloaded plates (example plate id {sorted(plate_names)[0]})")
    smi_col = "Metadata_SMILES" if "Metadata_SMILES" in comp.columns else None
    if smi_col is None:
        raise RuntimeError(f"compound.csv has no Metadata_SMILES column (columns: {list(comp.columns)})")
    ct = chem.smiles_table(comp[smi_col].fillna("").tolist(), "JUMP compounds -> InChIKey")
    comp["ik14"], comp["std_smiles"], comp["scaffold"] = ct["ik14"].values, ct["std_smiles"].values, ct["scaffold"].values
    if "Metadata_InChIKey" in comp.columns:                       # fall back to the supplied key where standardisation failed
        fb = comp["Metadata_InChIKey"].astype(str).str.split("-").str[0]
        comp["ik14"] = comp["ik14"].fillna(fb.where(fb.str.len() == 14))
    jc2ik = dict(zip(comp.Metadata_JCP2022, comp.ik14))
    neg_ids, pos_ids = _control_ids(res)
    if not neg_ids or not (well.Metadata_JCP2022.isin(neg_ids)).any():
        top = well.Metadata_JCP2022.value_counts()
        cand = top[~top.index.isin(jc2ik)].index[:1].tolist() or top.index[:1].tolist()
        neg_ids = set(cand)
        LOG.warning(f"[jump] no negative-control ids matched the wells; using the most frequent perturbation {cand} as DMSO/negcon")
    well["ik14"] = well.Metadata_JCP2022.map(jc2ik)
    n_comp_wells = int(well.ik14.notna().sum())
    LOG.info(f"[jump] wells mapped to a compound structure: {n_comp_wells:,} / {len(well):,}; negcon ids: {sorted(neg_ids)[:3]}; poscon ids: {len(pos_ids)}")
    if n_comp_wells == 0:
        raise RuntimeError("no JUMP well could be mapped to a compound InChIKey (check compound.csv / well.csv)")
    by_plate = {p: g.set_index("well") for p, g in well.groupby("Metadata_Plate")}
    uniq = sorted(set(well.ik14.dropna()))
    idx = {k: i for i, k in enumerate(uniq)}
    # ---------------------------------------------------------------- feature screen (NaN-heavy / constant columns dropped)
    probe = [pd.read_parquet(p, columns=feats_all).to_numpy(np.float32) for p in plates[: min(8, len(plates))]]
    P = np.vstack(probe); P[~np.isfinite(P)] = np.nan
    ok = (np.isnan(P).mean(0) < 0.1) & (np.nanstd(P, 0) > 1e-8)
    sel = [f for f, k in zip(feats_all, ok) if k][:4000]
    LOG.info(f"[jump] {len(sel)} usable features after NaN/constant screening")
    del P, probe; force_cleanup()
    S = np.zeros((len(uniq), len(sel)), np.float32); N = np.zeros(len(uniq), np.int32)
    keep_w, keep_ik = [], []
    cnt_w = np.zeros(len(uniq), np.int32)
    rng = np.random.default_rng(cfg.seed)
    wells_out, skipped = [], 0
    from concurrent.futures import ThreadPoolExecutor
    rd_cols = ["Metadata_Well"] + sel
    pool = ThreadPoolExecutor(max_workers=min(6, max(2, cfg.io_workers // 4)))
    ahead = 6                                                       # read-ahead: plate k+1..k+6 load while plate k is being normalised
    futs = [pool.submit(pd.read_parquet, p, columns=rd_cols) for p in plates[:ahead]]
    for pi, p in enumerate(tq(plates, "jump: streaming plates")):
        try:
            check_memory(cfg, "jump plates")
        except MemoryLimitExceeded as e:
            LOG.warning(f"[jump] stopping early: {e}")
            break
        if pi + ahead < len(plates):
            futs.append(pool.submit(pd.read_parquet, plates[pi + ahead], columns=rd_cols))
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            df = futs[pi].result()
        except Exception as e:
            LOG.warning(f"[jump] plate {stem} unreadable ({e!r}); skipped")
            skipped += 1
            futs[pi] = None
            continue
        futs[pi] = None
        wk = by_plate.get(stem)
        if wk is None:
            skipped += 1
            continue
        wn = df.Metadata_Well.map(norm_well)
        j = wn.map(wk["Metadata_JCP2022"]); ik = wn.map(wk["ik14"])
        if j.notna().sum() == 0:
            skipped += 1
            if skipped <= 2:
                LOG.warning(f"[jump] plate {stem}: no well matched (parquet wells {wn.head(3).tolist()} vs table {list(wk.index[:3])})")
            continue
        X = np.array(df[sel].to_numpy(), dtype=np.float32, copy=True); X[~np.isfinite(X)] = np.nan
        negm = j.isin(neg_ids).to_numpy()
        ref = X[negm] if negm.sum() >= 4 else X
        med = np.nanmedian(ref, 0); mad = np.nanmedian(np.abs(ref - med), 0) * 1.4826
        sd = np.nanstd(ref, 0)
        mad = np.where(mad < 1e-6, sd + 1e-6, mad)
        Z = np.nan_to_num(np.clip((X - med) / mad, -10, 10), nan=0.0).astype(np.float32)
        act = np.sqrt((Z ** 2).mean(1))
        r_c = [parse_well(w) for w in wn]
        ctype = np.where(negm, "negcon", np.where(j.isin(pos_ids).to_numpy(), "poscon", np.where(ik.notna().to_numpy(), "compound", "other")))
        wells_out.append(pd.DataFrame(dict(plate=stem, well=wn.values, row=[a for a, _ in r_c], col=[b for _, b in r_c], jcp=j.values, ik14=ik.values, ctrl=ctype, activity=act)))
        good = np.where(ik.notna().to_numpy() & ~negm)[0]
        if len(good):
            ii = np.fromiter((idx[k] for k in ik.to_numpy()[good]), dtype=np.int64, count=len(good))
            np.add.at(S, ii, Z[good]); np.add.at(N, ii, 1)
            # reservoir-style cap on stored well profiles per compound
            for g, i_ in zip(good, ii):
                if cnt_w[i_] < max_wells_per_compound:
                    keep_w.append(Z[g].astype(np.float16)); keep_ik.append(uniq[i_]); cnt_w[i_] += 1
                else:
                    cnt_w[i_] += 1
        del df, X, Z; force_cleanup()
    pool.shutdown(wait=False)
    if skipped:
        LOG.warning(f"[jump] {skipped}/{len(plates)} plates could not be joined to the well table and were skipped")
    if N.sum() == 0:
        raise RuntimeError("JUMP streaming finished with zero compound wells - check plate ids / well names")
    keepc = N > 0
    morph = pd.DataFrame(S[keepc] / N[keepc, None], columns=sel)
    morph.insert(0, "ik14", np.array(uniq)[keepc]); morph.insert(1, "n_wells", N[keepc])
    morph = downcast_df(morph)
    wells = pd.concat(wells_out, ignore_index=True)
    WX, WI = np.stack(keep_w), np.array(keep_ik)
    cmeta_df = comp[["Metadata_JCP2022", "ik14", "std_smiles", "scaffold"]].dropna(subset=["ik14"]).drop_duplicates("ik14")
    morph.to_parquet(cm); wells.to_parquet(cw); cmeta_df.to_parquet(cmeta)
    np.savez_compressed(cx, X=WX, ik=WI, feat=np.array(sel))
    LOG.info(f"[jump] compound profiles {morph.shape[0]:,} x {len(sel)}; well profiles {WX.shape}; QC well table {wells.shape}; plates ok {len(wells_out)}/{len(plates)}")
    return dict(morph=morph, well=wells, feat_names=sel, compound_meta=cmeta_df, well_X=WX, well_ik=WI)
