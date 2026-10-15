"""main.py - ChipScreen end-to-end entry point. Call from Jupyter cells (no argparse / sys.argv used):

    import sys; sys.path.insert(0, "/kaggle/working/chipscreen")     # folder that holds these modules
    from main import run_pipeline
    out = run_pipeline()                                              # everything
    out = run_pipeline(stages=["catalog", "chiptox"])                 # or selected stages
    # stage names: download, catalog, jump, chiptox, dosecurve, glow, chipcell, llm, agent
#
# Data: with cfg.auto_download=True (default) the Google Drive folder in config.py (gdrive_folder) is synced into cfg.root first, using the API key
# stored in the Kaggle Secret / environment variable named cfg.gdrive_secret ("GDRIVE_API_KEY"). The sync is incremental. Disable: run_pipeline(download=False).
Every stage runs inside its own try/except (graceful degradation); results of finished stages are kept in the returned dict."""
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())

import bootstrap
bootstrap.ensure_all(with_llm=True)      # pip-install any missing module BEFORE the rest of the pipeline imports it

import matplotlib
if "inline" not in matplotlib.get_backend().lower() and "ipympl" not in matplotlib.get_backend().lower():
    try:
        matplotlib.use("Agg")            # non-interactive backend outside notebooks; plt.show() then simply no-ops
    except Exception:
        pass

import numpy as np
import pandas as pd

from config import Config
import utils
from utils import LOG, Timer, force_cleanup, log_mem, save_json
import catalog as CAT

ALL_STAGES = ["download", "catalog", "jump", "chiptox", "dosecurve", "glow", "chipcell", "llm", "agent"]


def _guard(name, fn, out, errors):
    t0 = time.time()
    try:
        out[name] = fn()
        LOG.info(f"[main] stage '{name}' finished in {time.time()-t0:.0f}s")
    except Exception as e:
        errors[name] = traceback.format_exc()
        LOG.error(f"[main] stage '{name}' FAILED: {e!r} - continuing with remaining stages")
        out[name] = None
    finally:
        force_cleanup(name); log_mem(f"after {name}")


def run_pipeline(root=None, out_dir=None, stages=None, refresh_catalog=False, download=None, refresh_manifest=False,
                 show_plots=True, request="Assess the toxicity of the screened compounds, check plate quality and dose-response, and propose the next experiment.",
                 **cfg_overrides):
    stages = list(stages or ALL_STAGES)
    # root / out_dir default to whatever config.py says (Kaggle, Colab/Drive, local ...); pass them here only to override
    paths = {k: v for k, v in (("root", root), ("out", out_dir)) if v is not None}
    cfg = Config(show_plots=show_plots, **paths, **cfg_overrides)
    LOG.info(f"[main] root={cfg.root} | out={cfg.out}")
    utils.set_seed(cfg.seed)
    gpu = bootstrap.configure_gpu(cfg)
    LOG.info(f"[main] device info: {gpu}")
    out, errors = {"cfg": cfg}, {}
    t_all = time.time()

    # ---- 0a. data: incremental sync from Google Drive (API key from Kaggle Secrets / env) into cfg.root
    if (cfg.auto_download if download is None else download) and cfg.gdrive_folder:
        import gdrive
        _guard("download", lambda: gdrive.sync_dataset(cfg, refresh_manifest=refresh_manifest), out, errors)
        rep = out.get("download")
        if rep is None:
            have = os.path.isdir(cfg.root) and any(os.scandir(cfg.root))
            if not have:
                raise RuntimeError("Drive download failed and no local data exists in " + cfg.root + ":\n" + errors.get("download", "").strip().splitlines()[-1])
            LOG.warning("[main] Drive sync failed - continuing with the data already present in " + cfg.root)
        elif rep.get("downloaded", 0) > 0:
            refresh_catalog = True                                  # new files arrived -> the cached file catalog is stale
        if rep and rep.get("failed"):
            LOG.warning(f"[main] {rep['failed']} files failed to download; stages that need them will report it")
    # ---- 0b. recursive catalog + schema probing (+ role resolution)
    missing = [n for n in ("bootstrap", "catalog", "chem", "chipagent", "chipcell", "chiptox", "config", "dosecurve", "features", "glow", "jump", "models", "utils")
               if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".", n + ".py"))]
    if missing:
        raise FileNotFoundError(f"These module files are missing next to main.py: {missing}")
    cat = CAT.build_catalog(cfg.root, cfg, refresh=refresh_catalog)
    res = CAT.Resolver(cat, cfg.root)
    out["catalog"], out["resolver"] = cat, res
    if "catalog" in stages:
        _guard("schema", lambda: CAT.schema_report(res, cfg), out, errors)
    log_mem("catalog")

    # ---- 1. JUMP morphology profiles (feeds ChipTox multimodal, DoseCurve plate-QC, ChipCell MoA)
    morph = None
    if any(s in stages for s in ("jump", "chiptox", "dosecurve", "chipcell", "agent")):
        import jump
        _guard("jump", lambda: jump.build_morph(res, cfg), out, errors)
        morph = out.get("jump")

    # ---- 2. ChipTox
    if "chiptox" in stages or "agent" in stages or "llm" in stages:
        import chiptox
        _guard("tox", lambda: chiptox.run_chiptox(res, cfg, morph), out, errors)
    # ---- 3. DoseCurve QC Workbench
    if any(s in stages for s in ("dosecurve", "agent", "llm")):
        import dosecurve
        _guard("dose", lambda: dosecurve.run_dosecurve(res, cfg, morph), out, errors)
    # ---- 4. GlowFromLight
    if "glow" in stages or "agent" in stages:
        import glow
        _guard("glow", lambda: glow.run_glow(res, cfg), out, errors)
    # ---- 5. ChipCell Atlas
    if "chipcell" in stages or "agent" in stages:
        import chipcell
        _guard("cell", lambda: chipcell.run_chipcell(res, cfg, morph), out, errors)
    # ---- 6. LLM fine-tuning (LoRA, permissively licensed open model)
    if "llm" in stages and out.get("tox"):
        import chipagent
        _guard("llm", lambda: chipagent.finetune_llm(out["tox"], out.get("dose"), cfg), out, errors)
    # ---- 7. Agent (plan -> execute -> critique -> cite -> propose)
    if "agent" in stages:
        import chipagent
        def _agent():
            ag = chipagent.ChipAgent(cfg, res, dict(tox=out.get("tox"), dose=out.get("dose"), glow=out.get("glow"), cell=out.get("cell")))
            out["agent_obj"] = ag
            return ag.run(request)
        _guard("agent", _agent, out, errors)

    # ---- final consolidated summary
    if out.get("download") is not None or (cfg.auto_download if download is None else download):
        try:
            import gdrive
            out["download"] = gdrive.wait_background(cfg, out.get("download"))
            if out["download"] and out["download"].get("background_downloaded"):
                CAT.build_catalog(cfg.root, cfg, refresh=True)           # refresh file_catalog.csv with the full folder
        except Exception as e:                                           # never lose pipeline results because of the optional remainder
            LOG.warning(f"[main] background download wrap-up failed: {e!r}")
    try:
        summary = summarize(out, errors, cfg, time.time() - t_all)
    except Exception as e:                       # the summary must never hide the results of the stages that succeeded
        LOG.warning(f"[main] summary generation failed: {e!r}")
        summary = {"errors": {k: v.strip().splitlines()[-1] for k, v in errors.items()}, "summary_error": repr(e)}
    out["summary"], out["errors"] = summary, errors
    return out


def summarize(out, errors, cfg, seconds):
    S = {"download": out.get("download"), "seconds": round(seconds), "errors": {k: v.strip().splitlines()[-1] for k, v in errors.items()}, "plots": [p for _, p in utils.PLOT_REGISTRY]}
    t = out.get("tox")
    if t:
        b = t["best_name"]
        S["ChipTox"] = dict(best_model=b, split_sizes=t["split_report"]["sizes"], model_table=t["summary"].round(3).to_dict("index"),
                            holdout_macro_auroc=float(t["summary"].loc[b, "holdout_auroc"]), holdout_bal_acc=float(t["summary"].loc[b, "holdout_balacc"]),
                            gap=t["gap"][b]["gap_train_holdout"], overfit=t["gap"][b]["overfit"], recommendations=t["gap"][b]["recommendations"],
                            cv_train_val_only=t.get("cv"), poor_tasks=t["poor_tasks"][:15], n_outliers_downweighted=t["n_outliers"],
                            multimodal=({k: v for k, v in t["multimodal"].items() if k not in ("table",)} if isinstance(t.get("multimodal"), dict) else None),
                            uncertainty=t["uncertainty_holdout"])
    d = out.get("dose")
    if d:
        S["DoseCurve"] = dict(plates=(len(d["plate_qc"]) if d.get("plate_qc") is not None else 0), plates_passing=(int(d["plate_qc"].passed.sum()) if d.get("plate_qc") is not None else 0),
                              median_zprime=(float(d["plate_qc"].zprime.median()) if d.get("plate_qc") is not None else None), bbbc021=d.get("bbbc021_status"),
                              curve_flags=d.get("curve_flags"), quality_model=d.get("quality_model"), note=d.get("note"),
                              dose_tables_with_readout=len(d.get("dose_tables_with_readout", [])), dose_tables_metadata_only=len(d.get("dose_tables_metadata_only", [])))
    g = out.get("glow")
    if g:
        S["GlowFromLight"] = dict(mode=g.get("mode"), baseline_holdout={k: round(v["pearson"], 3) for k, v in g.get("baseline_holdout", {}).items()}, targets=g["targets"], holdout=g["metrics"].xs("holdout", level="split").round(3).to_dict("index"), trust=g["trust"], weak=g["weak"],
                                  gap={k: v["gap_train_holdout"] for k, v in g["gap"].items()})
    c = out.get("cell")
    if c:
        S["ChipCell"] = dict(seg=c["seg_summary"].round(3).to_dict("index"), seg_gap=c["seg_gap"]["gap_train_holdout"], weak_sources=c["weak_sources"],
                             moa=({k: v for k, v in c["moa"].items() if k not in ("table", "split_report")} if isinstance(c.get("moa"), dict) else None))
    if out.get("llm"):
        S["LLM"] = out["llm"]
    save_json(S, cfg.reports / "pipeline_summary.json")
    # text report
    L = ["CHIPSCREEN PIPELINE SUMMARY", "=" * 70, f"runtime {S['seconds']}s | plots saved: {len(S['plots'])} in {cfg.plots}", ""]
    if "ChipTox" in S:
        x = S["ChipTox"]; L += [f"ChipTox  best={x['best_model']}  holdout macro-AUROC={x['holdout_macro_auroc']:.3f}  bal.acc={x['holdout_bal_acc']:.3f}  train-holdout gap={x['gap']:.3f}  overfit={x['overfit']}"]
    if "DoseCurve" in S:
        x = S["DoseCurve"]; L += [f"DoseCurve plates={x['plates']} pass={x['plates_passing']} median Z'={x['median_zprime']} BBBC021={x['bbbc021']}"]
    if "GlowFromLight" in S:
        L += ["GlowFromLight holdout: " + "; ".join(f"{k}: r={v['pearson']:.2f} ssim={v['ssim']:.2f}" for k, v in S["GlowFromLight"]["holdout"].items())]
    if "ChipCell" in S:
        L += [f"ChipCell holdout: {S['ChipCell']['seg'].get('holdout')}"]
    for k, v in S["errors"].items():
        L += [f"STAGE FAILED {k}: {v}"]
    (cfg.reports / "pipeline_summary.txt").write_text("\n".join(L))
    LOG.info("\n" + "\n".join(L))
    return S


if __name__ == "__main__":     # works with `%run main.py` in a notebook cell (no argparse)
    OUT = run_pipeline()
