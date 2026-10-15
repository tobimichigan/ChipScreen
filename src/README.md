# ChipScreen pipeline

Copy this whole folder to the notebook machine (e.g. /kaggle/working/chipscreen) - all 15 .py files must sit side by side.

    import sys; sys.path.insert(0, "/kaggle/working/chipscreen")
    from main import run_pipeline
    out = run_pipeline()                      # sync Drive -> catalog -> jump -> chiptox -> dosecurve -> glow -> chipcell -> llm -> agent

Kaggle: enable Internet and add a Secret named GDRIVE_API_KEY (tick it for the notebook). Optional secret HF_TOKEN for the LLM stage.
Data: the full Drive folder (8.76 GB) is synced into /kaggle/working/open_datasets. Phase 1 fetches the files the stages read (~4.9 GB) and the pipeline
starts; phase 2 fetches the remainder in a background thread and is joined at the end. Progress is logged every 20 s. Re-running resumes (complete files are skipped).
Options: run_pipeline(stages=[...], download=False, refresh_catalog=True, refresh_manifest=True, root=..., out_dir=..., <any Config field>=...)

Files
  main.py        entry point (stages, error isolation, summary)        config.py     all settings (environment-aware paths)
  gdrive.py      Drive API-key sync (selective, resumable)             catalog.py    parallel recursive file discovery + role resolver
  jump.py        JUMP Cell Painting profiles (streamed)                chem.py       RDKit featurisation
  features.py    leak-free preprocessing, outlier weights, EDA          models.py     deep/classical models, training loop, EMA
  chiptox.py     ChipTox: CV-bagged toxicity ensemble                  dosecurve.py  DoseCurve QC: plate QC + 4PL dose-response
  glow.py        GlowFromLight: bright-field -> fluorescence + trust   chipcell.py    ChipCell Atlas: segmentation + MoA / cell phenotypes
  chipagent.py   ChipAgent: planner/executor/critic/librarian/gatekeeper + LoRA LLM fine-tune
  utils.py       splits, leakage checks, metrics, parallel IO, secrets  bootstrap.py   pip-installs missing packages

Outputs: <out>/plots, <out>/reports (pipeline_summary.json/txt, per-stage CSVs, agent report), <out>/models, <out>/cache
