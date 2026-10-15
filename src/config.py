"""config.py - single configuration object shared by every module."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


def _env():
    """Kaggle -> /kaggle/working ; Colab -> local /content disk (NOT the Drive mount: Drive FUSE is ~50x slower) ; else current directory."""
    if os.path.exists("/kaggle/working"):
        return dict(root="/kaggle/working/open_datasets", out="/kaggle/working/chipscreen_out",
                    hf="/kaggle/input/datasets/researchdevelopment2/access-tkns/access_tkns/hf.token.txt")
    if os.path.exists("/content"):
        return dict(root="/content/open_datasets", out="/content/chipscreen_out", hf="/content/hf.token.txt")
    cwd = Path.cwd()
    return dict(root=str(cwd / "open_datasets"), out=str(cwd / "chipscreen_out"), hf=str(cwd / "hf.token.txt"))


@dataclass
class Config:
    # ---- locations ----
    root: str = field(default_factory=lambda: _env()["root"])
    out: str = field(default_factory=lambda: _env()["out"])
    hf_token_path: str = field(default_factory=lambda: _env()["hf"])

    # ---- Google Drive sync (public / link-shared folders, read with an API key stored as a Kaggle Secret / env var) ----
    gdrive_folder: str = "https://drive.google.com/drive/folders/1WDwh3wGxof-P4Ja1x24hqr_yig6kRk6H"
    gdrive_secret: str = "GDRIVE_API_KEY"    # name of the Kaggle Secret / environment variable that holds the API key
    auto_download: bool = True               # sync before the pipeline runs (incremental: complete files are skipped)
    download_profile: str = "all"            # "all" = the full folder (8.76 GB) | "pipeline" = only the files the stages read (~4.9 GB)
    download_background_rest: bool = True    # with "all": fetch the files the pipeline reads FIRST, then the remainder in a background thread while the stages run
    download_extra: Tuple[str, ...] = ()     # extra regexes (matched on the relative path) to download on top of the profile
    download_workers: int = 24
    gdrive_api_base: str = "https://www.googleapis.com/drive/v3"

    # ---- I/O ----
    io_workers: int = 24             # threads for file reads (Google Drive / network disks are latency-bound)
    cache_catalog: bool = True

    # ---- reproducibility / plotting ----
    seed: int = 42
    show_plots: bool = True          # plt.show() after every plt.savefig()
    dpi: int = 160

    # ---- splits (strict: holdout never used for fitting, tuning, early stopping or model selection) ----
    split_fracs: Tuple[float, float, float, float] = (0.40, 0.15, 0.15, 0.30)  # train, val, test, holdout
    cv_folds: int = 5                # CV bagging runs on train+val ONLY (test and holdout are never touched)
    overfit_gap_warn: float = 0.05   # absolute metric gap that triggers an overfitting warning

    # ---- memory safety ----
    mem_limit_frac: float = 0.88     # stop an operation when system RAM use exceeds this fraction
    gpu_mem_fraction: float = 0.90
    chunk_rows: int = 8000           # row chunk for large CSV/TSV reads
    max_jump_plates: Optional[int] = None   # None = all plates found
    max_morph_feats: int = 1200      # morphological features kept after unsupervised filtering

    # ---- ChipTox ----
    toxcast_min_pos: int = 40
    toxcast_max_tasks: int = 40
    fp_bits: int = 2048
    deep_epochs: int = 120
    deep_patience: int = 14
    deep_batch: int = 256
    search_iters: int = 8           # randomized-search iterations per classical model
    mc_passes: int = 25
    chiptox_models: Tuple[str, ...] = ("lgbm", "rf", "logreg", "attn_mlp", "cnn_attn")
    min_bit_freq: float = 0.01       # fingerprint bits present in <1% of training molecules are dropped (noise)
    select_gap_penalty: float = 0.10 # selection score = mean(test, OOF AUROC) - penalty * max(0, train-OOF gap - 0.10)

    # ---- GlowFromLight ----
    glow_size: int = 192
    glow_z_per_volume: int = 18
    glow_epochs: int = 60
    glow_patience: int = 10
    glow_batch: int = 12
    glow_base: int = 24
    glow_mc: int = 10
    glow_tol: float = 0.10           # tolerance (normalised intensity units) for tolerance-percentage metric

    # ---- ChipCell ----
    seg_size: int = 256
    seg_epochs: int = 60
    seg_patience: int = 10
    seg_batch: int = 12
    seg_base: int = 24
    max_ctc_per_seq: int = 30
    max_cells_per_image: int = 200
    min_moa_compounds: int = 6
    max_moa_classes: int = 15

    # ---- LLM / agent ----
    run_llm_finetune: bool = True
    llm_max_steps: int = 150
    llm_candidates: Tuple[str, ...] = (
        "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct")
    allowed_licenses: Tuple[str, ...] = ("apache-2.0", "mit")
    approvals: dict = field(default_factory=dict)   # {"action-name": "approval-token"} for the Gatekeeper
    quote_word_limit: int = 25

    # ---- derived paths ----
    def __post_init__(self):
        self.root_p = Path(self.root)
        self.out_p = Path(self.out)
        self.plots = self.out_p / "plots"
        self.models = self.out_p / "models"
        self.reports = self.out_p / "reports"
        self.cache = self.out_p / "cache"
        for p in (self.out_p, self.plots, self.models, self.reports, self.cache):
            p.mkdir(parents=True, exist_ok=True)
