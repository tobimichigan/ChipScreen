"""bootstrap.py - install-if-absent for every third-party module the pipeline imports."""
import importlib
import subprocess
import sys

# import-name -> pip-name
CORE = {
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "sklearn": "scikit-learn",
    "matplotlib": "matplotlib", "tqdm": "tqdm", "psutil": "psutil", "joblib": "joblib",
    "pyarrow": "pyarrow", "openpyxl": "openpyxl", "lxml": "lxml", "tifffile": "tifffile",
    "skimage": "scikit-image", "cv2": "opencv-python-headless", "PIL": "pillow",
    "torch": "torch", "rdkit": "rdkit", "lightgbm": "lightgbm", "requests": "requests",
    "seaborn": "seaborn", "imagecodecs": "imagecodecs",
}
# needed for the LLM fine-tuning / retrieval stages (installed too, failures are tolerated)
LLM = {
    "transformers": "transformers", "peft": "peft", "accelerate": "accelerate",
    "huggingface_hub": "huggingface_hub", "sentence_transformers": "sentence-transformers",
}
OPTIONAL = {"umap": "umap-learn"}

_INSTALLED = set()


def _pip(pkg):
    cmd = [sys.executable, "-m", "pip", "install", "-q", pkg]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:  # externally-managed environments
        r = subprocess.run(cmd + ["--break-system-packages"], capture_output=True, text=True)
    return r.returncode == 0, (r.stderr or "")[-400:]


def ensure(mods, required=True, verbose=True):
    """Import each module; pip-install it first if it is missing. Returns list of unavailable modules."""
    missing_after = []
    for imp, pip_name in mods.items():
        if imp in _INSTALLED:
            continue
        try:
            importlib.import_module(imp)
            _INSTALLED.add(imp)
            continue
        except ImportError:
            pass
        if verbose:
            print(f"[bootstrap] '{imp}' not found -> pip install {pip_name}")
        ok, err = _pip(pip_name)
        importlib.invalidate_caches()
        try:
            importlib.import_module(imp)
            _INSTALLED.add(imp)
        except ImportError:
            missing_after.append(imp)
            msg = f"[bootstrap] could not install '{pip_name}': {err.strip()[:200]}"
            if required:
                raise ImportError(msg)
            print(msg)
    return missing_after


def ensure_all(with_llm=True, with_optional=True):
    ensure(CORE, required=True)
    missing = []
    if with_llm:
        missing += ensure(LLM, required=False)
    if with_optional:
        missing += ensure(OPTIONAL, required=False)
    return missing


def configure_gpu(cfg):
    """Cap GPU memory (PyTorch) and enable memory-growth (TensorFlow, only if it happens to be installed)."""
    info = {"cuda": False}
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda"] = True
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            torch.cuda.set_per_process_memory_fraction(float(cfg.gpu_mem_fraction))
            torch.backends.cudnn.benchmark = True
    except Exception as e:  # pragma: no cover
        info["torch_err"] = repr(e)
    try:
        import tensorflow as tf  # optional
        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass
    return info
