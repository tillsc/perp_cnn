import torch
from pathlib import Path

def get_best_device():
    """Select best available device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        print("🚀 Using CUDA GPU:", torch.cuda.get_device_name(0))
        return "cuda"
    elif torch.backends.mps.is_available():
        print("🚀 Using Apple MPS GPU")
        return "mps"
    else:
        print("⚙️ Using CPU")
        return "cpu"

def get_run_path(run_name: str = None, prefix: str = "bowtips_detect_") -> Path:
    """
    Resolve the path to a specific or latest run directory.

    Args:
        run_name (str): Optional run directory name.
        prefix (str): Prefix used for naming runs.

    Returns:
        Path to run directory or raises FileNotFoundError.
    """
    runs_path = Path("runs/detect")

    if run_name:
        run_dir = runs_path / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"❌ Specified run directory not found: {run_dir}")
    else:
        run_dirs = sorted(runs_path.glob(f"{prefix}*"), key=lambda d: d.stat().st_mtime, reverse=True)
        if not run_dirs:
            raise FileNotFoundError("❌ No trained detection runs found.")
        run_dir = run_dirs[0]

    return run_dir
