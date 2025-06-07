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
    Get path to a specific or latest detection run, based on run folder name.

    Args:
        run_name (str): Optional specific run name.
        prefix (str): Prefix to identify run directories.

    Returns:
        Path: Path to selected run directory.
    """
    runs_path = Path("runs/detect")

    if run_name:
        run_dir = runs_path / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"❌ Specified run directory not found: {run_dir}")
    else:
        run_dirs = sorted(
            runs_path.glob(f"{prefix}*"),
            key=lambda d: d.name,
            reverse=True  # newest name first (e.g. by date+time string)
        )
        if not run_dirs:
            raise FileNotFoundError("❌ No trained detection runs found.")
        run_dir = run_dirs[0]

    return run_dir
