import argparse
import subprocess
import sys
from pathlib import Path


def find_latest_checkpoint(results_dir: Path) -> Path:
    """Find the most recent result directory with a checkpoint."""
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    checkpoint_dirs = []
    for result_path in results_dir.iterdir():
        if result_path.is_dir():
            checkpoint_file = result_path / "gibbs_checkpoint_step.npz"
            if checkpoint_file.exists():
                checkpoint_dirs.append(result_path)
    
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint files found in: {results_dir}")
    
    # Sort by modification time, latest first
    latest = max(checkpoint_dirs, key=lambda p: (p / "gibbs_checkpoint_step.npz").stat().st_mtime)
    return latest


def main():
    parser = argparse.ArgumentParser(description="Restart ABC-Gibbs from checkpoint")
    parser.add_argument(
        "--results-dir", 
        type=str, 
        default="./results",
        help="Directory containing result subdirectories (default: ./results)"
    )
    parser.add_argument(
        "--resume-from", 
        type=str,
        help="Specific result directory to resume from (overrides auto-detection)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json", 
        help="Configuration file path (default: config.json)"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List available checkpoint directories and exit"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if args.list:
        print(f"Searching for checkpoints in: {results_dir}")
        checkpoint_dirs = []
        for result_path in results_dir.iterdir():
            if result_path.is_dir():
                checkpoint_file = result_path / "gibbs_checkpoint_step.npz"
                if checkpoint_file.exists():
                    # Get checkpoint info
                    import numpy as np
                    try:
                        ckpt = np.load(checkpoint_file, allow_pickle=True)
                        step = int(ckpt["step"]) + 1
                        checkpoint_dirs.append((result_path.name, step, checkpoint_file.stat().st_mtime))
                    except Exception as e:
                        checkpoint_dirs.append((result_path.name, "unknown", checkpoint_file.stat().st_mtime))
        
        if not checkpoint_dirs:
            print("No checkpoint files found.")
            return
        
        # Sort by modification time, latest first
        checkpoint_dirs.sort(key=lambda x: x[2], reverse=True)
        
        print("\nAvailable checkpoints:")
        print("=" * 80)
        for dirname, step, mtime in checkpoint_dirs:
            import datetime
            time_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"{dirname:<50} Step: {step:<10} Modified: {time_str}")
        
        return
    
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.is_absolute():
            resume_path = results_dir / resume_path
    else:
        try:
            resume_path = find_latest_checkpoint(results_dir)
            print(f"Auto-detected latest checkpoint: {resume_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Use --list to see available checkpoints")
            sys.exit(1)
    
    # Show checkpoint info
    checkpoint_file = resume_path / "gibbs_checkpoint_step.npz"
    try:
        import numpy as np
        ckpt = np.load(checkpoint_file, allow_pickle=True)
        completed_step = int(ckpt["step"])
        print(f"Resuming from step {completed_step + 1} (completed {completed_step + 1} steps)")
    except Exception as e:
        print(f"Warning: Could not read checkpoint info: {e}")
    
    # Build command
    cmd = ["python", "main.py", "--config", args.config, "--resume-from", str(resume_path)]
    
    print(f"Executing: {' '.join(cmd)}")
    print("=" * 80)
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print("✓ Restart completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Restart failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n✗ Restart interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()