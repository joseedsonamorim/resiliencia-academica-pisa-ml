#!/usr/bin/env python3
"""
Clean up project: remove cache, old outputs, and temporary files.
Usage: python cleanup_project.py
"""

import shutil
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_project(project_path: str = "project"):
    """Clean up project directories."""
    print("\n" + "="*70)
    print("CLEANING UP PROJECT")
    print("="*70 + "\n")

    project_root = Path(project_path)

    patterns_to_remove = [
        "**/__pycache__",
        "**/*.pyc",
        "**/.pytest_cache",
        "**/.DS_Store",
        "**/._*",
        "**/*.egg-info",
        "**/.streamlit",
    ]

    dirs_to_clean = [
        project_root / "logs",
        project_root / "outputs" / "reports",
        project_root / "outputs" / "tables",
        project_root / "outputs" / "figures",
    ]

    removed_count = 0

    # Remove pattern-matched files
    for pattern in patterns_to_remove:
        for path in project_root.glob(pattern):
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                    logger.info(f"Removed directory: {path}")
                else:
                    path.unlink()
                    logger.info(f"Removed file: {path}")
                removed_count += 1
            except Exception as e:
                logger.warning(f"Could not remove {path}: {str(e)}")

    # Clean directories (keep structure)
    for dir_path in dirs_to_clean:
        if dir_path.exists():
            try:
                for file_path in dir_path.glob("*"):
                    if file_path.is_file():
                        file_path.unlink()
                        logger.info(f"Removed file: {file_path}")
                        removed_count += 1
            except Exception as e:
                logger.warning(f"Could not clean {dir_path}: {str(e)}")

    print(f"\n✓ Cleanup completed: {removed_count} items removed")
    print("Project structure is clean and ready for fresh run.\n")

if __name__ == "__main__":
    try:
        cleanup_project()
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        exit(1)
