"""Utility functions for the pipeline."""
import os
import json
import hashlib
import logging
import psutil
from pathlib import Path
from typing import Any, Dict
import numpy as np
from datetime import datetime

# Setup logging
def setup_logger(name: str, config_path: str = "config.yaml") -> logging.Logger:
    """Setup logger with file and console handlers.

    Args:
        name: Logger name
        config_path: Path to config file (to get log path)

    Returns:
        Configured logger instance
    """
    from src.config import ConfigManager

    config = ConfigManager(config_path)
    log_file = config.get_path('outputs_reproducibility') / 'pipeline.log'
    log_level = config.get('logging.level', 'INFO')

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    formatter = logging.Formatter(config.get('logging.format'))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def compute_file_checksum(file_path: Path, algorithm: str = 'sha256') -> str:
    """Compute file checksum for data integrity verification.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha256', etc.)

    Returns:
        Hexadecimal checksum string
    """
    hasher = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_checksum(file_path: Path, checksum: str, metadata_file: Path):
    """Save checksum to metadata file.

    Args:
        file_path: Original file path
        checksum: Computed checksum
        metadata_file: Path to metadata JSON file
    """
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

    metadata[str(file_path)] = {
        'checksum_sha256': checksum,
        'timestamp': datetime.now().isoformat(),
        'file_size_bytes': file_path.stat().st_size
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics.

    Returns:
        Dictionary with memory stats in GB:
        - rss: Resident set size (actual physical memory)
        - vms: Virtual memory size
        - percent: Percentage of total system memory
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_percent = process.memory_percent()

    return {
        'rss_gb': mem_info.rss / (1024 ** 3),
        'vms_gb': mem_info.vms / (1024 ** 3),
        'percent': mem_percent,
        'available_gb': psutil.virtual_memory().available / (1024 ** 3)
    }


def check_memory_threshold(threshold_gb: float = 6.5, logger=None) -> bool:
    """Check if memory usage exceeds threshold.

    Args:
        threshold_gb: Threshold in GB (default 6.5 for M1 8GB)
        logger: Optional logger instance

    Returns:
        True if under threshold, False if exceeded
    """
    mem = get_memory_usage()
    under_threshold = mem['rss_gb'] < threshold_gb

    status_msg = f" Memory: {mem['rss_gb']:.2f}GB / {threshold_gb}GB ({mem['percent']:.1f}%)"

    if logger:
        if under_threshold:
            logger.info(status_msg)
        else:
            logger.warning(f"  {status_msg}")
    else:
        print(status_msg)

    return under_threshold


def save_metadata(data: Dict[str, Any], output_path: Path, description: str = ""):
    """Save metadata dictionary to JSON file.

    Args:
        data: Dictionary to save
        output_path: Output file path
        description: Optional description
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        'timestamp': datetime.now().isoformat(),
        'description': description,
        'data': data
    }

    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_metadata(file_path: Path) -> Dict[str, Any]:
    """Load metadata from JSON file.

    Args:
        file_path: Path to metadata JSON file

    Returns:
        Data dictionary
    """
    with open(file_path, 'r') as f:
        metadata = json.load(f)
    return metadata.get('data', metadata)


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    import random
    random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def format_size_bytes(size_bytes: int) -> str:
    """Format bytes as human readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def create_summary_report(data: Dict[str, Any], output_path: Path):
    """Create a text summary report.

    Args:
        data: Dictionary with report content
        output_path: Path to output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PIPELINE SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        for section, content in data.items():
            f.write(f"\n{'─' * 80}\n")
            f.write(f"{section}\n")
            f.write(f"{'─' * 80}\n")

            if isinstance(content, dict):
                for key, value in content.items():
                    f.write(f"  {key}: {value}\n")
            elif isinstance(content, list):
                for item in content:
                    f.write(f"  - {item}\n")
            else:
                f.write(f"  {content}\n")


class Timer:
    """Simple timer context manager for performance tracking."""

    def __init__(self, name: str = "Operation", logger=None):
        """Initialize timer.

        Args:
            name: Operation name
            logger: Optional logger instance
        """
        self.name = name
        self.logger = logger
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        if self.logger:
            self.logger.info(f"⏱  Starting {self.name}...")
        return self

    def __exit__(self, *args):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        msg = f" {self.name} completed in {elapsed:.2f}s"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
