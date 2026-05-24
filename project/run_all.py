#!/usr/bin/env python3
"""
Run all pipeline steps for PISA Creative Resilience analysis.
Usage: python run_all.py
"""

import subprocess
import sys
import time
from pathlib import Path

def run_pipeline():
    """Execute the complete pipeline."""
    print("\n" + "="*70)
    print("PISA CREATIVE RESILIENCE - COMPLETE PIPELINE")
    print("="*70 + "\n")

    start_time = time.time()

    try:
        # Execute main pipeline
        print("Executing main pipeline...")
        result = subprocess.run(
            [sys.executable, "project/main.py"],
            cwd=Path(__file__).parent,
            capture_output=False
        )

        if result.returncode != 0:
            print("\n❌ Pipeline execution failed!")
            sys.exit(1)

        # Start dashboard
        print("\n" + "="*70)
        print("STARTING STREAMLIT DASHBOARD")
        print("="*70)
        print("\nOpening dashboard at: http://localhost:8501")
        print("To stop: Press Ctrl+C\n")

        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "project/dashboard/app.py"],
            cwd=Path(__file__).parent
        )

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)
    finally:
        elapsed_time = time.time() - start_time
        print(f"\nTotal execution time: {elapsed_time/60:.2f} minutes")

if __name__ == "__main__":
    run_pipeline()
