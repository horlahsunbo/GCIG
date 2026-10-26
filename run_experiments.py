# ===== FILE: run_experiments.py =====
#!/usr/bin/env python3
"""
Quick runner script for GCf-IG experiments.

Usage:
    python run_experiments.py --quick                    # Quick test
    python run_experiments.py --datasets german compas   # Multiple datasets
    python run_experiments.py --rq rq1                   # Run specific RQ
"""

import sys
from pathlib import Path

# Ensure local imports work when running this script from the project folder.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comprehensive_experiments import main

if __name__ == "__main__":
    main()
