"""Reuse frozen four-way results and add shared 250-label fits."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments.run_geoai_rqs import main

if __name__ == '__main__':
    main('rq1')
