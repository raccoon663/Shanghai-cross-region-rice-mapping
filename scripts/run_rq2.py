"""Run controlled simple domain adaptation comparisons."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments.run_geoai_rqs import main

if __name__ == '__main__':
    main('rq2')
