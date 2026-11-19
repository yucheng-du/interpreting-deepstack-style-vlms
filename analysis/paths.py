"""Local inputs and outputs for the plotting and table commands."""
import os
from pathlib import Path

ROOT = Path(os.environ.get('DS_DATA_ROOT', 'work')).expanduser().resolve()
INPUT = ROOT / 'presentation_inputs'
OUT = ROOT / 'rendered'
GENERATED_CSV = ROOT / 'supplementary_data'
BUNDLED_CSV = Path(__file__).resolve().parents[1] / 'data'
default_csv = GENERATED_CSV if GENERATED_CSV.is_dir() else BUNDLED_CSV
CSV = Path(os.environ.get('DS_SUPPLEMENTARY_DATA', default_csv)).expanduser()

def output_dir():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'mpl-cache'))
    return OUT
