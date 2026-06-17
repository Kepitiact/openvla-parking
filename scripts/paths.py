"""
Single source of truth for dataset and checkpoint paths.

Everything is derived from the repository root so the pipeline is portable:
no hardcoded home directory, works for anyone who clones the repo, and works
for any dataset placed under DATA_DIR (not just the current CARLA parking one).

Import the constants you need; every data/eval script defaults to these.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset dir (generic concept — currently the CARLA parking set lives here).
DATA_DIR = PROJECT_ROOT / "data_carla"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Processed artifacts.
INFOS_PKL = PROCESSED_DIR / "parking_infos_temporal.pkl"
CACHED_INFO = PROCESSED_DIR / "cached_parking_info.pkl"
CONVERSATIONS = PROCESSED_DIR / "carla_conversations.json"
FEATURES_DIR = PROCESSED_DIR / "uniad_features"

# NuScenes-format DB the model loads (maps/expansion/basemap live alongside).
NUSC_ROOT = PROJECT_ROOT / "data" / "nuscenes"
NUSC_DB = NUSC_ROOT / "v1.0-carla"
NUSC_VERSION = "v1.0-carla"

# Model checkpoints.
CHECKPOINTS = PROJECT_ROOT / "checkpoints"
