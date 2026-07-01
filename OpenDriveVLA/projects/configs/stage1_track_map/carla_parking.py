"""
OpenDriveVLA config for CARLA parking dataset.

Overrides only the data paths. All model/pipeline settings are inherited from
base_track_map_mini_full.py (which inherits from base_track_map.py).

Set CACHED_DATA_PATH env var when launching so the LLaVA dataset loads the
CARLA cache instead of the nuScenes one:
  CACHED_DATA_PATH=data_carla/processed/cached_parking_info.pkl

Run from OpenDriveVLA/ directory.
"""

_base_ = ["./base_track_map_mini_full.py"]

import os

# Paths are parametrized so the config is portable (no hardcoded home dir).
#   CARLA_DATA_ROOT -> the data_carla dir (processed infos live under it)
#   NUSC_DATA_ROOT  -> the nuScenes-format DB root (maps/expansion/basemap alongside)
# The .sh wrappers export these; both default relative to the repo root, which
# is the parent of the OpenDriveVLA/ directory the scripts are launched from.
_REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
_CARLA = os.environ.get("CARLA_DATA_ROOT") or os.path.join(_REPO_ROOT, "data_carla")
_carla_ann = os.path.join(_CARLA, "processed", "parking_infos_temporal.pkl")

# Absolute path so NuScenes init never depends on CWD or symlink resolution.
# The real v1.0-mini DB and maps live here; CARLA image paths are absolute so
# img_root (derived from data_root) is unused for CARLA images.
data_root = os.environ.get("NUSC_DATA_ROOT") or os.path.join(_REPO_ROOT, "data", "nuscenes")
data_root = data_root.rstrip("/") + "/"

data = dict(
    val=dict(
        ann_file=_carla_ann,
        data_root=data_root,
        is_debug=False,
    ),
    test=dict(
        ann_file=_carla_ann,
        data_root=data_root,
        is_debug=False,
    ),
    test_llava_with_track_gt=dict(
        ann_file=_carla_ann,
        data_root=data_root,
        is_debug=False,
    ),
)
