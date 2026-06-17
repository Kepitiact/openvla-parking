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

_CARLA = "/home/s0002438/projects/openvla_nuscenes/data_carla"
_carla_ann = _CARLA + "/processed/parking_infos_temporal.pkl"

# Absolute path so NuScenes init never depends on CWD or symlink resolution.
# The real v1.0-mini DB and maps live here; CARLA image paths are absolute so
# img_root (derived from data_root) is unused for CARLA images.
data_root = "/home/s0002438/projects/openvla_nuscenes/data/nuscenes/"

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
