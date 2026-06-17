_base_ = ["./base_track_map.py"]

info_root = "data/infos/"
ann_file_mini = info_root + "nuscenes_infos_temporal_mini.pkl"

data = dict(
    val=dict(
        ann_file=ann_file_mini,
        is_debug=False,
    ),
    test=dict(
        ann_file=ann_file_mini,
        is_debug=False,
    ),
    test_llava_with_track_gt=dict(
        ann_file=ann_file_mini,
        is_debug=False,
    ),
)
