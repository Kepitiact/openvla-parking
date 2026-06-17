_base_ = ["./base_track_map.py"]

# Mini smoke config for quick local validation without full trainval.
info_root = "data/infos/"
ann_file_mini = info_root + "nuscenes_infos_temporal_mini.pkl"

data = dict(
    val=dict(
        ann_file=ann_file_mini,
        is_debug=True,
        len_debug=16,
    ),
    test=dict(
        ann_file=ann_file_mini,
        is_debug=True,
        len_debug=16,
    ),
    test_llava_with_track_gt=dict(
        ann_file=ann_file_mini,
        is_debug=True,
        len_debug=16,
    ),
)
