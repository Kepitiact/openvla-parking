"""
# Adapted from https://huggingface.co/MILVLG/imp-v1-3b/blob/main/vision_encoder.py
"""

from typing import Optional, Union
import torch
import torch.utils.checkpoint
from torch import nn
import os
import os.path as osp

from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from llava.utils import rank0_print

from mmengine import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger('shapely.geos').setLevel(logging.ERROR)

class UniadTrackMapConfig(PretrainedConfig):
    model_type = "uniad_track_map_model"

    def __init__(self, uniad_config_dict: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)

        self.uniad_config_dict = uniad_config_dict

class UniadTrackMapModel(PreTrainedModel):
    config_class = UniadTrackMapConfig
    base_model_prefix = "uniad_track_map"
    supports_gradient_checkpointing = True
    main_input_name = "pixel_values"
    _no_split_modules = ["UniAD"]

    def __init__(self, config: UniadTrackMapConfig, load_mmdet3d_weights=False, vision_tower_test_mode=False):
        super().__init__(config)

        self.config = config
        self.load_mmdet3d_weights = load_mmdet3d_weights
        self.vision_tower_test_mode = vision_tower_test_mode
        # build the UniAD model
        self.vision_model = self.build_uniad_track_map_model()

    def build_uniad_track_map_model(self):
        uniad_config_mmlab = Config()
        uniad_config_mmlab.merge_from_dict(self.config.uniad_config_dict)
        # import modules from plguin/xx, registry will be updated
        if hasattr(uniad_config_mmlab, 'plugin'):
            if uniad_config_mmlab.plugin:
                import importlib
                plugin_dir = uniad_config_mmlab.plugin_dir
                _module_dir = osp.dirname(plugin_dir)
                _module_dir = str(_module_dir).split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

        uniad_config_mmlab.model.pretrained = None
        uniad_config_mmlab.model.train_cfg = None
        model = build_model(uniad_config_mmlab.model, test_cfg=uniad_config_mmlab.get('test_cfg'))

        if self.load_mmdet3d_weights:
            # Explicit at the call site, like the geometry. Historically this path was
            # hardcoded to 'checkpoints/uniad_base_track_map.pth' -- a filename shared by
            # TWO different models (the 200 MB nuScenes warm-start and the CARLA-trained
            # model), so pointing it at the wrong one silently reverted the detector to
            # nuScenes weights. Set UNIAD_CKPT to the trained checkpoint.
            ckpt = os.environ.get("UNIAD_CKPT", "checkpoints/uniad_carla_trained.pth")
            rank0_print(f"[UniAD] loading vision-tower weights: {ckpt}")
            checkpoint = load_checkpoint(model, ckpt, map_location='cpu')

            if 'CLASSES' in checkpoint.get('meta', {}):
                model.CLASSES = checkpoint['meta']['CLASSES']
            if 'PALETTE' in checkpoint.get('meta', {}):
                model.PALETTE = checkpoint['meta']['PALETTE']

        return model

    def _init_weights(self, module):
        """Initialize the weights"""
        pass

    def forward(self, data):
        if self.vision_tower_test_mode:
            _, results_for_vlm = self.vision_model(return_loss=False, rescale=True, **data)
        else:
            _, results_for_vlm = self.vision_model(return_loss=True, rescale=True, return_vlm=True, **data)
        return results_for_vlm

# The config UniAD was TRAINED with. It is the single source of truth for the BEV
# geometry: if the tower samples BEV features at a different point_cloud_range than
# training used, the detector is silently wrong (this is exactly how the +/-51.2 bug
# survived -- the tower read a hardcoded config nobody passed in or reviewed).
UNIAD_TRAIN_CONFIG = "projects/configs/stage1_track_map/carla_parking_stage1.py"
_GEOMETRY_KEYS = ("point_cloud_range", "voxel_size", "patch_size", "bev_h_", "bev_w_")


def resolve_uniad_config(vision_tower_cfg=None) -> str:
    """Which UniAD config the vision tower should build from. Explicit at the call
    site, in precedence order: caller attr -> UNIAD_CONFIG env -> the training config."""
    cfg = getattr(vision_tower_cfg, "uniad_config", None) if vision_tower_cfg else None
    return cfg or os.environ.get("UNIAD_CONFIG") or UNIAD_TRAIN_CONFIG


# Configs whose geometry MUST agree with the training config. point_cloud_range is
# declared independently in several of these (mmcv 1.x can't reference a _base_ var from
# a child, so it can't be de-duplicated by inheritance) — and that duplication is the
# entire root cause of the +/-51.2 bug. Since we can't make divergence impossible, we
# make it LOUD: every config on the VLA data/model path is checked at tower-build time.
_MUST_AGREE = (
    "projects/configs/stage1_track_map/base_track_map.py",   # roots the VLA data configs
    "projects/configs/stage1_track_map/carla_parking.py",    # VLA train/extract data cfg
)


def _geometry_diff(cfg, train):
    return {k: (cfg.get(k), train.get(k)) for k in _GEOMETRY_KEYS
            if cfg.get(k) != train.get(k)}


def assert_geometry_matches_training(cfg, cfg_path: str) -> None:
    """Crash if any config on the UniAD path disagrees with the TRAINING geometry.

    A wrong point_cloud_range does not raise — it quietly produces garbage detections
    (near-field recall went 1.00 -> 0.04 and nobody noticed). Fail loudly instead.
    """
    train = Config.fromfile(UNIAD_TRAIN_CONFIG)

    to_check = [(cfg_path, cfg)]
    for p in _MUST_AGREE:
        if os.path.exists(p) and os.path.abspath(p) != os.path.abspath(cfg_path):
            try:
                to_check.append((p, Config.fromfile(p)))
            except Exception:      # a config we can't parse isn't on the hot path
                continue

    problems = []
    for path, c in to_check:
        d = _geometry_diff(c, train)
        if d:
            problems.append(f"  {path}\n" + "\n".join(
                f"    {k}: got={got!r}  training={want!r}" for k, (got, want) in d.items()))
    if problems:
        raise ValueError(
            "UniAD BEV geometry mismatch — features would be sampled at a different "
            "scale than the model was trained with, silently corrupting every "
            f"detection.\n  training config (source of truth): {UNIAD_TRAIN_CONFIG}\n"
            + "\n".join(problems))


class UniadTrackMapVisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
        super().__init__()

        # The UniAD config is a PARAMETER now, not a hardcoded filename: both things the
        # detector depends on -- geometry (here) and weights (UNIAD_CKPT, see
        # build_uniad_track_map_model) -- are explicit and overridable at the call site.
        uniad_config_path = resolve_uniad_config(vision_tower_cfg)
        _cfg = Config.fromfile(uniad_config_path)
        assert_geometry_matches_training(_cfg, uniad_config_path)
        rank0_print(f"[UniAD] config={uniad_config_path} "
                    f"point_cloud_range={_cfg.get('point_cloud_range')}")
        self.config = UniadTrackMapConfig(uniad_config_dict=_cfg.to_dict())

        self.vision_tower_name = vision_tower
        self.vision_tower: nn.Module = None
        self.is_loaded = False
        self.vision_tower_pretrained = vision_tower_cfg.vision_tower_pretrained
        if hasattr(vision_tower_cfg, "vision_tower_test_mode"):
            self.vision_tower_test_mode = vision_tower_cfg.vision_tower_test_mode
        else:  # set to False when in training mode
            self.vision_tower_test_mode = False

        self.image_processor = None

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()

        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()

        elif hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()

        else:
            self.cfg_only = self.config

    def load_model(self, device_map="auto"):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        if self.vision_tower_pretrained:
            # Check if vision_tower_name points to a valid pretrained model path
            load_from_transformers_pretrained = (
                isinstance(self.vision_tower_pretrained, str) and 
                (os.path.exists(self.vision_tower_pretrained) or 
                self.vision_tower_pretrained.startswith('https://') or
                self.vision_tower_pretrained.startswith('http://'))
            )

            if load_from_transformers_pretrained:
                rank0_print(f"Loading UniAD transformers checkpoint from: {self.vision_tower_pretrained}")
                self.vision_tower = UniadTrackMapModel.from_pretrained(
                    self.vision_tower_pretrained, 
                    device_map=device_map
                )
                self.vision_tower.vision_tower_test_mode = self.vision_tower_test_mode
            else: # load from mmdet3d checkpoint
                rank0_print("Loading UniAD from mmdet3d checkpoint")
                self.vision_tower = UniadTrackMapModel(self.config, load_mmdet3d_weights=True, vision_tower_test_mode=self.vision_tower_test_mode)
        else:
            # only build the model, not load the weights
            rank0_print("Building UniAD from config. Weights will be loaded from the llava checkpoint.")
            self.vision_tower = UniadTrackMapModel(self.config, load_mmdet3d_weights=False, vision_tower_test_mode=self.vision_tower_test_mode)

        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def forward(self, data):
        return self.vision_tower(data)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device
