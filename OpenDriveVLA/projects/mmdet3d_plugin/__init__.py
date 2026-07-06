# torch 2.1 compat: mmcv 1.7.2's Scatter passes int GPU indices to torch's
# _get_stream, but torch>=2.0 requires a torch.device (it reads device.type).
# Wrap mmcv's reference to convert int -> torch.device('cuda', idx).
try:
    import torch as _torch
    import mmcv.parallel._functions as _mmcv_fns
    _orig_get_stream = _mmcv_fns._get_stream

    def _get_stream_compat(device):
        if isinstance(device, int):
            device = _torch.device('cuda', device)
        return _orig_get_stream(device)

    _mmcv_fns._get_stream = _get_stream_compat
except Exception:
    pass

from .core.bbox.assigners.hungarian_assigner_3d import HungarianAssigner3D
from .core.bbox.coders.nms_free_coder import NMSFreeCoder
from .core.bbox.match_costs import BBox3DL1Cost, DiceCost
from .core.evaluation.eval_hooks import CustomDistEvalHook
from .datasets.pipelines import (
  PhotoMetricDistortionMultiViewImage, PadMultiViewImage, 
  NormalizeMultiviewImage,  CustomCollect3D)
from .models.backbones.vovnet import VoVNet
from .models.utils import *
from .models.opt.adamw import AdamW2
from .uniad import *
from .losses import *
