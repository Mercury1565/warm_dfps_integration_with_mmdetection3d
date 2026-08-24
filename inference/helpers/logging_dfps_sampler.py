from __future__ import annotations

from mmcv.ops import furthest_point_sample
from torch import Tensor, nn


class LoggingDFPSSampler(nn.Module):
    """Drop-in replacement for mmcv's stock DFPSSampler: same CUDA op, same
    call, bit-for-bit identical output -- it only additionally records the
    sampled point positions of the most recent forward() call, so the
    baseline run can be compared against warm-D-FPS in BEV.
    """

    def __init__(self):
        super().__init__()
        self.last_S = None  # (npoint, 3) float32, set by the most recent forward()

    def forward(self, points: Tensor, features: Tensor, npoint: int) -> Tensor:
        idx = furthest_point_sample(points.contiguous(), npoint)
        self.last_S = points[0][idx[0].long()].detach().cpu().numpy()
        return idx
