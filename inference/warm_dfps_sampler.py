from __future__ import annotations

import torch
from mmcv.ops import furthest_point_sample
from torch import Tensor, nn

from warm_dfps_manager import WarmStartManager, fps_refill


class WarmDFPSSampler(nn.Module):
    """
    Same call signature as mmcv.ops.points_sampler.DFPSSampler:
    forward(points, features, npoint) -> (B, npoint) int32 indices.
    Refer: default_points_sampler.py in mmcv.ops.points_sampler.

    Cold frames (no carried state, or a discontinuity) delegate straight to
    the real `furthest_point_sample` CUDA op, so they are bit-for-bit
    identical to stock D-FPS -- only genuinely warm frames take the NumPy
    continuation path, since mmcv ships no "continue from a seed set" CUDA
    op (unlike the old vendored 3DSSD TF ops).
    """

    def __init__(self, manager: WarmStartManager):
        super().__init__()
        self.manager = manager
        self.last_result = None  # StepResult of the most recent forward()
        self.last_S = None  # (npoint, 3) float32 sample positions, same frame
        self._pending_transform = None  # ego-motion transform for the next forward()

    def reset(self) -> None:
        self.manager.reset()
        self._pending_transform = None

    def set_transform(self, transform) -> None:
        """
        Set the previous-velo -> current-velo ego-motion transform (a 4x4
        homogeneous matrix, or None) to apply on the next forward() call.
        Call this once per frame, before running inference on it.
        """
        self._pending_transform = transform

    def forward(self, points: Tensor, features: Tensor, npoint: int) -> Tensor:
        assert points.shape[0] == 1, \
            "WarmDFPSSampler only supports batch_size=1"
        
        assert npoint == self.manager.num_samples, (
            f"npoint ({npoint}) != manager.num_samples "
            f"({self.manager.num_samples})")

        P = points[0].detach().cpu().numpy()
        res = self.manager.step(P, transform=self._pending_transform)
        self.last_result = res

        if res.cold:
            # Bit-for-bit stock D-FPS: real op, no NumPy involved.
            idx = furthest_point_sample(points.contiguous(), npoint)
            S = points[0][idx[0].long()].detach().cpu().numpy()
        else:
            idx_np = fps_refill(P, res.preidx, npoint)
            idx = torch.as_tensor(
                idx_np, dtype=torch.int32, device=points.device
            ).unsqueeze(0)
            S = P[idx_np]

        self.manager.commit(S)
        self.last_S = S
        return idx
