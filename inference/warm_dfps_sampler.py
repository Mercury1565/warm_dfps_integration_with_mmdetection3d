from __future__ import annotations

import torch
from mmcv.ops import furthest_point_sample
from torch import Tensor, nn

from fps_with_preidx import farthest_point_sample_with_preidx
from warm_dfps_manager import WarmStartManager


class WarmDFPSSampler(nn.Module):
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

        res = self.manager.step_gpu(points[0], transform=self._pending_transform)
        self.last_result = res

        if res.cold:
            # Bit-for-bit stock D-FPS: real op, no NumPy involved.
            idx = furthest_point_sample(points.contiguous(), npoint)
            S = points[0][idx[0].long()].detach().cpu().numpy()
        else:
            preidx = torch.as_tensor(
                res.preidx, dtype=torch.int64, device=points.device)
            idx0 = farthest_point_sample_with_preidx(points[0], preidx, npoint)
            idx = idx0.to(torch.int32).unsqueeze(0)
            S = points[0][idx0].detach().cpu().numpy()

        self.manager.commit(S)
        self.last_S = S
        return idx
