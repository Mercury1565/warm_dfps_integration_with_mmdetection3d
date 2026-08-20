"""Turns consecutive KITTI raw oxts (GPS/IMU) packets into the frame-to-frame
relative ego-motion transform that WarmStartManager.step(P, transform=...)
expects, so carried samples get checked against the new cloud at the
position they actually now occupy instead of their stale previous-frame
position. Mirrors the KITTI devkit's convertOxtsToPose.m conversion.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_EARTH_RADIUS = 6378137.0  # WGS84 equatorial radius (m); matches the devkit's Mercator projection


def _mercator_scale(lat0_deg: float) -> float:
    return float(np.cos(lat0_deg * np.pi / 180.0))


def _latlon_to_mercator(lat_deg: float, lon_deg: float, scale: float) -> tuple[float, float]:
    mx = scale * lon_deg * np.pi * _EARTH_RADIUS / 180.0
    my = scale * _EARTH_RADIUS * np.log(np.tan((90.0 + lat_deg) * np.pi / 360.0))
    return mx, my


def _rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Devkit convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def parse_oxts_packet(line: str) -> dict:
    """One line of oxts/data/*.txt -> the fields needed for pose. The full
    packet has 30 space-separated values; lat/lon/alt/roll/pitch/yaw are
    always the first 6."""
    v = [float(x) for x in line.split()]
    keys = ('lat', 'lon', 'alt', 'roll', 'pitch', 'yaw')
    return dict(zip(keys, v[:6]))


def oxts_to_pose(packet: dict, scale: float) -> np.ndarray:
    """4x4 homogeneous pose mapping IMU-frame coordinates at this frame into
    a sequence-anchored world frame (anchored at whichever frame `scale` was
    derived from -- normally the drive's first frame, per the devkit)."""
    tx, ty = _latlon_to_mercator(packet['lat'], packet['lon'], scale)
    tz = packet['alt']
    R = _rotation_from_rpy(packet['roll'], packet['pitch'], packet['yaw'])
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R
    pose[:3, 3] = [tx, ty, tz]
    return pose


def parse_imu_to_velo_calib(path) -> np.ndarray:
    """calib_imu_to_velo.txt -> 4x4 homogeneous transform mapping IMU/GPS
    coordinates into the velodyne frame (the R, T documented by the KITTI
    raw-data devkit)."""
    R = T = None
    with open(path) as f:
        for line in f:
            if line.startswith('R:'):
                R = np.array([float(x) for x in line.split()[1:]], dtype=np.float64).reshape(3, 3)
            elif line.startswith('T:'):
                T = np.array([float(x) for x in line.split()[1:]], dtype=np.float64)
    if R is None or T is None:
        raise ValueError(f'{path}: missing R:/T: lines')
    velo_T_imu = np.eye(4, dtype=np.float64)
    velo_T_imu[:3, :3] = R
    velo_T_imu[:3, 3] = T
    return velo_T_imu


class OxtsPoseTracker:
    """Feed one oxts packet per frame via `step`; get back the homogeneous
    transform mapping the *previous* frame's velodyne coordinates into the
    *current* frame's velodyne coordinates (or None on the first frame of a
    sequence, when there's nothing yet to compensate).
    """

    def __init__(self, imu_to_velo_calib_path: Path | str):
        self.velo_T_imu = parse_imu_to_velo_calib(imu_to_velo_calib_path)
        self.imu_T_velo = np.linalg.inv(self.velo_T_imu)
        self._scale: float | None = None
        self._pose_prev: np.ndarray | None = None

    def reset(self) -> None:
        """Call at a sequence/drive boundary, alongside WarmStartManager.reset()."""
        self._scale = None
        self._pose_prev = None

    def step(self, oxts_line: str) -> np.ndarray | None:
        packet = parse_oxts_packet(oxts_line)
        if self._scale is None:
            self._scale = _mercator_scale(packet['lat'])

        pose_curr = oxts_to_pose(packet, self._scale)

        if self._pose_prev is None:
            self._pose_prev = pose_curr
            return None

        # velo(prev) -> imu(prev) -> world -> imu(curr) -> velo(curr)
        transform = (self.velo_T_imu
                     @ np.linalg.inv(pose_curr)
                     @ self._pose_prev
                     @ self.imu_T_velo)
        self._pose_prev = pose_curr
        return transform
