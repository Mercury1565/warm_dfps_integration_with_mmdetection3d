from __future__ import annotations
from pathlib import Path
import numpy as np


def _rotx(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _roty(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rotz(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _mercator_scale(lat0_deg: float) -> float:
    return float(np.cos(lat0_deg * np.pi / 180.0))


def _oxts_packet_to_pose(packet: np.ndarray, scale: float) -> np.ndarray:
    """4x4 world<-imu pose for one oxts packet (KITTI devkit convention:
    Mercator-projected lat/lon + altitude for translation, roll/pitch/yaw
    composed as Rz @ Ry @ Rx for rotation)."""
    lat, lon, alt, roll, pitch, yaw = packet[:6]
    er = 6378137.0  # WGS84 equatorial radius -- matches the KITTI devkit
    tx = scale * lon * np.pi * er / 180.0
    ty = scale * er * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
    tz = alt
    R = _rotz(yaw) @ _roty(pitch) @ _rotx(roll)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def load_drive_poses(drive_dir: Path) -> dict[str, np.ndarray] | None:
    """One world<-imu 4x4 pose per frame, keyed by file stem (matches the
    corresponding velodyne_points/data/<stem>.bin). Returns None if this
    drive has no oxts data -- callers should fall back to
    transform=None (uncompensated), the pre-existing behaviour.
    """
    oxts_dir = drive_dir / 'oxts' / 'data'
    if not oxts_dir.is_dir():
        return None
    files = sorted(oxts_dir.glob('*.txt'))
    if not files:
        return None

    packets = {f.stem: np.loadtxt(f) for f in files}
    scale = _mercator_scale(next(iter(packets.values()))[0])
    return {stem: _oxts_packet_to_pose(p, scale) for stem, p in packets.items()}


def load_imu_to_velo(calib_dir: Path) -> np.ndarray:
    """4x4 transform mapping IMU-frame points into the Velodyne frame
    (p_velo = T @ p_imu), from calib_imu_to_velo.txt. Falls back to identity
    (treats IMU and Velodyne as co-located/co-oriented) if the calibration
    file isn't found -- a reasonable first-order approximation since the true
    lever arm between the two sensors is small relative to
    WarmStartManager's thresholds, which are already expressed as generous
    multiples of the sample spacing.
    """
    calib_path = calib_dir / 'calib_imu_to_velo.txt'
    T = np.eye(4)
    if not calib_path.is_file():
        return T
    R, t = None, None
    for line in calib_path.read_text().splitlines():
        if line.startswith('R:'):
            R = np.array([float(x) for x in line.split()[1:]]).reshape(3, 3)
        elif line.startswith('T:'):
            t = np.array([float(x) for x in line.split()[1:]])
    if R is not None and t is not None:
        T[:3, :3] = R
        T[:3, 3] = t
    return T


def relative_velo_transform(poses: dict[str, np.ndarray], t_velo_imu: np.ndarray,
                             stem_prev: str, stem_curr: str) -> np.ndarray | None:
    """4x4 transform mapping Velodyne-frame points at frame `stem_prev` into
    Velodyne-frame coordinates at frame `stem_curr`. Returns None if either
    frame is missing from `poses` (caller should fall back to uncompensated).
    """
    if stem_prev not in poses or stem_curr not in poses:
        return None
    t_imu_velo = np.linalg.inv(t_velo_imu)
    t_rel_imu = np.linalg.inv(poses[stem_curr]) @ poses[stem_prev]
    return t_velo_imu @ t_rel_imu @ t_imu_velo
