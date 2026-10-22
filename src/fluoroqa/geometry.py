"""Current-pose local neighborhood; this API deliberately has no GT input."""
import numpy as np
from scipy.spatial.transform import Rotation


def current_neighbors(current_pose, center_model_mm, translation_mm=2., rotation_deg=2.):
    pose = np.asarray(current_pose, dtype=float)
    center = np.asarray(center_model_mm, dtype=float)
    if pose.shape != (4, 4) or center.shape != (3,) or not np.isfinite(pose).all() or not np.isfinite(center).all():
        raise ValueError("Expected finite 4x4 pose and 3D model-space center")
    r = pose[:3, :3]
    if not np.allclose(pose[3], [0, 0, 0, 1]) or not np.allclose(r.T@r, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1):
        raise ValueError("Pose must be a rigid model-to-camera transformation")
    if not np.isfinite([translation_mm, rotation_deg]).all() or min(translation_mm, rotation_deg) <= 0:
        raise ValueError("Neighborhood amplitudes must be positive and finite")
    pivot = r@center + pose[:3, 3]
    output = []
    for axis in range(6):
        for sign in (-1, 1):
            delta = np.eye(4)
            if axis < 3:
                delta[axis, 3] = sign*translation_mm
            else:
                rot = Rotation.from_rotvec(np.eye(3)[axis-3]*np.deg2rad(sign*rotation_deg)).as_matrix()
                delta[:3, :3] = rot
                delta[:3, 3] = pivot-rot@pivot
            output.append(delta@pose)
    return np.stack(output)
