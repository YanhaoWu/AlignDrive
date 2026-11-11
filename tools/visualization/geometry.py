"""Array geometry used by visualization without loading the training plugin."""

import numpy as np


def to_numpy(values):
    """Convert arrays or detached model tensors to a CPU NumPy view."""
    if hasattr(values, 'detach'):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def prediction_modes(result, object_index):
    """Normalize single- or multimodal agent predictions to (M, T, 2) and (M,)."""
    trajectories = to_numpy(result['trajs_3d'][object_index])
    if trajectories.ndim == 2:
        return trajectories[None], np.ones(1)
    if trajectories.ndim != 3:
        raise ValueError(f'Unsupported agent trajectory shape: {trajectories.shape}')
    scores = result.get('trajs_score')
    scores = (
        np.ones(len(trajectories)) if scores is None else to_numpy(scores[object_index]).reshape(-1)
    )
    if len(scores) != len(trajectories):
        raise ValueError('Agent mode scores must match the number of trajectories')
    return trajectories, scores


def projective_flatten(values):
    """Normalize homogeneous coordinates along their final coordinate row."""
    return values / values[-1]


def box3d_to_corners(boxes):
    """Convert decoded boxes [x, y, z, width, length, height, yaw, ...] to (N, 8, 3)."""
    if hasattr(boxes, 'detach'):
        boxes = boxes.detach().cpu().numpy()
    boxes = np.asarray(boxes)
    normalized = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    normalized = normalized[[0, 1, 3, 2, 4, 5, 7, 6]] - 0.5
    corners = boxes[:, None, 3:6] * normalized[None]
    rotation = np.tile(np.eye(3)[None], (len(boxes), 1, 1))
    cosine, sine = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    rotation[:, 0, 0] = cosine
    rotation[:, 0, 1] = -sine
    rotation[:, 1, 0] = sine
    rotation[:, 1, 1] = cosine
    return (rotation[:, None] @ corners[..., None]).squeeze(-1) + boxes[:, None, :3]
