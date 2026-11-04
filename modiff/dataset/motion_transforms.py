"""Rigid-transform helpers for the ``v1_ait`` motion representation.

This module is purely additive: nothing in the existing training / inference code
imports it. It exists so that overlapping windows of a long sequence -- each one
canonicalized around its own first frame -- can be mapped into a single common
frame and back.

Layout of a ``v1_ait`` motion tensor ``x`` (see
:func:`modiff.dataset.representation_utils.sequence_to_repre_v1_ait`)::

    x[..., 0:198]     22 joints x (6D global rotation | 3D global position)
    x[..., 198:207]   head / aria (6D global rotation | 3D global position)
    x[..., 207:209]   left / right foot contact

Everything lives in the *canonical* world frame: Z up, floor at z = 0, and the
first frame of the window has its head at the XY origin facing +X (see
:func:`modiff.dataset.canonicalization.canonicalize_trajectory`). Two windows of
the same long sequence therefore differ by a yaw rotation about Z plus an XY
translation -- an SE(2) motion -- which is exactly what the helpers below apply.
Because such a transform preserves heights and vertical velocities, the two foot
contact channels are carried through unchanged.
"""

import torch

import modiff.utils.rotation_conversions as rc
from modiff.dataset.canonicalization import rot_trans_to_matrix, rotation_to_make_this_forward_batch


N_JOINTS = 22
JOINT_BLOCK = 9  # 6D rotation + 3D position
JOINT_END = N_JOINTS * JOINT_BLOCK  # 198
HEAD_END = JOINT_END + 9  # 207
CONTACT_END = HEAD_END + 2  # 209


def _transform_rot6d_and_pos(rot6d, pos, R, t):
    """Apply ``(R, t)`` to a 6D rotation / 3D position pair.

    The 6D representation stores the first two rows of a rotation matrix, so the
    third row is needed to left-multiply by ``R``. Recovering it goes through
    ``rotation_6d_to_matrix``, which Gram-Schmidts the input -- i.e. the result is
    projected onto SO(3). That is intentional and matches what
    ``repre_to_sequence`` does when decoding, but it does mean the transform is
    only an identity for inputs that already hold valid rotations.
    """
    mat = rc.rotation_6d_to_matrix(rot6d)  # ... x 3 x 3
    mat = R @ mat
    new_pos = (R @ pos[..., None])[..., 0] + t
    return rc.matrix_to_rotation_6d(mat), new_pos


def transform_v1_ait_motion(x, T4):
    """Apply a rigid transform to a ``v1_ait`` motion tensor.

    :param x: ``(..., F, 209)`` motion in some canonical frame (denormalized).
    :param T4: ``(..., 4, 4)``, broadcastable to ``x.shape[:-1] + (4, 4)``. Note
        that per-window transforms must be unsqueezed over the frame axis, e.g.
        ``G[:, None]`` for ``x`` of shape ``(W, F, 209)``.
    :return: ``x`` expressed in the frame reached by ``T4``, same shape.
    """
    assert x.shape[-1] == CONTACT_END, f"expected a v1_ait motion tensor, got {x.shape[-1]} channels"

    lead = x.shape[:-1]
    T4 = torch.broadcast_to(T4, lead + (4, 4))
    R = T4[..., :3, :3]  # lead x 3 x 3
    t = T4[..., :3, 3]  # lead x 3

    joints = x[..., :JOINT_END].reshape(*lead, N_JOINTS, JOINT_BLOCK)
    joint_rot, joint_pos = _transform_rot6d_and_pos(
        joints[..., :6], joints[..., 6:9], R[..., None, :, :], t[..., None, :]
    )
    joints = torch.cat([joint_rot, joint_pos], dim=-1).reshape(*lead, JOINT_END)

    head_rot, head_pos = _transform_rot6d_and_pos(
        x[..., JOINT_END : JOINT_END + 6], x[..., JOINT_END + 6 : HEAD_END], R, t
    )

    return torch.cat([joints, head_rot, head_pos, x[..., HEAD_END:CONTACT_END]], dim=-1)


def head_T_from_v1_ait(x):
    """Head pose channels of a ``v1_ait`` motion tensor as ``(..., F, 4, 4)``."""
    assert x.shape[-1] == CONTACT_END
    rot = rc.rotation_6d_to_matrix(x[..., JOINT_END : JOINT_END + 6])
    trans = x[..., JOINT_END + 6 : HEAD_END]
    return rot_trans_to_matrix(rot, trans)


def canonicalizing_transform(head_T):
    """The 4x4 that canonicalizes a head pose, i.e. ``canonicalize_trajectory``'s ``tsfm_T``.

    Given a head pose expressed in some frame, returns the yaw + XY translation
    that moves that frame to the one where this head sits at the XY origin facing
    +X (height and gravity untouched). Applying it to a whole sequence reproduces
    :func:`modiff.dataset.canonicalization.canonicalize_trajectory` exactly.

    :param head_T: ``(..., 4, 4)``.
    :return: ``(..., 4, 4)`` on the same device / dtype as ``head_T``.
    """
    shape = head_T.shape[:-2]
    flat = head_T.reshape(-1, 4, 4)

    # rotation_to_make_this_forward_batch builds its constant vectors on the CPU
    # without a .to(device), so it only runs on CPU tensors. The number of
    # windows is tiny, so just do the whole thing there.
    flat_cpu = flat.detach().to(device="cpu", dtype=torch.float32)
    R = rotation_to_make_this_forward_batch(flat_cpu[:, :3, :3])  # B x 3 x 3
    trans = flat_cpu[:, :3, 3] * torch.tensor([-1.0, -1.0, 0.0])
    t = (R @ trans[:, :, None])[:, :, 0]
    G = rot_trans_to_matrix(R, t)

    return G.reshape(*shape, 4, 4).to(device=head_T.device, dtype=head_T.dtype)


def make_window_starts(n_frames, window, stride):
    """Start indices of overlapping windows covering ``n_frames``.

    The last window is snapped to end exactly at ``n_frames``, so the final stride
    may be shorter than requested.
    """
    assert stride >= 1
    if n_frames <= window:
        return [0]
    starts = list(range(0, n_frames - window + 1, stride))
    if starts[-1] != n_frames - window:
        starts.append(n_frames - window)
    return starts


def window_taper_weights(window, taper, eps=1e-2, device=None, dtype=torch.float32):
    """Trapezoidal per-frame fusion weights, ramping over ``taper`` frames at each end.

    Clamped away from zero so that frames covered by a single window (the two ends
    of the sequence) still get a well-defined weighted average.
    """
    taper = max(int(taper), 1)
    i = torch.arange(window, device=device, dtype=dtype)
    w = torch.minimum((i + 1) / taper, (window - i) / taper)
    return w.clamp(eps, 1.0)


def align_se3_trajectories(X, Y, alpha=0.1, with_scale=False):
    """Estimate ``A`` (4x4) such that ``A @ Y_t`` best matches ``X_t``.

    Closed-form blend of translation and rotation energies.

    :param X: ``(T, 4, 4)`` target poses.
    :param Y: ``(T, 4, 4)`` source poses.
    :param alpha: weight on rotation consistency; 0 means translation only.
    """
    device, dtype = X.device, X.dtype

    t_x, t_y = X[:, :3, 3], Y[:, :3, 3]
    R_x, R_y = X[:, :3, :3], Y[:, :3, :3]

    mu_x, mu_y = t_x.mean(dim=0), t_y.mean(dim=0)
    Xc, Yc = t_x - mu_x, t_y - mu_y

    S_pos = Yc.T @ Xc
    S_rot = torch.bmm(R_y, R_x.transpose(1, 2)).mean(dim=0)
    position_energy = (Yc**2).sum()
    if position_energy < 1e-6:
        position_energy = torch.tensor(1.0, device=device, dtype=dtype)
    S_rot = S_rot * position_energy
    S = (1 - alpha) * S_pos + alpha * S_rot

    U, _, Vh = torch.linalg.svd(S)
    R = Vh.T @ U.T
    if torch.det(R) < 0:
        mask = torch.ones(3, device=device, dtype=dtype)
        mask[-1] = -1
        R = Vh.T @ torch.diag(mask) @ U.T

    if with_scale:
        s = torch.trace(S_pos @ R) / (Yc**2).sum()
    else:
        s = torch.tensor(1.0, device=device, dtype=dtype)

    A = torch.eye(4, device=device, dtype=dtype)
    A[:3, :3] = s * R
    A[:3, 3] = mu_x - s * (R @ mu_y)
    return A, R, A[:3, 3], s
