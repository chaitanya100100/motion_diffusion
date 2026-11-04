"""Long-sequence sampling for trajectory-conditioned ``v1_ait`` motion models.

The checkpoints are trained at a fixed ``cfg.DATA.WINDOW`` (80 frames = 8 s @ 10 fps)
and ``UEM_Module.sample`` hardcodes that length. This module adds ways to generate
an arbitrarily long motion from an arbitrarily long head trajectory, without
retraining and without touching any existing code path.

Three modes are available (see :func:`sample_long`):

``"multidiffusion"`` (default)
    MultiDiffusion / SyncTweedies style joint windowed denoising. Overlapping
    windows are denoised in parallel as one batch and their ``pred_xstart``
    predictions are fused in a common frame after every diffusion step.

``"autoregressive"``
    Baseline. Windows are generated one after another; each one's overlap with the
    previous result is pinned using the model's existing ``repaint_mask`` /
    ``repaint_value`` inpainting hook.

``"naive"``
    Baseline. One single pass at full length. The denoiser is fully convolutional
    in time so this runs, but the canonicalization only holds near frame 0 and the
    result drifts out of the training distribution.

Frames and the unknown gravity
------------------------------
Each training window is canonicalized so that its first head pose sits at the XY
origin facing +X. The transform doing that is a yaw rotation about Z plus an XY
translation -- an SE(2) motion -- so two windows of the same sequence differ by a
known-shape rigid transform and *share* gravity and floor height.

At inference only the relative head trajectory is trustworthy, so the head pose in
the global canonical frame ``C_ref`` is ``H_t = M @ A_t`` for one unknown rigid
``M``. Only three of its degrees of freedom are actually free (two of gravity tilt
plus head height); yaw and XY are pinned by the definition of ``C_ref``. ``M`` is
estimated from the model's own prediction, see
:func:`estimate_traj_to_canonical`. Note that the window frames derived from it
only carry yaw and XY -- tilt and height are not encoded in them at all and get
synchronized for free by the cross-window fusion.
"""

import torch
from loguru import logger

import modiff.utils.rotation_conversions as rc
from modiff.dataset.motion_transforms import (
    align_se3_trajectories,
    canonicalizing_transform,
    head_T_from_v1_ait,
    make_window_starts,
    transform_v1_ait_motion,
    window_taper_weights,
)
from modiff.dataset.canonicalization import rot_trans_to_matrix
from modiff.dataset.representation_utils import center_aria_traj


def _check_repre(module):
    repre = module.cfg.DATA.REPRE_TYPE
    if repre != "v1_ait":
        raise NotImplementedError(
            f"long sampling assumes the v1_ait channel layout and its SE(2) canonicalization, got {repre!r}"
        )


def _bla(dtype=torch.float32, device="cpu"):
    """The fixed frame swap ``center_aria_traj`` applies (+90 deg about X)."""
    return torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=dtype, device=device)


def traj_T_from_repre(traj_repre):
    """Inverse of :func:`traj_repre_from_T`: ``T x 9`` conditioning -> ``T x 4 x 4`` poses.

    Recovers ``T_0^-1 @ T_i``, i.e. the trajectory in its own first frame. That is
    all the downstream math needs, since every step is invariant to which frame the
    poses are expressed in.
    """
    centered = rot_trans_to_matrix(rc.rotation_6d_to_matrix(traj_repre[:, :6]), traj_repre[:, 6:9])
    return torch.inverse(_bla(traj_repre.dtype, traj_repre.device))[None] @ centered


def _as_traj_T(aria_traj_T):
    """Accept either ``T x 4 x 4`` head poses or the ``T x 9`` conditioning."""
    if not torch.is_tensor(aria_traj_T):
        aria_traj_T = torch.as_tensor(aria_traj_T)
    aria_traj_T = aria_traj_T.detach()
    # aria_traj_T = aria_traj_T.detach().to(device="cpu", dtype=torch.float32)
    if aria_traj_T.ndim == 3 and aria_traj_T.shape[0] == 1 and aria_traj_T.shape[-1] == 9:
        aria_traj_T = aria_traj_T[0]  # a batched (1, T, 9) conditioning tensor
    if aria_traj_T.ndim == 2 and aria_traj_T.shape[-1] == 9:
        return traj_T_from_repre(aria_traj_T)
    assert aria_traj_T.ndim == 3 and aria_traj_T.shape[1:] == (
        4,
        4,
    ), f"expected T x 4 x 4 head poses or T x 9 conditioning, got {tuple(aria_traj_T.shape)}"
    return aria_traj_T


def traj_repre_from_T(traj_T):
    """``T x 4 x 4`` head poses -> ``T x 9`` v1_ait conditioning.

    ``center_aria_traj`` makes this relative to the window's own first frame, so it is invariant to whatever frame
    ``traj_T`` is expressed in.
    """
    centered = center_aria_traj(traj_T)
    return torch.cat([rc.matrix_to_rotation_6d(centered[:, :3, :3]), centered[:, :3, 3]], dim=1)


def build_window_conditioning(traj_T, starts, window):
    """``W x window x 9`` conditioning, one entry per window."""
    return torch.stack([traj_repre_from_T(traj_T[s : s + window]) for s in starts], dim=0)


def _pin_reference_frame(M, traj_T):
    """Re-canonicalize ``M`` so that frame 0 defines the reference frame ``C_ref``.

    A Procrustes fit gets the tilt and height right but leaves yaw / XY arbitrary.
    Composing with the canonicalizing transform of the first head pose pins them,
    which makes the first window's frame the identity and matches what a plain
    80-frame ``sample()`` would produce.
    """
    return canonicalizing_transform(M @ traj_T[0]) @ M


def _fit_M(pred_motion, traj_T):
    """Fit the trajectory -> canonical-frame transform from a generated motion."""
    pred_head = head_T_from_v1_ait(pred_motion).detach().to(traj_T)  # match traj_T's device/dtype
    M, _, _, _ = align_se3_trajectories(pred_head, traj_T[: pred_head.shape[0]])
    return _pin_reference_frame(M, traj_T)


@torch.no_grad()
def estimate_traj_to_canonical(module, traj_T, mode="window0", cond_scale=None, **kwargs):
    """Estimate the rigid ``M`` mapping input trajectory poses into ``C_ref``.

    :param mode: ``"window0"`` runs one ordinary 80-frame sample on the first
        window and fits against it -- cheap, and enough because the three free
        degrees of freedom are global. ``"autoregressive"`` runs the full
        autoregressive rollout first and fits over the whole length instead.
    """
    window = module.window
    if mode == "window0":
        cond = traj_repre_from_T(traj_T[:window])[None]
        motion = module.sample({"traj": cond}, B=1, cond_scale=cond_scale)[0]
        return _fit_M(motion, traj_T)
    if mode == "autoregressive":
        motion, _ = sample_long_autoregressive(module, traj_T, cond_scale=cond_scale, **kwargs)
        return _fit_M(motion, traj_T)
    raise ValueError(f"unknown frame_init {mode!r}")


@torch.no_grad()
def sample_long_naive(module, traj_T, cond_scale=None, progress=False):
    """Baseline: a single full-length pass through the denoiser."""
    _check_repre(module)
    traj_T = _as_traj_T(traj_T)
    n_frames = traj_T.shape[0]
    device, D = traj_T.device, module.model.input_feats

    cond = traj_repre_from_T(traj_T)[None]
    y = {
        "traj": module.normalizer.normalize(cond, "traj"),
        "valid_frames": torch.ones(1, n_frames, device=device),
    }
    x = module.diffusion.p_sample_loop(
        module.model,
        (1, n_frames, D),
        model_kwargs={"y": y, "cond_scale": cond_scale, "diffusion": module.diffusion},
        clip_denoised=False,
        noise=None,
        progress=progress,
    )
    return module.normalizer.denormalize(x, None)[0], {"starts": [0], "n_windows": 1}


@torch.no_grad()
def sample_long_autoregressive(module, traj_T, stride=None, cond_scale=None, progress=False):
    """Baseline: sequential windows, overlap pinned with the model's repaint hook.

    Needs no estimate of ``M``: window ``s``'s frame is read off the head pose that
    the already-generated part of the sequence predicts at frame ``s``, so the
    rollout bootstraps itself. That is also why it doubles as the more robust
    ``frame_init`` for :func:`sample_long_multidiffusion`.
    """
    _check_repre(module)
    traj_T = _as_traj_T(traj_T)
    window = module.window
    n_frames = traj_T.shape[0]
    device, D = traj_T.device, module.model.input_feats
    stride = window // 2 if stride is None else stride
    assert stride < window, "windows must overlap so each one's frame can be read off the previous result"
    starts = make_window_starts(n_frames, window, stride)

    out = torch.zeros(n_frames, D, device=device)
    frames = []
    filled = 0
    for wi, s in enumerate(starts):
        cond = traj_repre_from_T(traj_T[s : s + window])[None]
        y = {
            "traj": module.normalizer.normalize(cond, "traj"),
            "valid_frames": torch.ones(1, window, device=device),
        }

        if wi == 0:
            G = torch.eye(4, device=device)
        else:
            G = canonicalizing_transform(head_T_from_v1_ait(out[s])).to(device)
            n_known = filled - s
            known = transform_v1_ait_motion(out[s : s + n_known], G)  # C_ref -> C_s
            value = torch.zeros(1, window, D, device=device)
            value[0, :n_known] = known
            mask = torch.zeros(1, window, D, device=device)
            mask[:, :n_known] = 1.0
            y["repaint_mask"] = mask
            # The repaint hook overwrites the model's *normalized* xstart prediction,
            # so the value has to be normalized here -- module.sample only normalizes
            # y["traj"].
            y["repaint_value"] = module.normalizer.normalize(value, None)

        motion = module.sample(y, B=1, cond_scale=cond_scale)[0]  # window x D, in C_s
        motion_ref = transform_v1_ait_motion(motion, torch.inverse(G))
        # The pinned overlap comes back unchanged (at t=0 the posterior mean is
        # exactly xstart), so only the frames past `filled` need writing.
        new_from = max(filled, s)
        out[new_from : s + window] = motion_ref[new_from - s :]
        filled = s + window
        frames.append(G)
        if progress:
            logger.info(f"autoregressive window {wi + 1}/{len(starts)} (frames {s}:{s + window})")

    return out, {"starts": starts, "frames": torch.stack(frames), "n_windows": len(starts)}


@torch.no_grad()
def sample_long_multidiffusion(
    module,
    traj_T,
    stride=20,
    taper=None,
    n_passes=2,
    frame_init="window0",
    cond_scale=None,
    M=None,
    progress=False,
):
    """Joint windowed denoising with per-step fusion of the window predictions.

    Each window keeps its own latent in its own canonical frame; only the
    ``pred_xstart`` predictions are mapped into ``C_ref``, averaged, and mapped
    back. Noisy latents are never transformed -- doing so would need a
    ``sqrt(alpha_bar_t)`` scaling of the translation *and* would break noise
    isotropy, since ``diag(1/sigma) R diag(sigma)`` is not orthogonal when the
    per-channel normalizer stds differ.

    The fusion is injected as ``denoised_fn``, which the existing sampler already
    threads through ``p_sample_loop -> p_sample -> p_mean_variance`` and applies to
    ``pred_xstart`` before the posterior step.

    :param stride: window hop in frames; 20 gives 75% overlap at window 80.
    :param taper: ramp length of the trapezoidal fusion weights (default ``window // 2``).
    :param n_passes: passes over the whole schedule. After the first, ``M`` is
        refit from the full-length result, which is cleaner than mutating window
        frames mid-loop.
    :param frame_init: how the initial ``M`` is bootstrapped, see
        :func:`estimate_traj_to_canonical`.
    """
    traj_T = _as_traj_T(traj_T)
    _check_repre(module)
    traj_T = _as_traj_T(traj_T)
    window = module.window
    n_frames = traj_T.shape[0]
    device, D = traj_T.device, module.model.input_feats
    starts = make_window_starts(n_frames, window, stride)
    n_win = len(starts)

    cond = build_window_conditioning(traj_T, starts, window)

    y = {
        "traj": module.normalizer.normalize(cond, "traj"),
        "valid_frames": torch.ones(n_win, window, device=device),
    }

    if M is None:
        M = estimate_traj_to_canonical(module, traj_T, mode=frame_init, cond_scale=cond_scale)

    taper = window // 2 if taper is None else taper
    weights = window_taper_weights(window, taper, device=device)[None, :, None]  # 1 x window x 1
    index = torch.stack([torch.arange(s, s + window) for s in starts]).to(device)  # W x window
    flat_index = index.reshape(-1)
    flat_weights = weights.expand(n_win, window, 1).reshape(-1, 1)

    motion = None
    for p in range(max(1, n_passes)):
        head_ref = M[None] @ traj_T  # head poses in C_ref
        G = canonicalizing_transform(head_ref[starts]).to(device)  # W x 4 x 4, C_ref -> C_s
        G_inv = torch.inverse(G)
        holder = {}

        def fuse(x0_norm):
            x0 = module.normalizer.denormalize(x0_norm, None)  # W x window x D, each in its own C_s
            x0_ref = transform_v1_ait_motion(x0, G_inv[:, None])

            num = torch.zeros(n_frames, D, device=device, dtype=x0_ref.dtype)
            den = torch.zeros(n_frames, 1, device=device, dtype=x0_ref.dtype)
            num.index_add_(0, flat_index, (weights * x0_ref).reshape(-1, D))
            den.index_add_(0, flat_index, flat_weights)
            glob = num / den
            holder["glob"] = glob

            return module.normalizer.normalize(transform_v1_ait_motion(glob[index], G[:, None]), None)

        module.diffusion.p_sample_loop(
            module.model,
            (n_win, window, D),
            model_kwargs={"y": y, "cond_scale": cond_scale, "diffusion": module.diffusion},
            clip_denoised=False,
            noise=None,
            progress=progress,
            denoised_fn=fuse,
        )
        motion = holder["glob"]
        if p + 1 < n_passes:
            M = _fit_M(motion, traj_T)

    return motion, {"starts": starts, "frames": G, "M": M, "n_windows": n_win}


@torch.no_grad()
def sample_long(module, aria_traj_T, mode="multidiffusion", seed=None, cond_scale=None, **kwargs):
    """Generate a long motion conditioned on a long head trajectory.

    :param module: a loaded ``UEM_Module`` with ``REPRE_TYPE == "v1_ait"``, already
        moved to the target device and in eval mode.
    :param aria_traj_T: ``T x 4 x 4`` head poses, or the ``T x 9`` (or ``1 x T x 9``)
        conditioning you would otherwise pass to ``sample`` as ``y["traj"]``. Any
        device. Only the *relative* trajectory is used, so the frame the poses are
        expressed in does not matter and no gravity alignment or floor height is
        required.
    :param mode: ``"multidiffusion"`` | ``"autoregressive"`` | ``"naive"``.
    :param seed: optional ``torch.manual_seed`` for reproducibility.
    :param kwargs: forwarded to the chosen mode (``stride``, ``taper``,
        ``n_passes``, ``frame_init``, ``progress``).
    :return: ``(motion, info)`` where ``motion`` is a ``T x 209`` denormalized
        ``v1_ait`` tensor on the model's device, in a single canonical frame --
        feed it straight to
        ``repre_to_sequence(cfg.DATA.REPRE_TYPE, motion, None, smpl, None, None)``.
        The caller aligns the input trajectory into that frame rather than the other way round.
    """
    _check_repre(module)
    if seed is not None:
        torch.manual_seed(seed)

    traj_T = _as_traj_T(aria_traj_T)
    device = traj_T.device
    n_frames = traj_T.shape[0]
    if n_frames <= module.window:
        # Nothing to stitch; fall back to the ordinary fixed-window path, zero-padding
        # the conditioning the same way Motion_Dataset.pad_to_window does. If the
        # caller already handed us the 9-d conditioning, reuse it verbatim so this is
        # bit-for-bit the call they were making before.
        if torch.is_tensor(aria_traj_T) and aria_traj_T.shape[-1] == 9:
            cond = aria_traj_T.detach().reshape(-1, 9).to(device, torch.float32)
        else:
            cond = traj_repre_from_T(traj_T)
        pad = module.window - n_frames
        y = {"valid_frames": torch.ones(1, module.window, device=device)}
        if pad:
            cond = torch.cat([cond, cond.new_zeros(pad, cond.shape[-1])], dim=0)
            y["valid_frames"][:, n_frames:] = 0
        y["traj"] = cond[None]
        motion = module.sample(y, B=1, cond_scale=cond_scale)[0]
        return motion[:n_frames], {"starts": [0], "n_windows": 1}

    fns = {
        "multidiffusion": sample_long_multidiffusion,
        "autoregressive": sample_long_autoregressive,
        "naive": sample_long_naive,
    }
    if mode not in fns:
        raise ValueError(f"unknown mode {mode!r}, expected one of {sorted(fns)}")
    return fns[mode](module, traj_T, cond_scale=cond_scale, **kwargs)
