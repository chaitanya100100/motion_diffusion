import copy

import numpy as np
import torch

import modiff.utils.rotation_conversions as rc
from modiff.dataset.canonicalization import rot_trans_to_matrix, rotation_to_make_this_forward_batch
from modiff.dataset.egoego_utils import local2global_pose, mat_ik_torch
from modiff.dataset.smpl_utils import saved_sequence_to_full_sequence, pca_to_matrix, matrix_to_pca


MOTION_DIM = {
    "v1_ait": 209,
    "v1": 224,
    "v1f": 224,
    "v1_beta": 234,
    "v4": 233,
    "v4_beta": 243,
    "v5": 233,
    "v5_beta": 243,
}
TRAJ_DIM = {
    "v1_ait": 9,
    "v1": 9,
    "v1f": 9,
    "v1_beta": 9,
    "v4": 18,
    "v4_beta": 18,
    "v5": 18,
    "v5_beta": 18,
}


def foot_detect(kp3d):
    # kp3d: T x 22 x 3
    fid_l, fid_r = 10, 11
    velfactor, heightfactor = 0.02, 0.10

    vel_l = kp3d[1:, fid_l] - kp3d[:-1, fid_l]
    vel_l = vel_l.square().sum(-1).sqrt()
    vel_l = (vel_l < velfactor).float()
    vel_l = torch.cat([vel_l[:1], vel_l], dim=0)

    vel_r = kp3d[1:, fid_r] - kp3d[:-1, fid_r]
    vel_r = vel_r.square().sum(-1).sqrt()
    vel_r = (vel_r < velfactor).float()
    vel_r = torch.cat([vel_r[:1], vel_r], dim=0)

    height_l = (kp3d[:, fid_l, 2] < heightfactor).float()
    height_r = (kp3d[:, fid_r, 2] < heightfactor).float()

    feet_l = vel_l * height_l
    feet_r = vel_r * height_r
    return feet_l, feet_r


def get_aria_traj_T(global_T):
    # Load ARIA trajectory
    head_idx, leye_idx, reye_idx = 15, 23, 24
    # global_pose = local2global_pose(full_pose, list(smpl.parents))
    # aria_rot = global_pose[:, head_idx]  # T x 3 x 3 head rotation
    # # aria_trans = kp3d[:, [leye_idx, reye_idx]].mean(1)  # T x 3 supposed aria translation
    # aria_trans = kp3d[:, [head_idx]].mean(1)  # T x 3 supposed aria translation
    # aria_traj_T = rot_trans_to_matrix(aria_rot, aria_trans)  # T x 4 x 4
    aria_traj_T = global_T[:, head_idx]  # T x 4 x 4
    return aria_traj_T


def smpl_params_to_global_T(smpl_params, smpl):
    from modiff.dataset.smpl_utils import evaluate_smpl

    kp3d, verts, full_pose = evaluate_smpl(smpl, smpl_params)
    global_rotmat = local2global_pose(full_pose[:, :22], list(smpl.parents)[:22])
    global_trans = kp3d[:, :22]
    global_T = rot_trans_to_matrix(global_rotmat, global_trans)  # T x 22 x 4 x 4
    return global_T


# ------------------------------------------------------------------------------------------------
# Common functions for all representations.
# ------------------------------------------------------------------------------------------------
def common_sequence_to_repre(aria_traj, smpl_params, kp3d, floor_height, smpl):
    # Some common operations for all representations. Avoids code duplication.
    # global_T is body joint transformations in global frame, not wrt parent joint.

    # left_hand_pca = smpl_params["left_hand_pose"].clone()
    # right_hand_pca = smpl_params["right_hand_pose"].clone()
    aria_traj_T, smpl_params = saved_sequence_to_full_sequence(aria_traj, smpl_params, smpl)
    if "left_hand_pose" in smpl_params:
        left_hand_pca = matrix_to_pca(smpl_params["left_hand_pose"], smpl.left_hand_components)
        right_hand_pca = matrix_to_pca(smpl_params["right_hand_pose"], smpl.right_hand_components)
    else:
        T = smpl_params["global_orient"].shape[0]
        left_hand_pca = torch.zeros(T, 12)
        right_hand_pca = torch.zeros(T, 12)

    if floor_height == 0:
        # print(f"Floor height is not available: {floor_height}")
        floor_height = kp3d[:, [7, 8, 10, 11], 2].min().item() - 0.02

    # apply offset for floor height
    offset = torch.tensor([0, 0, floor_height]).float().to(aria_traj.device)
    aria_traj_T = aria_traj_T.clone()
    aria_traj_T[:, :3, 3] = aria_traj_T[:, :3, 3] - offset
    kp3d = kp3d - offset
    smpl_params["transl"] = smpl_params["transl"] - offset
    del offset

    feet_l, feet_r = foot_detect(kp3d)  # T

    # Get global transformations for each joint
    T = aria_traj_T.shape[0]
    full_pose = torch.cat(
        [
            smpl_params["global_orient"][:, None],
            smpl_params["body_pose"],
            # torch.eye(3)[None, None].repeat(T, 1, 1, 1).to(aria_traj.device),  # jaw
            # torch.eye(3)[None, None].repeat(T, 1, 1, 1).to(aria_traj.device),  # leye
            # torch.eye(3)[None, None].repeat(T, 1, 1, 1).to(aria_traj.device),  # reye
            # smpl_params["left_hand_pose"],
            # smpl_params["right_hand_pose"],
        ],
        dim=1,
    )  # T x 22 x 3 x 3
    global_rotmat = local2global_pose(full_pose, list(smpl.parents)[:22])  # T x 22 x 3 x 3
    global_trans = kp3d[:, :22]
    global_T = rot_trans_to_matrix(global_rotmat, global_trans)  # T x 22 x 4 x 4

    return global_T, feet_l, feet_r, aria_traj_T, smpl_params, left_hand_pca, right_hand_pca


def center_aria_traj(aria_traj_T):
    # Transform invariant aria trajectory. We additionally apply `bla` to align it nicely
    # with our motion representation. Although network could learn without this, this provides
    # a better visualization for debugging.
    aria_traj_T_centered = aria_traj_T[..., :1, :, :].inverse() @ aria_traj_T  # T x 4 x 4
    bla = (
        torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
        )
        .to(aria_traj_T.dtype)
        .to(aria_traj_T.device)
    )
    aria_traj_T_centered = bla @ aria_traj_T_centered

    return aria_traj_T_centered


# ------------------------------------------------------------------------------------------------
# v1_ait: Same as V1 but aria trajectory is affine invariant i.e. doesn't have floor and gravity information.
# ------------------------------------------------------------------------------------------------
def sequence_to_repre_v1_ait(aria_traj, smpl_params, kp3d, floor_height, smpl):
    global_T, feet_l, feet_r, aria_traj_T, smpl_params, left_hand_pca, right_hand_pca = common_sequence_to_repre(
        aria_traj, smpl_params, kp3d, floor_height, smpl
    )

    T = global_T.shape[0]
    global_rot6d = rc.matrix_to_rotation_6d(global_T[:, :22, :3, :3])  # T x 22 x 6
    global_rottrans = torch.cat([global_rot6d, global_T[:, :22, :3, 3]], dim=2)  # T x 22 x 9

    # get aria from pose
    head_idx, leye_idx, reye_idx = 15, 23, 24
    aria_rot_from_pose = global_T[:, head_idx, :3, :3]  # T x 3 x 3 head rotation
    aria_trans_from_pose = global_T[:, [head_idx], :3, 3].mean(1)  # T x 3 supposed aria translation
    aria_rottrans_from_pose = torch.cat(
        [rc.matrix_to_rotation_6d(aria_rot_from_pose), aria_trans_from_pose], dim=1
    )  # T x 9

    # representation
    x = torch.cat(
        [
            global_rottrans.view(T, 198),  # T x 198
            aria_rottrans_from_pose,  # T x 9
            feet_l.view(T, 1),  # T x 1
            feet_r.view(T, 1),  # T x 1
        ],
        dim=1,
    )  # T x (198+9+1+1)
    x = x.view(T, 209)

    aria_traj_T_centered = center_aria_traj(aria_traj_T)

    aria_traj_repre = torch.cat(
        [rc.matrix_to_rotation_6d(aria_traj_T_centered[:, :3, :3]), aria_traj_T_centered[:, :3, 3]], dim=1
    )  # T x 9
    return x, aria_traj_repre


def repre_v1_ait_to_sequence(x, aria_traj_repre, smpl, betas, body_root_offset):
    T = aria_traj_repre.shape[0] if aria_traj_repre is not None else x.shape[0]
    device = aria_traj_repre.device if aria_traj_repre is not None else x.device
    dtype = aria_traj_repre.dtype if aria_traj_repre is not None else x.dtype

    # ignore betas and body root offset
    betas = None
    body_root_offset = None
    betas = torch.zeros(10, device=device, dtype=dtype)

    # We will ignore aria trajectory but putting it here for debugging purposes.
    if aria_traj_repre is not None:
        aria_traj_centered_rotmat = rc.rotation_6d_to_matrix(aria_traj_repre[:, :6])
        aria_traj_centered_trans = aria_traj_repre[:, 6:9]
        aria_traj_T_centered = rot_trans_to_matrix(aria_traj_centered_rotmat, aria_traj_centered_trans)
    else:
        aria_traj_T_centered = None

    # Get smpl parameters from x
    global_rottrans = x[:, :198].view(T, 22, 9)  # T x 22 x 9
    global_rotmat = rc.rotation_6d_to_matrix(global_rottrans[:, :, :6])  # T x 22 x 3 x 3
    global_trans = global_rottrans[:, :, 6:9]  # T x 22 x 3
    local_rotmat = mat_ik_torch(global_rotmat, list(smpl.parents)[:22])  # T x 22 x 3 x 3

    smpl_params = {}
    smpl_params["global_orient"] = local_rotmat[:, 0]  # T x 3 x 3
    smpl_params["body_pose"] = local_rotmat[:, 1:]  # T x 21 x 3 x 3
    smpl_params["betas"] = betas.view(-1, 10).repeat(T, 1)  # T x 10
    if body_root_offset is None:
        body_root_offset = smpl.forward(betas=betas.view(-1, 10)).joints[0, 0]  # 3
    smpl_params["transl"] = global_trans[:, 0] - body_root_offset[None]  # T x 3

    # Get aria trajectory from x
    aria_traj_rottrans = x[:, 198:207]  # T x 9
    aria_traj_rotmat = rc.rotation_6d_to_matrix(aria_traj_rottrans[:, :6])  # T x 3 x 3
    aria_traj_trans = aria_traj_rottrans[:, 6:9]  # T x 3
    aria_traj_T = rot_trans_to_matrix(aria_traj_rotmat, aria_traj_trans)  # T x 4 x 4

    j3d = global_trans

    return aria_traj_T, smpl_params, j3d


# ------------------------------------------------------------------------------------------------
# V1: Global rotation and translation of each joint of each frame.
# ------------------------------------------------------------------------------------------------
def sequence_to_repre_v1(aria_traj, smpl_params, kp3d, floor_height, smpl):
    # Explanation of representations is at the top.

    global_T, feet_l, feet_r, aria_traj_T, smpl_params, left_hand_pca, right_hand_pca = common_sequence_to_repre(
        aria_traj, smpl_params, kp3d, floor_height, smpl
    )

    global_rot6d = rc.matrix_to_rotation_6d(global_T[:, :22, :3, :3])  # T x 22 x 6
    global_rottrans = torch.cat([global_rot6d, global_T[:, :22, :3, 3]], dim=2)  # T x 22 x 9
    T = aria_traj_T.shape[0]
    x = torch.cat(
        [
            global_rottrans.view(T, 198),  # T x 198
            left_hand_pca,  # T x 12
            right_hand_pca,  # T x 12
            feet_l.view(T, 1),  # T x 1
            feet_r.view(T, 1),  # T x 1
        ],
        dim=1,
    )  # T x (198+12+12+1+1)
    x = x.view(T, 224)

    aria_traj_rot6d = rc.matrix_to_rotation_6d(aria_traj_T[:, :3, :3])  # T x 6
    aria_traj_trans = aria_traj_T[:, :3, 3]  # T x 3
    aria_traj_repre = torch.cat([aria_traj_rot6d, aria_traj_trans], dim=1)  # T x 9

    return x, aria_traj_repre


def sequence_to_repre_v1f(aria_traj, smpl_params, kp3d, floor_height, smpl):
    x, aria_traj_repre = sequence_to_repre_v1(aria_traj, smpl_params, kp3d, floor_height, smpl)

    fh = aria_traj_repre[0, -1]
    aria_traj_repre[:, -1] = aria_traj_repre[:, -1] - fh
    return x, aria_traj_repre


def repre_v1_to_sequence(x, aria_traj_repre, smpl, betas, body_root_offset):
    # Explanation of representations is at the top.
    T = aria_traj_repre.shape[0] if aria_traj_repre is not None else x.shape[0]
    if aria_traj_repre is not None:
        aria_traj_rotmat = rc.rotation_6d_to_matrix(aria_traj_repre[:, :6])
        aria_traj_trans = aria_traj_repre[:, 6:9]
        aria_traj_T = rot_trans_to_matrix(aria_traj_rotmat, aria_traj_trans)
    else:
        aria_traj_T = None

    if x is None:
        return aria_traj_T, None, None

    if betas is None:
        betas = torch.zeros(10, device=x.device, dtype=x.dtype)

    global_rottrans = x[:, :198].view(T, 22, 9)  # T x 22 x 9
    left_hand_pca = x[:, 198:210]  # T x 12
    right_hand_pca = x[:, 210:222]  # T x 12

    global_rotmat = rc.rotation_6d_to_matrix(global_rottrans[:, :, :6])  # T x 22 x 3 x 3
    global_trans = global_rottrans[:, :, 6:9]  # T x 22 x 3
    local_rotmat = mat_ik_torch(global_rotmat, list(smpl.parents)[:22])  # T x 22 x 3 x 3

    smpl_params = {}
    smpl_params["global_orient"] = local_rotmat[:, 0]  # T x 3 x 3
    smpl_params["body_pose"] = local_rotmat[:, 1:]  # T x 21 x 3 x 3
    smpl_params["betas"] = betas.view(-1, 10).repeat(T, 1)
    smpl_params["left_hand_pose"] = pca_to_matrix(left_hand_pca, smpl.left_hand_components)
    smpl_params["right_hand_pose"] = pca_to_matrix(right_hand_pca, smpl.right_hand_components)

    if body_root_offset is None:
        body_root_offset = smpl.forward(betas=betas.view(-1, 10)).joints[0, 0]  # 3
    smpl_params["transl"] = global_trans[:, 0] - body_root_offset[None]  # T x 3

    j3d = global_trans

    return aria_traj_T, smpl_params, j3d


def repre_v1f_to_sequence(x, aria_traj_repre, smpl, betas, body_root_offset):
    return repre_v1_to_sequence(x, aria_traj_repre, smpl, betas, body_root_offset)


# ------------------------------------------------------------------------------------------------
# V1_beta: V1 with betas
# ------------------------------------------------------------------------------------------------
def sequence_to_repre_v1_beta(aria_traj, smpl_params, kp3d, floor_height, smpl):
    # Explanation of representations is at the top.
    x, aria_traj_repre = sequence_to_repre_v1(aria_traj, smpl_params, kp3d, floor_height, smpl)
    # append beta to x
    betas = smpl_params["betas"]  # T x 10
    assert betas.shape[-1] == 10
    x = torch.cat([x, betas], dim=1)  # T x 234
    return x, aria_traj_repre


def repre_v1_beta_to_sequence(x, aria_traj_repre, smpl, betas, body_root_offset):
    # Explanation of representations is at the top.
    # Modify gt betas with the mean of the predicted betas
    x, x_betas = x[:, :-10], x[:, -10:]
    x_betas = x_betas.mean(0)[None]  # 1 x 10
    body_root_offset = None
    return repre_v1_to_sequence(x, aria_traj_repre, smpl, x_betas, body_root_offset)


# ------------------------------------------------------------------------------------------------
# V4: Head trajectory is projected onto floor and encoded as residuals from previous frame. Rest
# of the body pose is w.r.t floor-projected head trajectory.
# ------------------------------------------------------------------------------------------------
def sequence_to_repre_v4(aria_traj, smpl_params, kp3d, floor_height, smpl, canon_root_idx=15):  # 23 is leye
    # Explanation of representations is at the top.

    global_T, feet_l, feet_r, aria_traj_T, smpl_params, left_hand_pca, right_hand_pca = common_sequence_to_repre(
        aria_traj, smpl_params, kp3d, floor_height, smpl
    )
    T = aria_traj_T.shape[0]

    # Canonicalization of leye
    if canon_root_idx >= global_T.shape[1]:
        raise
        leye_T = aria_traj_T.clone()
    else:
        leye_T = global_T[:, canon_root_idx]  # T x 4 x 4
    tsfm_rot = rotation_to_make_this_forward_batch(leye_T[:, :3, :3])  # T x 3 x 3
    tsfm_trans = leye_T[:, :3, 3] * torch.tensor([-1.0, -1, 0])  # T x 3
    tsfm_trans = (tsfm_rot @ tsfm_trans[..., None])[..., 0]  # T x 3
    tsfm_T = rot_trans_to_matrix(tsfm_rot, tsfm_trans)  # T x 4 x 4
    local_T = tsfm_T[:, None] @ global_T  # T x 22 x 4 x 4

    # Canonicalization as delta
    invtsfm_T = tsfm_T.inverse()
    delta_invtsfm_T = invtsfm_T[:-1].inverse() @ invtsfm_T[1:]  # (T-1) x 4 x 4
    delta_invtsfm_T = torch.cat([invtsfm_T[:1], delta_invtsfm_T], dim=0)  # T x 4 x 4

    # representation
    local_rottrans = torch.cat(
        [rc.matrix_to_rotation_6d(local_T[:, :22, :3, :3]), local_T[:, :22, :3, 3]], dim=2
    )  # T x nj x 9
    delta_invtsfm_rottrans = torch.cat(
        [rc.matrix_to_rotation_6d(delta_invtsfm_T[:, :3, :3]), delta_invtsfm_T[:, :3, 3]], dim=1
    )  # T x 9

    x = torch.cat(
        [
            # local_rottrans.view(T, 225),  # T x 225
            local_rottrans.view(T, 198),  # T x 198
            delta_invtsfm_rottrans,  # T x 9
            left_hand_pca,  # T x 12
            right_hand_pca,  # T x 12
            feet_l.view(T, 1),  # T x 1
            feet_r.view(T, 1),  # T x 1
        ],
        dim=1,
    )
    x = x.view(T, 233)

    # Aria trajectory canonicalization
    aria_tsfm_rot = rotation_to_make_this_forward_batch(aria_traj_T[:, :3, :3])  # T x 3 x 3
    aria_tsfm_trans = aria_traj_T[:, :3, 3] * torch.tensor([-1.0, -1, 0])  # T x 3
    aria_tsfm_trans = (aria_tsfm_rot @ aria_tsfm_trans[..., None])[..., 0]  # T x 3
    aria_tsfm_T = rot_trans_to_matrix(aria_tsfm_rot, aria_tsfm_trans)  # T x 4 x 4
    aria_local_T = aria_tsfm_T @ aria_traj_T  # T x 4 x 4

    # Canonical frame as delta
    aria_invtsfm_T = aria_tsfm_T.inverse()
    delta_aria_invtsfm_T = aria_invtsfm_T[:-1].inverse() @ aria_invtsfm_T[1:]  # (T-1) x 4 x 4
    delta_aria_invtsfm_T = torch.cat([aria_invtsfm_T[:1], delta_aria_invtsfm_T], dim=0)  # T x 4 x 4

    # representation
    aria_local_rottrans = torch.cat(
        [rc.matrix_to_rotation_6d(aria_local_T[:, :3, :3]), aria_local_T[:, :3, 3]], dim=1
    )  # T x 9
    delta_aria_invtsfm_rottrans = torch.cat(
        [rc.matrix_to_rotation_6d(delta_aria_invtsfm_T[:, :3, :3]), delta_aria_invtsfm_T[:, :3, 3]], dim=1
    )  # T x 9
    aria_traj_repre = torch.cat([aria_local_rottrans, delta_aria_invtsfm_rottrans], dim=1)  # T x 18

    return x, aria_traj_repre


def repre_v4_to_sequence(x, aria_traj_repre, smpl, betas, body_root_offset):
    # Explanation of representations is at the top.
    T = aria_traj_repre.shape[0] if aria_traj_repre is not None else x.shape[0]

    if aria_traj_repre is not None:
        # Get aria trajectory
        aria_local_T = rot_trans_to_matrix(rc.rotation_6d_to_matrix(aria_traj_repre[:, :6]), aria_traj_repre[:, 6:9])
        delta_aria_invtsfm_T = rot_trans_to_matrix(
            rc.rotation_6d_to_matrix(aria_traj_repre[:, 9:15]), aria_traj_repre[:, 15:18]
        )
        aria_invtsfm_T = [delta_aria_invtsfm_T[0]]
        for i in range(1, T):
            aria_invtsfm_T.append(aria_invtsfm_T[-1] @ delta_aria_invtsfm_T[i])
        aria_invtsfm_T = torch.stack(aria_invtsfm_T, dim=0)  # T x 4 x 4
        aria_traj_T = aria_invtsfm_T @ aria_local_T
    else:
        aria_traj_T = None

    if x is None:
        return aria_traj_T, None, None

    if betas is None:
        betas = torch.zeros(10, device=x.device, dtype=x.dtype)

    # Get body
    local_rottrans = x[:, :198].view(T, 22, 9)  # T x 22 x 9
    local_T = rot_trans_to_matrix(
        rc.rotation_6d_to_matrix(local_rottrans[:, :, :6]), local_rottrans[:, :, 6:9]
    )  # T x 22 x 4 x 4
    delta_invtsfm_rottrans = x[:, 198 : 198 + 9]  # T x 9
    delta_invtsfm_T = rot_trans_to_matrix(
        rc.rotation_6d_to_matrix(delta_invtsfm_rottrans[:, :6]), delta_invtsfm_rottrans[:, 6:9]
    )  # T x 4 x 4
    invtsfm_T = [delta_invtsfm_T[0]]
    for i in range(1, T):
        invtsfm_T.append(invtsfm_T[-1] @ delta_invtsfm_T[i])
    invtsfm_T = torch.stack(invtsfm_T, dim=0)  # T x 4 x 4
    global_T = invtsfm_T[:, None] @ local_T  # T x 22 x 4 x 4

    global_trans = global_T[:, :, :3, 3]  # T x 22 x 3
    local_rotmat = mat_ik_torch(global_T[:, :, :3, :3], list(smpl.parents)[: global_T.shape[1]])  # T x 22 x 3 x 3
    left_hand_pca = x[:, 198 + 9 : 198 + 9 + 12]  # T x 12
    right_hand_pca = x[:, 198 + 9 + 12 : 198 + 9 + 24]  # T x 12

    smpl_params = {}
    smpl_params["global_orient"] = local_rotmat[:, 0]  # T x 3 x 3
    smpl_params["body_pose"] = local_rotmat[:, 1:22]  # T x 21 x 3 x 3
    if local_rotmat.shape[1] > 22:
        smpl_params["jaw_pose"] = local_rotmat[:, 22]
        smpl_params["leye_pose"] = local_rotmat[:, 23]
        smpl_params["reye_pose"] = local_rotmat[:, 24]
    smpl_params["betas"] = betas.view(-1, 10).repeat(T, 1)
    smpl_params["left_hand_pose"] = pca_to_matrix(left_hand_pca, smpl.left_hand_components)
    smpl_params["right_hand_pose"] = pca_to_matrix(right_hand_pca, smpl.right_hand_components)

    if body_root_offset is None:
        body_root_offset = smpl.forward(betas=betas.view(-1, 10)).joints[0, 0]  # 3
    smpl_params["transl"] = global_trans[:, 0] - body_root_offset[None]  # T x 3

    j3d = global_trans
    if aria_traj_T is None:
        aria_traj_T = get_aria_traj_T(global_T)
    return aria_traj_T, smpl_params, j3d


# ------------------------------------------------------------------------------------------------
# V4_beta: V4 with betas
# ------------------------------------------------------------------------------------------------
def sequence_to_repre_v4_beta(aria_traj, smpl_params, kp3d, floor_height, smpl):
    # Explanation of representations is at the top.
    x, aria_traj_repre = sequence_to_repre_v4(aria_traj, smpl_params, kp3d, floor_height, smpl)
    # append beta to x
    betas = smpl_params["betas"]  # T x 10
    assert betas.shape[-1] == 10
    x = torch.cat([x, betas], dim=1)  # T x 234
    return x, aria_traj_repre


def repre_v4_beta_to_sequence(x, aria_traj_repre, smpl, betas, body_root_offset):
    # Explanation of representations is at the top.
    # Modify gt betas with the mean of the predicted betas
    x, x_betas = x[:, :-10], x[:, -10:]
    x_betas = x_betas.mean(0)[None]  # 1 x 10
    body_root_offset = None
    return repre_v4_to_sequence(x, aria_traj_repre, smpl, x_betas, body_root_offset)


# ------------------------------------------------------------------------------------------------
# V5: Same as v4 but pelvis centric instead of head centric.
# ------------------------------------------------------------------------------------------------
def sequence_to_repre_v5(aria_traj, smpl_params, kp3d, floor_height, smpl):
    return sequence_to_repre_v4(aria_traj, smpl_params, kp3d, floor_height, smpl, canon_root_idx=0)


def repre_v5_to_sequence(x, aria_traj_repre, smpl, betas, body_root_offset):
    return repre_v4_to_sequence(x, aria_traj_repre, smpl, betas, body_root_offset)


# ------------------------------------------------------------------------------------------------
# V5_beta: Same as v4_beta but pelvis centric instead of head centric.
# ------------------------------------------------------------------------------------------------
def sequence_to_repre_v5_beta(aria_traj, smpl_params, kp3d, floor_height, smpl):
    # Explanation of representations is at the top.
    x, aria_traj_repre = sequence_to_repre_v4(aria_traj, smpl_params, kp3d, floor_height, smpl, canon_root_idx=0)
    # append beta to x
    betas = smpl_params["betas"]  # T x 10
    assert betas.shape[-1] == 10
    x = torch.cat([x, betas], dim=1)  # T x 234
    return x, aria_traj_repre


def repre_v5_beta_to_sequence(x, aria_traj_repre, smpl, betas, body_root_offset):
    # Explanation of representations is at the top.
    # Modify gt betas with the mean of the predicted betas
    x, x_betas = x[:, :-10], x[:, -10:]
    x_betas = x_betas.mean(0)[None]  # 1 x 10
    body_root_offset = None
    return repre_v4_to_sequence(x, aria_traj_repre, smpl, x_betas, body_root_offset)


# ------------------------------------------------------------------------------------------------
# Interface functions
# ------------------------------------------------------------------------------------------------
def sequence_to_repre(repre_type, *args, **kwargs):
    # Explanation of representations is at the top.
    if repre_type == "v1":
        return sequence_to_repre_v1(*args, **kwargs)
    if repre_type == "v1f":
        return sequence_to_repre_v1f(*args, **kwargs)
    if repre_type == "v1_beta":
        return sequence_to_repre_v1_beta(*args, **kwargs)
    if repre_type == "v4":
        return sequence_to_repre_v4(*args, **kwargs)
    if repre_type == "v4_beta":
        return sequence_to_repre_v4_beta(*args, **kwargs)
    if repre_type == "v5":
        return sequence_to_repre_v5(*args, **kwargs)
    if repre_type == "v5_beta":
        return sequence_to_repre_v5_beta(*args, **kwargs)
    if repre_type == "v1_ait":
        return sequence_to_repre_v1_ait(*args, **kwargs)
    raise ValueError(f"Unknown repre_type: {repre_type}")


def repre_to_sequence(repre_type, *args, **kwargs):
    # Explanation of representations is at the top.
    if repre_type == "v1":
        return repre_v1_to_sequence(*args, **kwargs)
    if repre_type == "v1f":
        return repre_v1f_to_sequence(*args, **kwargs)
    if repre_type == "v1_beta":
        return repre_v1_beta_to_sequence(*args, **kwargs)
    if repre_type == "v4":
        return repre_v4_to_sequence(*args, **kwargs)
    if repre_type == "v4_beta":
        return repre_v4_beta_to_sequence(*args, **kwargs)
    if repre_type == "v5":
        return repre_v5_to_sequence(*args, **kwargs)
    if repre_type == "v5_beta":
        return repre_v5_beta_to_sequence(*args, **kwargs)
    if repre_type == "v1_ait":
        return repre_v1_ait_to_sequence(*args, **kwargs)
    raise ValueError(f"Unknown repre_type: {repre_type}")


if __name__ == "__main__":
    from loguru import logger

    from modiff.dataset.smpl_utils import get_smpl, evaluate_smpl
    from modiff.dataset.canonicalization import get_a_canonicalized_segment
    from modiff.utils.vis_utils import visualize_sequence, save_video
    from modiff.dataset.representation_utils import sequence_to_repre

    repre_type = "v1f"
    smpl = get_smpl("smplx_n")
    motion_data = torch.load(
        "/vision/u/chpatel/data/modiff/amass_smplx_n_zerobeta_from_smplx_ng_mM.pt", weights_only=False
    )

    # ------------------------------------------------------------------------------------------------
    # Get a segment
    # ------------------------------------------------------------------------------------------------
    seq_name = list(motion_data.keys())[11000]
    st = 0
    en = st + 100
    floor_height = motion_data[seq_name]["floor_height"]

    seg_data = get_a_canonicalized_segment(
        motion_data[seq_name]["smpl_params"],
        motion_data[seq_name]["aria_traj"],
        motion_data[seq_name]["kp3d"],
        smpl,
        st,
        en,
    )
    aria_traj_T, smpl_params = saved_sequence_to_full_sequence(
        seg_data["can_aria_traj"], seg_data["can_smpl_params"], smpl
    )
    kp3d, verts, full_pose = evaluate_smpl(smpl, smpl_params)
    if floor_height == 0:
        logger.warning(f"Floor height is not available: {floor_height}")
        floor_height = kp3d[:, [7, 8, 10, 11], 2].min().item() - 0.02
    T = aria_traj_T.shape[0]

    # fmt: off
    # ------------------------------------------------------------------------------------------------
    # Visualize original segment
    # ------------------------------------------------------------------------------------------------
    imgs = visualize_sequence(aria_traj=aria_traj_T, global_jpos=kp3d[:, :55], verts=verts, faces=smpl.faces, floor_height=floor_height)
    final_imgs = imgs  # T x 3 x H x W

    # ------------------------------------------------------------------------------------------------
    # Visualize representation
    # ------------------------------------------------------------------------------------------------
    x, aria_traj_repre = sequence_to_repre(repre_type, seg_data["can_aria_traj"], seg_data["can_smpl_params"], seg_data["can_kp3d"], floor_height, smpl)

    if repre_type == "v1" or repre_type == "v1f":
        global_rottrans = x[:, :198].view(T, 22, 9)  # T x 22 x 9
        global_T = rot_trans_to_matrix(rc.rotation_6d_to_matrix(global_rottrans[:, :, :6]), global_rottrans[:, :, 6:9])  # T x 22 x 4 x 4
        aria_traj_T = rot_trans_to_matrix(rc.rotation_6d_to_matrix(aria_traj_repre[:, :6]), aria_traj_repre[:, 6:9])
        imgs = visualize_sequence(aria_traj=aria_traj_T, global_jpos=global_T[:, :, :3, 3], global_rotmat=global_T[:, :, :3, :3])
        final_imgs = np.concatenate([final_imgs, imgs], axis=-2)
    elif repre_type == "v1_ait":
        global_rottrans = x[:, :198].view(T, 22, 9)  # T x 22 x 9
        global_T = rot_trans_to_matrix(rc.rotation_6d_to_matrix(global_rottrans[:, :, :6]), global_rottrans[:, :, 6:9])  # T x 22 x 4 x 4

        aria_traj_T_from_pose = rot_trans_to_matrix(rc.rotation_6d_to_matrix(x[:, 198:207][:, :6]), x[:, 198:207][:, 6:9])

        aria_traj_T_centered = rot_trans_to_matrix(rc.rotation_6d_to_matrix(aria_traj_repre[:, :6]), aria_traj_repre[:, 6:9])

        imgs = visualize_sequence(aria_traj=aria_traj_T_from_pose, pred_aria_traj=aria_traj_T_centered, global_jpos=global_T[:, :, :3, 3], global_rotmat=global_T[:, :, :3, :3], mv_params=dict(add_floor=False))
        final_imgs = np.concatenate([final_imgs, imgs], axis=-2)
    elif repre_type == "v4" or repre_type == "v5":
        local_rottrans = x[:, :198].view(T, 22, 9)  # T x 22 x 9
        local_T = rot_trans_to_matrix(rc.rotation_6d_to_matrix(local_rottrans[:, :, :6]), local_rottrans[:, :, 6:9])  # T x 22 x 4 x 4
        delta_invtsfm_rottrans = x[:, 198 : 198 + 9]  # T x 9
        delta_invtsfm_T = rot_trans_to_matrix(rc.rotation_6d_to_matrix(delta_invtsfm_rottrans[:, :6]), delta_invtsfm_rottrans[:, 6:9])  # T x 4 x 4
        invtsfm_T = [delta_invtsfm_T[0]]
        for i in range(1, T):
            invtsfm_T.append(invtsfm_T[-1] @ delta_invtsfm_T[i])
        invtsfm_T = torch.stack(invtsfm_T, dim=0)  # T x 4 x 4
        global_T = invtsfm_T[:, None] @ local_T  # T x 22 x 4 x 4

        imgs = visualize_sequence(global_rotmat=local_T[:, :, :3, :3], global_jpos=local_T[:, :, :3, 3], aria_traj=delta_invtsfm_T, pred_aria_traj=invtsfm_T)
        final_imgs = np.concatenate([final_imgs, imgs], axis=-2)

    else:
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------
    # Get sequence from representation and visualize
    # ------------------------------------------------------------------------------------------------
    aria_traj_T, smpl_params, kp3d = repre_to_sequence(repre_type, x, aria_traj_repre, smpl, None, None)
    kp3d2, verts, _ = evaluate_smpl(smpl, smpl_params)
    imgs = visualize_sequence(aria_traj=aria_traj_T, global_jpos=kp3d[:, :55], pred_global_jpos=kp3d2[:, :55], verts=verts, faces=smpl.faces)
    final_imgs = np.concatenate([final_imgs, imgs], axis=-2)

    # fmt: on

    save_video(final_imgs[..., ::-1], f"{repre_type}", "/vision/u/chpatel/test", fps=10)

    import IPython

    IPython.embed()
