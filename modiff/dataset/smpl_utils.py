from loguru import logger
import smplx
import torch
import copy
import modiff.utils.rotation_conversions as rc


def get_smpl(smpl_type="smplx_n"):

    smpl_path = "/vision/u/chpatel/smpl_data"  # TODO: change this to your own path
    logger.warning(f"Loading SMPL model: {smpl_type}")
    assert smpl_type in ["smplx_n"]
    smpl_type, gender = {
        "smplx_n": ("smplx", "NEUTRAL"),
    }[smpl_type]

    # Load SMPL
    # joint_mapper = JointMapper(smpl_to_openpose(smpl_type, use_hands=args.use_hands, use_face=False))
    # Don't use joint mapper because we need root joint to be at pelvis. We can sample the needed joints later.
    # This will give all 145 joints as shown here: https://github.com/vchoutas/smplx/blob/1265df7ba545e8b00f72e7c557c766e15c71632f/smplx/joint_names.py#L19
    # First 55 SMPLX kintree joints are 22 body + 3 face + 15 left hand + 15 right hand. 76 joints cover body and hands. Rest are faces.
    # Rest of the joints are other helper joints (e.g. for openpose) and face contours.
    smpl_layer_cls = {"smpl": smplx.SMPLLayer, "smplx": smplx.SMPLXLayer, "smplh": smplx.SMPLHLayer}[smpl_type]
    smpl = smpl_layer_cls(
        f"{smpl_path}/{smpl_type}",
        gender=gender,
        use_pca=True,
        # joint_mapper=joint_mapper,
        num_pca_comps=12,
    )
    return smpl


def evaluate_smpl(smpl, smpl_params, max_parallel_smpl_evals=1000):
    kp3d = []
    verts = []
    full_pose = []
    bs = max_parallel_smpl_evals
    nf = len(smpl_params["body_pose"])
    for i in range(0, nf, bs):
        smpl_params_batch = {
            "body_pose": smpl_params["body_pose"][i : i + bs],
            "global_orient": smpl_params["global_orient"][i : i + bs],
            "betas": smpl_params["betas"][i : i + bs],
            "transl": smpl_params["transl"][i : i + bs],
        }
        if "left_hand_pose" in smpl_params:
            smpl_params_batch["left_hand_pose"] = smpl_params["left_hand_pose"][i : i + bs]
            smpl_params_batch["right_hand_pose"] = smpl_params["right_hand_pose"][i : i + bs]

        smpl_output = smpl(**smpl_params_batch, return_full_pose=True)

        kp3d.append(smpl_output.joints)
        verts.append(smpl_output.vertices)
        full_pose.append(smpl_output.full_pose)

    kp3d = torch.cat(kp3d, dim=0)
    verts = torch.cat(verts, dim=0)
    full_pose = torch.cat(full_pose, dim=0)

    return kp3d, verts, full_pose


def matrix_to_pca(hand_pose, hand_components):
    # hand_pose: (B, 15, 3, 3), hand_components: (C, 45)
    hand_pose = rc.matrix_to_axis_angle(hand_pose)  # (B, 15, 3)
    hand_pose = hand_pose.flatten(-2, -1)  # (B, 45)
    hand_pose = torch.einsum("...i,ij->...j", [hand_pose, hand_components.T])
    diags = hand_components.square().sum(-1)  # divide by eigenvalues
    hand_pose = hand_pose / diags
    return hand_pose


def pca_to_matrix(hand_pose, hand_components):
    # hand_pose: (B, C), hand_components: (C, 45)
    hand_pose = torch.einsum("...i,ij->...j", [hand_pose, hand_components])  # (B, 45)
    hand_pose = hand_pose.unflatten(-1, (15, 3))  # (B, 15, 3)
    hand_pose = rc.axis_angle_to_matrix(hand_pose)  # (B, 15, 3, 3)
    return hand_pose


def full_sequence_to_saved_sequence(smpl_params, aria_traj, smpl):
    # Make Things Small
    smpl_params["global_orient"] = rc.matrix_to_rotation_6d(smpl_params["global_orient"])  # T x 6
    smpl_params["body_pose"] = rc.matrix_to_rotation_6d(smpl_params["body_pose"])  # T x (21*6)
    assert torch.allclose(smpl_params["betas"], smpl_params["betas"][0:1])
    smpl_params["betas"] = smpl_params["betas"][:1]  # 1 x 10 because betas are same
    if "left_hand_pose" in smpl_params:
        smpl_params["left_hand_pose"] = matrix_to_pca(smpl_params["left_hand_pose"], smpl.left_hand_components)
        smpl_params["right_hand_pose"] = matrix_to_pca(smpl_params["right_hand_pose"], smpl.right_hand_components)

    if aria_traj is not None:
        aria_traj_rot6d = rc.matrix_to_rotation_6d(aria_traj[:, :3, :3])  # T x 6
        aria_traj = torch.cat([aria_traj_rot6d, aria_traj[:, :3, 3]], dim=1)  # T x (6 + 3)
    return smpl_params, aria_traj


def saved_sequence_to_full_sequence(aria_traj, smpl_params, smpl):
    # Saved smpl params are in in 6D rotation format and body shape is not repeated for all frames.
    # Convert them to full sequence format for easier use.

    if smpl_params["global_orient"].ndim == 3:  # maybe already full sequence. check it.
        T = smpl_params["global_orient"].shape[0]
        if aria_traj is not None:
            assert aria_traj.shape == (T, 4, 4)
            aria_traj = aria_traj.clone()
        assert smpl_params["global_orient"].shape == (T, 3, 3)
        assert smpl_params["body_pose"].shape == (T, 21, 3, 3)
        assert smpl_params["transl"].shape == (T, 3)
        assert smpl_params["betas"].shape == (T, 10)
        if "left_hand_pose" in smpl_params:
            assert smpl_params["left_hand_pose"].shape == (T, 15, 3, 3)
            assert smpl_params["right_hand_pose"].shape == (T, 15, 3, 3)
        copy.deepcopy(smpl_params)
        return aria_traj, smpl_params

    aria_traj_T = None
    if aria_traj is not None:
        aria_traj_T = torch.eye(4)[None].repeat(aria_traj.shape[0], 1, 1).to(aria_traj.device)
        aria_traj_T[:, :3, :3] = rc.rotation_6d_to_matrix(aria_traj[:, :6])
        aria_traj_T[:, :3, 3] = aria_traj[:, 6:9]

    smpl_params = copy.deepcopy(smpl_params)
    smpl_params["global_orient"] = rc.rotation_6d_to_matrix(smpl_params["global_orient"])
    smpl_params["body_pose"] = rc.rotation_6d_to_matrix(smpl_params["body_pose"])
    smpl_params["betas"] = smpl_params["betas"].repeat(aria_traj.shape[0], 1)
    if "left_hand_pose" in smpl_params:
        smpl_params["left_hand_pose"] = pca_to_matrix(smpl_params["left_hand_pose"], smpl.left_hand_components)
        smpl_params["right_hand_pose"] = pca_to_matrix(smpl_params["right_hand_pose"], smpl.right_hand_components)

    return aria_traj_T, smpl_params


def slerp_quaternions(
    q1: torch.Tensor, q2: torch.Tensor, t: float or torch.Tensor, dot_threshold: float = 0.9995
) -> torch.Tensor:
    """
    Performs Spherical Linear Interpolation (SLERP) between two quaternions (or batches of quaternions).

    Args:
        q1: Starting quaternion(s) tensor of shape (..., 4). Assumed to be normalized.
        q2: Ending quaternion(s) tensor of shape (..., 4). Assumed to be normalized.
        t: Interpolation factor(s) in [0, 1]. Can be a scalar or a tensor that broadcasts
           with the batch dimensions of q1/q2.
        dot_threshold: Threshold for the absolute dot product to fall back to Linear Interpolation (LERP).
                       This prevents division by zero/near-zero when q1 and q2 are nearly colinear.

    Returns:
        The interpolated quaternion(s) tensor of shape (..., 4).
    """

    # 1. Compute the dot product (cosine of the angle)
    # The sum is over the last dimension (the 4 quaternion components)
    dot = (q1 * q2).sum(dim=-1, keepdim=True)

    # 2. Ensure we take the shortest path by flipping the sign if the dot product is negative
    # This addresses the double-cover property of quaternions (q and -q represent the same rotation)
    q2_prime = q2.clone()
    q2_prime[dot < 0] = -q2_prime[dot < 0]
    dot_abs = torch.abs(dot)  # Re-calculate absolute dot product after potential flip

    # 3. Handle the near-collinear case (dot_abs approx 1) with LERP
    # This prevents numerical instability from division by sin(omega) approx 0

    # Create mask for LERP fallback
    should_lerp = dot_abs > dot_threshold

    # 4. SLERP calculation (for cases where they are not too close)

    # Clip dot product to avoid NaN from acos for values slightly > 1 due to floating point error
    dot_clipped = torch.clamp(dot_abs, -1.0, 1.0)

    omega = torch.acos(dot_clipped)
    sin_omega = torch.sin(omega)

    # Calculate the scale factors
    # The '1.0' in the denominator handles the near-collinear case gracefully due to the LERP fallback,
    # but strictly speaking, it's just `sin_omega`
    s0 = torch.sin((1.0 - t) * omega) / (sin_omega + 1e-10)
    s1 = torch.sin(t * omega) / (sin_omega + 1e-10)

    # Interpolated result
    slerped = s0 * q1 + s1 * q2_prime

    # 5. LERP calculation (for the near-collinear case)
    lerped = (1.0 - t) * q1 + t * q2_prime

    # 6. Combine LERP and SLERP results
    # Use torch.where to select between LERP and SLERP based on the mask
    result = torch.where(should_lerp, lerped, slerped)

    # 7. Final normalization is good practice, especially after LERP fallback
    # The SLERP result should technically be normalized already, but this ensures correctness.
    result = torch.nn.functional.normalize(result, p=2, dim=-1)

    return result
