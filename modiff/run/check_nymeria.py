import IPython
import numpy as np
import torch
from modiff.utils.pc_viewer_general import PointCloudViewer
from modiff.utils.vis_utils import visualize_sequence, save_video
from modiff.utils.torch_utils import to_tensor
from modiff.dataset.smpl_utils import get_smpl, evaluate_smpl
from modiff.dataset.egoego_utils import mat_ik_torch

if __name__ == "__main__":

    data_dir = "/vision/u/chpatel/test/nymeria/to_smpl/20230607_s0_james_johnson_act0_e72nhq/"
    smpl_data = np.load(f"{data_dir}/smpl_data.npz")

    global_rotmat = smpl_data["global_poses"][::100][:100]  # T x 24 x 3 x 3
    root_transl = smpl_data["root_positions"][::100][:100]  # T x 3

    global_rotmat = to_tensor(global_rotmat)
    root_transl = to_tensor(root_transl)
    smpl = get_smpl("smpl_n")
    T = global_rotmat.shape[0]

    local_rotmat = mat_ik_torch(global_rotmat, list(smpl.parents)[:24])  # T x 24 x 3 x 3
    smpl_params = {
        "global_orient": local_rotmat[:, 0],  # T x 3 x 3
        "body_pose": local_rotmat[:, 1:],  # T x 23 x 3 x 3
        "transl": root_transl,  # T x 3
        "betas": torch.zeros(T, 10),  # T x 10
    }
    kp3d, verts, full_pose = evaluate_smpl(smpl, smpl_params)

    mv_params = {"cam_up": [0.0, 1.0, 0.0], "cam_pos": [4, 4, 4], "cam_lookat": [0, 2, 0]}
    imgs = visualize_sequence(verts=verts, faces=smpl.faces, mv_params=mv_params)
    save_video(imgs[..., ::-1], "check", "/vision/u/chpatel/test/nymeria/to_smpl", fps=10)
