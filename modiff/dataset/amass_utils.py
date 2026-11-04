import os
import sys
import numpy as np
from loguru import logger
import glob
import json
import copy


"""
Some info about SMPLH dataset:
SMPLH has BMLhandball (649 seqs). But SMPLX doesn't (in both G and N).
SMPLH has 2 are from DanceDB and 99 CMU rory-justin couple sequences. But SMPLX doesn't.
SMPLH DFaust says 60fps but SMPLX DFaust says 120fps for all sequences.

Some info about SMPLX dataset (G and N):
SMPLX has CNRS (79 seqs) and WEIZMANN (2222 seqs) dataset. But SMPLH doesn't.
SMPLX has smplx_n/BMLmovi/Subject_[7/26/49]_F_MoSh subjects (62 seqs). But SMPLH doesn't.
SMPLX DFaust says 120fps but SMPLH DFaust says 60fps for all sequences.
"""


# TODO: path on our cluster. If you want to use this, set it to your own path.
amass_root = "/vision/u/chpatel/data/amass/"

smplh_g_dataset_names = [
    "ACCAD",
    "BMLmovi",
    "BioMotionLab_NTroje",
    "BMLhandball",
    "CMU",
    "DanceDB",
    "DFaust_67",
    "EKUT",
    "Eyes_Japan_Dataset",
    "HumanEva",
    "KIT",
    "MPI_HDM05",
    "MPI_Limits",
    "MPI_mosh",
    "SFU",
    "SSM_synced",
    "TCD_handMocap",
    "TotalCapture",
    "Transitions_mocap",
    # Added later
    "HUMAN4D",
    "SOMA",
    "MOYO_smplh_gendered",
]
smplx_n_dataset_names = [
    "ACCAD",
    "BMLmovi",
    "BMLrub",
    "CMU",
    "CNRS",
    "DFaust",
    "EKUT",
    "Eyes_Japan_Dataset",
    "HDM05",
    "HumanEva",
    "KIT",
    "MOYO_smplx_neutral",
    "MoSh",
    "PosePrior",
    "SFU",
    "SOMA",
    "SSM",
    "TCDHands",
    "TotalCapture",
    "Transitions",
    "WEIZMANN",
]
smplx_g_dataset_names = [
    "BMLhandball",
    "DanceDB",
    "HUMAN4D",
]

same_dataset_name_mapping = {
    "BioMotionLab_NTroje": "BMLrub",
    "BMLrub": "BioMotionLab_NTroje",
    "MOYO_smplh_gendered": "MOYO_smplx_gendered",
    "MOYO_smplx_gendered": "MOYO_smplh_gendered",
    "DFaust": "DFaust_67",
    "DFaust_67": "DFaust",
    "HDM05": "MPI_HDM05",
    "MPI_HDM05": "HDM05",
    "PosePrior": "MPI_Limits",
    "MPI_Limits": "PosePrior",
    "MPI_mosh": "MoSh",
    "MoSh": "MPI_mosh",
    "SSM": "SSM_synced",
    "SSM_synced": "SSM",
    "TCDHands": "TCD_handMocap",
    "TCD_handMocap": "TCDHands",
    "Transitions_mocap": "Transitions",
    "Transitions": "Transitions_mocap",
}


def amass_loop(ds_types, return_what, fix_fps, ignore_non_floor_seqs, mirrored=False, verbose=True):
    assert return_what in ["seq_path", "bdata"]
    mirror_prefix = "M_" if mirrored else ""
    for ds_type in ds_types:
        for ds_path in glob.glob(f"{amass_root}/{mirror_prefix}{ds_type}/*"):
            for sub_path in glob.glob(f"{ds_path}/*"):

                # ignore non-directories like .tar.gz2 or .zip or .txt files
                if not os.path.isdir(sub_path):
                    if verbose:
                        logger.warning(f"Skipping {sub_path}")
                    continue

                for seq_path in glob.glob(f"{sub_path}/*.npz"):

                    # ignore non-npz files like .txt
                    if not seq_path.endswith(".npz"):
                        if verbose:
                            logger.warning(f"Skipping {seq_path}")
                        continue

                    # some subjects have intermediate stage 1 results.
                    # if seq_path.endswith("neutral_stagei.npz") or seq_path.endswith("/shape.npz"):
                    if any(
                        seq_path.endswith(x)
                        for x in ["neutral_stagei.npz", "female_stagei.npz", "male_stagei.npz", "/shape.npz"]
                    ):
                        if verbose:
                            logger.warning(f"Skipping {seq_path}")
                        continue

                    # some bad zip files
                    if "smplx_n/BMLmovi/Subject_49_F_MoSh/Subject_49_F_19_stageii" in seq_path:
                        if verbose:
                            logger.warning(f"Skipping {seq_path}")
                        continue

                    # ignore non-floor sequences
                    if ignore_non_floor_seqs:
                        if "BMLrub" in ds_path or "BioMotionLab_NTroje" in ds_path:
                            motion_type = seq_path.split("/")[-1].split("_")[1]
                            if motion_type == "treadmill" or motion_type == "normal":
                                if verbose:
                                    logger.warning(f"Skipping treadmill or normal sequence {seq_path}")
                                continue

                        if "HDM05" in ds_path or "MPI_HDM05" in ds_path:
                            if "dg/HDM_dg_07-01" in seq_path:
                                if verbose:
                                    logger.warning(f"Skipping skating sequence {seq_path}")
                                continue

                    # return only sequence path if requested
                    if return_what == "seq_path":
                        yield seq_path
                        continue

                    # load npz file
                    try:
                        bdata = np.load(seq_path, allow_pickle=True)
                    except:
                        logger.error(f"Error loading {seq_path}")
                        raise

                    # verify gender
                    try:
                        gender = bdata["gender"].item()
                        if isinstance(gender, bytes):
                            gender = gender.decode()
                        if ds_type == "smplx_n":
                            assert gender == "neutral"
                        elif ds_type == "smplx_g" or ds_type == "smplh_g":
                            assert gender in ["male", "female"]
                        else:
                            raise ValueError(f"Invalid ds_type: {ds_type}")
                    except Exception as e:
                        logger.error(f"Error verifying gender for {seq_path}")
                        raise

                    # get fps
                    try:
                        # Almost all smplh data has mocap_frame_rate but SOMA has mocap_framerate
                        # All smplx data has mocap_framerate.
                        if "mocap_frame_rate" in bdata:
                            fps_key = "mocap_frame_rate"
                            assert "mocap_framerate" not in bdata
                        else:
                            fps_key = "mocap_framerate"
                            assert "mocap_frame_rate" not in bdata
                        fps = bdata[fps_key].item()
                    except:
                        logger.error(f"Error getting fps for {seq_path}")
                        raise

                    # fix fps for some sequences
                    if fix_fps:
                        if "BMLhandball" in ds_path:
                            if verbose:
                                logger.warning(f"Fixing fps for BMLhandball sequence {seq_path}")
                            fps = 240

                        if "DFaust" in ds_path:
                            if verbose:
                                logger.warning(f"Fixing fps for DFaust sequence {seq_path}")
                            fps = 60

                        if "20160930_50032" in seq_path or "20161014_50033" in seq_path:
                            if verbose:
                                logger.warning(f"Fixing fps for sequence {seq_path}")
                            fps = 59

                        # npz file doesn't support inplace modification, return the fps instead
                        # bdata[fps_key] = fps

                    # return bdata if requested
                    if return_what == "bdata":
                        yield seq_path, fps, bdata
                    else:
                        raise ValueError(f"Invalid return_what: {return_what}")


def save_nframes_fps():
    out_dict = {}
    out_file = f"{amass_root}/numframes_fps.json"
    for seq_path, fps, bdata in amass_loop(
        ["smplx_n", "smplx_g", "smplh_g"],
        return_what="bdata",
        fix_fps=False,
        ignore_non_floor_seqs=False,
    ):
        # get nframes
        nframes = bdata["poses"].shape[0]

        # key to save
        key = seq_path.replace(amass_root, "").replace(".npz", "")
        if key.startswith("/"):
            key = key[1:]
        out_dict[key] = {
            "nframes": nframes,
            "fps": fps,
        }
        logger.info(f"Saving {key}")

    with open(out_file, "w") as f:
        json.dump(out_dict, f)


"""
SMPLH and SMPLX have the same number of frames. FPS needs to be corrected for DFaust.
Otherwise both are same.

BMLhandball sequences in SMPLH are missing in SMPLX.
Humanact12 dataset is not available in any smpl amass.
Some CMU two actor sequences are missing in SMPLX.
"""


def replace_last_n(text, target, replacement, n):
    parts = text.rsplit(target, n)
    return replacement.join(parts)


def smplh_key_to_smplx_key(smplh_key):
    not_available_in_smplx = [
        "humanact12/",  # humanact dataset is not available in any smpl amass
        "BMLhandball/",
        "CMU/18_19_Justin/",
        "CMU/18_19_rory/",
        "CMU/20_21_Justin1/",
        "CMU/20_21_rory1/",
        "CMU/22_23_Rory/",
        "CMU/22_23_justin/",
    ]
    for k in not_available_in_smplx:
        if k in smplh_key:
            return None, None
    smplx_gender = "n"
    smplx_key = copy.deepcopy(smplh_key)
    smplx_key = smplx_key.replace("smplh_g", "smplx_n")
    smplx_key = smplx_key.replace(" ", "_")
    # smplx_key = smplx_key.replace("_poses", "_stageii")
    smplx_key = replace_last_n(smplx_key, "_poses", "_stageii", 1)
    smplx_key = smplx_key.replace("BioMotionLab_NTroje", "BMLrub")
    smplx_key = smplx_key.replace("Transitions_mocap/", "Transitions/")
    smplx_key = smplx_key.replace("MPI_mosh/", "MoSh/")
    smplx_key = smplx_key.replace("SSM_synced/", "SSM/")
    smplx_key = smplx_key.replace("TCD_handMocap/", "TCDHands/")
    smplx_key = smplx_key.replace("MPI_Limits/", "PosePrior/")
    smplx_key = smplx_key.replace("MPI_HDM05/", "HDM05/")
    smplx_key = smplx_key.replace("DFaust_67/", "DFaust/")
    if "HUMAN4D/" in smplx_key or "DanceDB/" in smplx_key:
        smplx_key = smplx_key.replace("smplx_n", "smplx_g")
        smplx_gender = "g"
    return smplx_key, smplx_gender


def process_text_anns(text_anns):
    # Process keys to match the format of the out_dict

    def modify_key(key):
        ds_name, sub_name, seq_name = key.split("/")[-3:]
        if key.startswith("M/"):
            seq_name = f"M_{seq_name}"
        if seq_name.endswith("_poses"):
            seq_name = seq_name[:-6]
        if seq_name.endswith("_stageii"):
            seq_name = seq_name[:-8]
        return f"{ds_name}___{sub_name}___{seq_name}"

    new_text_anns = {}
    for key, ann in text_anns.items():
        if "humanact12" in key:
            continue
        new_key = modify_key(key)
        new_text_anns[new_key] = ann
    return new_text_anns


if __name__ == "__main__":
    save_nframes_fps()
    pass
