import os
from typing import Literal

from ._base import SplittingParameters

NOISE_MODEL_ROOT_PATH = "/group/jug/ashesh/training/noise_model/2406/"
NM_2_DIR_NUM = {
    "2ms": {
        1: "10",
        2: "11",
        3: "1",
    },
    "3ms": {
        1: "3",
        2: "4",
        3: "5",
    },
    "5ms": {
        0: "6",
        1: "7",
        1: "8",
        3: "9",
    },
    "20ms": {
        1: "13",
        2: "14",
        3: "15",
    },
    "500ms": {
        1: "17",
        2: "18",
        3: "19",
    },
}

def _get_nm_paths(
    dset_type: Literal["2ms", "3ms", "5ms", "20ms", "500ms"],
    channel_idx_list: list[Literal[1, 2, 3, 17]],    
) -> list[str]:
    nm_paths = []
    for channel_idx in channel_idx_list:
        if channel_idx == 17:
            channel_idx = 0
        fname = f"GMMNoiseModel_nikola_denoising_input-uSplit_20240531_{dset_type}SNR_channel{channel_idx}__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz"
        nm_paths.append(os.path.join(NOISE_MODEL_ROOT_PATH, NM_2_DIR_NUM[dset_type][channel_idx], fname))
    return nm_paths


def get_musplit_parameters(
    channel_idx_list: list[int],
) -> dict:
    return SplittingParameters(
        algorithm="musplit",
        img_size=(64, 64),
        target_channels=len(channel_idx_list),
        multiscale_count=3,
        predict_logvar="pixelwise",
        loss_type="musplit",
        kl_type="kl_restricted",
    ).model_dump()


def get_microsplit_parameters(
    dset_type: Literal["2ms", "3ms", "5ms", "20ms", "500ms"],
    channel_idx_list: list = [1, 2, 3, 17] ,
) -> dict:
    nm_paths = _get_nm_paths(dset_type, channel_idx_list)
    return SplittingParameters(
        algorithm="denoisplit",
        img_size=(64, 64),
        target_channels=len(channel_idx_list),
        multiscale_count=3,
        predict_logvar="pixelwise",
        loss_type="denoisplit_musplit",
        nm_paths=nm_paths,
        kl_type="kl_restricted",
    ).model_dump()


def get_eval_params() -> dict:
    raise NotImplementedError("Evaluation parameters not implemented for HT_LIF24.")
    return SplittingParameters().model_dump()