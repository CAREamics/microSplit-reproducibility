import os
from typing import Literal

from ._base import SplittingParameters

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
    nm_path: str, 
) -> list[str]:
    nm_paths = []
    if dset_type == "20ms":
        channel_idx_list = [1, 2, 3] # TODO this is beyond any logic and absolutely disgusting !!!!!
    for channel_idx in channel_idx_list:
        if channel_idx == 17:
            channel_idx = 0
        fname = f"nm_ht_lif24_ch{channel_idx}_{dset_type}.npz"
        nm_paths.append(os.path.join(nm_path, fname))
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
    nm_path: str,
    channel_idx_list: list[Literal[1, 2, 3, 17]] = [1, 2, 3, 17],
) -> dict:
    nm_paths = _get_nm_paths(dset_type, nm_path=nm_path)
    return SplittingParameters(
        algorithm="denoisplit",
        img_size=(64, 64),
        target_channels=len(channel_idx_list) - 1,
        multiscale_count=3,
        predict_logvar="pixelwise",
        loss_type="denoisplit_musplit",
        nm_paths=nm_paths,
        kl_type="kl_restricted",
    ).model_dump()


def get_eval_params() -> dict:
    raise NotImplementedError("Evaluation parameters not implemented for HT_LIF24.")
    return SplittingParameters().model_dump()