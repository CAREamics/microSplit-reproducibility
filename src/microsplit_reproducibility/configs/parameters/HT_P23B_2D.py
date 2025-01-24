from ._base import SplittingParameters
from microsplit_reproducibility.utils.io import get_noise_models


def get_microsplit_parameters() -> dict:
    nm_paths=[
            "noise_models/nm_ht_p23b_2d_channel_1.npz",
            "noise_models/nm_ht_p23b_2d_channel_2.npz",
        ]
    get_noise_models(nm_paths)
    return SplittingParameters(
        algorithm="denoisplit",
        img_size=(64, 64),
        target_channels=2,
        multiscale_count=1,
        predict_logvar="pixelwise",
        loss_type="denoisplit_musplit",
        nm_paths=nm_paths,
        kl_type="kl_restricted",
    ).model_dump()


def get_eval_params() -> dict:
    return SplittingParameters().model_dump()
