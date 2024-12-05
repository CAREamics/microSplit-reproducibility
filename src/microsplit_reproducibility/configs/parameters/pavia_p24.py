from ._base import SplittingParameters
from microsplit_reproducibility.utils.io import get_noise_models


def get_denoisplit_parameters() -> dict:
    nm_paths=[
            "noise_models/noise_model_pavia_p24_channel_0.npz",
            "noise_models/noise_model_pavia_p24_channel_1.npz",
        ]
    get_noise_models(nm_paths)
    return SplittingParameters(
        algorithm="denoisplit",
        img_size=(64, 64),
        target_channels=2,
        multiscale_count=3,
        predict_logvar="pixelwise",
        loss_type="denoisplit_musplit",
        nm_paths=nm_paths,
        kl_type="kl_restricted",
    ).model_dump()


def get_eval_params() -> dict:
    return SplittingParameters().model_dump()
