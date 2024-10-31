from ._base import SplittingParameters


def get_musplit_parameters() -> dict:
    return SplittingParameters(
        algorithm="musplit",
        img_size=(64, 64),
        target_channels=2,
        multiscale_count=3,
        predict_logvar="pixelwise",
        loss_type="musplit",
    ).model_dump()

def get_denoisplit_parameters() -> dict:
    return SplittingParameters().model_dump()