from ._base import SplittingParameters


def get_musplit_parameters() -> dict:
    return SplittingParameters(
        algorithm="musplit",
        img_size=(64, 64),
        target_channels=2,
        multiscale_count=3,
        predict_logvar="pixelwise",
        loss_type="musplit",
        kl_type="kl_restricted",
    ).model_dump()

def get_denoisplit_parameters() -> dict:
    return SplittingParameters(
        algorithm="denoisplit",
        img_size=(64, 64),
        target_channels=2,
        multiscale_count=3,
        predict_logvar="pixelwise",
        loss_type="denoisplit_musplit",
        nm_paths=[
            "/group/jug/ashesh/training/noise_model/2405/37/GMMNoiseModel_20230327_Ki67_and_Iba1_trainingdata-Iba1__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz",
            "/group/jug/ashesh/training/noise_model/2405/38/GMMNoiseModel_20230327_Ki67_and_Iba1_trainingdata-Iba1__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz"
        ],
        kl_type="kl_restricted",
    ).model_dump()
    
def get_eval_params() -> dict:
    return SplittingParameters().model_dump()