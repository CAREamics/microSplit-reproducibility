from ._base import SplittingParameters

def get_musplit_parameters() -> dict:
    return SplittingParameters(
        algorithm="musplit",
        img_size=(64, 64),
        target_channels=2,
        multiscale_count=1,
        predict_logvar="pixelwise",
        loss_type="musplit",
    ).model_dump()


def get_microsplit_parameters() -> dict:
    return SplittingParameters(
        algorithm="denoisplit",
        img_size=(64, 64),
        target_channels=2,
        multiscale_count=1,
        predict_logvar="pixelwise",
        loss_type="denoisplit_musplit",
        nm_paths=[
            "/home/ashesh.ashesh/training/noise_model/2407/3/GMMNoiseModel_N2V_Elisa-n2v_input__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz",
            "/home/ashesh.ashesh/training/noise_model/2407/3/GMMNoiseModel_N2V_Elisa-n2v_input__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz"
        ]
    ).model_dump()
    
def get_eval_params() -> dict:
    return SplittingParameters().model_dump()
