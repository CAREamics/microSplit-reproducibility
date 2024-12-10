from ._base import SplittingParameters

def get_musplit_parameters(depth3D=9) -> dict:
    return SplittingParameters(
        algorithm="musplit",
        img_size=(depth3D,64, 64),
        target_channels=3,
        multiscale_count=1,
        predict_logvar="pixelwise",
        loss_type="musplit",
        encoder_conv_strides=[1,2,2],
        decoder_conv_strides=[1,2,2],

    ).model_dump()


def get_microsplit_parameters(subdset_type='zebrafish', depth3D=9) -> dict:
    if subdset_type == 'liver':
        noise_model_ch1_fpath = '/home/ashesh.ashesh/training/noise_model/2411/9/GMMNoiseModel_n2v_inputs-channel_234_1_ch0__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz'
        noise_model_ch2_fpath = '/home/ashesh.ashesh/training/noise_model/2411/10/GMMNoiseModel_n2v_inputs-channel_234_1_ch1__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz'
        noise_model_ch3_fpath = '/home/ashesh.ashesh/training/noise_model/2411/11/GMMNoiseModel_n2v_inputs-channel_234_1_ch2__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz'
    elif subdset_type == 'zebrafish':
        noise_model_ch1_fpath = '/home/ashesh.ashesh/training/noise_model/2411/4/GMMNoiseModel_n2v_inputs-farred_RFP_GFP_2109172__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz'
        noise_model_ch2_fpath = '/home/ashesh.ashesh/training/noise_model/2411/6/GMMNoiseModel_n2v_inputs-farred_RFP_GFP_2109172__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz'
        noise_model_ch3_fpath = '/home/ashesh.ashesh/training/noise_model/2411/5/GMMNoiseModel_n2v_inputs-farred_RFP_GFP_2109172__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz'
    else:
        raise ValueError(f"subdset_type {subdset_type} not recognized. Please choose between 'zebrafish' and 'liver'.")
    
    nm_paths = [noise_model_ch1_fpath, noise_model_ch2_fpath, noise_model_ch3_fpath]

    return SplittingParameters(
        algorithm="denoisplit",
        img_size=(depth3D,64, 64),
        target_channels=3,
        multiscale_count=1,
        predict_logvar="pixelwise",
        loss_type="denoisplit_musplit",
        nm_paths=nm_paths,
        encoder_conv_strides=[1,2,2],
        decoder_conv_strides=[1,2,2],
    ).model_dump()
    
def get_eval_params() -> dict:
    return SplittingParameters().model_dump()
