from careamics.lvae_training.dataset import DataSplitType, DataType

from microsplit_reproducibility.datasets.multifile_raw_dataloader import SubDsetType
from .exp_microscopy_v1 import ExpMicroscopyConfig


def get_data_configs() -> tuple[ExpMicroscopyConfig, ExpMicroscopyConfig]:
    train_data_config = ExpMicroscopyConfig(
        datasplit_type=DataSplitType.Train,
        data_type=DataType.ExpMicroscopyV2,
        image_size=[64, 64],
        grid_size=32,
        subdset_type=SubDsetType.MultiChannel,
        depth3D=1,
        mode_3D=False,
        random_flip_z_3D=False,
        multiscale_lowres_count=1,
        poisson_noise_factor=-1,
        enable_gaussian_noise=False,
        synthetic_gaussian_scale=228,
        input_has_dependant_noise=True,
        use_one_mu_std=True,
        train_aug_rotate=True,
        target_separate_normalization=True,
        input_is_sum=False,
        padding_kwargs={"mode": "reflect"},
        overlapping_padding_kwargs={"mode": "reflect"},
    )

    val_data_config = train_data_config.model_copy(
        update=dict(
            datasplit_type=DataSplitType.Val,
            allow_generation=False,  # No generation during validation
            enable_random_cropping=False,  # No random cropping on validation.
        )
    )

    test_data_config = train_data_config.model_copy(
        update=dict(
            datasplit_type=DataSplitType.Val,
            allow_generation=False,  # No generation during validation
            enable_random_cropping=False,  # No random cropping on validation.
        )
    )
    return train_data_config, val_data_config, test_data_config
