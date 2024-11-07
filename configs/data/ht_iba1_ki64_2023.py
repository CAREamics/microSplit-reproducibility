from careamics.lvae_training.dataset import DatasetConfig, DataSplitType, DataType

from datasets.ht_iba1_ki64_2023 import SubDsetType


class HTIBA1Ki64Config(DatasetConfig):
    subdset_type: SubDsetType


def get_data_configs() -> tuple[HTIBA1Ki64Config, HTIBA1Ki64Config]:
    train_data_config = HTIBA1Ki64Config(
        datasplit_type=DataSplitType.Train,
        image_size=64,
        data_type=DataType.HTIba1Ki67,
        subdset_type=SubDsetType.OnlyIba1,
        poisson_noise_factor=-1,
        enable_gaussian_noise=False,
        synthetic_gaussian_scale=6675,
        input_has_dependant_noise=True,
        multiscale_lowres_count=3,
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
    
    # TODO: add eval config using Percent 30, 50, 70 datasets

    return train_data_config, val_data_config
