from typing import Optional

from careamics.lvae_training.dataset import DatasetConfig, DataSplitType, DataType

from src.datasets.multifile_raw_dataloader import SubDsetType
from src.datasets.pavia3 import Pavia3SeqPowerLevel, Pavia3SeqAlpha


class PaviaDataConfig(DatasetConfig):
    power_level: Pavia3SeqPowerLevel
    alpha_level: Pavia3SeqAlpha
    val_idx: Optional[list[int]] = None
    test_idx: Optional[list[int]] = None
    subdset_type: SubDsetType = SubDsetType.MultiChannel


def get_data_configs() -> tuple[PaviaDataConfig, PaviaDataConfig]:
    train_data_config = PaviaDataConfig(
        datasplit_type=DataSplitType.Train,
        image_size=64,
        num_channels=2,
        power_level=Pavia3SeqPowerLevel.Low,
        alpha_level=Pavia3SeqAlpha.Balanced,
        val_idx=[2, 12],
        test_idx=[0, 10],
        subdset_type=SubDsetType.MultiChannel,
        multiscale_lowres_count=3,
        data_type=DataType.Pavia3SeqData,
        poisson_noise_factor=-1,
        enable_gaussian_noise=False,
        synthetic_gaussian_scale=None,
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

    return train_data_config, val_data_config
