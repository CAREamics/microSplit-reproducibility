from functools import partial

from careamics.lvae_training.dataset.types import DataType, DataSplitType

from .multifile_raw_dataloader import (
    get_train_val_data as get_train_val_data_twochannels,
    SubDsetType,
)


def get_multi_channel_files():
    return ["SIM201-263_merged.tif", "SIM1-49_merged.tif"]


def get_multi_channel_files_with_input(noise_level):
    if noise_level == "low":
        return ["SIM_3color_1channel_group1.tif"]
    elif noise_level == "high":
        return ["SIM_3color_1channel_group2.tif"]  # This is a different noise level.


def get_train_val_data(
    data_config,
    datadir,
    datasplit_type: DataSplitType,
    val_fraction=None,
    test_fraction=None,
    **kwargs,
):
    assert data_config.subdset_type == SubDsetType.MultiChannel
    if data_config.data_type == DataType.Dao3Channel:
        return get_train_val_data_twochannels(
            datadir,
            data_config,
            datasplit_type,
            get_multi_channel_files,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
    elif data_config.data_type == DataType.Dao3ChannelWithInput:
        return get_train_val_data_twochannels(
            datadir,
            data_config,
            datasplit_type,
            partial(get_multi_channel_files_with_input, data_config.noise_level),
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
    else:
        raise NotImplementedError(f"Data type {data_config.data_type} not implemented.")
