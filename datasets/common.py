from typing import Callable

import torch
from torch.utils.data import Dataset
from numpy.typing import NDArray

from careamics.lvae_training.dataset import DatasetConfig, DataType
from careamics.lvae_training.dataset import (
    LCMultiChDloader,
    MultiChDloader,
    MultiFileDset,
)


def create_train_val_datasets(
    datapath: str,
    train_config: DatasetConfig,
    val_config: DatasetConfig,
    load_data_func: Callable[..., NDArray],
) -> tuple[Dataset, Dataset, tuple[float, float]]:
    if train_config.data_type in [
        DataType.TavernaSox2Golgi,
        DataType.Dao3Channel,
        DataType.Dao3ChannelWithInput,
        DataType.ExpMicroscopyV1,
        DataType.ExpMicroscopyV2,
        DataType.TavernaSox2GolgiV2,
        DataType.Pavia3SeqData,
    ]:
        dataset_class = MultiFileDset
    elif train_config.multiscale_lowres_count > 1:
        dataset_class = LCMultiChDloader
    else:
        dataset_class = MultiChDloader

    train_data = dataset_class(
        train_config,
        datapath,
        load_data_fn=load_data_func,
        val_fraction=0.1,
        test_fraction=0.1,
    )
    max_val = train_data.get_max_val()
    val_config.max_val = max_val
    val_data = dataset_class(
        val_config,
        datapath,
        load_data_fn=load_data_func,
        val_fraction=0.1,
        test_fraction=0.1,
    )

    mean_val, std_val = train_data.compute_mean_std()
    train_data.set_mean_std(mean_val, std_val)
    val_data.set_mean_std(mean_val, std_val)
    data_stats = train_data.get_mean_std()

    # NOTE: "input" mean & std are computed over the entire dataset and repeated for each channel.
    # On the contrary, "target" mean & std are computed separately for each channel.
    # manipulate data stats to only have one mean and std for the target
    assert isinstance(data_stats, tuple)
    assert isinstance(data_stats[0], dict)

    data_stats = (
        torch.tensor(data_stats[0]["target"]),
        torch.tensor(data_stats[1]["target"]),
    )

    return train_data, val_data, data_stats
