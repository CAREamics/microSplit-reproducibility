import os

import numpy as np

from careamics.lvae_training.dataset import DataSplitType
from careamics.lvae_training.dataset.utils.data_utils import get_datasplit_tuples

# from .mrc_reader import get_mrc_data


def get_train_val_data(
    data_config,
    fpath,
    datasplit_type: DataSplitType,
    val_fraction=None,
    test_fraction=None,
    **kwargs,
):
    num_channels = data_config.num_channels
    fpaths = []
    data_list = []
    for i in range(num_channels):
        fpath1 = os.path.join(fpath, getattr(data_config, f"ch{i + 1}_fname"))
        fpaths.append(fpath1)
        data = get_mrc_data(fpath1)[..., None]
        data_list.append(data)

    dirname = os.path.dirname(os.path.dirname(fpaths[0])) + "/"

    msg = ",".join([x[len(dirname) :] for x in fpaths])
    print(f"Loaded from {dirname} Channels:{len(fpaths)} {msg} Mode:{datasplit_type}")
    N = data_list[0].shape[0]
    for data in data_list:
        N = min(N, data.shape[0])

    cropped_data = []
    for data in data_list:
        cropped_data.append(data[:N])

    data = np.concatenate(cropped_data, axis=3)

    if datasplit_type == DataSplitType.All:
        return data.astype(np.float32)

    train_idx, val_idx, test_idx = get_datasplit_tuples(
        val_fraction, test_fraction, len(data), starting_test=True
    )
    if datasplit_type == DataSplitType.Train:
        return data[train_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Val:
        return data[val_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Test:
        return data[test_idx].astype(np.float32)
