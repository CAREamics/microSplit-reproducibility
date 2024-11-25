import os

import numpy as np
from careamics.lvae_training.dataset.multifile_dataset import MultiChannelData
# from czifile import imread as imread_czi

from careamics.lvae_training.dataset import DataType, DataSplitType


def get_multi_channel_files_v1():
    return {
        DataSplitType.Train: [
            "Experiment-155.czi",  # 722,18 => very good.
            "Experiment-156.czi",  # 722,18 => good.
            "Experiment-164.czi",  # 521, => good.
            "Experiment-162.czi",  # 362,9 => good
        ],
        DataSplitType.Val: [
            "Experiment-163.czi",
            # 362,9 => good. it just has 3 bright spots which should be removed.
        ],
        DataSplitType.Test: [
            "Experiment-165.czi",  # 361,9 => good.
        ],
        # 'Experiment-140.czi', #400 => issue.
        # 'Experiment-160.czi', #521 => shift in between.
        # 'Experiment-157.czi', #561 => okay. the problem is a shift in between. This could be a good candidate for test val split.
        # 'Experiment-159.czi', #161,4 => does not make sense to use this.
        # 'Experiment-166.czi', #201
    }


def get_multi_channel_files_v2():
    return [
        "Experiment-447.czi",
        "Experiment-449.czi",
        "Experiment-448.czi",
        # 'Experiment-452.czi'
    ]


def get_multi_channel_files_v3():
    return [
        "Experiment-493.czi",
        "Experiment-494.czi",
        "Experiment-495.czi",
        "Experiment-496.czi",
        "Experiment-497.czi",
    ]


def load_data(fpath):
    # (4, 1, 4, 22, 512, 512, 1)
    data = imread_czi(fpath)
    clean_data = data[3, :, [0, 2], ..., 0]
    clean_data = np.swapaxes(clean_data[..., None], 0, 5)[0]
    return clean_data


def get_train_val_data(
    data_config,
    datadir,
    datasplit_type: DataSplitType,
    val_fraction=None,
    test_fraction=None,
    **kwargs
):
    if data_config.data_type == DataType.ExpMicroscopyV1:
        fnames = get_multi_channel_files_v1()[datasplit_type]
    elif data_config.data_type == DataType.ExpMicroscopyV2:
        fnames = get_multi_channel_files_v2()
        assert len(fnames) == 3
        if datasplit_type == DataSplitType.Train:
            fnames = fnames[:-1]
        elif datasplit_type in [DataSplitType.Val, DataSplitType.Test]:
            fnames = fnames[-1:]

    fpaths = [os.path.join(datadir, fname) for fname in fnames]
    data = [load_data(fpath) for fpath in fpaths]
    if (
        datasplit_type in [DataSplitType.Val, DataSplitType.Test]
        and data_config.data_type == DataType.ExpMicroscopyV2
    ):
        assert len(data) == 1
        zN = data[0].shape[1]
        if datasplit_type == DataSplitType.Val:
            data[0] = data[0][:, : zN // 2]
        elif datasplit_type == DataSplitType.Test:
            data[0] = data[0][:, zN // 2 :]

    data = MultiChannelData(data, paths=fpaths)
    return data
