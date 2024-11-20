import os
from enum import Enum
from pathlib import Path
from typing import Sequence, Union

import numpy as np
from numpy.typing import NDArray
from careamics.lvae_training.dataset import DataSplitType, DatasetConfig
from careamics.lvae_training.dataset.utils.data_utils import (
    get_datasplit_tuples,
    load_tiff,
)
from czifile import imread as imread_czi


class SubDsetType(Enum):
    OnlyIba1 = "Iba1"
    Iba1Ki67 = "Iba1_Ki67"
    OnlyIba1P30 = "Iba1NucPercent30"
    OnlyIba1P50 = "Iba1NucPercent50"
    OnlyIba1P70 = "Iba1NucPercent70"


def get_iba1_ki67_files() -> list[str]:
    return [f'{i}.czi' for i in range(1, 31)]

def get_iba1_only_files() -> list[str]:
    return [f'Iba1only_{i}.czi' for i in range(1, 16)]


def load_czi(fpaths: Sequence[Union[str, Path]]) -> NDArray:
    imgs = []
    for fpath in fpaths:
        img = imread_czi(fpath)
        assert img.shape[3] == 1
        img = np.swapaxes(img, 0, 3)
        # the first dimension of img stored in imgs will have dim of 1, where the contenation will happen
        imgs.append(img)
    return np.concatenate(imgs, axis=0)


def get_subdir(subdset_type: SubDsetType) -> str:
    if subdset_type in [
        SubDsetType.OnlyIba1P30,
        SubDsetType.OnlyIba1P50,
        SubDsetType.OnlyIba1P70,
    ]:
        subdset_type = SubDsetType.OnlyIba1
    return f"{subdset_type.value}"


def get_train_val_data(
    data_config: DatasetConfig,
    datadir: str,
    datasplit_type: DataSplitType,
    val_fraction: float = None,
    test_fraction: float = None,
    **kwargs,
):
    dset_subtype = data_config.subdset_type
    subdir = get_subdir(dset_subtype)

    if dset_subtype in [
        SubDsetType.OnlyIba1,
        SubDsetType.OnlyIba1P30,
        SubDsetType.OnlyIba1P50,
        SubDsetType.OnlyIba1P70,
    ]:
        fnames = get_iba1_only_files()
    elif dset_subtype == SubDsetType.Iba1Ki67:
        fnames = get_iba1_ki67_files()
    else:
        raise Exception(f"Invalid dset subtype: {dset_subtype}")

    train_idx, val_idx, test_idx = get_datasplit_tuples(
        val_fraction, test_fraction, len(fnames)
    )
    if datasplit_type == DataSplitType.All:
        fpaths = [os.path.join(datadir, subdir, x) for x in fnames]
        data = load_czi(fpaths)
    elif datasplit_type == DataSplitType.Train:
        print(train_idx)
        fnames = [fnames[i] for i in train_idx]
        fpaths = [os.path.join(datadir, subdir, x) for x in fnames]
        # print(f"Files dir: {os.path.join(datadir, subdir)}")
        # print(f"File names: {fnames}")
        data = load_czi(fpaths)
    elif datasplit_type == DataSplitType.Val:
        print(val_idx)
        fnames = [fnames[i] for i in val_idx]
        fpaths = [os.path.join(datadir, subdir, x) for x in fnames]
        # print(f"Files dir: {os.path.join(datadir, subdir)}")
        # print(f"File names: {fnames}")
        data = load_czi(fpaths)
    elif datasplit_type == DataSplitType.Test:
        print(test_idx)
        fnames_iba1 = [fnames[i] for i in test_idx]
        fpaths_iba1 = [os.path.join(datadir, subdir, x) for x in fnames_iba1]
        # print(f"Iba1 Files dir: {os.path.join(datadir, subdir)}")
        # print(f"Iba1 File names: {fnames_iba1}")
        # it contains iba1/iba1ki67 and DAPI.
        data = load_czi(fpaths_iba1)

        data_nuc = None
        if dset_subtype in [
            SubDsetType.OnlyIba1P30,
            SubDsetType.OnlyIba1P50,
            SubDsetType.OnlyIba1P70,
        ]:
            datadir_nuc = os.path.join(
                datadir, SubDsetType.OnlyIba1.value, f"synthetic_test/{dset_subtype.value}"
            )
            fnames_nuc = sorted(os.listdir(datadir_nuc))
            fpaths_nuc = [os.path.join(datadir_nuc, x) for x in fnames_nuc]
            fpaths_nuc = [fpath for fpath in fpaths_nuc if not os.path.isdir(fpath)]
            # print("Nuc File dir:", datadir_nuc)
            # print(f"Nuc File names: {[os.path.basename(x) for x in fpaths_nuc]}")
            data_nuc = np.concatenate(
                [load_tiff(fpath_)[None] for fpath_ in fpaths_nuc], axis=0
            )[..., None]
            # TODO: fix this
            data = np.tile(data[..., 1:], (len(data_nuc), 1, 1, 1))
            data = np.concatenate([data_nuc, data], axis=3)

    else:
        raise Exception("invalid datasplit")

    # fpaths = [os.path.join(datadir, dset_subtype, x) for x in fnames]
    # data = load_czi(fpaths)
    print("Loaded from", SubDsetType(dset_subtype), datadir, data.shape)
    if dset_subtype == SubDsetType.Iba1Ki67:
        # We just need the combined channel. we don't need the nuclear channel.
        # in order for the whole setup to work well, I'm just copying the channel twice.
        # when creating the input, the average of these channels will still be exactly this channel, which is what we want.
        # we want this channel as input to the network.
        # Note that mean and the stdev used to normalize this data will be different, but we can try to do that initially.
        data = data[..., 1:]
        data = np.tile(data, (1, 1, 1, 2))
    return data


# if __name__ == '__main__':
#     from ml_collections.config_dict import ConfigDict
#     data_config = ConfigDict()
#     data_config.subdset_type = SubDsetType.OnlyIba1P50
#     datadir = '/group/jug/ashesh/data/Stefania/20230327_Ki67_and_Iba1_trainingdata/'
#     data = get_train_val_data(
#         data_config=data_config, 
#         datadir=datadir,
#         datasplit_type=DataSplitType.Test, 
#         val_fraction=0.1, 
#         test_fraction=0.1
#     )
