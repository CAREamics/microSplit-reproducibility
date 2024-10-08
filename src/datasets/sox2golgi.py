from enum import Enum

from careamics.lvae_training.dataset import DataSplitType

from .multifile_raw_dataloader import (
    get_train_val_data as get_train_val_data_twochannels,
    SubDsetType,
)


class Sox2GolgiChannelList(Enum):
    Sox2 = "SOX2"
    Golgi = "GOLGI"


def get_two_channel_files():
    arr = [71, 89, 92, 93, 94, 95, 96, 97, 98, 99, 100, 1752, 1757, 1758, 1760, 1761]
    sox2 = [f"SOX2/C2-Experiment-{i}.tif" for i in arr]
    golgi = [f"GOLGI/C1-Experiment-{i}.tif" for i in arr]
    return sox2, golgi


def get_one_channel_files():
    c2exp = [1267, 1268, 1269, 1270, 1272, 1273, 1274]
    fpaths = [f"SOX2-Golgi/C2-Experiment-{i}.tif" for i in c2exp]

    c2osvz = [1294, 1295, 1296, 1297]
    fpaths += [f"SOX2-Golgi/C2-oSVZ-Experiment-{i}.tif" for i in c2osvz]

    c2Osvz = [1286, 1287]
    fpaths += [f"SOX2-Golgi/C2-OSVZ-Experiment-{i}.tif" for i in c2Osvz]

    c2svz = [1290, 1291, 1292, 1293]
    fpaths += [f"SOX2-Golgi/C2-SVZ-Experiment-{i}.tif" for i in c2svz]

    fpaths += [
        "SOX2-Golgi/C2-SVZ-Experiment-1282-Substack-9-12.tif",
        "SOX2-Golgi/C2-SVZ-Experiment-1283-Substack-8-20.tif",
        "SOX2-Golgi/C2-SVZ-Experiment-1285-Substack-13-32.tif",
    ]
    return fpaths


def get_train_val_data(
    data_config,
    datadir,
    datasplit_type: DataSplitType,
    val_fraction=None,
    test_fraction=None,
    **kwargs,
):
    if data_config.subdset_type == SubDsetType.OneChannel:
        files_fn = get_one_channel_files
    elif data_config.subdset_type == SubDsetType.TwoChannel:
        files_fn = get_two_channel_files

    assert (
        data_config.channel_1 == Sox2GolgiChannelList.Sox2
    ), "Filenames have this order"
    assert data_config.channel_2 == Sox2GolgiChannelList.Golgi

    return get_train_val_data_twochannels(
        datadir,
        data_config,
        datasplit_type,
        files_fn,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
