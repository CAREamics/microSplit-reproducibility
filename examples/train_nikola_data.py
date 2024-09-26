from datasets import create_train_val_datasets, load_train_val_data_nikola
from configs.nikolaData import get_data_configs


train_data_config, val_data_config = get_data_configs()

train_dset, val_dset, data_stats = create_train_val_datasets(
    datapath='/group/jug/ashesh/data/nikola_data/20240531/',
    train_config=train_data_config,
    val_config=val_data_config,
    load_data_func=load_train_val_data_nikola
)
