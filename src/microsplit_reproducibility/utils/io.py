import argparse
import glob
import json
import pickle
import os
import socket
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union, TYPE_CHECKING

from pytorch_lightning.loggers import WandbLogger
import torch

if TYPE_CHECKING:
    from careamics.config import (
        VAEAlgorithmConfig,
        DataConfig,
        LVAELossConfig,
        TrainingConfig,
    )

Config = Union["VAEAlgorithmConfig", "DataConfig", "LVAELossConfig", "TrainingConfig"]


def get_new_model_version(model_dir: Union[Path, str]) -> int:
    """Create a unique version ID for a new model run.
    
    Parameters
    ----------
    model_dir : Union[Path, str]
        The directory where the model logs are stored.
        
    Returns
    -------
    int
        The new version ID.
    """
    versions = []
    for version_dir in os.listdir(model_dir):
        try:
            versions.append(int(version_dir))
        except:
            print(
                f"Invalid subdirectory:{model_dir}/{version_dir}. Only integer versions are allowed"
            )
            exit()
    if len(versions) == 0:
        return "0"
    return f"{max(versions) + 1}"


def get_workdir(
    root_dir: str,
    model_name: str,
) -> tuple[Path, Path]:
    """Get the workdir for the current model.

    Workdir path has the following structure: "root_dir/YYMM/model_name/version".
    
    Parameters
    ----------
    root_dir : str
        The root directory where all model logs are stored.
    model_name : str
        The name of the model.
        
    Returns
    -------
    cur_workdir : Path
        The current work directory.
    rel_path : Path
        The relative path of the work directory.
    """
    rel_path = datetime.now().strftime("%y%m")
    cur_workdir = os.path.join(root_dir, rel_path)
    Path(cur_workdir).mkdir(exist_ok=True, parents=True)

    rel_path = os.path.join(rel_path, model_name)
    cur_workdir = os.path.join(root_dir, rel_path)
    Path(cur_workdir).mkdir(exist_ok=True)

    rel_path = os.path.join(rel_path, get_new_model_version(cur_workdir))
    cur_workdir = os.path.join(root_dir, rel_path)
    try:
        Path(cur_workdir).mkdir(exist_ok=False)
    except FileExistsError:
        print(f"Workdir {cur_workdir} already exists.")
    return cur_workdir, rel_path


def log_config(
    config: Config, 
    name: Literal["algorithm", "training", "data", "loss"],
    log_dir: Union[Path, str],
    logger: Optional[WandbLogger] = None
) -> None:
    """Save the `pydantic` configuration in a JSON file.
    
    Parameters
    ----------
    config : Config
        The configuration to save.
    name : str
        The name of the configuration.
    save_dir : Union[Path, str]
        The directory where the configuration file is saved.
    logger : Optional[WandbLogger], optional
        The logger to save the configuration in WANDB, by default None.
        
    Returns
    -------
    None
    """
    with open(os.path.join(log_dir, f"{name}_config.json"), "w") as f:
        f.write(config.model_dump_json(indent=4))

    if logger:
        logger.experiment.config.update({f"{name}": config.model_dump()})


def log_configs(
    configs: Sequence[Config],
    names: Sequence[Literal["algorithm", "training", "data", "loss"]],
    log_dir: Union[Path, str],
    wandb_project: Optional[WandbLogger] = None
) -> Optional[WandbLogger]:
    """Save the `pydantic` configurations in JSON files.
    
    Parameters
    ----------
    configs : Sequence[Config, Config, Config, Config]
        The configurations to save.
    names : Sequence[Literal["algorithm", "training", "data", "loss"]]
        The names of the configurations.
    log_dir : Union[Path, str]
        The directory where the configuration files are logged.
    logger : Optional[WandbLogger], optional
        The logger to save the configurations in WANDB, by default None.
        
    Returns
    -------
    Optional[WandbLogger]
        The logger instance.
    """
    # Define the logger
    if wandb_project:
        name = "/".join(log_dir.split("/")[-2:])
        logger = WandbLogger(
            name=os.path.join(socket.gethostname(), name),
            save_dir=log_dir,
            project=wandb_project,
        )
    else:
        logger = None
    
    for config, name in zip(configs, names):
        log_config(config, name, log_dir, logger)
        
    return logger


def get_noise_models(paths: list[str]):
    """Load the noise models if they aren't provided

    Parameters
    ----------
    pats : list[str]
        The paths to the noise models.
    """
    for path in paths:
        if not os.path.isfile(path):
            print(f"Downloading {path.split('/')[-1]}") # TODO get from url 
            

def get_model_checkpoint(
    ckpt_dir: str, mode: Literal['best', 'last'] = 'best'
) -> str:
    """Get the model checkpoint path.
    
    Parameters
    ----------
    ckpt_dir : str
        Checkpoint directory.
    mode : Literal['best', 'last'], optional
        Mode to get the checkpoint, by default 'best'.
    
    Returns
    -------
    str
        Checkpoint path.
    """
    output = []
    for fpath in glob.glob(ckpt_dir + "/*.ckpt"):
        fname = os.path.basename(fpath)
        if mode == 'best':
            if fname.startswith('best'):
                output.append(fpath)
        elif mode == 'last':
            if fname.startswith('last'):
                output.append(fpath)
    assert len(output) == 1, '\n'.join(output)
    return output[0]


def load_model_checkpoint(
    ckpt_dir: str, 
    mode: Literal['best', 'last'] = 'best'
) -> dict[str, Any]:
    """Load a model checkpoint.
    
    Parameters
    ----------
    ckpt_path : str
        Checkpoint path.
    mode : Literal['best', 'last'], optional
        Mode to get the checkpoint, by default 'best'.
    
    Returns
    -------
    dict[str, Any]
        Model checkpoint.
    """
    if os.path.isdir(ckpt_dir):
        ckpt_fpath = get_model_checkpoint(ckpt_dir, mode=mode)
    else:
        assert os.path.isfile(ckpt_dir)
        ckpt_fpath = ckpt_dir

    print(f"Loading checkpoint from: '{ckpt_fpath}'")
    return torch.load(ckpt_fpath)


def _load_file(file_path: str) -> Any:
    """Load a file with the appropriate method based on the file extension.
    
    Parameters
    ----------
    file_path : str
        File path.
        
    Returns
    -------
    Any
        Loaded file content.
    """
    # Get the file extension
    _, ext = os.path.splitext(file_path)

    # Check the extension and load the file accordingly
    if ext == '.pkl':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif ext == '.json':
        with open(file_path) as f:
            return json.load(f)
    else:
        raise ValueError(
            f"Unsupported file extension: {ext}. Only .pkl and .json are supported."
        )


def load_config(
    config_fpath: str, 
    config_type: Literal['algorithm', 'training', 'data']
) -> dict:
    """Load a configuration file.
    
    Parameters
    ----------
    config_fpath : str
        Configuration file path.
    config_type : Literal['algorithm', 'training', 'data']
        Configuration type.
        
    Returns
    -------
    dict
        Configuration dictionary.
    """
    fname = glob.glob(os.path.join(config_fpath, f'{config_type}_config.*'))
    if fname:
        return _load_file(os.path.join(config_fpath, fname[0]))
    else:
        raise FileNotFoundError(f"Config file not found in {config_fpath}.")


def load_checkpoint(ckpt_dir: Union[str, Path], best: bool = True) -> dict:
    """Load the checkpoint from the given directory.
    
    Parameters
    ----------
    ckpt_dir : Union[str, Path]
        The path to the checkpoint directory.
    best : bool, optional
        Whether to load the best checkpoint, by default True.
        If False, the last checkpoint will be loaded.
    
    Returns
    -------
    dict
        The loaded checkpoint.
    """
    if os.path.isdir(ckpt_dir):
        ckpt_fpath = get_model_checkpoint(ckpt_dir, mode="best" if best else "last")
    else:
        assert os.path.isfile(ckpt_dir)
        ckpt_fpath = ckpt_dir

    ckpt = torch.load(ckpt_fpath)
    print(f"Loading checkpoint from: '{ckpt_fpath}' - Epoch: {ckpt['epoch']}")
    return ckpt

def _prepare_log_info(
    algorithm: str,
    metrics: dict[str, Any],
    eval_info: dict[str, Any],
    ckpt_dir: Union[str, Path],
) -> dict[str, Any]:
    """Prepare the log information for the evaluation of the current experiment.
    
    Parameters
    ----------
    algorithm : str
        The algorithm identifier.
    metrics: dict[str, Any]
        The evaluation metrics.
    eval_info : dict[str, Any]
        Evaluation information.
    ckpt_dir : Union[str, Path]
        Checkpoint directory.
    
    Returns
    -------
    dict[str, Any]
        Logged information about current experiment.
    """
    log_info = {"timestamp": datetime.now().strftime("%d/%m/%y_%H:%M")}
    log_info["user_id"] = socket.gethostname()
    log_info["algorithm"] = algorithm
    log_info["ckpt_dir"] = ckpt_dir
    log_info["metrics"] = metrics
    log_info["eval_info"] = eval_info
    return log_info  


def log_experiment(
    log_dir: Union[str, Path],
    dset_id: str,
    algorithm: Literal["denoisplit", "musplit"],
    metrics: dict[str, Any],
    eval_info: dict[str, Any],
    ckpt_dir: Union[str, Path],
) -> None:
    """Log the evaluation information for the current experiment.
    
    Parameters
    ----------
    log_dir : Union[str, Path]
        The directory where the evaluation information is saved.
    dset_id : str
        The dataset identifier.
    algorithm : Literal["denoisplit", "musplit"]
        The algorithm identifier.
    metrics: dict[str, Any]
        The evaluation metrics.
    eval_info : dict[str, Any]
        Evaluation information.
    ckpt_dir : Union[str, Path]
        Checkpoint directory.
        
    Returns
    -------
    None
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{dset_id}_eval.json")
    if not os.path.isfile(log_file_path):
        print(f"Creating new log file at {log_file_path}")
        with open(log_file_path, "w") as f:
            json.dump({"evals": []}, f, indent=2)
    
    info = _prepare_log_info(algorithm, metrics, eval_info, ckpt_dir)
    
    with open(log_file_path, "r") as f:
        data = json.load(f)
        data["evals"].append(info)
    with open(log_file_path, "w") as f:
        print(f"Updating log file at {log_file_path}")
        json.dump(data, f, indent=2)
        

def get_training_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        type=str,
        help="The algorithm to use. Pick between 'denoiSplit' and 'muSplit'.",
        required=False,
    )
    parser.add_argument(
        "--root_path",
        type=str,
        help="The root path for the training experiments.",
        required=True,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="The path to the data directory.",
        required=True,
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        help="The name of the wandb project.",
        required=False,
        default=None,
    )
    return parser.parse_args()


def get_evaluation_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="The path to the training checkpoints.",
        required=True,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="The path to the data directory.",
        required=True,
    )
    parser.add_argument(
        "--mmse_count",
        type=int,
        help="The number of samples to draw from the model.",
        default=2,
    )
    parser.add_argument(
        "--subdset_type",
        type=str,
        help="The sub-dataset to use for evaluation.",
        default="OnlyIba1",
        choices=[
            "Iba1",
            "Iba1_Ki67",
            "Iba1NucPercent30",
            "Iba1NucPercent50",
            "Iba1NucPercent70",
        ]
        )
    return parser.parse_args()
    