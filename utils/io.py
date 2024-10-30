from datetime import datetime
import os
from pathlib import Path
from typing import Literal, Optional, Sequence, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from pytorch_lightning.loggers import WandbLogger
    from careamics.config import (
        VAEAlgorithmConfig,
        DataConfig,
        LVAELossConfig,
        TrainingConfig,
    )
    Config = Union[VAEAlgorithmConfig, DataConfig, LVAELossConfig, TrainingConfig]


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


def save_config(
    config: Config, 
    name: Literal["algorithm", "training", "data", "loss"],
    save_dir: Union[Path, str],
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
    with open(os.path.join(save_dir, f"{name}_config.json"), "w") as f:
        f.write(config.model_dump_json(indent=4))

    if logger:
        logger.experiment.config.update({f"{name}": config.model_dump()})


def save_configs(
    configs: Sequence[Config, Config, Config, Config],
    names: Sequence[Literal["algorithm", "training", "data", "loss"]],
    save_dir: Union[Path, str],
    logger: Optional[WandbLogger] = None
) -> None:
    """Save the `pydantic` configurations in JSON files.
    
    Parameters
    ----------
    configs : Sequence[Config, Config, Config, Config]
        The configurations to save.
    names : Sequence[Literal["algorithm", "training", "data", "loss"]]
        The names of the configurations.
    save_dir : Union[Path, str]
        The directory where the configuration files are saved.
    logger : Optional[WandbLogger], optional
        The logger to save the configurations in WANDB, by default None.
        
    Returns
    -------
    None
    """
    for config, name in zip(configs, names):
        save_config(config, name, save_dir, logger)
    