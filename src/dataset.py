import os
import shutil
from pathlib import Path

import hydra
import kagglehub  # type: ignore
import typer
from dotenv import load_dotenv
from hydra.utils import get_original_cwd
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm  # type: ignore

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def validate_input_path(path: Path) -> Path:
    """Validate input paths."""
    if not path.exists():
        raise typer.BadParameter(f"Input path {path} does not exist.")
    return path


def validate_output_path(path: Path) -> Path:
    """Validate output paths."""
    if not path.parent.exists():
        raise typer.BadParameter(f"Output directory {path.parent} does not exist.")
    return path


def validate_paths(input_path: Path, output_path: Path) -> None:
    """Validate input and output paths."""
    validate_input_path(input_path)
    validate_output_path(output_path)


def download_dataset(url: str, data_cfg: DictConfig, cfg: DictConfig, output_path: Path) -> None:
    logger.info(f"Downloading dataset from {url} to {output_path}...")

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    load_dotenv()
    os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")  # type: ignore
    os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_API_TOKEN")  # type: ignore

    path = kagglehub.dataset_download(data_cfg.dataset_name)

    files = os.listdir(path)
    csv_files = [f for f in files if f.endswith(".csv")]

    if not csv_files:
        csv_files = files

    for csv_file in csv_files:
        source_file = Path(path) / csv_file
        destination = output_path / csv_file

        shutil.copy(source_file, destination)
        logger.info(f"Copied {csv_file} to {destination}")

    logger.success("Dataset downloaded successfully.")


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to download the dataset."""

    data_config_dir = Path(get_original_cwd()) / "config" / "data"
    dataset_configs = []
    print(data_config_dir)
    for yaml_file in sorted(data_config_dir.glob("*.yaml")):
        data_cfg = OmegaConf.load(yaml_file)
        dataset_configs.append(data_cfg)

    logger.info(f"Datasets to download: {[dc.dataset_name for dc in dataset_configs]}")

    for data_cfg in dataset_configs:
        download_dataset(data_cfg.dataset_name, data_cfg, cfg, RAW_DATA_DIR)


if __name__ == "__main__":
    main()
