import os
import shutil
from pathlib import Path

import hydra
import kagglehub  # type: ignore
import typer
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm  # type: ignore

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

def validate_input_path(path: Path) -> Path:
    '''Validate input paths.'''
    if not path.exists():
        raise typer.BadParameter(f"Input path {path} does not exist.")
    return path

def validate_output_path(path: Path) -> Path:
    '''Validate output paths.'''
    if not path.parent.exists():
        raise typer.BadParameter(f"Output directory {path.parent} does not exist.")
    return path

def validate_paths(input_path: Path, output_path: Path) -> None:
    '''Validate input and output paths.'''
    validate_input_path(input_path)
    validate_output_path(output_path)
    

def download_dataset(url: str, cfg:DictConfig, output_path: Path) -> None:
    '''Download dataset from a URL.'''
    logger.info(f"Downloading dataset from {url} to {output_path}...")
    load_dotenv()
    os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME") # type: ignore
    os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_API_TOKEN") # type: ignore

    path = kagglehub.competition_download(cfg.data.dataset_name)

    files = os.listdir(path)
    csv_files = [f for f in files if f.endswith(".csv")]

    if not csv_files:
        csv_files = files

    os.makedirs(cfg.data.raw_data_path, exist_ok=True)

    for csv_file in csv_files:
        source_file = os.path.join(path, csv_file)
        destination = os.path.join(cfg.data.raw_data_path, csv_file)
        shutil.copy(source_file, destination)
        logger.info(f"Copied {csv_file} to {destination}")
        
    logger.success("Dataset downloaded successfully.")

@hydra.main(config_path="../../config", config_name="config")
def main(
    cfg: DictConfig
) -> None:
    '''Main function to download the dataset.'''
    validate_paths(Path(cfg.data.raw_data_path), Path(cfg.data.processed_data_path))
    download_dataset(cfg.data.dataset_url, cfg, Path(cfg.data.raw_data_path))


if __name__ == "__main__":
    main()
