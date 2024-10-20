"""
This module sets up project paths and configurations by loading 
environment variables from a .env file. It also configures logging using the 
Loguru library and integrates progress bar support with tqdm if available.

The module defines the following directory paths for organizing project files:
- PROJ_ROOT: Root directory of the project.
- DATA_DIR: Directory for storing data.
- RAW_DATA_DIR: Subdirectory for raw data.
- INTERIM_DATA_DIR: Subdirectory for interim data.
- PROCESSED_DATA_DIR: Subdirectory for processed data.
- EXTERNAL_DATA_DIR: Subdirectory for external data.
- MODELS_DIR: Directory for storing machine learning models.
- RESOURCES_DIR: Directory for storing additional resources.
- REPORTS_DIR: Directory for generating reports.
- FIGURES_DIR: Subdirectory for storing figures related to reports.

If tqdm is installed, it configures the logger to support progress bars.
"""

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

RESOURCES_DIR = PROJ_ROOT / "resources"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
