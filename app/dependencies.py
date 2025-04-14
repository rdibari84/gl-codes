from functools import lru_cache
from typing import Optional

from app.data_loader import DataLoader
from app.logger import logger

# Global instance storage - not exposed outside this module
_data_loader_instance: Optional[DataLoader] = None


@lru_cache(maxsize=1)
def get_or_initialize_data_loader() -> DataLoader:
    """
    Singleton pattern implementation for DataLoader.
    Uses lru_cache to ensure only one instance is created.
    """
    global _data_loader_instance

    # If instance already exists, return it
    if _data_loader_instance is not None:
        logger.info("Returning instance")
        return _data_loader_instance

    # Create new instance
    logger.info("Creating new data loader instance")
    data_loader = DataLoader()

    # Only load data the first time
    data_loader.load_data()

    # Store instance for future use
    _data_loader_instance = data_loader

    return data_loader
