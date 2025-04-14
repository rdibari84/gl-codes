from typing import Dict

from app.config import get_settings
from app.logger import logger
from app.schemas import SearchResult
from app.vector_store.manager import VectorStoreManager


class DataLoader:
    """
    This class wraps the Vector Store.
    This enables decoupling and flexibility - if in the future we want tp swap out the vector database implementation,
    we can do so by updating this file and adding the new store
    """

    def __init__(self):
        self.settings = get_settings()
        self.vector_store = VectorStoreManager()

    def load_data(self):
        """
        Load data into the vector store
        """
        logger.info("-- Load data --")
        try:
            self.vector_store.load_data()
        except Exception as e:
            logger.info(f"Error loading data into vector store: {str(e)}")
            raise

    def get_most_frequent_code_for_merchant(
        self, org_id: str, merchant_name: str, transaction_amount: float
    ) -> Dict:
        """
        Get historical transactions for a specific merchant in an organization
        """
        return self.vector_store.get_most_frequent_code_for_merchant(
            org_id, merchant_name, transaction_amount
        )

    def search_similar_transactions(
        self, org_id: str, query: str, k: int = 10
    ) -> SearchResult:
        """
        Search for similar transactions using vector similarity
        """
        return self.vector_store.search(org_id, query, k)
