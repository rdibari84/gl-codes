from typing import Dict

from app.config import get_settings
from app.logger import logger
from app.schemas import SearchResult
from app.vector_store.manager import VectorStoreManager


class DataLoader:
    """
    Service to load and process data for GL code prediction
    FAISS Vector Database implementation
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
            # # Load data from CSV files
            # gl_codes_path = self.settings.GL_CODES_PATH
            # if not Path(gl_codes_path).exists():
            #     gl_codes_path = Path(Path(__file__).parent, self.settings.GL_CODES_PATH)
            # transactions_path = self.settings.TRANSACTIONS_PATH
            # if not Path(transactions_path).exists():
            #     transactions_path = Path(
            #         Path(__file__).parent, self.settings.TRANSACTIONS_PATH
            #     )
            # gl_codes_df = pd.read_csv(gl_codes_path)
            # transactions_df = pd.read_csv(transactions_path)
            # transactions_df.columns = [col.lower() for col in transactions_df.columns]
            # gl_codes_df.columns = [col.lower() for col in gl_codes_df.columns]

            # Load data into vector store
            # self.vector_store.load_data(gl_codes_df, transactions_df)

            # logger.info(
            #     f"-- Loaded {len(gl_codes_df)} GL codes and {len(transactions_df)} transactions into vector store. -- \n"
            # )
            self.vector_store.load_data()
        except Exception as e:
            logger.info(f"Error loading data into vector store: {str(e)}")
            raise

    # def get_organization_data(self) -> Dict:
    #     """
    #     Return organization data (for compatibility with in-memory implementation)
    #     """
    #     # This method is maintained for compatibility but not used in FAISS implementation
    #     return {}

    def get_most_frequent_code_for_merchant(
        self, org_id: str, merchant_name: str, transaction_amount: float
    ) -> Dict:
        """
        Get historical transactions for a specific merchant in an organization
        """
        return self.vector_store.get_most_frequent_code_for_merchant(
            org_id, merchant_name, transaction_amount
        )

    # def get_transactions_for_merchant(
    #     self, org_id: str, merchant_name: str
    # ) -> List[Dict]:
    #     """
    #     Get historical transactions for a specific merchant in an organization
    #     """
    #     return self.vector_store.get_transactions_for_merchant(org_id, merchant_name)

    # def get_gl_code_details(self, org_id: str, gl_code: str) -> Optional[Dict]:
    #     """
    #     Get details for a specific GL code in an organization
    #     """
    #     return self.vector_store.get_gl_code_details(org_id, gl_code)

    # def get_merchant_frequency(self, org_id: str, merchant_name: str) -> Dict:
    #     """
    #     Get frequency of GL codes used for a specific merchant
    #     """
    #     return self.vector_store.get_merchant_frequency(org_id, merchant_name)

    # def check_organization_exists(self, org_id: str) -> bool:
    #     """
    #     Check if an organization exists
    #     """
    #     return self.vector_store.check_organization_exists(org_id)

    def search_similar_transactions(
        self, org_id: str, query: str, k: int = 10
    ) -> SearchResult:
        """
        Search for similar transactions using vector similarity
        """
        return self.vector_store.search(org_id, query, k)
