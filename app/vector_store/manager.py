from pathlib import Path
from typing import Dict

import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from app.config import get_settings
from app.logger import logger
from app.schemas import SearchResult
from app.vector_store.organization_store import OrganizationVectorStore


class VectorStoreManager:
    """
    Manages vector stores for multiple organizations
    Creates an organization Store for each org
    """

    def __init__(self):
        self.settings = get_settings()

        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(self.settings.EMBEDDING_MODEL)

        # Dictionary of organization vector stores
        self.org_stores: Dict[str, OrganizationVectorStore] = {}

        # Load data from CSV files
        gl_codes_path = self.settings.GL_CODES_PATH
        if not Path(gl_codes_path).exists():
            gl_codes_path = Path(
                Path(__file__).parent.parent, self.settings.GL_CODES_PATH
            )
        transactions_path = self.settings.TRANSACTIONS_PATH
        if not Path(transactions_path).exists():
            transactions_path = Path(
                Path(__file__).parent.parent, self.settings.TRANSACTIONS_PATH
            )
        self.gl_codes_df = pd.read_csv(gl_codes_path)
        self.transactions_df = pd.read_csv(transactions_path)
        self.transactions_df.columns = [
            col.lower() for col in self.transactions_df.columns
        ]
        self.gl_codes_df.columns = [col.lower() for col in self.gl_codes_df.columns]

    def load_data(self) -> None:
        """
        Load data for all organizations
        """
        # Get unique organizations
        organizations = self.transactions_df["organization_id"].unique()

        # Find organizations that need to be processed
        for org_id in tqdm(organizations, desc="Processing organizations .........."):
            print("\n")
            logger.info(f"---- Organization id {org_id} -----")
            # Get or create the org store
            org_store = self._get_org_store(org_id)

            # Skip if already exists
            if org_store.exists():
                logger.info(f"Index already exists for organization. Loading...")
                org_store.load()
                continue

            # Filter data for this organization
            org_transactions = self.transactions_df[
                self.transactions_df["organization_id"] == org_id
            ]
            org_gl_codes = self.gl_codes_df[
                self.gl_codes_df["organization_id"] == org_id
            ]

            # Build the index
            logger.info(f"Need to build the index for organization...")
            org_store.build_index(org_gl_codes, org_transactions)

        logger.info(
            f"-- Successfully Loaded {len(self.gl_codes_df)} GL codes and {len(self.transactions_df)} transactions into vector store. -- \n"
        )

    def _get_org_store(self, org_id: str) -> OrganizationVectorStore:
        """
        Get or create an organization vector store
        """
        if org_id not in self.org_stores:
            self.org_stores[org_id] = OrganizationVectorStore(
                org_id=org_id,
                embedding_model=self.embedding_model,
            )

        return self.org_stores[org_id]

    # ===== API Methods =====

    def search(self, org_id: str, query: str, k: int = 10) -> SearchResult:
        """
        Search for similar items in an organization
        """
        org_store = self._get_org_store(org_id)
        return org_store.search(query, k)

    def get_most_frequent_code_for_merchant(
        self, org_id: str, merchant_name: str, transaction_amount: float
    ) -> Dict:
        """
        Get historical transactions for a specific merchant in an organization
        """
        org_store = self._get_org_store(org_id)
        return org_store.get_most_frequent_code_for_merchant(
            merchant_name, transaction_amount
        )

    def check_organization_exists(self, org_id: str) -> bool:
        """
        Check if an organization exists
        """
        # Try to get or create the store
        org_store = self._get_org_store(org_id)

        # Check if it exists on disk
        return org_store.exists()
