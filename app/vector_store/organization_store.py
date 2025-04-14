import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.logger import logger
from app.schemas import Metadata, SearchResult, SourceNode


class OrganizationVectorStore:
    """
    A simplified vector store for a single organization
    """

    def __init__(self, org_id: str, embedding_model: SentenceTransformer):
        """Initialize the vector store"""
        self.settings = get_settings()
        self.org_id = org_id
        self.data_dir = self.settings.VECTOR_STORE_DIR
        self.embedding_model = embedding_model

        # The FAISS index and metadata (loaded on demand)
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Metadata] = []

        # File paths
        self.index_path = Path(self.settings.VECTOR_STORE_DIR, f"{org_id}_index.faiss")
        if not Path(self.index_path).exists():
            self.index_path = Path(
                Path(__file__).parent,
                self.settings.VECTOR_STORE_DIR,
                f"{org_id}_index.faiss",
            )
        self.metadata_path = Path(
            self.settings.VECTOR_STORE_DIR, f"{org_id}_metadata.json"
        )
        if not Path(self.metadata_path).exists():
            self.metadata_path = Path(
                Path(__file__).parent,
                self.settings.VECTOR_STORE_DIR,
                f"{org_id}_metadata.json",
            )

        # Ensure directories exist before saving
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.metadata_path).parent.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        """Check if this organization's index exists on disk"""
        return self.index_path.exists() and self.metadata_path.exists()

    def is_loaded(self) -> bool:
        """Check if the index is loaded in memory"""
        return self.index is not None and len(self.metadata) > 0

    # ===== INDEX CREATION METHODS =====

    def build_index(
        self, gl_codes_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> bool:
        """Build a new index for this organization"""
        try:
            logger.info(f"build_index called")
            # Create documents and metadata
            documents, self.metadata = self._prepare_documents(
                gl_codes_df, transactions_df
            )

            # Create the FAISS index
            self.index = self._create_index(documents)

            # Save to disk
            self._save_index()

            return True
        except Exception as e:
            print(f"Error building index for organization {self.org_id}: {e}")
            return False

    def _prepare_documents(
        self, gl_codes_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> Tuple[List[str], List[Dict]]:
        """Prepare documents and metadata for indexing"""
        logger.info("Prepare documents")
        documents = []
        metadata = []

        merged_df = pd.merge(
            transactions_df,
            gl_codes_df[["organization_id", "gl_code_id", "gl_name"]],
            on=["organization_id", "gl_code_id"],
            how="left",
        )
        missing_gl_names = merged_df["gl_name"].isna().sum()
        if missing_gl_names > 0:
            logger.warning(
                f"WARNING: Some transactions do not link back to master GL list. Percentage of transactions missing gl_name: {missing_gl_names / len(merged_df) * 100:.2f}%"
            )
            logger.warning(
                "These GL codes were likely deleted from the master list. Keeping these for historical information, with gl_name 'is_deleted'"
            )
        merged_df["gl_name"] = merged_df["gl_name"].fillna("is_deleted")

        # Step 1: Calculate merchant-GL code frequencies
        logger.info("Calculate merchant-GL code frequencies")
        merchant_gl_counts = (
            merged_df.groupby(["merchant_name", "gl_code_id"])
            .size()
            .reset_index(name="count")
        )

        # Step 2: Add merchant-specific documents
        logger.info("Add merchant-specific documents")
        for _, row in merged_df.iterrows():
            merchant = row["merchant_name"]
            gl_code = row["gl_code_id"]
            gl_name = row["gl_name"]
            frequency = merchant_gl_counts[
                (merchant_gl_counts["merchant_name"] == merchant)
                & (merchant_gl_counts["gl_code_id"] == gl_code)
            ]["count"].iloc[0]
            amount = row["line_item_amount"]

            documents.append(
                f"Transaction with merchant {merchant} for amount {amount}"
            )

            # Add metadata
            metadata.append(
                Metadata(
                    gl_code=gl_code,
                    gl_name=gl_name,
                    frequency=frequency,
                    merchant=merchant,
                    amount=amount,
                )
            )

        return documents, metadata

    def _create_index(self, documents: List[str]) -> Tuple[faiss.Index, List[Dict]]:
        """Create a FAISS index from documents"""
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)

        # Create index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        return index

    def _save_index(self) -> None:
        """Save the index and metadata to disk"""
        if self.index is None or not self.metadata:
            return

        # Save FAISS index
        if not self.exists() or self.settings.FORCE_INDEX_OVERWRITE:
            logger.info(f"... saving index for {self.org_id}")
            faiss.write_index(self.index, str(self.index_path))
            logger.info(f"Successfully saved index for {self.org_id}")

        # Save metadata
        logger.info(f"Saving metadata for {self.org_id}")
        with open(self.metadata_path, "w") as f:
            # For each item, ensure it's a simple dict with serializable values
            metadata_to_save = []

            for item in self.metadata:
                metadata_to_save.append(item.to_json_str())

            # Write the JSON
            json.dump(metadata_to_save, f)

        logger.info(f"Successfully saved metadata for {self.org_id}")

    # ===== QUERY METHODS =====

    def load(self) -> bool:
        """Load the index and metadata from disk"""
        if not self.exists():
            return False

        try:
            # Load index
            logger.info("reading faiss index...")
            self.index = faiss.read_index(str(self.index_path))
        except Exception as e:
            logger.error(f"Error loading index for organization {self.org_id}: {e}")
            return False

        try:
            # Load metadata
            logger.info("loading metadata...")
            with open(self.metadata_path, "r") as f:
                metadata = json.load(f)
            self.metadata = Metadata.from_dict(metadata)
        except Exception as e:
            logger.error(f"Error loading metdata for organization {self.org_id}: {e}")
            return False

        logger.info("Successfully loaded data.")
        return True

    def search(self, query: str, k: int = 10) -> SearchResult:
        """Search the index for similar items"""
        if self.index is None:
            logger.error("Index not loaded")
            return SearchResult(source_nodes=[])

        if self.embedding_model is None:
            logger.error("Embedding model not provided")
            return SearchResult(source_nodes=[])

        # Generate query embedding
        logger.info(f"Searching for data relevant to {query}")
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)

        # Search
        distances, indices = self.index.search(query_embedding, k)

        if len(indices) == 0 or len(indices[0]) == 0:
            logger.warning(f"Query returned no results")
            return SearchResult(source_nodes=[])

        # Process results
        source_nodes = []

        # Use indices[0] to get the first row of the 2D indices array
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                # Get the metadata
                metadata_item = self.metadata[idx]

                # Generate document text from metadata
                if isinstance(metadata_item, Metadata):
                    merchant = (
                        metadata_item.merchant
                        if metadata_item.merchant is not None
                        else "Unknown"
                    )
                    gl_code = metadata_item.gl_code
                    gl_name = metadata_item.gl_name
                    amount = (
                        metadata_item.amount
                        if metadata_item.amount is not None
                        else "0"
                    )
                    document_text = f"Transaction with merchant {merchant} for gl code {gl_code} and GL name {gl_name} with amount {amount}"
                else:
                    logger.warning("metadata is not an instance of Metadata class...")
                    document_text = str(metadata_item)

                # Calculate relevance score
                score = 1.0 / (1.0 + float(distances[0][i]))

                # Create SourceNode
                source_node = SourceNode(
                    text=document_text,
                    metadata=metadata_item,
                    score=score,
                )

                source_nodes.append(source_node)

        return SearchResult(source_nodes=source_nodes)

    def get_most_frequent_code_for_merchant(
        self, merchant_name: str, transaction_amount: float
    ) -> Dict:
        """
        Get the most frequently used GL code for a merchant.
        If multiple GL codes tie for frequency, select the one with amount closest to transaction_amount.
        """
        logger.info(
            f"Get most frequent GL code for merchant. org_id {self.org_id}, merchant_name: {merchant_name}"
        )

        # Load if needed
        if not self.is_loaded():
            if not self.load():
                return {}

        highest_freq = 0
        candidates = []  # Store all candidates with the highest frequency

        for item in self.metadata:
            if item.merchant is not None and item.merchant == merchant_name:
                gl_code = item.gl_code
                amount = float(item.amount) if item.amount is not None else 0.0
                frequency = item.frequency

                if frequency > highest_freq:
                    # Found a new highest frequency - clear previous candidates
                    highest_freq = frequency
                    candidates = [
                        {
                            "gl_code": gl_code,
                            "gl_name": item.gl_name,
                            "merchant_name": merchant_name,
                            "transaction_amount": amount,
                            "frequency": frequency,
                            "amount_diff": abs(amount - transaction_amount),
                        }
                    ]
                elif frequency == highest_freq:
                    # Tied with current highest frequency - add to candidates
                    candidates.append(
                        {
                            "gl_code": gl_code,
                            "gl_name": item.gl_name,
                            "merchant_name": merchant_name,
                            "transaction_amount": amount,
                            "frequency": frequency,
                            "amount_diff": abs(amount - transaction_amount),
                        }
                    )

        if not candidates:
            return {}

        # If there are multiple candidates with the same frequency,
        # choose the one with the closest transaction amount
        if len(candidates) > 1:
            candidates.sort(key=lambda x: x["amount_diff"])
            logger.info(
                f"Multiple GL codes with frequency {highest_freq} found. Selected closest amount match."
            )

        result = candidates[0]
        # Remove the temporary diff field
        del result["amount_diff"]
        return result
