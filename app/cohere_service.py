from typing import Any, Dict, List

import cohere

from app.config import get_settings
from app.logger import logger
from app.schemas import SourceNode


class CohereService:
    """
    Service to interact with Cohere API for semantic search and reranking
    """

    def __init__(self):
        self.settings = get_settings()
        self.client = cohere.Client(self.settings.COHERE_API_KEY)

    def rerank(
        self, query: str, source_nodes: List[SourceNode], top_n: int = None
    ) -> List[Dict[str, Any]]:
        """
        Use Cohere rerank API to rank documents based on relevance to query

        Args:
            query: The query text
            source_nodes: List of SourceNode objects from vector search
            top_n: Number of top results to return (default from settings)

        Returns:
            List of ranked documents with scores
        """
        logger.debug("Calling cohere rerank")
        if not source_nodes:
            logger.warning("No source nodes provided for reranking")
            return []

        if top_n is None:
            top_n = self.settings.TOP_K_RESULTS

        try:
            # Convert SourceNode objects to dictionary format required by Cohere
            documents = []
            for node in source_nodes:
                documents.append(
                    {
                        "text": node.text,
                        "metadata": node.metadata,
                    }
                )

            # Call Cohere rerank API
            response = self.client.rerank(
                query=query,
                documents=documents,
                top_n=top_n,
                model="rerank-v3.5",
            )

            # Process results
            results = []
            for idx, result in enumerate(response.results):
                # Make sure the index is valid
                if result.index >= len(source_nodes):
                    logger.warning(f"Invalid index {result.index} in result {idx}")
                    continue

                # Get the original source node
                original_node = source_nodes[result.index]

                # Create a dictionary with all the necessary information
                result_doc = {
                    "gl_code": original_node.metadata.gl_code,
                    "gl_name": original_node.metadata.gl_name,
                    "rank": idx + 1,
                    "relevance_score": result.relevance_score,
                    "frequency": original_node.metadata.frequency,
                    "amount": original_node.metadata.amount,
                    "reasoning": f"This GL code matches the merchant transaction based on semantic similarity with a score of {result.relevance_score:.4f}",
                    "metadata": original_node.metadata,
                }

                results.append(result_doc)

            return results

        except Exception as e:
            logger.error(f"Error calling Cohere rerank API: {str(e)}")
            logger.exception(e)  # Log the full stack trace
            raise e
