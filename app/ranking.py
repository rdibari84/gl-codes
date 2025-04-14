from app.cohere_service import CohereService
from app.config import get_settings
from app.data_loader import DataLoader
from app.logger import logger
from app.schemas import (
    GLCodeMatch,
    PredictionResponse,
    SearchResult,
    SearchType,
    TransactionRequest,
)


class HybridRanker:
    """
    Service that implements hybrid ranking using both FAISS vector similarity and Cohere rerank
    """

    def __init__(self, data_loader: DataLoader):
        self.settings = get_settings()
        self.min_vector_score_threshold = self.settings.VECTOR_SCORE_THHRESHOLD
        self.min_semantic_score_threshold = self.settings.SEMANTIC_SCORE_THHRESHOLD
        self.cohere_service = CohereService()
        self.data_loader = data_loader

    def rank_gl_codes(
        self, organization_id: str, merchant_name: str, transaction_amount: float
    ) -> PredictionResponse:
        """
        Rank GL codes for a given transaction using hybrid approach combining:
        1. Vector similarity from FAISS
        2. Semantic similarity from Cohere
        3. Historical frequency data for the merchant

        Args:
            organization_id: The organization ID
            merchant_name: The merchant name
            transaction_amount: The transaction amount

        Returns:
            PredictionResponse with vector, semantic, and frequency-based results
        """
        # Initialize empty result lists
        vector_search_results = []
        semantic_search_results = []
        frequency_search_results = []

        # Original transaction for response
        original_transaction = TransactionRequest(
            organization_id=organization_id,
            merchant_name=merchant_name,
            transaction_amount=transaction_amount,
        )

        # Generate query for vector search
        vector_query = (
            f"Transaction with merchant {merchant_name} for amount {transaction_amount}"
        )

        # 1. VECTOR SEARCH
        # ----------------
        logger.info("- Perform vector search -")
        vector_results: SearchResult = self.data_loader.search_similar_transactions(
            organization_id, vector_query, k=self.settings.TOP_K_RESULTS
        )
        # logger.info(
        #     f"Vector search returned {len(vector_results.source_nodes) if vector_results and hasattr(vector_results, 'source_nodes') else 0} results"
        # )

        if (
            vector_results
            and hasattr(vector_results, "source_nodes")
            and vector_results.source_nodes
        ):
            # Check if any results are good enough
            good_vector_results = any(
                node.score > self.settings.VECTOR_SCORE_THHRESHOLD
                for node in vector_results.source_nodes
            )

            if not good_vector_results:
                logger.warning(
                    f"Vector similarity search returned low-quality results. All relevancy scores were below the threshold {self.settings.VECTOR_SCORE_THHRESHOLD}. Continue anyway"
                )

            # Add vector search results to the list
            for item in vector_results.source_nodes:
                result = GLCodeMatch(
                    gl_code=item.metadata.gl_code,
                    gl_name=item.metadata.gl_name,
                    search_type=SearchType.VECTOR_SEARCH,
                    relevance_score=item.score if item.score else 0.0,
                    reasoning=f"Vector similarity search returned a relevancy score of {item.score:.4f}",
                )
                vector_search_results.append(result)
        else:
            logger.warning("Vector similarity search returned no results")

        # 2. SEMANTIC SEARCH (using Cohere reranking)
        # ------------------------------------------
        logger.info("- Perform semantic search -")
        if (
            vector_results
            and hasattr(vector_results, "source_nodes")
            and vector_results.source_nodes
        ):
            try:
                semantic_result_dicts = self.cohere_service.rerank(
                    vector_query, vector_results.source_nodes, top_n=3
                )

                for result_dict in semantic_result_dicts:
                    result = GLCodeMatch(
                        gl_code=result_dict["gl_code"],
                        gl_name=result_dict["gl_name"],
                        search_type=SearchType.SEMANTIC_SEARCH,
                        relevance_score=result_dict["relevance_score"],
                        reasoning=result_dict["reasoning"],
                    )
                    semantic_search_results.append(result)

                # Check if any results are good enough
                good_semantic_results = any(
                    result.relevance_score > self.settings.SEMANTIC_SCORE_THHRESHOLD
                    for result in semantic_search_results
                )

                if not good_semantic_results:
                    logger.warning(
                        f"Semantic search returned low-quality results. All relevancy scores were below the threshold {self.settings.SEMANTIC_SCORE_THHRESHOLD}. Continue anyway"
                    )
            except Exception as e:
                logger.error(f"Error during semantic reranking: {str(e)}")
                # Continue with the process even if semantic search fails
        else:
            logger.warning(
                "Skipping semantic search due to lack of vector search results"
            )

        # 3. FREQUENCY SEARCH (historical data)
        # ------------------------------------
        logger.info("- Search frequency of codes for merchant -")
        try:
            # Get most frequent GL code for this merchant
            frequent_result = self.data_loader.get_most_frequent_code_for_merchant(
                org_id=organization_id,
                merchant_name=merchant_name,
                transaction_amount=transaction_amount,
            )

            if (
                frequent_result
                and "gl_code" in frequent_result
                and "gl_name" in frequent_result
            ):
                frequency = frequent_result.get("frequency", 0)
                # Calculate a normalized score based on frequency
                normalized_score = min(0.9, 0.5 + (frequency / 20))  # Cap at 0.9

                result = GLCodeMatch(
                    gl_code=frequent_result["gl_code"],
                    gl_name=frequent_result["gl_name"],
                    search_type=SearchType.FREQUENCY_SEARCH,
                    relevance_score=normalized_score,
                    reasoning=f"This GL code has been used {frequency} times previously for this merchant",
                )
                frequency_search_results.append(result)
                logger.info(
                    f"Frequency search found a result with {frequency} occurrences"
                )
            else:
                logger.info("No frequent GL codes found for this merchant")
        except Exception as e:
            logger.error(f"Error during frequency search: {str(e)}")

        # Determine the best overall result using the specified priority:
        # 1. Semantic search (first choice)
        # 2. Vector search (second choice)
        # 3. Frequency search (last resort)
        best_result = None

        if good_semantic_results:
            best_result = semantic_search_results[0]
            logger.info(
                f"Selected best result from semantic search: {best_result.gl_code}"
            )
        elif good_vector_results:
            best_result = vector_search_results[0]
            logger.info(
                f"Selected best result from vector search: {best_result.gl_code}"
            )
        elif frequency_search_results:
            best_result = frequency_search_results[0]
            logger.info(
                f"Selected best result from frequency data: {best_result.gl_code}"
            )
        else:
            best_result = None
            logger.warning("Could not determine a good result.")

        # Create and return the final response
        response = PredictionResponse(
            best_result=best_result,
            vector_search_results=vector_search_results,
            semantic_search_results=semantic_search_results,
            frequency_search_results=frequency_search_results,
            original_transaction=original_transaction,
        )

        return response
