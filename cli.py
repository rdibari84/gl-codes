import click

from app.dependencies import get_or_initialize_data_loader
from app.ranking import HybridRanker
from app.schemas import TransactionRequest


@click.command()
@click.option("--organization_id", "-o", required=True, help="Organization ID")
@click.option("--merchant_name", "-m", required=True, help="Merchant name")
@click.option("--transaction_amount", "-a", required=True, help="Transaction amount")
def predict_gl_code(organization_id, merchant_name, transaction_amount):
    """Predict GL code for a transaction using the hybrid ranking system."""
    print(f"Processing transaction: {merchant_name} for ${transaction_amount}")

    # Create transaction request
    transaction = TransactionRequest(
        organization_id=organization_id,
        merchant_name=merchant_name,
        transaction_amount=float(transaction_amount),
    )

    # Initialize dependencies
    data_loader = get_or_initialize_data_loader()
    ranker = HybridRanker(data_loader)

    # Get prediction
    gl_prediction = ranker.rank_gl_codes(
        transaction.organization_id,
        transaction.merchant_name,
        transaction.transaction_amount,
    )

    # logger.info(gl_prediction.model_dump_json())

    # Print a more readable output
    print("\nResults:")
    if hasattr(gl_prediction, "best_result") and gl_prediction.best_result:
        best = gl_prediction.best_result
        print(f"\nBest result: {best.gl_name} ({best.gl_code})")
        print(f"Search type: {best.search_type.value}")
        print(f"Score: {best.relevance_score:.4f}")
        print(f"Reasoning: {best.reasoning}")
    else:
        print(
            "Could not determine a good result. Did not get a good semantic search result or a vector result. No frequent GL codes were found for this merchant either."
        )
        print("None")


if __name__ == "__main__":
    predict_gl_code()
