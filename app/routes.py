from fastapi import APIRouter, Depends, HTTPException

from app.data_loader import DataLoader
from app.dependencies import get_or_initialize_data_loader
from app.logger import logger
from app.ranking import HybridRanker
from app.schemas import ErrorResponse, PredictionResponse, TransactionRequest

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Organization not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def predict_gl_code(
    transaction: TransactionRequest,
    data_loader: DataLoader = Depends(get_or_initialize_data_loader),
):
    logger.info("Predict GL Codes!")
    """
    Predict the most likely GL code for a transaction.

    This endpoint analyzes the transaction data and returns a ranked list of
    potential GL codes with confidence scores based on historical transactions.
    """
    try:
        # Initialize hybrid ranker
        ranker = HybridRanker(data_loader)

        # Get ranked GL codes
        gl_matches = ranker.rank_gl_codes(
            transaction.organization_id,
            transaction.merchant_name,
            transaction.transaction_amount,
        )

        # Return response
        return gl_matches

    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        # Log error and return 500
        print(f"Error predicting GL code: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error predicting GL code: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    logger.info("/health")
    return {"status": "healthy", "storage_type": "faiss"}
