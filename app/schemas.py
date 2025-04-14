import json
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class SearchType(Enum):
    VECTOR_SEARCH = "vector_search"
    SEMANTIC_SEARCH = "semantic_search"
    FREQUENCY_SEARCH = "frequency_search"


class TransactionRequest(BaseModel):
    """
    Request model for transaction prediction
    """

    organization_id: str = Field(..., description="Organization ID")
    merchant_name: str = Field(..., description="Merchant name")
    transaction_amount: float = Field(..., description="Transaction amount")


class GLCodeMatch(BaseModel):
    """
    Model for a GL code match result
    """

    gl_code: str = Field(..., description="GL code")
    gl_name: str = Field(..., description="GL code name/description")
    search_type: SearchType = Field(
        ..., description="the type of search that was performed to get the result"
    )
    relevance_score: float = Field(..., description="Semantic relevance score")
    reasoning: str = Field(..., description="Reasoning for this match")


class PredictionResponse(BaseModel):
    """
    Response model for transaction prediction
    """

    best_result: Optional[GLCodeMatch] = Field(default=None)
    vector_search_results: List[GLCodeMatch] = Field(
        ..., description="Ranked list of GL code matches"
    )
    semantic_search_results: List[GLCodeMatch] = Field(
        ..., description="Ranked list of GL code matches"
    )
    frequency_search_results: List[GLCodeMatch] = Field(
        ..., description="Ranked list of GL code matches"
    )
    original_transaction: TransactionRequest = Field(
        ..., description="Original transaction data"
    )

    def to_json(self):
        return self.model_dump_json()


class ErrorResponse(BaseModel):
    """
    Error response model
    """

    detail: str = Field(..., description="Error details")


class Metadata(BaseModel):
    gl_code: str
    gl_name: str
    frequency: int
    merchant: Optional[str] = None
    amount: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return self.model_dump()

    def to_json_str(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_dict(
        cls, data: Union[Dict[str, Any], List, str]
    ) -> Union["Metadata", List["Metadata"]]:
        """Create Metadata instance(s) from dictionary, list, or JSON string.

        Args:
            data: Dictionary, list of dictionaries, or JSON string containing metadata

        Returns:
            A single Metadata instance or a list of Metadata instances

        Note:
            This method handles dictionaries, lists of dictionaries, and JSON strings
        """
        # If data is a string, try to parse it as JSON
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                raise ValueError(f"Could not parse metadata string as JSON: {data}")

        # Handle list of dictionaries
        if isinstance(data, list):
            # Return a list of Metadata objects
            result = []
            for item in data:
                # Recursively call from_dict for each item
                # This will handle nested lists or dictionaries
                if isinstance(item, (dict, str)):
                    result.append(cls.from_dict(item))
                else:
                    raise TypeError(
                        f"Expected dict or JSON string in list, got {type(item).__name__}"
                    )
            return result

        # Handle single dictionary
        elif isinstance(data, dict):
            # Create a single Metadata instance
            return cls(
                gl_code=data.get("gl_code", ""),
                gl_name=data.get("gl_name", ""),
                frequency=data.get("frequency", 0),
                merchant=data.get("merchant"),  # Optional field
                amount=data.get("amount"),  # Optional field
            )

        else:
            raise TypeError(
                f"Expected dict, list, or JSON string, got {type(data).__name__}"
            )


class SourceNode(BaseModel):
    """Simple class to represent a source node with document and metadata"""

    text: str = Field(..., description="The document text content")
    metadata: Metadata = Field(..., description="Metadata for the document")
    score: Optional[float] = Field(None, description="Relevance score")

    def __str__(self):
        return f"SourceNode(text={self.text[:50]}..., score={self.score})"


class SearchResult(BaseModel):
    """Simple container for search results"""

    source_nodes: list[SourceNode]


# class SearchResult(BaseModel):
#     """Class representing a search result from the vector store"""

#     text: str = Field(..., description="The document text content")
#     distance: float = Field(..., description="Distance metric from the vector search")
#     relevance_score: float = Field(
#         ..., description="Calculated relevance score (1/(1+distance))"
#     )

#     # Metadata fields
#     gl_code: Optional[str] = Field(None, description="GL code identifier")
#     gl_name: Optional[str] = Field(None, description="Name of the GL code")
#     merchant: Optional[str] = Field(None, description="Merchant name")
#     amount: Optional[float] = Field(None, description="Transaction amount")
#     frequency: Optional[int] = Field(None, description="Frequency of occurrence")
#     is_merchant_specific: Optional[bool] = Field(
#         None, description="Whether this is a merchant-specific entry"
#     )

#     # Additional metadata that doesn't fit into predefined fields
#     additional_metadata: Dict[str, Any] = Field(
#         default_factory=dict,
#         description="Additional metadata not covered by specific fields",
#     )

#     def to_rerank_format(self) -> Dict[str, Any]:
#         """Convert the search result to a format suitable for reranking"""
#         return {
#             "text": self.text,
#             "id": f"{self.gl_code}_{self.merchant}"
#             if self.gl_code and self.merchant
#             else str(id(self)),
#         }

#     @classmethod
#     def from_metadata(
#         cls, document: str, metadata_obj: Any, distance: float
#     ) -> "SearchResult":
#         """Create a SearchResult from a document and metadata object"""
#         # Convert metadata object to dict if it's not already
#         if hasattr(metadata_obj, "__dict__"):
#             metadata = metadata_obj.__dict__
#         elif hasattr(metadata_obj, "dict"):
#             metadata = metadata_obj.dict()
#         else:
#             metadata = dict(metadata_obj)

#         # Calculate relevance score
#         relevance_score = 1.0 / (1.0 + distance)

#         # Extract known fields
#         known_fields = {
#             "gl_code",
#             "gl_name",
#             "merchant",
#             "amount",
#             "frequency",
#             "is_merchant_specific",
#         }

#         # Separate known fields from additional metadata
#         main_fields = {k: v for k, v in metadata.items() if k in known_fields}
#         additional_fields = {k: v for k, v in metadata.items() if k not in known_fields}

#         # Create the result
#         result = cls(
#             text=document,
#             distance=distance,
#             relevance_score=relevance_score,
#             additional_metadata=additional_fields,
#             **main_fields,
#         )

#         return result
