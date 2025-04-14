# GL Code Predictor

A microservice that predicts General Ledger (GL) codes for financial transactions using the Cohere rerank API and a hybrid ranking approach.

## Overview

This application addresses the challenge of accurately classifying financial transactions with appropriate GL codes. By analyzing historical transaction data and utilizing semantic similarity through the Cohere API alongside numerical similarity based on transaction amounts, it provides intelligent predictions.

## Features

- **API Service**: Accepts transaction data and returns ranked GL code predictions
- **Cohere Integration**: Leverages the Cohere rerank API for semantic matching
- **Hybrid Ranking**: Combines semantic and numerical similarity for better predictions
- **Detailed Reasoning**: Provides explanations for suggested matches
- **CLI Tool**: Command-line interface for testing without API calls
- **Docker Support**: Easy deployment with Docker and docker-compose

## Requirements
- Docker
- docker-compose
- Optional pyenv, python 3.12.4

*Note*
Consider using [colima](https://smallsharpsoftwaretools.com/tutorials/use-colima-to-run-docker-containers-on-macos/) as a free version to run docker, if Docker Desktop is unavailable

## Setup Instructions
1. Create a `.env` file to override any setting
   ```
   # API settings
   COHERE_API_KEY=fillin

   # Data paths - relative to app/
   GL_CODES_PATH=data/gl_codes_genai_takehome.csv
   TRANSACTIONS_PATH=data/transactions_genai_takehome.csv

   # FAISS settings - relative to app/vector_store
   VECTOR_STORE_DIR=embeddings/
   EMBEDDING_MODEL=all-MiniLM-L6-v2 

   # Ranking weights
   VECTOR_SCORE_THHRESHOLD=0.65
   SEMANTIC_SCORE_THHRESHOLD=0.2
   TOP_K_RESULTS=5
   ```
2. Stand up docker
   ```
   docker-compose -d up
   ```
3. If you want to run locally
   ```
   pyenv virtualenv 3.12.4 venv-mercury
   pyenv activate venv-mercury
   pip install poetry
   poetry install
   poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## Design Approach

### Why use RAG, semantic search and frequency
RAG (Retrieval-Augmented Generation) and semantic search are ideal for the GL code prediction problem because they effectively bridge the knowledge gap between different stakeholders. By semantically matching new transactions with historical ones, the system can understand the context beyond simple keyword matching - recognizing that "client dinner" and "business meal" represent similar concepts despite different terminology. This approach combines the strengths of both AI understanding (through Cohere's semantic ranking) and historical patterns (frequency analysis), providing explainable recommendations with confidence scores. The hybrid ranking method balances semantic similarity with numerical transaction data, enabling accurate GL code prediction when users have limited accounting knowledge.

*NOTE*
I intentially kept the historical transactions for each org in a separate vector store index. I thought it best to not accidently predict codes that an organization does not use. This is an assumption that we may want to change in the future

### Future Enhancements
- Need to handle a few cases 
1. A new organization - no historical transactions exist so this service might not predict anything
2. We have a new merchant that we haven't seen before -- RAG may or may not return with a good answer and the historical frequency won't match
   - maybe we should search other organizations to see if the merchant was used, determine a GL code and then figure out what active GL for the org matches the predicted one
3. Some of the historical transactions reference GL code that have been deleted. That historical transaction data is still worthwhile. Right now, the service uses those transactions to predict, but we'll need to make another call to determine what active GL code to use.
4. Consider transaction patterns and seasonality - aka use the date of the transaction in a meaningful way
5. Collect feedback on predictions to improve future recommendations
6. Secure the API with proper authentication
7. More comprehensive error responses for edge cases

### Data Processing

1. Data is loaded from CSV files during application startup
2. Transactions are organized by organization and stored in a vector database
3. A FAISS index is created per org to ensure we don't predict GL codes not associated to the organization
4. GL code frequency is calculated for each merchant-GL code pair and used in a worst case scenario

### Ranking Mechanism
The service suggests a best answer, while also returning results from semantic search, vector search and frequency search so the calling client can choose what it wants to use. 
The 'best answer' approach entails:

1. **Semantic Similarity**:
   - Uses Cohere rerank API to find semantically similar GL codes
   - Considers historical use of GL codes with specific merchants

2. **Vector Similarity**:
   - If Semantic Similarity fails or is bad, we'll default to the vector similarity

3. **Frequency**:
   - If Semantic similarity or vector similarity is bad, we'll look at the most frequented GL code for the org/ merchant
   - Compares transaction amount to historical amounts

## How to Run
Vist `http://localhost:8000/docs`
### CLI
```
poetry run python3 cli.py -o 8c2f4fde-a69d-11ee-8758-f709cc17f119j -m fake -a -1
-o 0261019e-f163-11ec-b45a-8b765a2c4fc8p -m Slack -a 75
```
#### API
`curl -X GET http://localhost:8000/api/health`
`curl -X POST http://localhost:8000/api/predict -H "Content-Type: application/json" -d '{"organization_id":"36e10e52-c4ff-11ee-80f8-436515a9f9a3f","merchant_name":"Amazon","transaction_amount":75}'`


## Data Classes
#### Input
```
{
   "organization_id": "8c2f4fde-a69d-11ee-8758-f709cc17f119j",
   "merchant_name": "Slack",
   "transaction_amount": 50
}
```  
#### Response
```
{
  "best_result": {
    "gl_code": "b386550a-1628-11ef-ae6b-0f7bc6a08613",
    "gl_name": "Professional Fees",
    "search_type": "semantic_search",
    "relevance_score": 0.23739068,
    "reasoning": "This GL code matches the merchant transaction based on semantic similarity with a score of 0.2374",
    "metadata": {
      "gl_code": "b386550a-1628-11ef-ae6b-0f7bc6a08613",
      "gl_name": "Professional Fees",
      "frequency": 3,
      "merchant": "Upwork",
      "amount": "50.4",
      "transaction_date": "50.4",
      "transaction_metadata_id": "149f9d4a-f2da-11ef-9cd5-d3e446ac44d5t",
      "hierarchy_manualness": "manual categorization"
    }
  },
  "vector_search_results": [
    {
      "gl_code": "b386550a-1628-11ef-ae6b-0f7bc6a08613",
      "gl_name": "Professional Fees",
      "search_type": "vector_search",
      "relevance_score": 0.6485603514033108,
      "reasoning": "Vector similarity search returned a relevancy score of 0.6486",
      "metadata": {
        "gl_code": "b386550a-1628-11ef-ae6b-0f7bc6a08613",
        "gl_name": "Professional Fees",
        "frequency": 3,
        "merchant": "Upwork",
        "amount": "50.4",
        "transaction_date": "50.4",
        "transaction_metadata_id": "149f9d4a-f2da-11ef-9cd5-d3e446ac44d5t",
        "hierarchy_manualness": "manual categorization"
      }
    },
    {
      "gl_code": "934a48c4-162c-11ef-b2e6-1b4bcaf260f4",
      "gl_name": "Software Subscriptions",
      "search_type": "vector_search",
      "relevance_score": 0.593541452129625,
      "reasoning": "Vector similarity search returned a relevancy score of 0.5935",
      "metadata": {
        "gl_code": "934a48c4-162c-11ef-b2e6-1b4bcaf260f4",
        "gl_name": "Software Subscriptions",
        "frequency": 1,
        "merchant": "Openai",
        "amount": "60",
        "transaction_date": "60",
        "transaction_metadata_id": "1dd60af4-03ae-11f0-a248-5f18b7aa74eat",
        "hierarchy_manualness": "manual categorization"
      }
    },
    {
      "gl_code": "934a48c4-162c-11ef-b2e6-1b4bcaf260f4",
      "gl_name": "Software Subscriptions",
      "search_type": "vector_search",
      "relevance_score": 0.5896360350876897,
      "reasoning": "Vector similarity search returned a relevancy score of 0.5896",
      "metadata": {
        "gl_code": "934a48c4-162c-11ef-b2e6-1b4bcaf260f4",
        "gl_name": "Software Subscriptions",
        "frequency": 1,
        "merchant": "Sendinblue.Com",
        "amount": "30",
        "transaction_date": "30",
        "transaction_metadata_id": "3c9b944a-f599-11ef-a17a-69c7d114aeaet",
        "hierarchy_manualness": "manual categorization"
      }
    },
    {
      "gl_code": "934a48c4-162c-11ef-b2e6-1b4bcaf260f4",
      "gl_name": "Software Subscriptions",
      "search_type": "vector_search",
      "relevance_score": 0.5871361138060006,
      "reasoning": "Vector similarity search returned a relevancy score of 0.5871",
      "metadata": {
        "gl_code": "934a48c4-162c-11ef-b2e6-1b4bcaf260f4",
        "gl_name": "Software Subscriptions",
        "frequency": 1,
        "merchant": "Chargeflow",
        "amount": "7.25",
        "transaction_date": "7.25",
        "transaction_metadata_id": "188518ec-106f-11f0-85cc-a7b37556186et",
        "hierarchy_manualness": "manual categorization"
      }
    }
  ],
  "semantic_search_results": [
    {
      "gl_code": "b386550a-1628-11ef-ae6b-0f7bc6a08613",
      "gl_name": "Professional Fees",
      "search_type": "semantic_search",
      "relevance_score": 0.23739068,
      "reasoning": "This GL code matches the merchant transaction based on semantic similarity with a score of 0.2374",
      "metadata": {
        "gl_code": "b386550a-1628-11ef-ae6b-0f7bc6a08613",
        "gl_name": "Professional Fees",
        "frequency": 3,
        "merchant": "Upwork",
        "amount": "50.4",
        "transaction_date": "50.4",
        "transaction_metadata_id": "149f9d4a-f2da-11ef-9cd5-d3e446ac44d5t",
        "hierarchy_manualness": "manual categorization"
      }
    },
    {
      "gl_code": "934a48c4-162c-11ef-b2e6-1b4bcaf260f4",
      "gl_name": "Software Subscriptions",
      "search_type": "semantic_search",
      "relevance_score": 0.09674864,
      "reasoning": "This GL code matches the merchant transaction based on semantic similarity with a score of 0.0967",
      "metadata": {
        "gl_code": "934a48c4-162c-11ef-b2e6-1b4bcaf260f4",
        "gl_name": "Software Subscriptions",
        "frequency": 1,
        "merchant": "Openai",
        "amount": "60",
        "transaction_date": "60",
        "transaction_metadata_id": "1dd60af4-03ae-11f0-a248-5f18b7aa74eat",
        "hierarchy_manualness": "manual categorization"
      }
    },
    {
      "gl_code": "934a48c4-162c-11ef-b2e6-1b4bcaf260f4",
      "gl_name": "Software Subscriptions",
      "search_type": "semantic_search",
      "relevance_score": 0.0842289,
      "reasoning": "This GL code matches the merchant transaction based on semantic similarity with a score of 0.0842",
      "metadata": {
        "gl_code": "934a48c4-162c-11ef-b2e6-1b4bcaf260f4",
        "gl_name": "Software Subscriptions",
        "frequency": 1,
        "merchant": "Chargeflow",
        "amount": "7.25",
        "transaction_date": "7.25",
        "transaction_metadata_id": "188518ec-106f-11f0-85cc-a7b37556186et",
        "hierarchy_manualness": "manual categorization"
      }
    }
  ],
  "frequency_search_results": [],
  "original_transaction": {
    "organization_id": "8c2f4fde-a69d-11ee-8758-f709cc17f119j",
    "merchant_name": "Amazon",
    "transaction_amount": 50
  }
}
```
