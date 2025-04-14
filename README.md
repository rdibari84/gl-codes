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
   COHERE_API_KEY=Gwiu5tTMOeV2uLLA13g30SUXYliv2p4FgInZ2Ki3

   # Data paths
   GL_CODES_PATH=data/gl_codes_genai_takehome.csv
   TRANSACTIONS_PATH=data/transactions_genai_takehome.csv

   # FAISS settings
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
   poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Design Approach

### Data Processing

1. Data is loaded from CSV files during application startup
2. Transactions are organized by organization and stored in a vector database
3. GL code frequency is calculated for each merchant-GL code pair

### Ranking Mechanism
The hybrid ranking approach combines:

1. **Semantic Similarity**:
   - Uses Cohere rerank API to find semantically similar GL codes
   - Considers historical use of GL codes with specific merchants

2. **Vector Similarity**:
   - If Semantic Similarity fails or is bad, we'll default to the vector similarity

2. **Frequency**:
   - Compares transaction amount to historical amounts
   - Normalizes differences for fair comparison

## Future Enhancements
- handle case for new merchant
- handle case where a historical GL code that matches the incoming transaction is no longer active
- Consider transaction patterns and seasonality
- Collect feedback on predictions to improve future recommendations
- Secure the API with proper authentication
- More comprehensive error responses for edge cases

### Use
Vist `http://localhost:8000/docs`
#### CLI
`poetry run python3 cli.py -o 8c2f4fde-a69d-11ee-8758-f709cc17f119j -m fake -a -1`
`-o 0261019e-f163-11ec-b45a-8b765a2c4fc8p -m Slack -a 75`
#### API
`curl -X GET http://localhost:8000/api/health`
`curl -X POST http://localhost:8000/api/predict -H "Content-Type: application/json" -d '{"organization_id":"36e10e52-c4ff-11ee-80f8-436515a9f9a3f","merchant_name":"Amazon","transaction_amount":75}'`