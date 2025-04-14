FROM python:3.12-slim

WORKDIR /work

# Install system dependencies needed for FAISS
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY app/ app/.
COPY pyproject.toml .

RUN pip install poetry
RUN poetry install

# Environment variables
ENV PYTHONPATH=/work
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]