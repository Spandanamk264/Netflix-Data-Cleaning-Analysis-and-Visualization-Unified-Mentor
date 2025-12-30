
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++

# Copy Requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Project
COPY . .

# Expose Ports (API and Dashboard)
EXPOSE 8000
EXPOSE 5006

# Command: Run API by default
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
