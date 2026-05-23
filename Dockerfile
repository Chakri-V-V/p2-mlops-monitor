FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python model/train.py && \
    python model/simulate_drift.py && \
    python monitoring/drift_check.py && \
    python monitoring/performance_estimate.py
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
