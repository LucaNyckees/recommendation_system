#!/bin/bash
make download-data
echo "datasets downloaded."

make load-data
echo "data loaded to db."

mlflow server --host 0.0.0.0 --port 5000 &
echo "mlflow server launched."

make sentiment
echo "sentiment classifiers deployed."

uvicorn src.fastapi_app.main:app --host 0.0.0.0 --port 8000 &
echo "fastapi app launched."

python3.10 /app/dashboard.py
echo "dash app launched."
