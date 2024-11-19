#!/bin/bash
echo "Waiting for database to be ready..."
while ! nc -z db 5432; do
  sleep 0.1
done
echo "Database is ready!"

# Run dataset download and load commands
alembic upgrade head
echo "alembic upgrade successful."

make download-data
echo "datasets downloaded."

make load-data
echo "data loaded to db."

uvicorn src.fastapi_app.main:app  --host 0.0.0.0 --port 8000
echo "fastapi app launched."

python3.10 . src/dashboard/dashboard.py
echo "dash app launched."
