#!/bin/bash
echo "Waiting for database to be ready..."
while ! nc -z db 5432; do
  sleep 0.1
done
echo "Database is ready!"

# Run dataset download and load commands
alembic upgrade head
make download-data
make load-data
uvicorn src.fastapi_app.main:app  --host 0.0.0.0 --port 8000
python3.10 . src/dashboard/dashboard.py