#!/bin/bash
make download-data
echo "datasets downloaded."

make load-data
echo "data loaded to db."

uvicorn src.fastapi_app.main:app &
echo "fastapi app launched."

python3.10 /app/dashboard.py
echo "dash app launched."
