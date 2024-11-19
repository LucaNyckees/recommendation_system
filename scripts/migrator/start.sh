#!/bin/bash
# Start PostgreSQL in the background

# Run Alembic migrations
echo "Running Alembic migrations..."
alembic upgrade head

# Keep the container running
wait