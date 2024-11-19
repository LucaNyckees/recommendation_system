FROM python:3.10

# Set the working directory
WORKDIR /app

# Install Python and Alembic
RUN pip install --upgrade pip
RUN pip install alembic toml "psycopg2[binary]" "psycopg[binary]" SQLAlchemy alembic-utils

# Copy the Alembic files
COPY alembic.ini /app/
COPY src/database /app/src/database
COPY src/paths.py /app/src/paths.py

# Define the entrypoint script
COPY scripts/migrator/start.sh /app/
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]