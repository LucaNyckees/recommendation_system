# Use Python base image
FROM python:3.10

# Set working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY . .

# Install the application in editable mode (if using setuptools)
RUN pip install -e .

# Expose ports for FastAPI and Dash
EXPOSE 8000 8501

# Set default command to a bash shell
CMD ["/bin/bash"]
