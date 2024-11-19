FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies and clean up apt cache
RUN apt install -y make


# Copy the requirements file first to leverage Docker's cache
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application files
COPY ./ /app

# Make sure the start.sh script is executable
RUN chmod +x scripts/start.sh

# Set the default command to run when the container starts
CMD ["scripts/start.sh"]
