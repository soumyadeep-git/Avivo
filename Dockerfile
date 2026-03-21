# Use an official Python runtime as a parent image (slim version for smaller footprint)
FROM python:3.10-slim

# Set environment variables to ensure Python output is logged directly to the terminal
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for building some python packages (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure the database and data directories exist
RUN mkdir -p /app/db /app/data

# Command to run the bot
CMD ["python", "bot.py"]
