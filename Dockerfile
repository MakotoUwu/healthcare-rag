# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY ./src /app/src
COPY .env.sample /app/.env.sample
# Note: .env should ideally be managed via Cloud Run secrets or env vars, not copied.

# Make port 8080 available to the world outside this container
# Cloud Run expects the container to listen on the port defined by the PORT env var ($PORT), default is 8080.
EXPOSE 8080

# Define environment variable (Cloud Run will set this)
ENV PORT=8080

# Run main.py when the container launches
# Ensure src/main.py is set up to listen on 0.0.0.0 and respect the PORT env var
CMD ["python", "src/main.py"]
