# Start from a small, official Python image (best practice for production)
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies first (for faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and model artifact
# This includes the Python script and the saved model
COPY app.py .
COPY model.pkl .

# Expose the port that Uvicorn will listen on
EXPOSE 8000

# Command to run the application using Uvicorn
# The command uses 'gunicorn' with uvicorn workers for production stability, 
# instead of the simple 'uvicorn app:app --reload' we used for development.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "app:app"]