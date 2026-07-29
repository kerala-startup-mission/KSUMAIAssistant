# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for audio processing (Whisper/Edge-TTS)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy your requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the container
COPY . .

# Expose port 8000 so the frontend can reach FastAPI
EXPOSE 8000

# Start the FastAPI server when the container boots
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]