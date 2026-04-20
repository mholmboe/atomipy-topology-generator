FROM python:3.11-slim

# Install git needed for any submodule or potential pip-git installs
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code and local library folder
COPY . .

# Set environment variables
ENV PORT=5001
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MALLOC_ARENA_MAX=2
ENV OMP_NUM_THREADS=1
ENV ATOMIPY_PROCESS_INLINE=true

# Expose the default port (documentation only)
EXPOSE 5001

# Run the app using gunicorn
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 1 --worker-class gthread --threads 2 --timeout 1800 app:app"]
