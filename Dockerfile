# 1. Use an official Python runtime as a parent image
FROM python:3.9-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your application's code into the container
COPY . .

# 5. Expose the port the app runs on (Gunicorn default is 8000)
EXPOSE 8000

# 6. Define the command to run your application
# This command runs Gunicorn, a production-ready web server.
# It specifies 4 worker processes and binds to port 8000.
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8000", "app_local:app"]