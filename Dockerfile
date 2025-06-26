FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Create streamlit config
RUN mkdir -p ~/.streamlit/
RUN echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = 8080\n\
" > ~/.streamlit/config.toml

# Expose port
EXPOSE 8080

# Run the application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"] 