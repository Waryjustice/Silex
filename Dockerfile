FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Create user with ID 1000 (required by HF Spaces)
RUN useradd -m -u 1000 user

# Switch to user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Set working directory for user
WORKDIR $HOME/app

# Copy project files with correct ownership
COPY --chown=user:user . $HOME/app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    pandas==2.1.3 \
    numpy==1.26.2 \
    httpx==0.25.2 \
    openai==1.3.0 \
    python-dotenv==1.0.0 \
    tabulate==0.9.0

# Expose port for HF Spaces
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/', timeout=5)"

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
