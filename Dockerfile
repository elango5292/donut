FROM python:3.9

# Create a non-root user with a home directory
RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"

# Install system dependencies as root
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory and create cache directory
WORKDIR /app
RUN mkdir -p /app/cache && chown -R user:user /app

# Switch to non-root user
USER user

# Copy requirements and install as the user
COPY --chown=user:user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user:user . /app

# Expose port
EXPOSE 7860

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]