FROM pytorch/pytorch

# Install system dependencies and upgrade pip
RUN apt-get update && apt-get install -y \
    libpython3-dev \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip

WORKDIR /working_dir

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

# Copy project files into the container
COPY . .

# Default command to keep the container running
CMD ["bash"]