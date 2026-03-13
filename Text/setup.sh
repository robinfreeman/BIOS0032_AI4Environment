#!/bin/bash

# Installing ollama dependencies
apt-get install zstd

# Installing ollama
curl -fsSL https://ollama.com/install.sh | sh

# Install ollama python package
uv pip install ollama

# Start server in background
nohup ollama serve > ollama.log 2>&1 &

# Wait for server to start
echo "Waiting for Ollama server to respond..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
  sleep 2
done
echo "Server is up!"

# Install models
ollama pull gemma3
ollama pull qwen3

# Download data
wget https://raw.githubusercontent.com/MScEcologyAndDataScienceUCL/BIOS0032_AI4Environment/refs/heads/main/Text/sample_abstracts.csv
wget https://github.com/MScEcologyAndDataScienceUCL/BIOS0032_AI4Environment/raw/refs/heads/main/Text/test_set.xlsx
