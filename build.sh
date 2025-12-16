#!/bin/bash
# Install system deps for compilation if needed (rare)
apt-get update && apt-get install -y curl build-essential

# Install solc-select
curl -L https://github.com/crytic/solc-select/releases/download/v0.3.0/solc-select-linux-amd64 -o /usr/local/bin/solc-select
chmod +x /usr/local/bin/solc-select

# Install solc versions (match common Solidity pragmas)
solc-select install 0.8.24 0.7.6 0.6.12

# Set default
solc-select use 0.8.24

# Install Python deps
pip install -r requirements.txt