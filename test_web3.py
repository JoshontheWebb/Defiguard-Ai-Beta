import os
from web3 import Web3

# Use environment variable for Infura project ID
infura_project_id = os.getenv("INFURA_PROJECT_ID")
if not infura_project_id:
    print("Error: INFURA_PROJECT_ID environment variable not set")
    exit(1)

infura_url = f"https://mainnet.infura.io/v3/{infura_project_id}"
w3 = Web3(Web3.HTTPProvider(infura_url))
print("Connected:", w3.is_connected())
print("Block Number:", w3.eth.block_number)