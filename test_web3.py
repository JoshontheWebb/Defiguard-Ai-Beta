from web3 import Web3
infura_url = "https://mainnet.infura.io/v3/6f6912f033c847b9aca653e8b246e639"
w3 = Web3(Web3.HTTPProvider(infura_url))
print("Connected:", w3.is_connected())
print("Block Number:", w3.eth.block_number)