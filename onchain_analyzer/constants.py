"""
DeFiGuard AI - On-Chain Analysis Constants
Storage slots, function selectors, ABIs, and chain configurations.
"""

# =============================================================================
# EIP-1967 STANDARD PROXY STORAGE SLOTS
# =============================================================================

# Implementation address slot: keccak256("eip1967.proxy.implementation") - 1
EIP1967_IMPLEMENTATION_SLOT = "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"

# Admin address slot: keccak256("eip1967.proxy.admin") - 1
EIP1967_ADMIN_SLOT = "0xb53127684a568b3173ae13b9f8a6016e243e63b6e8ee1178d6a717850b5d6103"

# Beacon address slot: keccak256("eip1967.proxy.beacon") - 1
EIP1967_BEACON_SLOT = "0xa3f0ad74e5423aebfd80d3ef4346578335a9a72aeaee59ff6cb3582b35133d50"

# =============================================================================
# OPENZEPPELIN STORAGE SLOTS (ERC-7201 Namespaced)
# =============================================================================

# Pausable storage: keccak256(abi.encode(uint256(keccak256("openzeppelin.storage.Pausable")) - 1)) & ~bytes32(uint256(0xff))
OZ_PAUSABLE_SLOT = "0xcd5ed15c6e187e77e9aee88184c21f4f2182ab5827cb3b7e07fbedcd63f03300"

# Ownable storage: keccak256(abi.encode(uint256(keccak256("openzeppelin.storage.Ownable")) - 1)) & ~bytes32(uint256(0xff))
OZ_OWNABLE_SLOT = "0x9016d09d72d40fdae2fd8ceac6b6234c7706214fd39c1cd1e609a0528c199300"

# =============================================================================
# EIP-1167 MINIMAL PROXY (CLONE) BYTECODE PATTERNS
# =============================================================================

MINIMAL_PROXY_PREFIX = "363d3d373d3d3d363d73"
MINIMAL_PROXY_SUFFIX = "5af43d82803e903d91602b57fd5bf3"

# Full minimal proxy bytecode template (45 bytes runtime)
# 363d3d373d3d3d363d73 <20-byte-address> 5af43d82803e903d91602b57fd5bf3

# =============================================================================
# FUNCTION SELECTORS (4-byte signatures)
# =============================================================================

FUNCTION_SELECTORS = {
    # ERC20 Standard
    "name": "06fdde03",
    "symbol": "95d89b41",
    "decimals": "313ce567",
    "totalSupply": "18160ddd",
    "balanceOf": "70a08231",
    "transfer": "a9059cbb",
    "approve": "095ea7b3",
    "allowance": "dd62ed3e",
    "transferFrom": "23b872dd",

    # Ownable
    "owner": "8da5cb5b",
    "transferOwnership": "f2fde38b",
    "renounceOwnership": "715018a6",
    "pendingOwner": "e30c3978",
    "acceptOwnership": "79ba5097",

    # Pausable
    "paused": "5c975abb",
    "pause": "8456cb59",
    "unpause": "3f4ba83a",

    # Proxy (EIP-1967)
    "implementation": "5c60da1b",
    "upgradeTo": "3659cfe6",
    "upgradeToAndCall": "4f1ef286",
    "admin": "f851a440",
    "changeAdmin": "8f283970",

    # UUPS (EIP-1822)
    "proxiableUUID": "52d1902d",

    # Diamond (EIP-2535)
    "facets": "7a0ed627",
    "facetAddress": "cdffacc6",
    "facetAddresses": "52ef6b2c",
    "facetFunctionSelectors": "adfca15e",
    "diamondCut": "1f931c1c",

    # Beacon
    "beacon": "59659e90",

    # Access Control
    "hasRole": "91d14854",
    "getRoleAdmin": "248a9ca3",
    "grantRole": "2f2ff15d",
    "revokeRole": "d547741f",
    "renounceRole": "36568abe",
    "DEFAULT_ADMIN_ROLE": "a217fddf",

    # Timelock
    "getMinDelay": "f27a0c92",
    "schedule": "01d5062a",
    "execute": "134008d3",
    "cancel": "c4d252f5",
}

# =============================================================================
# DANGEROUS FUNCTION SELECTORS (Backdoor Detection)
# =============================================================================

DANGEROUS_SELECTORS = {
    # Minting (can inflate supply)
    "40c10f19": {"name": "mint(address,uint256)", "risk": "HIGH", "category": "mint"},
    "a0712d68": {"name": "mint(uint256)", "risk": "HIGH", "category": "mint"},
    "1249c58b": {"name": "mint()", "risk": "HIGH", "category": "mint"},
    "4e6ec247": {"name": "mint(address,uint256)", "risk": "HIGH", "category": "mint"},
    "6a627842": {"name": "mint(address)", "risk": "HIGH", "category": "mint"},

    # Burning (can destroy tokens)
    "42966c68": {"name": "burn(uint256)", "risk": "MEDIUM", "category": "burn"},
    "9dc29fac": {"name": "burn(address,uint256)", "risk": "HIGH", "category": "burn"},
    "79cc6790": {"name": "burnFrom(address,uint256)", "risk": "HIGH", "category": "burn"},

    # Blacklist/Whitelist (centralization)
    "f9f92be4": {"name": "blacklist(address)", "risk": "HIGH", "category": "blacklist"},
    "0e136b19": {"name": "addToBlacklist(address)", "risk": "HIGH", "category": "blacklist"},
    "537df3b6": {"name": "removeFromBlacklist(address)", "risk": "MEDIUM", "category": "blacklist"},
    "fe575a87": {"name": "isBlacklisted(address)", "risk": "LOW", "category": "blacklist"},
    "44337ea1": {"name": "blacklistAddress(address,bool)", "risk": "HIGH", "category": "blacklist"},

    # Whitelist
    "0a3b0a4f": {"name": "addToWhitelist(address)", "risk": "HIGH", "category": "whitelist"},
    "e43252d7": {"name": "whitelist(address)", "risk": "HIGH", "category": "whitelist"},

    # Fee manipulation
    "8c0b5e22": {"name": "setFee(uint256)", "risk": "HIGH", "category": "fees"},
    "c0246668": {"name": "setTaxFee(uint256)", "risk": "HIGH", "category": "fees"},
    "8ee88c53": {"name": "setFeePercent(uint256)", "risk": "HIGH", "category": "fees"},
    "4fbee193": {"name": "setMaxFee(uint256)", "risk": "MEDIUM", "category": "fees"},
    "af465a27": {"name": "setSellFee(uint256)", "risk": "CRITICAL", "category": "fees"},
    "b8c61130": {"name": "setBuyFee(uint256)", "risk": "CRITICAL", "category": "fees"},

    # Arbitrary execution
    "b61d27f6": {"name": "execute(address,uint256,bytes)", "risk": "CRITICAL", "category": "execution"},
    "1cff79cd": {"name": "execute(address,bytes)", "risk": "CRITICAL", "category": "execution"},
    "a9059cbb": {"name": "transfer(address,uint256)", "risk": "LOW", "category": "transfer"},  # Normal but tracked

    # Withdrawal
    "3ccfd60b": {"name": "withdraw()", "risk": "MEDIUM", "category": "withdrawal"},
    "f3fef3a3": {"name": "withdraw(address,uint256)", "risk": "HIGH", "category": "withdrawal"},
    "51cff8d9": {"name": "withdraw(address)", "risk": "HIGH", "category": "withdrawal"},

    # Trading controls
    "8f70ccf7": {"name": "setTradingEnabled(bool)", "risk": "HIGH", "category": "trading"},
    "a9e75723": {"name": "enableTrading()", "risk": "MEDIUM", "category": "trading"},
    "c9567bf9": {"name": "openTrading()", "risk": "MEDIUM", "category": "trading"},

    # Max transaction/wallet limits
    "e517f2b9": {"name": "setMaxTxAmount(uint256)", "risk": "MEDIUM", "category": "limits"},
    "f7739b5f": {"name": "setMaxWalletSize(uint256)", "risk": "MEDIUM", "category": "limits"},
}

# =============================================================================
# DANGEROUS OPCODES
# =============================================================================

DANGEROUS_OPCODES = {
    "ff": {"name": "SELFDESTRUCT", "risk": "CRITICAL", "description": "Contract can be permanently destroyed"},
    "f4": {"name": "DELEGATECALL", "risk": "HIGH", "description": "Can execute arbitrary code in contract context"},
    "f2": {"name": "CALLCODE", "risk": "HIGH", "description": "Deprecated, similar to DELEGATECALL"},
}

# =============================================================================
# COMMON ABIs
# =============================================================================

# ERC20 ABI (minimal)
ERC20_ABI = [
    {"inputs": [], "name": "name", "outputs": [{"type": "string"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "symbol", "outputs": [{"type": "string"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "decimals", "outputs": [{"type": "uint8"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "totalSupply", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "account", "type": "address"}], "name": "balanceOf", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
]

# Ownable ABI
OWNABLE_ABI = [
    {"inputs": [], "name": "owner", "outputs": [{"type": "address"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "pendingOwner", "outputs": [{"type": "address"}], "stateMutability": "view", "type": "function"},
]

# Pausable ABI
PAUSABLE_ABI = [
    {"inputs": [], "name": "paused", "outputs": [{"type": "bool"}], "stateMutability": "view", "type": "function"},
]

# Proxy ABI
PROXY_ABI = [
    {"inputs": [], "name": "implementation", "outputs": [{"type": "address"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "admin", "outputs": [{"type": "address"}], "stateMutability": "view", "type": "function"},
]

# UUPS ABI
UUPS_ABI = [
    {"inputs": [], "name": "proxiableUUID", "outputs": [{"type": "bytes32"}], "stateMutability": "view", "type": "function"},
]

# Beacon ABI
BEACON_ABI = [
    {"inputs": [], "name": "implementation", "outputs": [{"type": "address"}], "stateMutability": "view", "type": "function"},
]

# Diamond Loupe ABI
DIAMOND_LOUPE_ABI = [
    {
        "inputs": [],
        "name": "facets",
        "outputs": [{
            "components": [
                {"name": "facetAddress", "type": "address"},
                {"name": "functionSelectors", "type": "bytes4[]"}
            ],
            "type": "tuple[]"
        }],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "facetAddresses",
        "outputs": [{"type": "address[]"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"name": "_functionSelector", "type": "bytes4"}],
        "name": "facetAddress",
        "outputs": [{"type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
]

# Access Control ABI
ACCESS_CONTROL_ABI = [
    {"inputs": [{"name": "role", "type": "bytes32"}, {"name": "account", "type": "address"}], "name": "hasRole", "outputs": [{"type": "bool"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "role", "type": "bytes32"}], "name": "getRoleAdmin", "outputs": [{"type": "bytes32"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "DEFAULT_ADMIN_ROLE", "outputs": [{"type": "bytes32"}], "stateMutability": "view", "type": "function"},
]

# =============================================================================
# WELL-KNOWN ROLE IDENTIFIERS
# =============================================================================

ROLE_IDENTIFIERS = {
    "DEFAULT_ADMIN_ROLE": "0x0000000000000000000000000000000000000000000000000000000000000000",
    "MINTER_ROLE": "0x9f2df0fed2c77648de5860a4cc508cd0818c85b8b8a1ab4ceeef8d981c8956a6",  # keccak256("MINTER_ROLE")
    "PAUSER_ROLE": "0x65d7a28e3265b37a6474929f336521b332c1681b933f6cb9f3376673440d862a",  # keccak256("PAUSER_ROLE")
    "UPGRADER_ROLE": "0x189ab7a9244df0848122154315af71fe140f3db0fe014031783b0946b8c9d2e3",  # keccak256("UPGRADER_ROLE")
    "BURNER_ROLE": "0x3c11d16cbaffd01df69ce1c404f6340ee057498f5f00246190ea54220576a848",  # keccak256("BURNER_ROLE")
}

# =============================================================================
# CHAIN CONFIGURATIONS
# =============================================================================

CHAIN_CONFIG = {
    1: {
        "name": "Ethereum",
        "short_name": "ETH",
        "rpc_env": "INFURA_PROJECT_ID",
        "rpc_template": "https://mainnet.infura.io/v3/{key}",
        "explorer_api": "https://api.etherscan.io/api",
        "explorer_key_env": "ETHERSCAN_API_KEY",
        "explorer_url": "https://etherscan.io",
        "native_symbol": "ETH",
        "native_decimals": 18,
        "block_time": 12,
        "supports_trace": True,
    },
    8453: {
        "name": "Base",
        "short_name": "BASE",
        "rpc_env": "ALCHEMY_API_KEY",
        "rpc_template": "https://base-mainnet.g.alchemy.com/v2/{key}",
        "explorer_api": "https://api.basescan.org/api",
        "explorer_key_env": "BASESCAN_API_KEY",
        "explorer_url": "https://basescan.org",
        "native_symbol": "ETH",
        "native_decimals": 18,
        "block_time": 2,
        "supports_trace": True,
    },
    42161: {
        "name": "Arbitrum One",
        "short_name": "ARB",
        "rpc_env": "ALCHEMY_API_KEY",
        "rpc_template": "https://arb-mainnet.g.alchemy.com/v2/{key}",
        "explorer_api": "https://api.arbiscan.io/api",
        "explorer_key_env": "ARBISCAN_API_KEY",
        "explorer_url": "https://arbiscan.io",
        "native_symbol": "ETH",
        "native_decimals": 18,
        "block_time": 0.25,
        "supports_trace": True,
    },
    137: {
        "name": "Polygon",
        "short_name": "MATIC",
        "rpc_env": "ALCHEMY_API_KEY",
        "rpc_template": "https://polygon-mainnet.g.alchemy.com/v2/{key}",
        "explorer_api": "https://api.polygonscan.com/api",
        "explorer_key_env": "POLYGONSCAN_API_KEY",
        "explorer_url": "https://polygonscan.com",
        "native_symbol": "MATIC",
        "native_decimals": 18,
        "block_time": 2,
        "supports_trace": True,
    },
    10: {
        "name": "Optimism",
        "short_name": "OP",
        "rpc_env": "ALCHEMY_API_KEY",
        "rpc_template": "https://opt-mainnet.g.alchemy.com/v2/{key}",
        "explorer_api": "https://api-optimistic.etherscan.io/api",
        "explorer_key_env": "OPTIMISM_API_KEY",
        "explorer_url": "https://optimistic.etherscan.io",
        "native_symbol": "ETH",
        "native_decimals": 18,
        "block_time": 2,
        "supports_trace": True,
    },
}

# Default chain if not specified
DEFAULT_CHAIN_ID = 1

# =============================================================================
# RISK SCORING
# =============================================================================

RISK_WEIGHTS = {
    "CRITICAL": 40,
    "HIGH": 25,
    "MEDIUM": 10,
    "LOW": 3,
}

# Max risk score adjustment from on-chain analysis
MAX_ONCHAIN_RISK_ADJUSTMENT = 30

# =============================================================================
# API RATE LIMITING
# =============================================================================

# Etherscan free tier: 5 requests/second
# We use 0.25s delay between requests for safety margin
ETHERSCAN_API_DELAY = 0.25  # seconds between API calls
ETHERSCAN_MAX_RETRIES = 3
ETHERSCAN_RETRY_DELAY = 1.0  # seconds
