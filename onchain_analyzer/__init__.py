"""
DeFiGuard AI - On-Chain Analysis Module

Comprehensive on-chain analysis for smart contract security:
- Proxy pattern detection (EIP-1967, UUPS, Transparent, Beacon, Diamond, Minimal)
- Storage slot analysis (owner, admin, paused state)
- Backdoor scanning (20+ dangerous patterns)
- Honeypot detection
- Token security analysis

Usage:
    from onchain_analyzer import OnChainAnalyzer

    analyzer = OnChainAnalyzer(web3_provider_url)
    result = await analyzer.analyze(contract_address)
"""

from .core import OnChainAnalyzer
from .proxy_detector import ProxyDetector
from .storage_analyzer import StorageAnalyzer
from .backdoor_scanner import BackdoorScanner, HoneypotDetector
from .constants import (
    CHAIN_CONFIG,
    DEFAULT_CHAIN_ID,
    EIP1967_IMPLEMENTATION_SLOT,
    EIP1967_ADMIN_SLOT,
    EIP1967_BEACON_SLOT,
    DANGEROUS_SELECTORS,
    DANGEROUS_OPCODES,
    FUNCTION_SELECTORS,
)

__all__ = [
    # Main analyzer
    "OnChainAnalyzer",

    # Component analyzers
    "ProxyDetector",
    "StorageAnalyzer",
    "BackdoorScanner",
    "HoneypotDetector",

    # Constants
    "CHAIN_CONFIG",
    "DEFAULT_CHAIN_ID",
    "EIP1967_IMPLEMENTATION_SLOT",
    "EIP1967_ADMIN_SLOT",
    "EIP1967_BEACON_SLOT",
    "DANGEROUS_SELECTORS",
    "DANGEROUS_OPCODES",
    "FUNCTION_SELECTORS",
]

__version__ = "1.0.0"
