"""
DeFiGuard AI - On-Chain Analysis Module

Comprehensive on-chain analysis for smart contract security:
- Proxy pattern detection (EIP-1967, UUPS, Transparent, Beacon, Diamond, Minimal)
- Storage slot analysis (owner, admin, paused state)
- Backdoor scanning (20+ dangerous patterns)
- Honeypot detection
- Token security analysis (ERC-20/721/1155)
- Live state monitoring
- Transaction history analysis (Enterprise)
- Event log analysis (Enterprise)

Usage:
    from onchain_analyzer import OnChainAnalyzer

    analyzer = OnChainAnalyzer(web3_provider_url)
    result = await analyzer.analyze(contract_address)
"""

from .core import OnChainAnalyzer, MultiChainAnalyzer
from .proxy_detector import ProxyDetector
from .storage_analyzer import StorageAnalyzer
from .backdoor_scanner import BackdoorScanner, HoneypotDetector
from .token_analyzer import TokenAnalyzer
from .state_checker import StateChecker, ContractMonitor
from .transaction_analyzer import TransactionAnalyzer
from .event_analyzer import EventAnalyzer
from .constants import (
    CHAIN_CONFIG,
    DEFAULT_CHAIN_ID,
    EIP1967_IMPLEMENTATION_SLOT,
    EIP1967_ADMIN_SLOT,
    EIP1967_BEACON_SLOT,
    DANGEROUS_SELECTORS,
    DANGEROUS_OPCODES,
    FUNCTION_SELECTORS,
    ERC20_ABI,
)

__all__ = [
    # Main analyzer
    "OnChainAnalyzer",
    "MultiChainAnalyzer",

    # Component analyzers (Pro)
    "ProxyDetector",
    "StorageAnalyzer",
    "BackdoorScanner",
    "HoneypotDetector",
    "TokenAnalyzer",
    "StateChecker",
    "ContractMonitor",

    # Enterprise analyzers
    "TransactionAnalyzer",
    "EventAnalyzer",

    # Constants
    "CHAIN_CONFIG",
    "DEFAULT_CHAIN_ID",
    "EIP1967_IMPLEMENTATION_SLOT",
    "EIP1967_ADMIN_SLOT",
    "EIP1967_BEACON_SLOT",
    "DANGEROUS_SELECTORS",
    "DANGEROUS_OPCODES",
    "FUNCTION_SELECTORS",
    "ERC20_ABI",
]

__version__ = "1.2.0"
