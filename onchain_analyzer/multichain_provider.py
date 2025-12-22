"""
DeFiGuard AI - Multi-Chain Provider
Unified interface for multi-chain on-chain analysis.
"""

import logging
import os
from typing import Any, Optional
from web3 import Web3
import asyncio

from .constants import CHAIN_CONFIG, DEFAULT_CHAIN_ID
from .proxy_detector import ProxyDetector
from .storage_analyzer import StorageAnalyzer
from .backdoor_scanner import BackdoorScanner, HoneypotDetector
from .token_analyzer import TokenAnalyzer
from .state_checker import StateChecker
from .transaction_analyzer import TransactionAnalyzer
from .event_analyzer import EventAnalyzer

logger = logging.getLogger(__name__)


class ChainConnection:
    """
    Represents a connection to a single blockchain.
    """

    def __init__(self, chain_id: int, config: dict):
        """
        Initialize chain connection.

        Args:
            chain_id: Chain ID
            config: Chain configuration dict
        """
        self.chain_id = chain_id
        self.config = config
        self.name = config.get("name", f"Chain {chain_id}")
        self.short_name = config.get("short_name", "???")
        self.w3: Optional[Web3] = None
        self.is_connected = False
        self.last_block = 0
        self.error: Optional[str] = None

        # Initialize Web3
        self._connect()

    def _connect(self) -> bool:
        """
        Establish Web3 connection.

        Returns:
            Success boolean
        """
        try:
            rpc_url = self._get_rpc_url()
            if not rpc_url:
                self.error = "No RPC URL configured"
                return False

            self.w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': 30}))

            if self.w3.is_connected():
                self.is_connected = True
                self.last_block = self.w3.eth.block_number
                logger.info(f"[MULTICHAIN] Connected to {self.name} (block {self.last_block})")
                return True
            else:
                self.error = "Connection failed"
                return False

        except Exception as e:
            self.error = str(e)
            logger.warning(f"[MULTICHAIN] Failed to connect to {self.name}: {e}")
            return False

    def _get_rpc_url(self) -> Optional[str]:
        """
        Get RPC URL from environment variables.

        Returns:
            RPC URL or None
        """
        # First check for direct chain-specific URL
        chain_env = f"{self.short_name}_RPC_URL"
        direct_url = os.getenv(chain_env)
        if direct_url:
            return direct_url

        # Then try template with API key
        key_env = self.config.get("rpc_env", "")
        template = self.config.get("rpc_template", "")

        api_key = os.getenv(key_env, "")
        if api_key and template:
            return template.format(key=api_key)

        # Fallback to generic provider
        generic_key = os.getenv("ALCHEMY_API_KEY") or os.getenv("INFURA_PROJECT_ID")
        if generic_key and template:
            return template.format(key=generic_key)

        return None

    def reconnect(self) -> bool:
        """
        Attempt to reconnect.

        Returns:
            Success boolean
        """
        self.is_connected = False
        self.error = None
        return self._connect()


class MultiChainProvider:
    """
    Unified provider for multi-chain on-chain analysis.

    Supports:
    - Ethereum (Chain ID: 1)
    - Base (Chain ID: 8453)
    - Arbitrum One (Chain ID: 42161)
    - Polygon (Chain ID: 137)
    - Optimism (Chain ID: 10)
    """

    SUPPORTED_CHAINS = [1, 8453, 42161, 137, 10]

    def __init__(self, auto_connect: bool = True):
        """
        Initialize multi-chain provider.

        Args:
            auto_connect: Whether to connect to all chains on init
        """
        self.connections: dict[int, ChainConnection] = {}
        self.analyzers: dict[int, dict] = {}

        if auto_connect:
            self._connect_all()

    def _connect_all(self) -> None:
        """Connect to all supported chains."""
        for chain_id in self.SUPPORTED_CHAINS:
            config = CHAIN_CONFIG.get(chain_id)
            if config:
                conn = ChainConnection(chain_id, config)
                self.connections[chain_id] = conn

                if conn.is_connected and conn.w3:
                    self.analyzers[chain_id] = self._create_analyzers(conn.w3, chain_id)

    def _create_analyzers(self, w3: Web3, chain_id: int) -> dict:
        """
        Create analyzer instances for a chain.

        Args:
            w3: Web3 instance
            chain_id: Chain ID

        Returns:
            Dict of analyzer instances
        """
        return {
            "proxy": ProxyDetector(w3),
            "storage": StorageAnalyzer(w3),
            "backdoor": BackdoorScanner(w3),
            "honeypot": HoneypotDetector(w3),
            "token": TokenAnalyzer(w3),
            "state": StateChecker(w3),
            "transaction": TransactionAnalyzer(w3, chain_id),
            "event": EventAnalyzer(w3, chain_id),
        }

    def get_connection(self, chain_id: int) -> Optional[ChainConnection]:
        """
        Get connection for a specific chain.

        Args:
            chain_id: Chain ID

        Returns:
            ChainConnection or None
        """
        return self.connections.get(chain_id)

    def get_analyzers(self, chain_id: int) -> Optional[dict]:
        """
        Get analyzers for a specific chain.

        Args:
            chain_id: Chain ID

        Returns:
            Dict of analyzers or None
        """
        return self.analyzers.get(chain_id)

    @property
    def connected_chains(self) -> list[dict]:
        """Get list of connected chains."""
        chains = []
        for chain_id, conn in self.connections.items():
            chains.append({
                "chain_id": chain_id,
                "name": conn.name,
                "short_name": conn.short_name,
                "connected": conn.is_connected,
                "last_block": conn.last_block,
                "error": conn.error,
                "explorer_url": conn.config.get("explorer_url", "")
            })
        return chains

    @property
    def active_chain_count(self) -> int:
        """Get number of connected chains."""
        return sum(1 for conn in self.connections.values() if conn.is_connected)

    def chain_name(self, chain_id: int) -> str:
        """Get chain name by ID."""
        conn = self.connections.get(chain_id)
        return conn.name if conn else f"Unknown ({chain_id})"

    async def analyze(
        self,
        address: str,
        chain_id: int = DEFAULT_CHAIN_ID,
        include_enterprise: bool = False,
        tier: str = "pro"
    ) -> dict[str, Any]:
        """
        Analyze a contract on a specific chain.

        Args:
            address: Contract address
            chain_id: Chain ID
            include_enterprise: Include transaction/event analysis
            tier: User tier

        Returns:
            Complete analysis result
        """
        conn = self.get_connection(chain_id)
        if not conn or not conn.is_connected:
            return {
                "error": f"Chain {chain_id} not connected",
                "chain": self.chain_name(chain_id),
                "address": address
            }

        analyzers = self.get_analyzers(chain_id)
        if not analyzers:
            return {
                "error": "Analyzers not initialized",
                "chain": conn.name,
                "address": address
            }

        w3 = conn.w3
        address = w3.to_checksum_address(address)

        result = {
            "address": address,
            "chain": conn.name,
            "chain_id": chain_id,
            "is_contract": False,
            "proxy": {},
            "storage": {},
            "backdoors": {},
            "honeypot": None,
            "token": None,
            "state": None,
            "transactions": None,
            "events": None,
            "overall_risk": {},
        }

        try:
            # Check if contract
            code = w3.eth.get_code(address)
            result["is_contract"] = len(code) > 2

            if not result["is_contract"]:
                result["overall_risk"] = {
                    "score": 0,
                    "level": "N/A",
                    "summary": "Address is an EOA, not a contract"
                }
                return result

            # Run core analyzers in parallel
            proxy_task = analyzers["proxy"].detect(address)
            storage_task = analyzers["storage"].analyze(address)
            backdoor_task = analyzers["backdoor"].scan(address)
            honeypot_task = analyzers["honeypot"].check(address)
            token_task = analyzers["token"].analyze(address)
            state_task = analyzers["state"].check(address)

            results = await asyncio.gather(
                proxy_task,
                storage_task,
                backdoor_task,
                honeypot_task,
                token_task,
                state_task,
                return_exceptions=True
            )

            # Assign results
            result["proxy"] = results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])}
            result["storage"] = results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])}
            result["backdoors"] = results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])}
            result["honeypot"] = results[3] if not isinstance(results[3], Exception) else {"error": str(results[3])}
            result["token"] = results[4] if not isinstance(results[4], Exception) else {"error": str(results[4])}
            result["state"] = results[5] if not isinstance(results[5], Exception) else {"error": str(results[5])}

            # Enterprise analysis
            if include_enterprise and tier == "enterprise":
                tx_task = analyzers["transaction"].analyze(address, days=30)
                event_task = analyzers["event"].analyze(address, days=30)

                enterprise_results = await asyncio.gather(
                    tx_task,
                    event_task,
                    return_exceptions=True
                )

                result["transactions"] = enterprise_results[0] if not isinstance(enterprise_results[0], Exception) else {"error": str(enterprise_results[0])}
                result["events"] = enterprise_results[1] if not isinstance(enterprise_results[1], Exception) else {"error": str(enterprise_results[1])}

            # Calculate overall risk
            result["overall_risk"] = self._calculate_overall_risk(result)

            return result

        except Exception as e:
            logger.error(f"Multi-chain analysis failed for {address} on {conn.name}: {e}")
            result["error"] = str(e)
            return result

    def _calculate_overall_risk(self, result: dict) -> dict[str, Any]:
        """
        Calculate overall risk from all analysis components.

        Args:
            result: Analysis result

        Returns:
            Overall risk assessment
        """
        score = 0
        factors = []

        # Proxy risk
        proxy = result.get("proxy", {})
        if proxy.get("is_proxy"):
            upgrade_risk = proxy.get("upgrade_risk", "LOW")
            if upgrade_risk == "CRITICAL":
                score += 25
                factors.append({"source": "proxy", "risk": "CRITICAL", "description": "Critical upgrade risk"})
            elif upgrade_risk == "HIGH":
                score += 15
                factors.append({"source": "proxy", "risk": "HIGH", "description": "High upgrade risk"})

        # Backdoor risk
        backdoors = result.get("backdoors", {})
        backdoor_level = backdoors.get("risk_level", "LOW")
        if backdoor_level in ["CRITICAL", "HIGH"]:
            score += 25 if backdoor_level == "CRITICAL" else 15
            factors.append({
                "source": "backdoors",
                "risk": backdoor_level,
                "description": backdoors.get("summary", "Backdoor patterns detected")
            })

        # Honeypot risk
        honeypot = result.get("honeypot", {})
        if honeypot and honeypot.get("is_honeypot"):
            confidence = honeypot.get("confidence", "LOW")
            score += 30 if confidence == "HIGH" else 15
            factors.append({
                "source": "honeypot",
                "risk": "CRITICAL" if confidence == "HIGH" else "HIGH",
                "description": "Honeypot indicators detected"
            })

        # Token risk
        token = result.get("token", {})
        if token and token.get("is_token"):
            token_score = token.get("risk_score", 0)
            if token_score >= 50:
                score += min(token_score // 3, 20)
                factors.append({
                    "source": "token",
                    "risk": "HIGH" if token_score >= 70 else "MEDIUM",
                    "description": f"Token risk score: {token_score}/100"
                })

        # State health
        state = result.get("state", {})
        if state:
            health = state.get("health_score", 100)
            if health < 50:
                score += 15
                factors.append({
                    "source": "state",
                    "risk": "HIGH",
                    "description": f"Low health score: {health}/100"
                })

        # Transaction risk (Enterprise)
        transactions = result.get("transactions", {})
        if transactions and not transactions.get("error"):
            tx_score = transactions.get("risk_score", 0)
            if tx_score >= 50:
                score += min(tx_score // 4, 15)
                factors.append({
                    "source": "transactions",
                    "risk": "HIGH" if tx_score >= 70 else "MEDIUM",
                    "description": f"Transaction risk: {tx_score}/100"
                })

        # Event risk (Enterprise)
        events = result.get("events", {})
        if events and not events.get("error"):
            event_score = events.get("risk_score", 0)
            if event_score >= 50:
                score += min(event_score // 4, 15)
                factors.append({
                    "source": "events",
                    "risk": "HIGH" if event_score >= 70 else "MEDIUM",
                    "description": f"Event risk: {event_score}/100"
                })

        # Cap score
        score = min(score, 100)

        # Determine level
        if score >= 70:
            level = "CRITICAL"
        elif score >= 50:
            level = "HIGH"
        elif score >= 25:
            level = "MEDIUM"
        else:
            level = "LOW"

        # Generate summary
        if not factors:
            summary = "No significant risks detected across all chains and analyzers."
        else:
            high_factors = [f for f in factors if f["risk"] in ["HIGH", "CRITICAL"]]
            if high_factors:
                summary = f"Found {len(high_factors)} high-severity risk(s): " + \
                         ", ".join(f["description"] for f in high_factors[:3])
            else:
                summary = f"Found {len(factors)} moderate risk factor(s)."

        return {
            "score": score,
            "level": level,
            "summary": summary,
            "factors": factors
        }

    async def analyze_multi_chain(
        self,
        address: str,
        chain_ids: Optional[list[int]] = None,
        tier: str = "pro"
    ) -> dict[str, Any]:
        """
        Analyze a contract across multiple chains.

        Useful for detecting if the same contract is deployed
        on multiple networks with different security profiles.

        Args:
            address: Contract address
            chain_ids: List of chain IDs to check (default: all connected)
            tier: User tier

        Returns:
            Multi-chain analysis result
        """
        if chain_ids is None:
            chain_ids = [
                chain_id for chain_id, conn in self.connections.items()
                if conn.is_connected
            ]

        result = {
            "address": address,
            "chains_analyzed": [],
            "deployments_found": [],
            "cross_chain_summary": {},
        }

        # Analyze on each chain
        tasks = []
        for chain_id in chain_ids:
            tasks.append(self.analyze(
                address,
                chain_id,
                include_enterprise=(tier == "enterprise"),
                tier=tier
            ))

        chain_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        highest_risk = 0
        deployments = []

        for i, chain_id in enumerate(chain_ids):
            chain_result = chain_results[i]

            if isinstance(chain_result, Exception):
                result["chains_analyzed"].append({
                    "chain_id": chain_id,
                    "chain": self.chain_name(chain_id),
                    "error": str(chain_result)
                })
                continue

            result["chains_analyzed"].append({
                "chain_id": chain_id,
                "chain": chain_result.get("chain", self.chain_name(chain_id)),
                "is_contract": chain_result.get("is_contract", False),
                "risk_score": chain_result.get("overall_risk", {}).get("score", 0),
                "risk_level": chain_result.get("overall_risk", {}).get("level", "N/A"),
            })

            if chain_result.get("is_contract"):
                deployments.append({
                    "chain_id": chain_id,
                    "chain": chain_result.get("chain"),
                    "full_analysis": chain_result
                })

                risk_score = chain_result.get("overall_risk", {}).get("score", 0)
                if risk_score > highest_risk:
                    highest_risk = risk_score

        result["deployments_found"] = deployments

        # Cross-chain summary
        result["cross_chain_summary"] = {
            "total_chains_checked": len(chain_ids),
            "deployments_found": len(deployments),
            "highest_risk_score": highest_risk,
            "deployment_chains": [d["chain"] for d in deployments]
        }

        return result

    def get_explorer_url(self, chain_id: int, address: str) -> str:
        """
        Get block explorer URL for an address.

        Args:
            chain_id: Chain ID
            address: Contract/wallet address

        Returns:
            Explorer URL
        """
        conn = self.connections.get(chain_id)
        if conn:
            base_url = conn.config.get("explorer_url", "https://etherscan.io")
            return f"{base_url}/address/{address}"
        return f"https://etherscan.io/address/{address}"

    def health_check(self) -> dict[str, Any]:
        """
        Perform health check on all connections.

        Returns:
            Health status dict
        """
        status = {
            "healthy": True,
            "chains": [],
            "active_count": 0,
            "total_count": len(self.connections)
        }

        for chain_id, conn in self.connections.items():
            chain_status = {
                "chain_id": chain_id,
                "name": conn.name,
                "connected": conn.is_connected,
                "error": conn.error
            }

            if conn.is_connected:
                try:
                    # Quick connectivity test
                    block = conn.w3.eth.block_number
                    chain_status["last_block"] = block
                    status["active_count"] += 1
                except Exception as e:
                    chain_status["connected"] = False
                    chain_status["error"] = str(e)
                    conn.is_connected = False

            status["chains"].append(chain_status)

        status["healthy"] = status["active_count"] >= 1
        return status
