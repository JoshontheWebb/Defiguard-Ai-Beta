"""
DeFiGuard AI - On-Chain Analysis Core
Main orchestrator that combines all on-chain analyzers.
"""

import logging
import os
from typing import Any, Optional
from web3 import Web3

from .proxy_detector import ProxyDetector
from .storage_analyzer import StorageAnalyzer
from .backdoor_scanner import BackdoorScanner, HoneypotDetector
from .constants import CHAIN_CONFIG, DEFAULT_CHAIN_ID, RISK_WEIGHTS, MAX_ONCHAIN_RISK_ADJUSTMENT

logger = logging.getLogger(__name__)


class OnChainAnalyzer:
    """
    Main on-chain analysis orchestrator for DeFiGuard AI.

    Combines:
    - Proxy pattern detection
    - Storage slot analysis
    - Backdoor scanning
    - Honeypot detection
    - Overall risk scoring

    Tier Access:
    - Pro users: Basic on-chain analysis (+$50/month add-on)
    - Enterprise users: Full on-chain analysis (included)
    """

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        chain_id: int = DEFAULT_CHAIN_ID
    ):
        """
        Initialize the on-chain analyzer.

        Args:
            rpc_url: Web3 RPC provider URL. If None, uses WEB3_PROVIDER_URL env var.
            chain_id: Chain ID for multi-chain support. Default: 1 (Ethereum mainnet)
        """
        self.chain_id = chain_id
        self.chain_config = CHAIN_CONFIG.get(chain_id, CHAIN_CONFIG[DEFAULT_CHAIN_ID])

        # Initialize Web3 connection
        if rpc_url:
            self.rpc_url = rpc_url
        else:
            self.rpc_url = self._get_rpc_url()

        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))

        # Verify connection
        if not self.w3.is_connected():
            logger.warning(f"Web3 not connected to {self.chain_config['name']}")

        # Initialize component analyzers
        self.proxy_detector = ProxyDetector(self.w3)
        self.storage_analyzer = StorageAnalyzer(self.w3)
        self.backdoor_scanner = BackdoorScanner(self.w3)
        self.honeypot_detector = HoneypotDetector(self.w3)

    def _get_rpc_url(self) -> str:
        """
        Get RPC URL from environment variables based on chain config.

        Returns:
            RPC URL string
        """
        # First try direct WEB3_PROVIDER_URL
        direct_url = os.getenv("WEB3_PROVIDER_URL")
        if direct_url:
            return direct_url

        # Otherwise construct from template
        key_env = self.chain_config.get("rpc_env", "INFURA_PROJECT_ID")
        template = self.chain_config.get("rpc_template", "")

        api_key = os.getenv(key_env, "")
        if api_key and template:
            return template.format(key=api_key)

        # Fallback to public RPC (not recommended for production)
        logger.warning("No API key found, using fallback RPC")
        return "https://eth.llamarpc.com"

    @property
    def is_connected(self) -> bool:
        """Check if Web3 is connected."""
        try:
            return self.w3.is_connected()
        except Exception:
            return False

    @property
    def chain_name(self) -> str:
        """Get current chain name."""
        return self.chain_config.get("name", "Unknown")

    async def analyze(
        self,
        address: str,
        include_honeypot: bool = True,
        tier: str = "pro"
    ) -> dict[str, Any]:
        """
        Perform comprehensive on-chain analysis.

        Args:
            address: Contract address to analyze
            include_honeypot: Whether to run honeypot detection
            tier: User tier ("pro" or "enterprise") for feature gating

        Returns:
            {
                "address": str,
                "chain": str,
                "chain_id": int,
                "is_contract": bool,
                "proxy": dict,  # Proxy detection results
                "storage": dict,  # Storage analysis results
                "backdoors": dict,  # Backdoor scan results
                "honeypot": dict | None,  # Honeypot detection (if requested)
                "overall_risk": dict,  # Combined risk assessment
                "ai_context": str,  # Summary for AI prompt injection
            }
        """
        if not self.is_connected:
            return {
                "error": "Web3 not connected",
                "address": address,
                "chain": self.chain_name
            }

        try:
            address = self.w3.to_checksum_address(address)
        except Exception as e:
            return {
                "error": f"Invalid address: {e}",
                "address": address
            }

        result = {
            "address": address,
            "chain": self.chain_name,
            "chain_id": self.chain_id,
            "is_contract": False,
            "proxy": {},
            "storage": {},
            "backdoors": {},
            "honeypot": None,
            "overall_risk": {},
            "ai_context": ""
        }

        try:
            # Check if address is a contract
            code = self.w3.eth.get_code(address)
            result["is_contract"] = len(code) > 2

            if not result["is_contract"]:
                result["overall_risk"] = {
                    "score": 0,
                    "level": "N/A",
                    "summary": "Address is an EOA (Externally Owned Account), not a contract"
                }
                result["ai_context"] = "Address is an EOA, not a smart contract."
                return result

            # Run all analyzers
            result["proxy"] = await self.proxy_detector.detect(address)
            result["storage"] = await self.storage_analyzer.analyze(address)
            result["backdoors"] = await self.backdoor_scanner.scan(address)

            if include_honeypot:
                result["honeypot"] = await self.honeypot_detector.check(address)

            # Calculate overall risk
            result["overall_risk"] = self._calculate_overall_risk(result)

            # Generate AI context summary
            result["ai_context"] = self._generate_ai_context(result)

            return result

        except Exception as e:
            logger.error(f"On-chain analysis failed for {address}: {e}")
            result["error"] = str(e)
            return result

    def _calculate_overall_risk(self, analysis: dict) -> dict[str, Any]:
        """
        Calculate overall risk score from all analysis components.

        Args:
            analysis: Full analysis results

        Returns:
            {
                "score": int,  # 0-100
                "level": str,  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
                "summary": str,
                "factors": list[dict]  # Contributing risk factors
            }
        """
        score = 0
        factors = []

        # Proxy risk contribution
        proxy = analysis.get("proxy", {})
        if proxy.get("is_proxy"):
            upgrade_risk = proxy.get("upgrade_risk", "LOW")
            weight = RISK_WEIGHTS.get(upgrade_risk, 0)
            score += weight
            factors.append({
                "source": "proxy",
                "risk": upgrade_risk,
                "description": f"{proxy.get('proxy_type', 'Unknown')} proxy detected"
            })

        # Storage/centralization risk contribution
        storage = analysis.get("storage", {})
        centralization = storage.get("centralization_risk", "LOW")
        weight = RISK_WEIGHTS.get(centralization, 0)
        score += weight
        if centralization != "LOW":
            factors.append({
                "source": "centralization",
                "risk": centralization,
                "description": f"Centralization risk: {centralization}"
            })

        # Backdoor risk contribution (major factor)
        backdoors = analysis.get("backdoors", {})
        backdoor_level = backdoors.get("risk_level", "LOW")
        weight = RISK_WEIGHTS.get(backdoor_level, 0)
        score += weight * 1.5  # Weight backdoors more heavily
        if backdoor_level != "LOW":
            factors.append({
                "source": "backdoors",
                "risk": backdoor_level,
                "description": backdoors.get("summary", "Dangerous patterns detected")
            })

        # Honeypot risk contribution
        honeypot = analysis.get("honeypot", {})
        if honeypot and honeypot.get("is_honeypot"):
            confidence = honeypot.get("confidence", "LOW")
            if confidence == "HIGH":
                score += 40
                factors.append({
                    "source": "honeypot",
                    "risk": "CRITICAL",
                    "description": "High confidence honeypot detected"
                })
            elif confidence == "MEDIUM":
                score += 25
                factors.append({
                    "source": "honeypot",
                    "risk": "HIGH",
                    "description": "Potential honeypot indicators"
                })

        # Pausable adds risk
        if storage.get("is_pausable"):
            score += 5
            if storage.get("is_paused"):
                score += 10
                factors.append({
                    "source": "paused",
                    "risk": "MEDIUM",
                    "description": "Contract is currently paused"
                })

        # Cap score at 100
        score = min(int(score), 100)

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
            summary = "No significant on-chain risks detected."
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

    def _generate_ai_context(self, analysis: dict) -> str:
        """
        Generate summary context for AI prompt injection.

        This text is added to the AI analysis prompt to provide
        on-chain context for the security assessment.

        Args:
            analysis: Full analysis results

        Returns:
            Context string for AI prompt
        """
        parts = []
        parts.append(f"[ON-CHAIN ANALYSIS - {analysis['chain']}]")

        # Proxy info
        proxy = analysis.get("proxy", {})
        if proxy.get("is_proxy"):
            parts.append(
                f"PROXY: {proxy.get('proxy_type')} proxy detected. "
                f"Implementation: {proxy.get('implementation', 'Unknown')}. "
                f"Upgrade risk: {proxy.get('upgrade_risk')}."
            )
        else:
            parts.append("PROXY: Not a proxy contract.")

        # Storage/ownership info
        storage = analysis.get("storage", {})
        if storage.get("owner"):
            parts.append(
                f"OWNERSHIP: Owner at {storage['owner'][:10]}...{storage['owner'][-4:]} "
                f"(source: {storage.get('owner_source', 'unknown')}). "
                f"Centralization: {storage.get('centralization_risk')}."
            )
        if storage.get("is_pausable"):
            paused_status = "PAUSED" if storage.get("is_paused") else "active"
            parts.append(f"PAUSABLE: Contract is pausable, currently {paused_status}.")

        # Backdoor summary
        backdoors = analysis.get("backdoors", {})
        if backdoors.get("has_backdoors"):
            parts.append(
                f"BACKDOORS: {backdoors.get('risk_level')} risk. "
                f"{backdoors.get('summary', '')}"
            )
        else:
            parts.append("BACKDOORS: No dangerous patterns detected.")

        # Honeypot summary
        honeypot = analysis.get("honeypot", {})
        if honeypot:
            if honeypot.get("is_honeypot"):
                parts.append(
                    f"HONEYPOT: {honeypot.get('confidence')} confidence. "
                    f"{honeypot.get('recommendation', '')}"
                )
            else:
                parts.append("HONEYPOT: No honeypot indicators detected.")

        # Overall risk
        overall = analysis.get("overall_risk", {})
        parts.append(
            f"OVERALL ON-CHAIN RISK: {overall.get('level', 'UNKNOWN')} "
            f"(score: {overall.get('score', 0)}/100). "
            f"{overall.get('summary', '')}"
        )

        return "\n".join(parts)

    async def get_implementation_code(self, address: str) -> Optional[str]:
        """
        For proxy contracts, get the implementation address.

        Useful for analyzing the actual logic contract.

        Args:
            address: Proxy contract address

        Returns:
            Implementation address or None
        """
        try:
            proxy_result = await self.proxy_detector.detect(address)
            return proxy_result.get("implementation")
        except Exception as e:
            logger.error(f"Failed to get implementation for {address}: {e}")
            return None

    def get_risk_adjustment(self, analysis: dict) -> int:
        """
        Get risk score adjustment for static analysis integration.

        This value adjusts the static analysis risk score based on
        on-chain findings.

        Args:
            analysis: Full analysis results

        Returns:
            Risk adjustment value (-30 to +30)
        """
        overall = analysis.get("overall_risk", {})
        score = overall.get("score", 0)

        # Map on-chain score to adjustment
        if score >= 70:
            return MAX_ONCHAIN_RISK_ADJUSTMENT  # +30
        elif score >= 50:
            return 20
        elif score >= 25:
            return 10
        elif score >= 10:
            return 5
        else:
            return 0  # No adjustment for low risk


class MultiChainAnalyzer:
    """
    Multi-chain on-chain analysis support.

    Allows analyzing contracts across multiple EVM chains.
    Enterprise feature.
    """

    def __init__(self):
        """Initialize multi-chain analyzer with all supported chains."""
        self.analyzers: dict[int, OnChainAnalyzer] = {}

        for chain_id in CHAIN_CONFIG.keys():
            try:
                self.analyzers[chain_id] = OnChainAnalyzer(chain_id=chain_id)
            except Exception as e:
                logger.warning(f"Failed to initialize chain {chain_id}: {e}")

    def get_analyzer(self, chain_id: int) -> Optional[OnChainAnalyzer]:
        """
        Get analyzer for specific chain.

        Args:
            chain_id: Chain ID

        Returns:
            OnChainAnalyzer for the chain or None
        """
        return self.analyzers.get(chain_id)

    async def analyze(
        self,
        address: str,
        chain_id: int = DEFAULT_CHAIN_ID,
        **kwargs
    ) -> dict[str, Any]:
        """
        Analyze contract on specific chain.

        Args:
            address: Contract address
            chain_id: Chain ID
            **kwargs: Additional args passed to analyzer

        Returns:
            Analysis results
        """
        analyzer = self.get_analyzer(chain_id)
        if not analyzer:
            return {
                "error": f"Chain {chain_id} not supported",
                "address": address
            }

        return await analyzer.analyze(address, **kwargs)

    @property
    def supported_chains(self) -> list[dict]:
        """Get list of supported chains with connection status."""
        chains = []
        for chain_id, analyzer in self.analyzers.items():
            config = CHAIN_CONFIG.get(chain_id, {})
            chains.append({
                "chain_id": chain_id,
                "name": config.get("name", "Unknown"),
                "short_name": config.get("short_name", "???"),
                "connected": analyzer.is_connected,
                "explorer_url": config.get("explorer_url", "")
            })
        return chains
