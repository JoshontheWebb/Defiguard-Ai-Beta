"""
DeFiGuard AI - Transaction Analyzer
Analyzes transaction history for suspicious patterns.
Enterprise feature.
"""

import logging
import os
import asyncio
from typing import Any, Optional
from datetime import datetime, timedelta
import aiohttp
from web3 import Web3

from .constants import (
    CHAIN_CONFIG,
    DEFAULT_CHAIN_ID,
    ETHERSCAN_API_DELAY,
    ETHERSCAN_MAX_RETRIES,
    ETHERSCAN_RETRY_DELAY,
)

logger = logging.getLogger(__name__)


class TransactionAnalyzer:
    """
    Analyze transaction history for suspicious patterns.
    Enterprise feature - requires Etherscan API key.

    Detects:
    - Large value transfers
    - Ownership transfers
    - Contract upgrades
    - Suspicious function calls
    - Rug pull patterns
    - Flash loan attacks
    """

    def __init__(self, w3: Web3, chain_id: int = DEFAULT_CHAIN_ID):
        """
        Initialize transaction analyzer.

        Args:
            w3: Web3 instance
            chain_id: Chain ID for explorer API
        """
        self.w3 = w3
        self.chain_id = chain_id
        self.chain_config = CHAIN_CONFIG.get(chain_id, CHAIN_CONFIG[DEFAULT_CHAIN_ID])
        self.api_key = os.getenv("ETHERSCAN_API_KEY", "")

    async def analyze(
        self,
        address: str,
        days: int = 30,
        limit: int = 100
    ) -> dict[str, Any]:
        """
        Analyze transaction history for an address.

        Args:
            address: Contract or wallet address
            days: Number of days to analyze
            limit: Maximum transactions to analyze

        Returns:
            {
                "address": str,
                "chain": str,
                "period_days": int,
                "total_transactions": int,
                "transaction_summary": dict,
                "suspicious_transactions": list[dict],
                "patterns": list[dict],
                "risk_indicators": list[dict],
                "risk_score": int,
            }
        """
        address = self.w3.to_checksum_address(address)

        result = {
            "address": address,
            "chain": self.chain_config.get("name", "Unknown"),
            "period_days": days,
            "total_transactions": 0,
            "transaction_summary": {},
            "suspicious_transactions": [],
            "patterns": [],
            "risk_indicators": [],
            "risk_score": 0
        }

        try:
            # Fetch transactions from explorer API
            transactions = await self._fetch_transactions(address, days, limit)

            if not transactions:
                result["error"] = "No transactions found or API unavailable"
                return result

            result["total_transactions"] = len(transactions)

            # Analyze transaction patterns
            result["transaction_summary"] = self._summarize_transactions(transactions)
            result["suspicious_transactions"] = self._find_suspicious_transactions(transactions, address)
            result["patterns"] = self._detect_patterns(transactions, address)
            result["risk_indicators"] = self._identify_risk_indicators(result)

            # Calculate risk score
            result["risk_score"] = self._calculate_risk_score(result)

            return result

        except Exception as e:
            logger.error(f"Transaction analysis failed for {address}: {e}")
            result["error"] = str(e)
            return result

    async def _fetch_transactions(
        self,
        address: str,
        days: int,
        limit: int
    ) -> list[dict]:
        """
        Fetch transactions from block explorer API.

        Args:
            address: Address to fetch transactions for
            days: Number of days
            limit: Maximum transactions

        Returns:
            List of transaction dicts
        """
        if not self.api_key:
            logger.warning("No Etherscan API key - using limited data")
            return await self._fetch_from_rpc(address, limit)

        try:
            # Calculate start block (approximate)
            current_block = self.w3.eth.block_number
            blocks_per_day = 7200  # ~12 second blocks
            start_block = max(0, current_block - (days * blocks_per_day))

            # Build API URL
            explorer_api = self.chain_config.get("explorer_api", "https://api.etherscan.io/api")
            url = (
                f"{explorer_api}?module=account&action=txlist"
                f"&address={address}"
                f"&startblock={start_block}"
                f"&endblock={current_block}"
                f"&page=1&offset={limit}"
                f"&sort=desc"
                f"&apikey={self.api_key}"
            )

            # Rate limiting with retry logic
            for attempt in range(ETHERSCAN_MAX_RETRIES):
                try:
                    # Add delay between API calls for rate limiting
                    await asyncio.sleep(ETHERSCAN_API_DELAY)

                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, timeout=30) as response:
                            if response.status == 200:
                                data = await response.json()
                                if data.get("status") == "1":
                                    return data.get("result", [])
                                elif data.get("message") == "NOTOK" and "rate limit" in data.get("result", "").lower():
                                    # Rate limited, wait and retry
                                    logger.warning(f"Rate limited, retry {attempt + 1}/{ETHERSCAN_MAX_RETRIES}")
                                    await asyncio.sleep(ETHERSCAN_RETRY_DELAY * (attempt + 1))
                                    continue
                                else:
                                    logger.warning(f"Explorer API error: {data.get('message')}")
                                    return []
                            elif response.status == 429:  # Too Many Requests
                                logger.warning(f"Rate limited (429), retry {attempt + 1}/{ETHERSCAN_MAX_RETRIES}")
                                await asyncio.sleep(ETHERSCAN_RETRY_DELAY * (attempt + 1))
                                continue
                            else:
                                logger.warning(f"Explorer API HTTP error: {response.status}")
                                return []
                except asyncio.TimeoutError:
                    logger.warning(f"API timeout, retry {attempt + 1}/{ETHERSCAN_MAX_RETRIES}")
                    await asyncio.sleep(ETHERSCAN_RETRY_DELAY)
                    continue

            return []

        except Exception as e:
            logger.error(f"Failed to fetch transactions: {e}")
            return []

    async def _fetch_from_rpc(self, address: str, limit: int) -> list[dict]:
        """
        Fallback: Fetch recent transactions from RPC.

        Args:
            address: Address
            limit: Max transactions

        Returns:
            List of transaction dicts (limited info)
        """
        # This is a simplified fallback - RPC doesn't provide easy tx history
        # In production, you'd use a proper indexer
        transactions = []

        try:
            current_block = self.w3.eth.block_number

            # Check last 100 blocks for transactions involving this address
            for block_num in range(current_block, max(0, current_block - 100), -1):
                if len(transactions) >= limit:
                    break

                try:
                    block = self.w3.eth.get_block(block_num, full_transactions=True)
                    for tx in block.transactions:
                        if tx.get("to") == address or tx.get("from") == address:
                            transactions.append({
                                "hash": tx.get("hash", b"").hex() if isinstance(tx.get("hash"), bytes) else tx.get("hash"),
                                "from": tx.get("from", ""),
                                "to": tx.get("to", ""),
                                "value": str(tx.get("value", 0)),
                                "input": tx.get("input", "0x"),
                                "blockNumber": str(block_num),
                                "timeStamp": str(block.timestamp),
                                "gasUsed": str(tx.get("gas", 0)),
                            })
                except Exception:
                    continue

        except Exception as e:
            logger.warning(f"RPC transaction fetch failed: {e}")

        return transactions

    def _summarize_transactions(self, transactions: list[dict]) -> dict[str, Any]:
        """
        Summarize transaction activity.

        Args:
            transactions: List of transactions

        Returns:
            Summary dict
        """
        summary = {
            "total_count": len(transactions),
            "incoming_count": 0,
            "outgoing_count": 0,
            "contract_calls": 0,
            "total_value_in": 0,
            "total_value_out": 0,
            "unique_senders": set(),
            "unique_receivers": set(),
            "first_tx_time": None,
            "last_tx_time": None,
            "avg_value": 0,
            "high_value_count": 0,  # > 1 ETH
        }

        if not transactions:
            return summary

        address = transactions[0].get("to", "").lower()  # Assuming first tx target is our address

        for tx in transactions:
            value_wei = int(tx.get("value", 0))
            value_eth = value_wei / 1e18

            tx_from = tx.get("from", "").lower()
            tx_to = tx.get("to", "").lower()
            tx_input = tx.get("input", "0x")

            # Determine direction
            if tx_to == address:
                summary["incoming_count"] += 1
                summary["total_value_in"] += value_eth
                summary["unique_senders"].add(tx_from)
            else:
                summary["outgoing_count"] += 1
                summary["total_value_out"] += value_eth
                summary["unique_receivers"].add(tx_to)

            # Contract call if has input data
            if tx_input and tx_input != "0x" and len(tx_input) > 10:
                summary["contract_calls"] += 1

            # High value
            if value_eth > 1:
                summary["high_value_count"] += 1

            # Timestamps
            timestamp = int(tx.get("timeStamp", 0))
            if timestamp:
                tx_time = datetime.fromtimestamp(timestamp)
                if summary["first_tx_time"] is None or tx_time < summary["first_tx_time"]:
                    summary["first_tx_time"] = tx_time
                if summary["last_tx_time"] is None or tx_time > summary["last_tx_time"]:
                    summary["last_tx_time"] = tx_time

        # Calculate averages
        total_value = summary["total_value_in"] + summary["total_value_out"]
        summary["avg_value"] = total_value / len(transactions) if transactions else 0

        # Convert sets to counts
        summary["unique_senders"] = len(summary["unique_senders"])
        summary["unique_receivers"] = len(summary["unique_receivers"])

        # Format timestamps
        if summary["first_tx_time"]:
            summary["first_tx_time"] = summary["first_tx_time"].isoformat()
        if summary["last_tx_time"]:
            summary["last_tx_time"] = summary["last_tx_time"].isoformat()

        return summary

    def _find_suspicious_transactions(
        self,
        transactions: list[dict],
        address: str
    ) -> list[dict]:
        """
        Identify suspicious transactions.

        Args:
            transactions: List of transactions
            address: Contract address

        Returns:
            List of suspicious transactions with reasons
        """
        suspicious = []
        address = address.lower()

        # Known suspicious function selectors
        suspicious_selectors = {
            "f2fde38b": "transferOwnership",
            "715018a6": "renounceOwnership",
            "3659cfe6": "upgradeTo",
            "4f1ef286": "upgradeToAndCall",
            "8f283970": "changeAdmin",
            "3ccfd60b": "withdraw",
            "51cff8d9": "withdrawAll",
            "40c10f19": "mint",
            "42966c68": "burn",
        }

        for tx in transactions:
            reasons = []
            tx_input = tx.get("input", "0x")
            value_wei = int(tx.get("value", 0))
            value_eth = value_wei / 1e18

            # Check for suspicious function calls
            if tx_input and len(tx_input) >= 10:
                selector = tx_input[2:10].lower()
                if selector in suspicious_selectors:
                    reasons.append(f"Called {suspicious_selectors[selector]}()")

            # Large value transfer
            if value_eth > 10:
                reasons.append(f"Large value transfer: {value_eth:.2f} ETH")

            # Contract creation (to is empty)
            if not tx.get("to"):
                reasons.append("Contract creation transaction")

            # Failed transaction with high gas
            if tx.get("isError") == "1":
                gas_used = int(tx.get("gasUsed", 0))
                if gas_used > 100000:
                    reasons.append(f"Failed transaction with high gas ({gas_used})")

            if reasons:
                suspicious.append({
                    "hash": tx.get("hash", ""),
                    "from": tx.get("from", ""),
                    "to": tx.get("to", ""),
                    "value_eth": value_eth,
                    "timestamp": tx.get("timeStamp", ""),
                    "reasons": reasons,
                    "function": suspicious_selectors.get(tx_input[2:10].lower() if len(tx_input) >= 10 else "", "unknown")
                })

        return suspicious

    def _detect_patterns(self, transactions: list[dict], address: str) -> list[dict]:
        """
        Detect transaction patterns.

        Args:
            transactions: List of transactions
            address: Contract address

        Returns:
            List of detected patterns
        """
        patterns = []
        address = address.lower()

        if len(transactions) < 5:
            return patterns

        # Pattern 1: Rapid successive transactions
        timestamps = sorted([int(tx.get("timeStamp", 0)) for tx in transactions if tx.get("timeStamp")])
        if len(timestamps) >= 2:
            time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            rapid_txs = sum(1 for diff in time_diffs if diff < 60)  # < 1 minute apart

            if rapid_txs > 5:
                patterns.append({
                    "type": "RAPID_TRANSACTIONS",
                    "severity": "MEDIUM",
                    "description": f"{rapid_txs} transactions within 1 minute of each other",
                    "risk": "May indicate bot activity or attack"
                })

        # Pattern 2: Large withdrawals after deposits
        values_by_direction = {"in": [], "out": []}
        for tx in transactions:
            tx_to = tx.get("to", "").lower()
            value = int(tx.get("value", 0)) / 1e18
            if value > 0:
                if tx_to == address:
                    values_by_direction["in"].append(value)
                else:
                    values_by_direction["out"].append(value)

        total_in = sum(values_by_direction["in"])
        total_out = sum(values_by_direction["out"])

        if total_in > 1 and total_out > total_in * 0.9:
            patterns.append({
                "type": "LARGE_WITHDRAWAL",
                "severity": "HIGH",
                "description": f"Large outflow detected: {total_out:.2f} ETH out vs {total_in:.2f} ETH in",
                "risk": "Potential rug pull or fund extraction"
            })

        # Pattern 3: Single address dominance
        senders = {}
        for tx in transactions:
            sender = tx.get("from", "").lower()
            senders[sender] = senders.get(sender, 0) + 1

        if senders:
            max_sender = max(senders.values())
            if max_sender > len(transactions) * 0.7:
                patterns.append({
                    "type": "SINGLE_SENDER_DOMINANCE",
                    "severity": "MEDIUM",
                    "description": f"One address sent {max_sender}/{len(transactions)} transactions",
                    "risk": "Centralized control or wash trading"
                })

        # Pattern 4: Ownership changes
        ownership_changes = sum(
            1 for tx in transactions
            if tx.get("input", "")[:10].lower() in ["0xf2fde38b", "0x715018a6"]
        )
        if ownership_changes > 0:
            patterns.append({
                "type": "OWNERSHIP_CHANGES",
                "severity": "HIGH",
                "description": f"{ownership_changes} ownership change(s) detected",
                "risk": "Contract control may have changed"
            })

        return patterns

    def _identify_risk_indicators(self, result: dict) -> list[dict]:
        """
        Identify overall risk indicators from analysis.

        Args:
            result: Analysis result

        Returns:
            List of risk indicators
        """
        indicators = []
        summary = result.get("transaction_summary", {})
        suspicious = result.get("suspicious_transactions", [])
        patterns = result.get("patterns", [])

        # High number of suspicious transactions
        if len(suspicious) > 5:
            indicators.append({
                "type": "HIGH_SUSPICIOUS_TX_COUNT",
                "severity": "HIGH",
                "description": f"{len(suspicious)} suspicious transactions detected"
            })

        # Recent ownership change
        for tx in suspicious:
            if "transferOwnership" in tx.get("function", "") or "renounceOwnership" in tx.get("function", ""):
                indicators.append({
                    "type": "RECENT_OWNERSHIP_CHANGE",
                    "severity": "CRITICAL",
                    "description": "Ownership was recently transferred or renounced"
                })
                break

        # Net negative value (more out than in)
        value_in = summary.get("total_value_in", 0)
        value_out = summary.get("total_value_out", 0)
        if value_out > value_in * 1.5 and value_out > 10:
            indicators.append({
                "type": "NET_OUTFLOW",
                "severity": "HIGH",
                "description": f"Net outflow: {value_out - value_in:.2f} ETH"
            })

        # Very few unique senders (potential wash trading)
        if summary.get("unique_senders", 0) < 3 and summary.get("total_count", 0) > 20:
            indicators.append({
                "type": "LOW_SENDER_DIVERSITY",
                "severity": "MEDIUM",
                "description": "Very few unique transaction senders"
            })

        # Add pattern-based indicators
        for pattern in patterns:
            if pattern.get("severity") in ["HIGH", "CRITICAL"]:
                indicators.append({
                    "type": pattern.get("type"),
                    "severity": pattern.get("severity"),
                    "description": pattern.get("description")
                })

        return indicators

    def _calculate_risk_score(self, result: dict) -> int:
        """
        Calculate overall transaction risk score.

        Args:
            result: Analysis result

        Returns:
            Risk score 0-100
        """
        score = 0

        # Score from suspicious transactions
        suspicious_count = len(result.get("suspicious_transactions", []))
        score += min(suspicious_count * 5, 30)

        # Score from patterns
        for pattern in result.get("patterns", []):
            severity = pattern.get("severity", "LOW")
            if severity == "CRITICAL":
                score += 25
            elif severity == "HIGH":
                score += 15
            elif severity == "MEDIUM":
                score += 8
            else:
                score += 3

        # Score from risk indicators
        for indicator in result.get("risk_indicators", []):
            severity = indicator.get("severity", "LOW")
            if severity == "CRITICAL":
                score += 20
            elif severity == "HIGH":
                score += 12
            elif severity == "MEDIUM":
                score += 6

        return min(score, 100)

    def get_summary(self, result: dict) -> str:
        """
        Generate human-readable transaction analysis summary.

        Args:
            result: Analysis result

        Returns:
            Summary string
        """
        parts = []

        total = result.get("total_transactions", 0)
        suspicious = len(result.get("suspicious_transactions", []))
        risk_score = result.get("risk_score", 0)

        parts.append(f"Analyzed {total} transactions")

        if suspicious > 0:
            parts.append(f"{suspicious} suspicious transaction(s)")

        indicators = result.get("risk_indicators", [])
        critical = [i for i in indicators if i.get("severity") == "CRITICAL"]
        high = [i for i in indicators if i.get("severity") == "HIGH"]

        if critical:
            parts.append(f"⚠️ {len(critical)} CRITICAL indicator(s)")
        if high:
            parts.append(f"🔴 {len(high)} HIGH risk indicator(s)")

        parts.append(f"Risk Score: {risk_score}/100")

        return " | ".join(parts)
