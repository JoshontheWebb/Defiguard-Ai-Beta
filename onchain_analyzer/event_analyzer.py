"""
DeFiGuard AI - Event Analyzer
Analyzes contract event logs for security insights.
Enterprise feature.
"""

import logging
import os
import asyncio
from typing import Any, Optional
from datetime import datetime
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


# Common event signatures (keccak256 hashes)
EVENT_SIGNATURES = {
    # ERC-20 Events
    "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef": {
        "name": "Transfer",
        "signature": "Transfer(address,address,uint256)",
        "params": ["from", "to", "value"]
    },
    "0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925": {
        "name": "Approval",
        "signature": "Approval(address,address,uint256)",
        "params": ["owner", "spender", "value"]
    },

    # Ownership Events
    "0x8be0079c531659141344cd1fd0a4f28419497f9722a3daafe3b4186f6b6457e0": {
        "name": "OwnershipTransferred",
        "signature": "OwnershipTransferred(address,address)",
        "params": ["previousOwner", "newOwner"]
    },
    "0x38d16b8cac22d99fc7c124b9cd0de2d3fa1faef420bfe791d8c362d765e22700": {
        "name": "OwnershipTransferStarted",
        "signature": "OwnershipTransferStarted(address,address)",
        "params": ["previousOwner", "newOwner"]
    },

    # Proxy Events
    "0xbc7cd75a20ee27fd9adebab32041f755214dbc6bffa90cc0225b39da2e5c2d3b": {
        "name": "Upgraded",
        "signature": "Upgraded(address)",
        "params": ["implementation"]
    },
    "0x7e644d79422f17c01e4894b5f4f588d331ebfa28653d42ae832dc59e38c9798f": {
        "name": "AdminChanged",
        "signature": "AdminChanged(address,address)",
        "params": ["previousAdmin", "newAdmin"]
    },
    "0x1cf3b03a6cf19fa2baba4df148e9dcabedea7f8a5c07840e207e5c089be95d3e": {
        "name": "BeaconUpgraded",
        "signature": "BeaconUpgraded(address)",
        "params": ["beacon"]
    },

    # Pausable Events
    "0x62e78cea01bee320cd4e420270b5ea74000d11b0c9f74754ebdbfc544b05a258": {
        "name": "Paused",
        "signature": "Paused(address)",
        "params": ["account"]
    },
    "0x5db9ee0a495bf2e6ff9c91a7834c1ba4fdd244a5e8aa4e537bd38aeae4b073aa": {
        "name": "Unpaused",
        "signature": "Unpaused(address)",
        "params": ["account"]
    },

    # DEX Events
    "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822": {
        "name": "Swap",
        "signature": "Swap(address,uint256,uint256,uint256,uint256,address)",
        "params": ["sender", "amount0In", "amount1In", "amount0Out", "amount1Out", "to"]
    },
    "0x4c209b5fc8ad50758f13e2e1088ba56a560dff690a1c6fef26394f4c03821c4f": {
        "name": "Mint",
        "signature": "Mint(address,uint256,uint256)",
        "params": ["sender", "amount0", "amount1"]
    },
    "0xdccd412f0b1252819cb1fd330b93224ca42612892bb3f4f789976e6d81936496": {
        "name": "Burn",
        "signature": "Burn(address,uint256,uint256,address)",
        "params": ["sender", "amount0", "amount1", "to"]
    },
}


class EventAnalyzer:
    """
    Analyze contract event logs for security insights.
    Enterprise feature - requires Etherscan API key.

    Analyzes:
    - Ownership changes
    - Proxy upgrades
    - Pause/unpause events
    - Large transfers
    - Liquidity events
    - Custom events
    """

    def __init__(self, w3: Web3, chain_id: int = DEFAULT_CHAIN_ID):
        """
        Initialize event analyzer.

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
        limit: int = 1000
    ) -> dict[str, Any]:
        """
        Analyze event logs for an address.

        Args:
            address: Contract address
            days: Number of days to analyze
            limit: Maximum events to analyze

        Returns:
            {
                "address": str,
                "chain": str,
                "period_days": int,
                "total_events": int,
                "event_breakdown": dict,  # Count by event type
                "critical_events": list[dict],  # Important events
                "ownership_timeline": list[dict],
                "upgrade_history": list[dict],
                "large_transfers": list[dict],
                "risk_events": list[dict],
                "risk_score": int,
            }
        """
        address = self.w3.to_checksum_address(address)

        result = {
            "address": address,
            "chain": self.chain_config.get("name", "Unknown"),
            "period_days": days,
            "total_events": 0,
            "event_breakdown": {},
            "critical_events": [],
            "ownership_timeline": [],
            "upgrade_history": [],
            "large_transfers": [],
            "risk_events": [],
            "risk_score": 0
        }

        try:
            # Fetch event logs
            logs = await self._fetch_logs(address, days, limit)

            if not logs:
                result["error"] = "No events found or API unavailable"
                return result

            result["total_events"] = len(logs)

            # Decode and categorize events
            decoded_events = self._decode_events(logs)

            # Build event breakdown
            result["event_breakdown"] = self._build_breakdown(decoded_events)

            # Extract critical events
            result["critical_events"] = self._extract_critical_events(decoded_events)

            # Build ownership timeline
            result["ownership_timeline"] = self._build_ownership_timeline(decoded_events)

            # Build upgrade history
            result["upgrade_history"] = self._build_upgrade_history(decoded_events)

            # Find large transfers
            result["large_transfers"] = self._find_large_transfers(decoded_events)

            # Identify risk events
            result["risk_events"] = self._identify_risk_events(decoded_events, result)

            # Calculate risk score
            result["risk_score"] = self._calculate_risk_score(result)

            return result

        except Exception as e:
            logger.error(f"Event analysis failed for {address}: {e}")
            result["error"] = str(e)
            return result

    async def _fetch_logs(
        self,
        address: str,
        days: int,
        limit: int
    ) -> list[dict]:
        """
        Fetch event logs from block explorer API.

        Args:
            address: Contract address
            days: Number of days
            limit: Maximum events

        Returns:
            List of event log dicts
        """
        if not self.api_key:
            logger.warning("No Etherscan API key - using RPC fallback")
            return await self._fetch_logs_rpc(address, days, limit)

        try:
            # Calculate start block
            current_block = self.w3.eth.block_number
            blocks_per_day = 7200
            start_block = max(0, current_block - (days * blocks_per_day))

            # Build API URL for getLogs
            explorer_api = self.chain_config.get("explorer_api", "https://api.etherscan.io/api")
            url = (
                f"{explorer_api}?module=logs&action=getLogs"
                f"&address={address}"
                f"&fromBlock={start_block}"
                f"&toBlock={current_block}"
                f"&page=1&offset={limit}"
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
            logger.error(f"Failed to fetch logs: {e}")
            return []

    async def _fetch_logs_rpc(
        self,
        address: str,
        days: int,
        limit: int
    ) -> list[dict]:
        """
        Fallback: Fetch logs via RPC.

        Args:
            address: Contract address
            days: Number of days
            limit: Maximum events

        Returns:
            List of log dicts
        """
        try:
            current_block = self.w3.eth.block_number
            blocks_per_day = 7200
            start_block = max(0, current_block - (days * blocks_per_day))

            # RPC get_logs has limits, so we chunk
            logs = []
            chunk_size = 2000
            from_block = start_block

            while from_block < current_block and len(logs) < limit:
                to_block = min(from_block + chunk_size, current_block)

                try:
                    chunk_logs = self.w3.eth.get_logs({
                        "address": address,
                        "fromBlock": from_block,
                        "toBlock": to_block
                    })

                    for log in chunk_logs:
                        logs.append({
                            "address": log.get("address", ""),
                            "topics": [t.hex() if isinstance(t, bytes) else t for t in log.get("topics", [])],
                            "data": log.get("data", "0x").hex() if isinstance(log.get("data"), bytes) else log.get("data", "0x"),
                            "blockNumber": hex(log.get("blockNumber", 0)),
                            "transactionHash": log.get("transactionHash", b"").hex() if isinstance(log.get("transactionHash"), bytes) else log.get("transactionHash", ""),
                            "logIndex": hex(log.get("logIndex", 0)),
                            "timeStamp": str(self.w3.eth.get_block(log.get("blockNumber", 0)).timestamp)
                        })

                        if len(logs) >= limit:
                            break

                except Exception as e:
                    logger.warning(f"RPC log fetch error for blocks {from_block}-{to_block}: {e}")

                from_block = to_block + 1

            return logs

        except Exception as e:
            logger.error(f"RPC log fetch failed: {e}")
            return []

    def _decode_events(self, logs: list[dict]) -> list[dict]:
        """
        Decode event logs using known signatures.

        Args:
            logs: Raw log dicts

        Returns:
            List of decoded event dicts
        """
        decoded = []

        for log in logs:
            topics = log.get("topics", [])
            if not topics:
                continue

            event_sig = topics[0] if topics else None
            event_info = EVENT_SIGNATURES.get(event_sig)

            event = {
                "raw": log,
                "block_number": int(log.get("blockNumber", "0x0"), 16),
                "transaction_hash": log.get("transactionHash", ""),
                "timestamp": log.get("timeStamp", ""),
                "log_index": int(log.get("logIndex", "0x0"), 16),
                "event_signature": event_sig,
                "decoded": False
            }

            if event_info:
                event["decoded"] = True
                event["event_name"] = event_info["name"]
                event["event_type"] = event_info["signature"]
                event["params"] = self._decode_params(topics[1:], log.get("data", "0x"), event_info["params"])
            else:
                event["event_name"] = "Unknown"
                event["event_type"] = event_sig[:10] + "..." if event_sig else "unknown"

            decoded.append(event)

        return decoded

    def _decode_params(
        self,
        indexed_topics: list[str],
        data: str,
        param_names: list[str]
    ) -> dict[str, Any]:
        """
        Decode event parameters.

        Args:
            indexed_topics: Indexed parameter topics
            data: Non-indexed data
            param_names: Parameter names

        Returns:
            Dict of parameter values
        """
        params = {}

        # Decode indexed parameters from topics
        for i, topic in enumerate(indexed_topics):
            if i < len(param_names):
                param_name = param_names[i]
                # Topics are 32 bytes, addresses are 20 bytes (right-padded)
                if param_name in ["from", "to", "owner", "spender", "sender", "previousOwner", "newOwner", "implementation", "previousAdmin", "newAdmin", "beacon", "account"]:
                    # Extract address from last 40 chars
                    params[param_name] = "0x" + topic[-40:]
                else:
                    # Numeric value
                    try:
                        params[param_name] = int(topic, 16)
                    except (ValueError, TypeError):
                        params[param_name] = topic

        # Decode non-indexed data (simplified - assumes 32-byte aligned values)
        if data and data != "0x" and len(data) > 2:
            data_hex = data[2:]  # Remove 0x
            chunks = [data_hex[i:i+64] for i in range(0, len(data_hex), 64)]

            for i, chunk in enumerate(chunks):
                param_idx = len(indexed_topics) + i
                if param_idx < len(param_names):
                    param_name = param_names[param_idx]
                    if param_name in ["value", "amount", "amount0", "amount1", "amount0In", "amount1In", "amount0Out", "amount1Out"]:
                        try:
                            params[param_name] = int(chunk, 16)
                        except (ValueError, TypeError):
                            params[param_name] = chunk

        return params

    def _build_breakdown(self, events: list[dict]) -> dict[str, int]:
        """
        Build event type breakdown.

        Args:
            events: Decoded events

        Returns:
            Dict of event counts
        """
        breakdown = {}
        for event in events:
            name = event.get("event_name", "Unknown")
            breakdown[name] = breakdown.get(name, 0) + 1
        return dict(sorted(breakdown.items(), key=lambda x: x[1], reverse=True))

    def _extract_critical_events(self, events: list[dict]) -> list[dict]:
        """
        Extract critical/important events.

        Args:
            events: Decoded events

        Returns:
            List of critical events
        """
        critical_event_names = [
            "OwnershipTransferred",
            "OwnershipTransferStarted",
            "Upgraded",
            "AdminChanged",
            "BeaconUpgraded",
            "Paused",
            "Unpaused"
        ]

        critical = []
        for event in events:
            if event.get("event_name") in critical_event_names:
                critical.append({
                    "event_name": event.get("event_name"),
                    "block_number": event.get("block_number"),
                    "transaction_hash": event.get("transaction_hash"),
                    "timestamp": event.get("timestamp"),
                    "params": event.get("params", {})
                })

        return sorted(critical, key=lambda x: x.get("block_number", 0), reverse=True)

    def _build_ownership_timeline(self, events: list[dict]) -> list[dict]:
        """
        Build ownership change timeline.

        Args:
            events: Decoded events

        Returns:
            List of ownership changes
        """
        ownership_events = []
        for event in events:
            if event.get("event_name") in ["OwnershipTransferred", "OwnershipTransferStarted"]:
                params = event.get("params", {})
                ownership_events.append({
                    "event": event.get("event_name"),
                    "block_number": event.get("block_number"),
                    "timestamp": event.get("timestamp"),
                    "previous_owner": params.get("previousOwner", "Unknown"),
                    "new_owner": params.get("newOwner", "Unknown"),
                    "transaction_hash": event.get("transaction_hash")
                })

        return sorted(ownership_events, key=lambda x: x.get("block_number", 0))

    def _build_upgrade_history(self, events: list[dict]) -> list[dict]:
        """
        Build proxy upgrade history.

        Args:
            events: Decoded events

        Returns:
            List of upgrades
        """
        upgrades = []
        for event in events:
            if event.get("event_name") in ["Upgraded", "AdminChanged", "BeaconUpgraded"]:
                params = event.get("params", {})
                upgrades.append({
                    "event": event.get("event_name"),
                    "block_number": event.get("block_number"),
                    "timestamp": event.get("timestamp"),
                    "implementation": params.get("implementation"),
                    "previous_admin": params.get("previousAdmin"),
                    "new_admin": params.get("newAdmin"),
                    "beacon": params.get("beacon"),
                    "transaction_hash": event.get("transaction_hash")
                })

        return sorted(upgrades, key=lambda x: x.get("block_number", 0))

    def _find_large_transfers(self, events: list[dict], threshold_eth: float = 10.0) -> list[dict]:
        """
        Find large Transfer events.

        Args:
            events: Decoded events
            threshold_eth: Minimum ETH equivalent to flag

        Returns:
            List of large transfers
        """
        large_transfers = []

        for event in events:
            if event.get("event_name") == "Transfer":
                params = event.get("params", {})
                value = params.get("value", 0)

                # Assuming 18 decimals (ERC-20 standard)
                value_eth = value / 1e18 if value else 0

                if value_eth >= threshold_eth:
                    large_transfers.append({
                        "block_number": event.get("block_number"),
                        "timestamp": event.get("timestamp"),
                        "from": params.get("from", "Unknown"),
                        "to": params.get("to", "Unknown"),
                        "value": value,
                        "value_formatted": f"{value_eth:.4f}",
                        "transaction_hash": event.get("transaction_hash")
                    })

        return sorted(large_transfers, key=lambda x: x.get("value", 0), reverse=True)[:20]

    def _identify_risk_events(self, events: list[dict], result: dict) -> list[dict]:
        """
        Identify risk-indicating events.

        Args:
            events: Decoded events
            result: Partial analysis result

        Returns:
            List of risk events
        """
        risk_events = []

        # Risk: Multiple ownership changes
        ownership_changes = len(result.get("ownership_timeline", []))
        if ownership_changes > 2:
            risk_events.append({
                "type": "MULTIPLE_OWNERSHIP_CHANGES",
                "severity": "HIGH",
                "description": f"{ownership_changes} ownership changes detected",
                "count": ownership_changes
            })

        # Risk: Recent upgrade
        upgrades = result.get("upgrade_history", [])
        if upgrades:
            latest_upgrade = upgrades[-1]
            risk_events.append({
                "type": "CONTRACT_UPGRADED",
                "severity": "MEDIUM",
                "description": f"Contract was upgraded at block {latest_upgrade.get('block_number')}",
                "block_number": latest_upgrade.get("block_number")
            })

        # Risk: Paused event without unpause
        pause_events = [e for e in events if e.get("event_name") == "Paused"]
        unpause_events = [e for e in events if e.get("event_name") == "Unpaused"]

        if len(pause_events) > len(unpause_events):
            risk_events.append({
                "type": "CONTRACT_PAUSED",
                "severity": "HIGH",
                "description": "Contract appears to be in paused state",
                "pause_count": len(pause_events),
                "unpause_count": len(unpause_events)
            })

        # Risk: Very high Transfer count (possible spam/wash)
        transfer_count = result.get("event_breakdown", {}).get("Transfer", 0)
        if transfer_count > 500:
            risk_events.append({
                "type": "HIGH_TRANSFER_VOLUME",
                "severity": "LOW",
                "description": f"{transfer_count} Transfer events (may indicate high activity or wash trading)",
                "count": transfer_count
            })

        # Risk: Large transfers detected
        large_transfers = result.get("large_transfers", [])
        if len(large_transfers) > 5:
            total_value = sum(t.get("value", 0) for t in large_transfers) / 1e18
            risk_events.append({
                "type": "LARGE_TRANSFERS_DETECTED",
                "severity": "MEDIUM",
                "description": f"{len(large_transfers)} large transfers totaling ~{total_value:.2f} tokens",
                "count": len(large_transfers)
            })

        return risk_events

    def _calculate_risk_score(self, result: dict) -> int:
        """
        Calculate event-based risk score.

        Args:
            result: Analysis result

        Returns:
            Risk score 0-100
        """
        score = 0

        # Score from risk events
        for event in result.get("risk_events", []):
            severity = event.get("severity", "LOW")
            if severity == "CRITICAL":
                score += 25
            elif severity == "HIGH":
                score += 15
            elif severity == "MEDIUM":
                score += 8
            else:
                score += 3

        # Score from critical events
        critical = result.get("critical_events", [])
        score += min(len(critical) * 5, 25)

        # Score from ownership changes
        ownership = result.get("ownership_timeline", [])
        if len(ownership) > 3:
            score += 15

        return min(score, 100)

    def get_summary(self, result: dict) -> str:
        """
        Generate human-readable event analysis summary.

        Args:
            result: Analysis result

        Returns:
            Summary string
        """
        parts = []

        total = result.get("total_events", 0)
        parts.append(f"Analyzed {total} events")

        # Key events
        breakdown = result.get("event_breakdown", {})
        transfers = breakdown.get("Transfer", 0)
        if transfers > 0:
            parts.append(f"{transfers} transfers")

        ownership = len(result.get("ownership_timeline", []))
        if ownership > 0:
            parts.append(f"{ownership} ownership change(s)")

        upgrades = len(result.get("upgrade_history", []))
        if upgrades > 0:
            parts.append(f"{upgrades} upgrade(s)")

        # Risk summary
        risk_events = result.get("risk_events", [])
        high_risk = [e for e in risk_events if e.get("severity") in ["HIGH", "CRITICAL"]]
        if high_risk:
            parts.append(f"⚠️ {len(high_risk)} high-risk indicator(s)")

        parts.append(f"Risk Score: {result.get('risk_score', 0)}/100")

        return " | ".join(parts)
