"""
DeFiGuard AI - State Checker
Live contract state monitoring and anomaly detection.
"""

import logging
from typing import Any, Optional
from web3 import Web3
from datetime import datetime

from .constants import (
    ERC20_ABI,
    OWNABLE_ABI,
    PAUSABLE_ABI,
    FUNCTION_SELECTORS,
)

logger = logging.getLogger(__name__)


class StateChecker:
    """
    Monitor live contract state for security anomalies.

    Checks:
    - Contract balance changes
    - Owner/admin changes
    - Pause state
    - Approval state
    - Liquidity state (DEX pairs)
    - Recent transactions
    """

    def __init__(self, w3: Web3):
        """
        Initialize state checker.

        Args:
            w3: Web3 instance connected to an Ethereum node
        """
        self.w3 = w3

    async def check(self, address: str) -> dict[str, Any]:
        """
        Perform comprehensive state check.

        Args:
            address: Contract address to check

        Returns:
            {
                "timestamp": str,  # ISO timestamp
                "block_number": int,
                "eth_balance": str,
                "contract_state": dict,  # Pausable, owner, etc.
                "liquidity_info": dict | None,  # For DEX tokens
                "anomalies": list[dict],  # Detected anomalies
                "health_score": int,  # 0-100 (100 = healthy)
            }
        """
        address = self.w3.to_checksum_address(address)

        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "block_number": 0,
            "eth_balance": "0",
            "contract_state": {},
            "liquidity_info": None,
            "anomalies": [],
            "health_score": 100
        }

        try:
            # Get current block
            result["block_number"] = self.w3.eth.block_number

            # Get ETH balance
            eth_balance = self.w3.eth.get_balance(address)
            result["eth_balance"] = str(self.w3.from_wei(eth_balance, 'ether'))

            # Get contract state
            result["contract_state"] = await self._get_contract_state(address)

            # Check for liquidity (if token)
            result["liquidity_info"] = await self._check_liquidity(address)

            # Detect anomalies
            result["anomalies"] = self._detect_anomalies(result)

            # Calculate health score
            result["health_score"] = self._calculate_health_score(result)

            return result

        except Exception as e:
            logger.error(f"State check failed for {address}: {e}")
            result["error"] = str(e)
            result["health_score"] = 0
            return result

    async def _get_contract_state(self, address: str) -> dict[str, Any]:
        """
        Get contract state information.

        Args:
            address: Contract address

        Returns:
            Contract state dict
        """
        state = {
            "is_pausable": False,
            "is_paused": False,
            "owner": None,
            "pending_owner": None,
            "has_renounced": False,
        }

        try:
            bytecode = self.w3.eth.get_code(address).hex().lower()

            # Check if pausable
            pause_selector = "8456cb59"
            paused_selector = "5c975abb"

            if pause_selector in bytecode or paused_selector in bytecode:
                state["is_pausable"] = True

                # Try to call paused()
                try:
                    contract = self.w3.eth.contract(address=address, abi=PAUSABLE_ABI)
                    state["is_paused"] = contract.functions.paused().call()
                except Exception:
                    pass

            # Try to get owner
            try:
                contract = self.w3.eth.contract(address=address, abi=OWNABLE_ABI)
                owner = contract.functions.owner().call()
                state["owner"] = owner

                # Check if renounced (owner is zero address)
                if owner == "0x0000000000000000000000000000000000000000":
                    state["has_renounced"] = True

            except Exception:
                pass

            # Try to get pending owner (Ownable2Step)
            try:
                pending_selector = "e30c3978"  # pendingOwner()
                if pending_selector in bytecode:
                    contract = self.w3.eth.contract(
                        address=address,
                        abi=[{
                            "inputs": [],
                            "name": "pendingOwner",
                            "outputs": [{"type": "address"}],
                            "stateMutability": "view",
                            "type": "function"
                        }]
                    )
                    pending = contract.functions.pendingOwner().call()
                    if pending != "0x0000000000000000000000000000000000000000":
                        state["pending_owner"] = pending
            except Exception:
                pass

            return state

        except Exception as e:
            logger.warning(f"Failed to get contract state: {e}")
            return state

    async def _check_liquidity(self, address: str) -> Optional[dict[str, Any]]:
        """
        Check liquidity information for token contracts.

        Args:
            address: Token address

        Returns:
            Liquidity info or None
        """
        # Common DEX factory addresses
        uniswap_v2_factory = "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f"
        sushiswap_factory = "0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac"

        # WETH address
        weth = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"

        try:
            # Try to get Uniswap V2 pair
            factory_abi = [{
                "inputs": [{"type": "address"}, {"type": "address"}],
                "name": "getPair",
                "outputs": [{"type": "address"}],
                "stateMutability": "view",
                "type": "function"
            }]

            for factory_addr in [uniswap_v2_factory, sushiswap_factory]:
                try:
                    factory = self.w3.eth.contract(
                        address=self.w3.to_checksum_address(factory_addr),
                        abi=factory_abi
                    )

                    pair_address = factory.functions.getPair(
                        self.w3.to_checksum_address(address),
                        self.w3.to_checksum_address(weth)
                    ).call()

                    if pair_address != "0x0000000000000000000000000000000000000000":
                        # Get pair reserves
                        pair_abi = [{
                            "inputs": [],
                            "name": "getReserves",
                            "outputs": [
                                {"type": "uint112", "name": "reserve0"},
                                {"type": "uint112", "name": "reserve1"},
                                {"type": "uint32", "name": "blockTimestampLast"}
                            ],
                            "stateMutability": "view",
                            "type": "function"
                        }, {
                            "inputs": [],
                            "name": "token0",
                            "outputs": [{"type": "address"}],
                            "stateMutability": "view",
                            "type": "function"
                        }]

                        pair = self.w3.eth.contract(
                            address=self.w3.to_checksum_address(pair_address),
                            abi=pair_abi
                        )

                        reserves = pair.functions.getReserves().call()
                        token0 = pair.functions.token0().call()

                        # Determine which reserve is ETH
                        if token0.lower() == weth.lower():
                            eth_reserve = reserves[0]
                            token_reserve = reserves[1]
                        else:
                            eth_reserve = reserves[1]
                            token_reserve = reserves[0]

                        eth_liquidity = self.w3.from_wei(eth_reserve, 'ether')

                        return {
                            "pair_address": pair_address,
                            "dex": "Uniswap V2" if factory_addr == uniswap_v2_factory else "SushiSwap",
                            "eth_liquidity": str(eth_liquidity),
                            "token_reserve": str(token_reserve),
                            "has_liquidity": float(eth_liquidity) > 0.1
                        }

                except Exception:
                    continue

            return None

        except Exception as e:
            logger.warning(f"Liquidity check failed: {e}")
            return None

    def _detect_anomalies(self, result: dict) -> list[dict]:
        """
        Detect state anomalies.

        Args:
            result: Current state check result

        Returns:
            List of detected anomalies
        """
        anomalies = []
        state = result.get("contract_state", {})
        liquidity = result.get("liquidity_info")

        # Anomaly: Contract is paused
        if state.get("is_paused"):
            anomalies.append({
                "type": "CONTRACT_PAUSED",
                "severity": "HIGH",
                "description": "Contract is currently paused - all transfers blocked"
            })

        # Anomaly: Pending ownership transfer
        if state.get("pending_owner"):
            anomalies.append({
                "type": "PENDING_OWNER_TRANSFER",
                "severity": "MEDIUM",
                "description": f"Ownership transfer pending to {state['pending_owner'][:10]}..."
            })

        # Anomaly: Very low liquidity
        if liquidity and not liquidity.get("has_liquidity"):
            eth_liq = float(liquidity.get("eth_liquidity", 0))
            if eth_liq < 0.1:
                anomalies.append({
                    "type": "LOW_LIQUIDITY",
                    "severity": "HIGH",
                    "description": f"Very low liquidity ({eth_liq:.4f} ETH) - high slippage risk"
                })
            elif eth_liq < 1:
                anomalies.append({
                    "type": "LOW_LIQUIDITY",
                    "severity": "MEDIUM",
                    "description": f"Low liquidity ({eth_liq:.2f} ETH)"
                })

        # Anomaly: Large ETH balance in contract
        eth_balance = float(result.get("eth_balance", 0))
        if eth_balance > 10:
            anomalies.append({
                "type": "HIGH_CONTRACT_BALANCE",
                "severity": "LOW",
                "description": f"Contract holds {eth_balance:.2f} ETH"
            })

        # Anomaly: Ownership not renounced but has critical functions
        if not state.get("has_renounced") and state.get("owner"):
            # This is informational, not necessarily bad
            pass

        return anomalies

    def _calculate_health_score(self, result: dict) -> int:
        """
        Calculate contract health score.

        Args:
            result: State check result

        Returns:
            Health score 0-100 (100 = healthy)
        """
        score = 100

        # Deduct for anomalies
        for anomaly in result.get("anomalies", []):
            severity = anomaly.get("severity", "LOW")
            if severity == "CRITICAL":
                score -= 40
            elif severity == "HIGH":
                score -= 25
            elif severity == "MEDIUM":
                score -= 10
            else:
                score -= 5

        # Deduct if paused
        if result.get("contract_state", {}).get("is_paused"):
            score -= 30

        # Deduct if no liquidity info (for tokens)
        if result.get("liquidity_info") is None:
            score -= 5  # Minor deduction

        # Ensure bounds
        return max(0, min(100, score))

    async def check_approval_risk(
        self,
        token_address: str,
        owner_address: str
    ) -> dict[str, Any]:
        """
        Check approval risks for a specific owner.

        Args:
            token_address: Token contract address
            owner_address: Address to check approvals for

        Returns:
            {
                "unlimited_approvals": list,  # Spenders with unlimited approval
                "high_risk_approvals": list,  # Known risky spenders
                "total_approved": int,  # Number of spenders
            }
        """
        result = {
            "unlimited_approvals": [],
            "high_risk_approvals": [],
            "total_approved": 0
        }

        # This would require event log analysis (Phase 3)
        # For now, return empty result
        logger.info(f"Approval check for {token_address}/{owner_address} - requires event analysis")

        return result


class ContractMonitor:
    """
    Continuous monitoring for contract state changes.
    Enterprise feature.
    """

    def __init__(self, w3: Web3, state_checker: StateChecker):
        """
        Initialize contract monitor.

        Args:
            w3: Web3 instance
            state_checker: StateChecker instance
        """
        self.w3 = w3
        self.state_checker = state_checker
        self.monitored_contracts: dict[str, dict] = {}

    def add_contract(self, address: str, webhook_url: Optional[str] = None) -> bool:
        """
        Add contract to monitoring list.

        Args:
            address: Contract address to monitor
            webhook_url: Optional webhook for alerts

        Returns:
            Success boolean
        """
        try:
            address = self.w3.to_checksum_address(address)
            self.monitored_contracts[address] = {
                "added_at": datetime.utcnow().isoformat(),
                "webhook_url": webhook_url,
                "last_check": None,
                "last_state": None
            }
            logger.info(f"Added {address} to monitoring")
            return True
        except Exception as e:
            logger.error(f"Failed to add contract to monitoring: {e}")
            return False

    def remove_contract(self, address: str) -> bool:
        """
        Remove contract from monitoring list.

        Args:
            address: Contract address

        Returns:
            Success boolean
        """
        try:
            address = self.w3.to_checksum_address(address)
            if address in self.monitored_contracts:
                del self.monitored_contracts[address]
                logger.info(f"Removed {address} from monitoring")
                return True
            return False
        except Exception:
            return False

    async def check_all(self) -> list[dict]:
        """
        Check all monitored contracts for state changes.

        Returns:
            List of alerts for contracts with changes
        """
        alerts = []

        for address, info in self.monitored_contracts.items():
            try:
                current_state = await self.state_checker.check(address)

                # Compare with last state
                last_state = info.get("last_state")
                if last_state:
                    changes = self._detect_changes(last_state, current_state)
                    if changes:
                        alerts.append({
                            "address": address,
                            "timestamp": current_state["timestamp"],
                            "changes": changes,
                            "current_state": current_state
                        })

                # Update last state
                info["last_check"] = datetime.utcnow().isoformat()
                info["last_state"] = current_state

            except Exception as e:
                logger.error(f"Monitor check failed for {address}: {e}")

        return alerts

    def _detect_changes(self, old_state: dict, new_state: dict) -> list[dict]:
        """
        Detect significant state changes.

        Args:
            old_state: Previous state
            new_state: Current state

        Returns:
            List of detected changes
        """
        changes = []

        # Check owner change
        old_owner = old_state.get("contract_state", {}).get("owner")
        new_owner = new_state.get("contract_state", {}).get("owner")
        if old_owner != new_owner:
            changes.append({
                "type": "OWNER_CHANGED",
                "severity": "CRITICAL",
                "old_value": old_owner,
                "new_value": new_owner,
                "description": f"Owner changed from {old_owner} to {new_owner}"
            })

        # Check pause state change
        old_paused = old_state.get("contract_state", {}).get("is_paused", False)
        new_paused = new_state.get("contract_state", {}).get("is_paused", False)
        if old_paused != new_paused:
            changes.append({
                "type": "PAUSE_STATE_CHANGED",
                "severity": "HIGH",
                "old_value": old_paused,
                "new_value": new_paused,
                "description": f"Contract {'paused' if new_paused else 'unpaused'}"
            })

        # Check significant ETH balance change
        old_eth = float(old_state.get("eth_balance", 0))
        new_eth = float(new_state.get("eth_balance", 0))
        if old_eth > 0:
            change_pct = abs(new_eth - old_eth) / old_eth * 100
            if change_pct > 50:
                changes.append({
                    "type": "BALANCE_CHANGE",
                    "severity": "MEDIUM",
                    "old_value": old_eth,
                    "new_value": new_eth,
                    "description": f"ETH balance changed by {change_pct:.1f}%"
                })

        # Check liquidity change
        old_liq = old_state.get("liquidity_info", {})
        new_liq = new_state.get("liquidity_info", {})
        if old_liq and new_liq:
            old_eth_liq = float(old_liq.get("eth_liquidity", 0))
            new_eth_liq = float(new_liq.get("eth_liquidity", 0))
            if old_eth_liq > 0:
                liq_change = (new_eth_liq - old_eth_liq) / old_eth_liq * 100
                if liq_change < -30:  # 30% liquidity decrease
                    changes.append({
                        "type": "LIQUIDITY_DECREASE",
                        "severity": "HIGH",
                        "old_value": old_eth_liq,
                        "new_value": new_eth_liq,
                        "description": f"Liquidity decreased by {abs(liq_change):.1f}%"
                    })

        return changes
