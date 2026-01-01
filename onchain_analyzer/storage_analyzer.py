"""
DeFiGuard AI - Storage Slot Analyzer
Reads contract storage to identify owner, admin, paused state, and other critical values.
"""

import logging
from typing import Any, Optional
from web3 import Web3

from .constants import (
    EIP1967_ADMIN_SLOT,
    OZ_PAUSABLE_SLOT,
    OZ_OWNABLE_SLOT,
    OWNABLE_ABI,
    PAUSABLE_ABI,
    ACCESS_CONTROL_ABI,
    ROLE_IDENTIFIERS,
)

logger = logging.getLogger(__name__)


class StorageAnalyzer:
    """
    Analyze contract storage to extract critical state information.

    Capabilities:
    - Owner/Admin address detection
    - Paused state detection
    - Pending ownership transfers
    - Access control role analysis
    - Balance and ETH holdings
    """

    def __init__(self, w3: Web3):
        """
        Initialize storage analyzer.

        Args:
            w3: Web3 instance connected to an Ethereum node
        """
        self.w3 = w3

    async def analyze(self, address: str) -> dict[str, Any]:
        """
        Perform comprehensive storage analysis.

        Args:
            address: Contract address to analyze

        Returns:
            {
                "owner": str | None,
                "owner_source": str,  # "owner()", "slot_0", "eip1967_admin", etc.
                "admin": str | None,
                "pending_owner": str | None,
                "is_pausable": bool,
                "is_paused": bool,
                "eth_balance": str,  # In ETH
                "eth_balance_wei": int,
                "privileged_addresses": list[dict],
                "access_control": dict | None,
                "storage_slots": dict,  # Raw slot data for advanced analysis
                "centralization_risk": str,  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
            }
        """
        address = self.w3.to_checksum_address(address)

        result = {
            "owner": None,
            "owner_source": None,
            "admin": None,
            "pending_owner": None,
            "is_pausable": False,
            "is_paused": False,
            "eth_balance": "0",
            "eth_balance_wei": 0,
            "privileged_addresses": [],
            "access_control": None,
            "storage_slots": {},
            "centralization_risk": "LOW"
        }

        try:
            # Get ETH balance
            balance_wei = self.w3.eth.get_balance(address)
            result["eth_balance_wei"] = balance_wei
            result["eth_balance"] = str(self.w3.from_wei(balance_wei, "ether"))

            # Detect owner using multiple methods
            owner_result = await self._detect_owner(address)
            result["owner"] = owner_result["owner"]
            result["owner_source"] = owner_result["source"]
            result["pending_owner"] = owner_result.get("pending_owner")

            if result["owner"]:
                result["privileged_addresses"].append({
                    "address": result["owner"],
                    "role": "Owner",
                    "source": result["owner_source"]
                })

            # Check EIP-1967 admin slot
            admin_result = await self._check_admin_slot(address)
            if admin_result:
                result["admin"] = admin_result
                result["privileged_addresses"].append({
                    "address": admin_result,
                    "role": "Proxy Admin",
                    "source": "EIP-1967 Admin Slot"
                })

            # Check pausable state
            pausable_result = await self._check_pausable(address)
            result["is_pausable"] = pausable_result["is_pausable"]
            result["is_paused"] = pausable_result["is_paused"]

            # Check access control roles
            access_result = await self._check_access_control(address)
            if access_result["has_access_control"]:
                result["access_control"] = access_result
                # Add role holders to privileged addresses
                for role_holder in access_result.get("role_holders", []):
                    result["privileged_addresses"].append(role_holder)

            # Read common storage slots for additional analysis
            result["storage_slots"] = await self._read_common_slots(address)

            # Calculate centralization risk
            result["centralization_risk"] = self._calculate_centralization_risk(result)

            return result

        except Exception as e:
            logger.error(f"Storage analysis failed for {address}: {e}")
            result["error"] = str(e)
            return result

    async def _detect_owner(self, address: str) -> dict[str, Any]:
        """
        Detect owner using multiple methods.

        Methods (in order of reliability):
        1. Call owner() function
        2. Check OZ Ownable storage slot
        3. Check slot 0 (common pattern)
        """
        result = {
            "owner": None,
            "source": None,
            "pending_owner": None
        }

        # Method 1: Try calling owner() function
        try:
            contract = self.w3.eth.contract(address=address, abi=OWNABLE_ABI)
            owner = contract.functions.owner().call()

            if owner and owner != "0x" + "0" * 40:
                result["owner"] = owner
                result["source"] = "owner() function"

                # Also try pendingOwner()
                try:
                    pending = contract.functions.pendingOwner().call()
                    if pending and pending != "0x" + "0" * 40:
                        result["pending_owner"] = pending
                except Exception:
                    pass  # pendingOwner() may not exist on all contracts

                return result

        except Exception as e:
            logger.debug(f"owner() call failed: {e}")

        # Method 2: Check OZ Ownable storage slot (ERC-7201 namespaced)
        try:
            ownable_data = self.w3.eth.get_storage_at(address, OZ_OWNABLE_SLOT)
            owner = self._extract_address(ownable_data)

            if owner:
                result["owner"] = owner
                result["source"] = "OpenZeppelin Ownable storage slot"
                return result

        except Exception as e:
            logger.debug(f"OZ Ownable slot check failed: {e}")

        # Method 3: Check slot 0 (traditional pattern)
        try:
            slot_0_data = self.w3.eth.get_storage_at(address, 0)
            potential_owner = self._extract_address(slot_0_data)

            if potential_owner and self._looks_like_owner(potential_owner):
                result["owner"] = potential_owner
                result["source"] = "Storage slot 0 (traditional)"
                return result

        except Exception as e:
            logger.debug(f"Slot 0 check failed: {e}")

        return result

    async def _check_admin_slot(self, address: str) -> Optional[str]:
        """
        Check EIP-1967 admin storage slot.

        Returns:
            Admin address or None
        """
        try:
            admin_data = self.w3.eth.get_storage_at(address, EIP1967_ADMIN_SLOT)
            return self._extract_address(admin_data)
        except Exception as e:
            logger.debug(f"Admin slot check failed: {e}")
            return None

    async def _check_pausable(self, address: str) -> dict[str, bool]:
        """
        Check if contract is pausable and current paused state.

        Returns:
            {
                "is_pausable": bool,
                "is_paused": bool
            }
        """
        result = {
            "is_pausable": False,
            "is_paused": False
        }

        # Method 1: Try calling paused() function
        try:
            contract = self.w3.eth.contract(address=address, abi=PAUSABLE_ABI)
            is_paused = contract.functions.paused().call()

            result["is_pausable"] = True
            result["is_paused"] = bool(is_paused)
            return result

        except Exception as e:
            logger.debug(f"paused() call failed: {e}")

        # Method 2: Check OZ Pausable storage slot
        try:
            pausable_data = self.w3.eth.get_storage_at(address, OZ_PAUSABLE_SLOT)
            # Paused state is stored as bool in first byte
            is_paused = int.from_bytes(pausable_data[:1], "big") == 1

            if pausable_data != b'\x00' * 32:
                result["is_pausable"] = True
                result["is_paused"] = is_paused

        except Exception as e:
            logger.debug(f"Pausable slot check failed: {e}")

        # Method 3: Check bytecode for pause/unpause selectors
        try:
            bytecode = self.w3.eth.get_code(address).hex().lower()
            pause_selector = "8456cb59"  # pause()
            unpause_selector = "3f4ba83a"  # unpause()

            if pause_selector in bytecode or unpause_selector in bytecode:
                result["is_pausable"] = True

        except Exception as e:
            logger.debug(f"Bytecode pause check failed: {e}")

        return result

    async def _check_access_control(self, address: str) -> dict[str, Any]:
        """
        Check for OpenZeppelin AccessControl implementation.

        Returns:
            {
                "has_access_control": bool,
                "roles_detected": list[str],
                "role_holders": list[dict]
            }
        """
        result = {
            "has_access_control": False,
            "roles_detected": [],
            "role_holders": []
        }

        try:
            contract = self.w3.eth.contract(address=address, abi=ACCESS_CONTROL_ABI)

            # Check for DEFAULT_ADMIN_ROLE
            try:
                admin_role = contract.functions.DEFAULT_ADMIN_ROLE().call()
                result["has_access_control"] = True
                result["roles_detected"].append("DEFAULT_ADMIN_ROLE")

            except Exception:
                # Contract doesn't have AccessControl interface
                return result

            # Try to identify role holders by checking common roles
            for role_name, role_hash in ROLE_IDENTIFIERS.items():
                try:
                    # We can't directly get role members without events
                    # But we can check the role admin
                    role_admin = contract.functions.getRoleAdmin(role_hash).call()
                    if role_admin:
                        result["roles_detected"].append(role_name)

                except Exception:
                    pass  # Role may not exist in this contract

        except Exception as e:
            logger.debug(f"Access control check failed: {e}")

        return result

    async def _read_common_slots(self, address: str) -> dict[str, str]:
        """
        Read common storage slots for advanced analysis.

        Returns:
            Dict mapping slot names to hex values
        """
        slots = {}

        # Read first 5 slots (common for simple contracts)
        for i in range(5):
            try:
                data = self.w3.eth.get_storage_at(address, i)
                if data != b'\x00' * 32:
                    slots[f"slot_{i}"] = data.hex()
            except Exception:
                pass  # Storage read may fail for various RPC reasons

        # Read EIP-1967 slots
        try:
            impl_data = self.w3.eth.get_storage_at(address, int(EIP1967_ADMIN_SLOT, 16))
            if impl_data != b'\x00' * 32:
                slots["eip1967_admin"] = impl_data.hex()
        except Exception:
            pass  # EIP-1967 slot may not exist

        return slots

    def _extract_address(self, storage_data: bytes) -> Optional[str]:
        """
        Extract address from 32-byte storage slot data.

        Args:
            storage_data: 32 bytes from storage slot

        Returns:
            Checksummed address or None if zero address
        """
        try:
            addr_bytes = storage_data[-20:]
            addr = "0x" + addr_bytes.hex()

            if addr == "0x" + "0" * 40:
                return None

            return self.w3.to_checksum_address(addr)

        except Exception:
            return None

    def _looks_like_owner(self, address: str) -> bool:
        """
        Heuristic to check if an address looks like it could be an owner.

        Checks:
        - Is not zero address
        - Is not the contract itself
        - Has some transaction history (optional, expensive check skipped)
        """
        if not address:
            return False

        if address == "0x" + "0" * 40:
            return False

        # Could add more checks here:
        # - Check if address has sent transactions
        # - Check if address is a known contract
        # For now, just validate it's a valid address

        return True

    def _calculate_centralization_risk(self, analysis: dict) -> str:
        """
        Calculate overall centralization risk based on analysis results.

        Returns:
            "LOW", "MEDIUM", "HIGH", or "CRITICAL"
        """
        risk_score = 0

        # Owner exists
        if analysis.get("owner"):
            risk_score += 20

        # Separate admin exists
        if analysis.get("admin"):
            risk_score += 15

        # Pending ownership transfer
        if analysis.get("pending_owner"):
            risk_score += 10

        # Contract is pausable
        if analysis.get("is_pausable"):
            risk_score += 15

        # Contract is currently paused
        if analysis.get("is_paused"):
            risk_score += 10

        # Multiple privileged addresses
        privileged_count = len(analysis.get("privileged_addresses", []))
        if privileged_count > 2:
            risk_score += 10
        elif privileged_count == 1:
            risk_score += 5

        # Access control with multiple roles
        if analysis.get("access_control", {}).get("has_access_control"):
            risk_score += 10

        # Significant ETH balance held
        eth_balance = float(analysis.get("eth_balance", 0))
        if eth_balance > 100:
            risk_score += 15
        elif eth_balance > 10:
            risk_score += 10
        elif eth_balance > 1:
            risk_score += 5

        # Convert score to risk level
        if risk_score >= 60:
            return "CRITICAL"
        elif risk_score >= 40:
            return "HIGH"
        elif risk_score >= 20:
            return "MEDIUM"
        else:
            return "LOW"

    def get_centralization_risk_description(self, risk_level: str) -> str:
        """
        Get human-readable description of centralization risk.

        Args:
            risk_level: "LOW", "MEDIUM", "HIGH", or "CRITICAL"

        Returns:
            Risk description string
        """
        descriptions = {
            "LOW": "Minimal centralization. No single point of failure detected.",
            "MEDIUM": "Moderate centralization. Contract has owner/admin but with standard protections.",
            "HIGH": "Significant centralization risk. Multiple privileged roles or pausable with substantial holdings.",
            "CRITICAL": "Severe centralization. Single entity controls critical functions with minimal checks."
        }
        return descriptions.get(risk_level, "Unknown risk level")
