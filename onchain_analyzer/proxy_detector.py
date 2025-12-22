"""
DeFiGuard AI - Proxy Pattern Detector
Detects all major proxy patterns: EIP-1967, UUPS, Transparent, Beacon, Diamond, Minimal.
"""

import logging
from typing import Any, Optional
from web3 import Web3

from .constants import (
    EIP1967_IMPLEMENTATION_SLOT,
    EIP1967_ADMIN_SLOT,
    EIP1967_BEACON_SLOT,
    MINIMAL_PROXY_PREFIX,
    MINIMAL_PROXY_SUFFIX,
    BEACON_ABI,
    DIAMOND_LOUPE_ABI,
    FUNCTION_SELECTORS,
)

logger = logging.getLogger(__name__)


class ProxyDetector:
    """
    Comprehensive proxy pattern detection for Ethereum smart contracts.

    Supports:
    - EIP-1967 Standard Proxy Storage Slots
    - UUPS (Universal Upgradeable Proxy Standard, EIP-1822)
    - Transparent Proxy (OpenZeppelin)
    - Beacon Proxy
    - Diamond Pattern (EIP-2535)
    - Minimal Proxy / Clones (EIP-1167)
    """

    def __init__(self, w3: Web3):
        """
        Initialize proxy detector.

        Args:
            w3: Web3 instance connected to an Ethereum node
        """
        self.w3 = w3

    async def detect(self, address: str) -> dict[str, Any]:
        """
        Detect proxy pattern for a contract address.

        Args:
            address: Contract address to analyze

        Returns:
            dict with proxy detection results:
            {
                "is_proxy": bool,
                "proxy_type": str | None,  # "EIP-1967", "UUPS", "Transparent", "Beacon", "Diamond", "Minimal", "Unknown"
                "implementation": str | None,  # Implementation contract address
                "admin": str | None,  # Admin address (for Transparent proxies)
                "beacon": str | None,  # Beacon address (for Beacon proxies)
                "is_upgradeable": bool,
                "upgrade_risk": str,  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
                "details": dict,  # Additional pattern-specific details
            }
        """
        address = self.w3.to_checksum_address(address)

        result = {
            "is_proxy": False,
            "proxy_type": None,
            "implementation": None,
            "admin": None,
            "beacon": None,
            "is_upgradeable": False,
            "upgrade_risk": "LOW",
            "details": {}
        }

        try:
            # Get contract bytecode
            bytecode = self.w3.eth.get_code(address).hex()

            if len(bytecode) <= 2:  # "0x" or empty
                result["details"]["error"] = "No bytecode found (EOA or self-destructed)"
                return result

            # Check patterns in order of likelihood

            # 1. Check EIP-1167 Minimal Proxy first (most specific bytecode pattern)
            minimal_result = self._check_minimal_proxy(bytecode)
            if minimal_result["is_proxy"]:
                result.update(minimal_result)
                return result

            # 2. Check EIP-1967 storage slots
            eip1967_result = await self._check_eip1967(address)

            # 3. Check Beacon Proxy
            beacon_result = await self._check_beacon_proxy(address)

            # 4. Check Diamond Pattern
            diamond_result = await self._check_diamond(address)

            # Determine proxy type based on findings
            if eip1967_result["implementation"]:
                result["is_proxy"] = True
                result["implementation"] = eip1967_result["implementation"]
                result["is_upgradeable"] = True

                if eip1967_result["admin"]:
                    # Has admin slot = Transparent Proxy
                    result["proxy_type"] = "Transparent"
                    result["admin"] = eip1967_result["admin"]
                    result["upgrade_risk"] = "MEDIUM"
                    result["details"]["pattern"] = "OpenZeppelin Transparent Proxy"
                elif self._has_uups_functions(bytecode):
                    # Has proxiableUUID = UUPS
                    result["proxy_type"] = "UUPS"
                    result["upgrade_risk"] = "MEDIUM"
                    result["details"]["pattern"] = "UUPS (EIP-1822)"
                else:
                    # Generic EIP-1967
                    result["proxy_type"] = "EIP-1967"
                    result["upgrade_risk"] = "MEDIUM"
                    result["details"]["pattern"] = "EIP-1967 Standard Proxy"

            elif beacon_result["is_beacon"]:
                result["is_proxy"] = True
                result["proxy_type"] = "Beacon"
                result["beacon"] = beacon_result["beacon"]
                result["implementation"] = beacon_result["implementation"]
                result["is_upgradeable"] = True
                result["upgrade_risk"] = "HIGH"  # All beacon proxies upgrade together
                result["details"]["pattern"] = "Beacon Proxy"
                result["details"]["beacon_address"] = beacon_result["beacon"]

            elif diamond_result["is_diamond"]:
                result["is_proxy"] = True
                result["proxy_type"] = "Diamond"
                result["is_upgradeable"] = True
                result["upgrade_risk"] = "HIGH"  # Complex upgrade mechanics
                result["details"]["pattern"] = "Diamond Pattern (EIP-2535)"
                result["details"]["facet_count"] = diamond_result["facet_count"]
                result["details"]["facets"] = diamond_result["facets"]

            # Add bytecode size info
            result["details"]["bytecode_size"] = len(bytecode) // 2 - 1  # Convert hex to bytes, minus "0x"

            return result

        except Exception as e:
            logger.error(f"Proxy detection failed for {address}: {e}")
            result["details"]["error"] = str(e)
            return result

    async def _check_eip1967(self, address: str) -> dict[str, Optional[str]]:
        """
        Check EIP-1967 standard storage slots.

        Returns:
            {
                "implementation": str | None,
                "admin": str | None,
                "beacon": str | None
            }
        """
        result = {
            "implementation": None,
            "admin": None,
            "beacon": None
        }

        try:
            # Read implementation slot
            impl_data = self.w3.eth.get_storage_at(address, EIP1967_IMPLEMENTATION_SLOT)
            impl_addr = self._extract_address(impl_data)
            if impl_addr and self._is_valid_contract(impl_addr):
                result["implementation"] = impl_addr

            # Read admin slot
            admin_data = self.w3.eth.get_storage_at(address, EIP1967_ADMIN_SLOT)
            admin_addr = self._extract_address(admin_data)
            if admin_addr:
                result["admin"] = admin_addr

            # Read beacon slot
            beacon_data = self.w3.eth.get_storage_at(address, EIP1967_BEACON_SLOT)
            beacon_addr = self._extract_address(beacon_data)
            if beacon_addr and self._is_valid_contract(beacon_addr):
                result["beacon"] = beacon_addr

        except Exception as e:
            logger.debug(f"EIP-1967 check failed: {e}")

        return result

    async def _check_beacon_proxy(self, address: str) -> dict[str, Any]:
        """
        Check if contract is a Beacon Proxy and get beacon implementation.

        Returns:
            {
                "is_beacon": bool,
                "beacon": str | None,
                "implementation": str | None
            }
        """
        result = {
            "is_beacon": False,
            "beacon": None,
            "implementation": None
        }

        try:
            # Check beacon slot from EIP-1967
            beacon_data = self.w3.eth.get_storage_at(address, EIP1967_BEACON_SLOT)
            beacon_addr = self._extract_address(beacon_data)

            if not beacon_addr or not self._is_valid_contract(beacon_addr):
                return result

            # Try to call implementation() on beacon
            try:
                beacon_contract = self.w3.eth.contract(
                    address=self.w3.to_checksum_address(beacon_addr),
                    abi=BEACON_ABI
                )
                implementation = beacon_contract.functions.implementation().call()

                result["is_beacon"] = True
                result["beacon"] = beacon_addr
                result["implementation"] = implementation

            except Exception as e:
                logger.debug(f"Beacon implementation call failed: {e}")

        except Exception as e:
            logger.debug(f"Beacon proxy check failed: {e}")

        return result

    async def _check_diamond(self, address: str) -> dict[str, Any]:
        """
        Check if contract implements Diamond Pattern (EIP-2535).

        Returns:
            {
                "is_diamond": bool,
                "facet_count": int,
                "facets": list[dict]  # [{address, selectors}]
            }
        """
        result = {
            "is_diamond": False,
            "facet_count": 0,
            "facets": []
        }

        try:
            # Try to call facets() from Diamond Loupe
            contract = self.w3.eth.contract(
                address=self.w3.to_checksum_address(address),
                abi=DIAMOND_LOUPE_ABI
            )

            facets = contract.functions.facets().call()

            if facets and len(facets) > 0:
                result["is_diamond"] = True
                result["facet_count"] = len(facets)

                for facet in facets:
                    facet_info = {
                        "address": facet[0],
                        "selector_count": len(facet[1]),
                        "selectors": [sel.hex() if isinstance(sel, bytes) else sel for sel in facet[1][:5]]  # First 5 only
                    }
                    result["facets"].append(facet_info)

        except Exception as e:
            logger.debug(f"Diamond check failed (expected for non-diamond contracts): {e}")

        return result

    def _check_minimal_proxy(self, bytecode: str) -> dict[str, Any]:
        """
        Check if bytecode matches EIP-1167 Minimal Proxy pattern.

        Args:
            bytecode: Contract bytecode as hex string

        Returns:
            {
                "is_proxy": bool,
                "proxy_type": str | None,
                "implementation": str | None,
                "is_upgradeable": bool,
                "upgrade_risk": str,
                "details": dict
            }
        """
        result = {
            "is_proxy": False,
            "proxy_type": None,
            "implementation": None,
            "is_upgradeable": False,
            "upgrade_risk": "LOW",
            "details": {}
        }

        # Remove 0x prefix if present
        bytecode = bytecode.lower().replace("0x", "")

        # Check for EIP-1167 pattern
        if MINIMAL_PROXY_PREFIX.lower() in bytecode and MINIMAL_PROXY_SUFFIX.lower() in bytecode:
            try:
                prefix_pos = bytecode.index(MINIMAL_PROXY_PREFIX.lower())
                addr_start = prefix_pos + len(MINIMAL_PROXY_PREFIX)
                addr_end = addr_start + 40  # 20 bytes = 40 hex chars

                implementation = "0x" + bytecode[addr_start:addr_end]

                if self.w3.is_address(implementation):
                    result["is_proxy"] = True
                    result["proxy_type"] = "Minimal"
                    result["implementation"] = self.w3.to_checksum_address(implementation)
                    result["is_upgradeable"] = False  # Clones are immutable
                    result["upgrade_risk"] = "LOW"
                    result["details"]["pattern"] = "EIP-1167 Minimal Proxy (Clone)"
                    result["details"]["immutable"] = True
                    result["details"]["bytecode_size"] = len(bytecode) // 2

            except (ValueError, IndexError) as e:
                logger.debug(f"Minimal proxy address extraction failed: {e}")

        return result

    def _has_uups_functions(self, bytecode: str) -> bool:
        """
        Check if bytecode contains UUPS-specific function signatures.

        Args:
            bytecode: Contract bytecode as hex string

        Returns:
            True if UUPS pattern detected
        """
        bytecode = bytecode.lower()

        # proxiableUUID selector
        return FUNCTION_SELECTORS["proxiableUUID"] in bytecode

    def _extract_address(self, storage_data: bytes) -> Optional[str]:
        """
        Extract address from 32-byte storage slot data.

        Args:
            storage_data: 32 bytes from storage slot

        Returns:
            Checksummed address or None if zero address
        """
        try:
            # Address is in the last 20 bytes
            addr_bytes = storage_data[-20:]
            addr = "0x" + addr_bytes.hex()

            # Check if zero address
            if addr == "0x" + "0" * 40:
                return None

            return self.w3.to_checksum_address(addr)

        except Exception:
            return None

    def _is_valid_contract(self, address: str) -> bool:
        """
        Check if address has contract code (not EOA or empty).

        Args:
            address: Address to check

        Returns:
            True if address has code
        """
        try:
            code = self.w3.eth.get_code(self.w3.to_checksum_address(address))
            return len(code) > 2  # More than just "0x"
        except Exception:
            return False

    def get_upgrade_risk_description(self, risk_level: str) -> str:
        """
        Get human-readable description of upgrade risk.

        Args:
            risk_level: "LOW", "MEDIUM", "HIGH", or "CRITICAL"

        Returns:
            Risk description string
        """
        descriptions = {
            "LOW": "Contract is immutable or has no upgrade mechanism",
            "MEDIUM": "Contract can be upgraded by admin. Standard proxy pattern with typical protections.",
            "HIGH": "Contract has complex upgrade mechanics or shared upgrade authority (Beacon/Diamond). Higher risk of coordinated attacks.",
            "CRITICAL": "Contract has dangerous upgrade capabilities with minimal protections."
        }
        return descriptions.get(risk_level, "Unknown risk level")
