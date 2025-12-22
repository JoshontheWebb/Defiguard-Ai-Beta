"""
DeFiGuard AI - Token Analyzer
Analyzes ERC-20/ERC-721/ERC-1155 tokens for security issues.
"""

import logging
from typing import Any, Optional
from web3 import Web3

from .constants import (
    ERC20_ABI,
    FUNCTION_SELECTORS,
    DANGEROUS_SELECTORS,
)

logger = logging.getLogger(__name__)


class TokenAnalyzer:
    """
    Analyze token contracts for security issues.

    Detects:
    - Token standard compliance (ERC-20, ERC-721, ERC-1155)
    - Supply manipulation risks
    - Transfer restrictions
    - Fee mechanisms
    - Owner privileges
    - Unusual token economics
    """

    def __init__(self, w3: Web3):
        """
        Initialize token analyzer.

        Args:
            w3: Web3 instance connected to an Ethereum node
        """
        self.w3 = w3

    async def analyze(self, address: str) -> dict[str, Any]:
        """
        Perform comprehensive token analysis.

        Args:
            address: Token contract address

        Returns:
            {
                "is_token": bool,
                "token_type": str,  # "ERC20", "ERC721", "ERC1155", "Unknown"
                "name": str,
                "symbol": str,
                "decimals": int,
                "total_supply": str,
                "supply_info": dict,  # Max supply, circulating, etc.
                "owner_privileges": list,  # Owner-only functions
                "transfer_restrictions": list,
                "fee_mechanisms": list,
                "risks": list[dict],  # Detected risks
                "risk_score": int,  # 0-100
            }
        """
        address = self.w3.to_checksum_address(address)

        result = {
            "is_token": False,
            "token_type": "Unknown",
            "name": None,
            "symbol": None,
            "decimals": None,
            "total_supply": None,
            "supply_info": {},
            "owner_privileges": [],
            "transfer_restrictions": [],
            "fee_mechanisms": [],
            "risks": [],
            "risk_score": 0
        }

        try:
            # Get bytecode for analysis
            bytecode = self.w3.eth.get_code(address).hex().lower()

            if len(bytecode) <= 2:
                return result

            bytecode = bytecode.replace("0x", "")

            # Detect token type
            result["token_type"] = self._detect_token_type(bytecode)
            result["is_token"] = result["token_type"] != "Unknown"

            if not result["is_token"]:
                return result

            # Get basic token info (ERC-20)
            if result["token_type"] in ["ERC20", "ERC20_EXTENDED"]:
                await self._get_erc20_info(address, result)

            # Analyze for risks
            self._analyze_supply_risks(bytecode, result)
            self._analyze_transfer_restrictions(bytecode, result)
            self._analyze_fee_mechanisms(bytecode, result)
            self._analyze_owner_privileges(bytecode, result)

            # Calculate risk score
            result["risk_score"] = self._calculate_risk_score(result)

            return result

        except Exception as e:
            logger.error(f"Token analysis failed for {address}: {e}")
            result["error"] = str(e)
            return result

    def _detect_token_type(self, bytecode: str) -> str:
        """
        Detect token standard from bytecode.

        Args:
            bytecode: Contract bytecode as hex string

        Returns:
            Token type: "ERC20", "ERC721", "ERC1155", "Unknown"
        """
        # ERC-20 signatures
        erc20_sigs = [
            "18160ddd",  # totalSupply()
            "70a08231",  # balanceOf(address)
            "a9059cbb",  # transfer(address,uint256)
            "dd62ed3e",  # allowance(address,address)
            "095ea7b3",  # approve(address,uint256)
            "23b872dd",  # transferFrom(address,address,uint256)
        ]

        # ERC-721 signatures
        erc721_sigs = [
            "6352211e",  # ownerOf(uint256)
            "b88d4fde",  # safeTransferFrom(address,address,uint256,bytes)
            "42842e0e",  # safeTransferFrom(address,address,uint256)
            "081812fc",  # getApproved(uint256)
            "e985e9c5",  # isApprovedForAll(address,address)
        ]

        # ERC-1155 signatures
        erc1155_sigs = [
            "00fdd58e",  # balanceOf(address,uint256)
            "4e1273f4",  # balanceOfBatch(address[],uint256[])
            "2eb2c2d6",  # safeBatchTransferFrom
            "f242432a",  # safeTransferFrom (1155 version)
        ]

        # Count matches
        erc20_count = sum(1 for sig in erc20_sigs if sig in bytecode)
        erc721_count = sum(1 for sig in erc721_sigs if sig in bytecode)
        erc1155_count = sum(1 for sig in erc1155_sigs if sig in bytecode)

        # Determine type based on signature count
        if erc1155_count >= 3:
            return "ERC1155"
        elif erc721_count >= 3:
            return "ERC721"
        elif erc20_count >= 4:
            # Check for extended ERC-20 (with mint/burn)
            has_mint = "40c10f19" in bytecode  # mint(address,uint256)
            has_burn = "42966c68" in bytecode or "9dc29fac" in bytecode  # burn variants
            if has_mint or has_burn:
                return "ERC20_EXTENDED"
            return "ERC20"

        return "Unknown"

    async def _get_erc20_info(self, address: str, result: dict) -> None:
        """
        Get ERC-20 token information.

        Args:
            address: Token address
            result: Result dict to update
        """
        try:
            contract = self.w3.eth.contract(address=address, abi=ERC20_ABI)

            # Name
            try:
                result["name"] = contract.functions.name().call()
            except Exception:
                result["name"] = "Unknown"

            # Symbol
            try:
                result["symbol"] = contract.functions.symbol().call()
            except Exception:
                result["symbol"] = "???"

            # Decimals
            try:
                result["decimals"] = contract.functions.decimals().call()
            except Exception:
                result["decimals"] = 18

            # Total supply
            try:
                raw_supply = contract.functions.totalSupply().call()
                decimals = result["decimals"] or 18
                result["total_supply"] = str(raw_supply / (10 ** decimals))
                result["supply_info"]["raw_total_supply"] = str(raw_supply)
            except Exception:
                result["total_supply"] = "Unknown"

        except Exception as e:
            logger.warning(f"Failed to get ERC-20 info for {address}: {e}")

    def _analyze_supply_risks(self, bytecode: str, result: dict) -> None:
        """
        Analyze supply manipulation risks.

        Args:
            bytecode: Contract bytecode
            result: Result dict to update
        """
        risks = []

        # Unlimited mint capability
        mint_selectors = ["40c10f19", "a0712d68", "4e6ec247"]
        has_mint = any(sel in bytecode for sel in mint_selectors)

        if has_mint:
            # Check if there's a max supply cap
            # Look for common patterns like maxSupply, cap, MAX_SUPPLY
            has_cap = "6d1c76c" in bytecode or "cap()" in bytecode or "355274ea" in bytecode

            if not has_cap:
                risks.append({
                    "type": "UNLIMITED_MINT",
                    "severity": "HIGH",
                    "description": "Contract has mint function without apparent supply cap",
                    "impact": "Owner can mint unlimited tokens, diluting existing holders"
                })
                result["owner_privileges"].append("Mint tokens without cap")
            else:
                risks.append({
                    "type": "MINTABLE",
                    "severity": "MEDIUM",
                    "description": "Contract has mint function (with supply cap)",
                    "impact": "Owner can mint tokens up to the cap"
                })
                result["owner_privileges"].append("Mint tokens (capped)")

        # Burn from any address
        burn_from_selector = "79cc6790"  # burnFrom(address,uint256)
        if burn_from_selector in bytecode:
            risks.append({
                "type": "BURN_FROM",
                "severity": "MEDIUM",
                "description": "Contract allows burning tokens from any address (with approval)",
                "impact": "Tokens can be burned from wallets with approval"
            })

        result["risks"].extend(risks)

    def _analyze_transfer_restrictions(self, bytecode: str, result: dict) -> None:
        """
        Analyze transfer restrictions (honeypot indicators).

        Args:
            bytecode: Contract bytecode
            result: Result dict to update
        """
        restrictions = []
        risks = []

        # Blacklist functionality
        blacklist_selectors = ["f9f92be4", "0e136b19", "44337ea1"]
        if any(sel in bytecode for sel in blacklist_selectors):
            restrictions.append("Blacklist functionality")
            risks.append({
                "type": "BLACKLIST",
                "severity": "HIGH",
                "description": "Contract can blacklist addresses from transferring",
                "impact": "Owner can block specific wallets from selling"
            })
            result["owner_privileges"].append("Blacklist addresses")

        # Whitelist functionality
        whitelist_selectors = ["9b19251a", "e43252d7"]
        if any(sel in bytecode for sel in whitelist_selectors):
            restrictions.append("Whitelist functionality")
            risks.append({
                "type": "WHITELIST",
                "severity": "MEDIUM",
                "description": "Contract restricts transfers to whitelisted addresses",
                "impact": "Only approved addresses may transfer"
            })
            result["owner_privileges"].append("Manage whitelist")

        # Trading enable/disable
        trading_selectors = ["8f70ccf7", "a9e75723", "c9567bf9"]
        if any(sel in bytecode for sel in trading_selectors):
            restrictions.append("Trading toggle")
            risks.append({
                "type": "TRADING_CONTROL",
                "severity": "CRITICAL",
                "description": "Owner can enable/disable trading",
                "impact": "Owner can prevent all sells (honeypot risk)"
            })
            result["owner_privileges"].append("Enable/disable trading")

        # Max transaction limit
        max_tx_selectors = ["e517f2b9", "8c0b5e22"]
        if any(sel in bytecode for sel in max_tx_selectors):
            restrictions.append("Max transaction limit")
            risks.append({
                "type": "MAX_TX_LIMIT",
                "severity": "LOW",
                "description": "Contract has maximum transaction amount limit",
                "impact": "Large transactions may be blocked"
            })

        # Max wallet limit
        max_wallet_selectors = ["f7739b5f", "49bd5a5e"]
        if any(sel in bytecode for sel in max_wallet_selectors):
            restrictions.append("Max wallet limit")
            risks.append({
                "type": "MAX_WALLET_LIMIT",
                "severity": "LOW",
                "description": "Contract has maximum wallet holding limit",
                "impact": "Wallets cannot hold above a certain amount"
            })

        result["transfer_restrictions"] = restrictions
        result["risks"].extend(risks)

    def _analyze_fee_mechanisms(self, bytecode: str, result: dict) -> None:
        """
        Analyze fee mechanisms (rug pull indicators).

        Args:
            bytecode: Contract bytecode
            result: Result dict to update
        """
        fees = []
        risks = []

        # Buy fee setter
        buy_fee_selectors = ["b8c61130", "c0246668"]
        if any(sel in bytecode for sel in buy_fee_selectors):
            fees.append("Adjustable buy fee")
            risks.append({
                "type": "ADJUSTABLE_BUY_FEE",
                "severity": "MEDIUM",
                "description": "Owner can adjust buy fee",
                "impact": "Buy fees could be increased"
            })
            result["owner_privileges"].append("Set buy fee")

        # Sell fee setter
        sell_fee_selectors = ["af465a27", "8ee88c53"]
        if any(sel in bytecode for sel in sell_fee_selectors):
            fees.append("Adjustable sell fee")
            risks.append({
                "type": "ADJUSTABLE_SELL_FEE",
                "severity": "HIGH",
                "description": "Owner can adjust sell fee",
                "impact": "Sell fees could be increased to 100% (rug pull)"
            })
            result["owner_privileges"].append("Set sell fee")

        # Fee exemption
        fee_exempt_selectors = ["31c2d847", "b62496f5"]
        if any(sel in bytecode for sel in fee_exempt_selectors):
            fees.append("Fee exemption list")
            result["owner_privileges"].append("Exempt addresses from fees")

        # Tax distribution
        tax_selectors = ["a457c2d7", "52f7c988"]
        if any(sel in bytecode for sel in tax_selectors):
            fees.append("Tax distribution mechanism")

        result["fee_mechanisms"] = fees
        result["risks"].extend(risks)

    def _analyze_owner_privileges(self, bytecode: str, result: dict) -> None:
        """
        Analyze additional owner privileges.

        Args:
            bytecode: Contract bytecode
            result: Result dict to update
        """
        risks = []

        # Pause functionality
        pause_selector = "8456cb59"
        if pause_selector in bytecode:
            result["owner_privileges"].append("Pause contract")
            risks.append({
                "type": "PAUSABLE",
                "severity": "MEDIUM",
                "description": "Owner can pause all transfers",
                "impact": "Trading can be halted at any time"
            })

        # Renounce ownership check
        renounce_selector = "715018a6"
        has_renounce = renounce_selector in bytecode

        # If has owner privileges but no renounce
        if result["owner_privileges"] and not has_renounce:
            risks.append({
                "type": "NO_RENOUNCE",
                "severity": "LOW",
                "description": "Contract may not have renounceOwnership function",
                "impact": "Owner cannot permanently give up control"
            })

        # Arbitrary token recovery
        recover_selectors = ["5a3b7e68", "00ae3bf8"]
        if any(sel in bytecode for sel in recover_selectors):
            result["owner_privileges"].append("Recover tokens from contract")
            risks.append({
                "type": "TOKEN_RECOVERY",
                "severity": "LOW",
                "description": "Owner can recover tokens sent to contract",
                "impact": "Generally positive - can rescue stuck tokens"
            })

        result["risks"].extend(risks)

    def _calculate_risk_score(self, result: dict) -> int:
        """
        Calculate overall token risk score.

        Args:
            result: Analysis result

        Returns:
            Risk score 0-100
        """
        score = 0

        for risk in result["risks"]:
            severity = risk.get("severity", "LOW")
            if severity == "CRITICAL":
                score += 30
            elif severity == "HIGH":
                score += 20
            elif severity == "MEDIUM":
                score += 10
            else:
                score += 3

        # Additional penalties
        if len(result["owner_privileges"]) > 5:
            score += 10  # Too many owner privileges

        if len(result["transfer_restrictions"]) > 2:
            score += 15  # Many transfer restrictions (honeypot risk)

        # Cap at 100
        return min(score, 100)

    def get_token_summary(self, result: dict) -> str:
        """
        Generate human-readable token summary.

        Args:
            result: Analysis result

        Returns:
            Summary string
        """
        if not result.get("is_token"):
            return "Not a recognized token contract."

        parts = []

        # Basic info
        name = result.get("name") or "Unknown"
        symbol = result.get("symbol") or "???"
        token_type = result.get("token_type", "Unknown")
        parts.append(f"{name} ({symbol}) - {token_type}")

        # Supply
        if result.get("total_supply"):
            parts.append(f"Total Supply: {result['total_supply']}")

        # Risks
        critical = [r for r in result.get("risks", []) if r.get("severity") == "CRITICAL"]
        high = [r for r in result.get("risks", []) if r.get("severity") == "HIGH"]

        if critical:
            parts.append(f"⚠️ {len(critical)} CRITICAL risk(s): " +
                        ", ".join(r.get("type", "Unknown") for r in critical))

        if high:
            parts.append(f"🔴 {len(high)} HIGH risk(s)")

        # Owner privileges
        if result.get("owner_privileges"):
            parts.append(f"Owner can: {', '.join(result['owner_privileges'][:3])}")

        return " | ".join(parts)
