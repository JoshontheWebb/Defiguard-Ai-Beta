"""
DeFiGuard AI - Backdoor Scanner
Detects dangerous functions, opcodes, and patterns in contract bytecode.
"""

import logging
from typing import Any
from web3 import Web3

from .constants import (
    DANGEROUS_SELECTORS,
    DANGEROUS_OPCODES,
    FUNCTION_SELECTORS,
)

logger = logging.getLogger(__name__)


class BackdoorScanner:
    """
    Scan contract bytecode for backdoors and dangerous patterns.

    Detection Categories:
    - Dangerous function selectors (mint, blacklist, fee manipulation)
    - Dangerous opcodes (SELFDESTRUCT, DELEGATECALL)
    - Hidden mint capabilities
    - Blacklist/whitelist mechanisms
    - Fee manipulation functions
    - Trading controls
    - Arbitrary execution patterns
    """

    def __init__(self, w3: Web3):
        """
        Initialize backdoor scanner.

        Args:
            w3: Web3 instance connected to an Ethereum node
        """
        self.w3 = w3

    async def scan(self, address: str) -> dict[str, Any]:
        """
        Perform comprehensive backdoor scan on contract bytecode.

        Args:
            address: Contract address to scan

        Returns:
            {
                "has_backdoors": bool,
                "risk_score": int,  # 0-100
                "risk_level": str,  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
                "dangerous_functions": list[dict],  # Detected dangerous functions
                "dangerous_opcodes": list[dict],  # Detected dangerous opcodes
                "patterns": dict,  # Categorized pattern findings
                "summary": str,  # Human-readable summary
            }
        """
        address = self.w3.to_checksum_address(address)

        result = {
            "has_backdoors": False,
            "risk_score": 0,
            "risk_level": "LOW",
            "dangerous_functions": [],
            "dangerous_opcodes": [],
            "patterns": {
                "mint": [],
                "burn": [],
                "blacklist": [],
                "whitelist": [],
                "fees": [],
                "trading": [],
                "withdrawal": [],
                "execution": [],
                "limits": [],
            },
            "summary": ""
        }

        try:
            # Get contract bytecode
            bytecode = self.w3.eth.get_code(address).hex().lower()

            if len(bytecode) <= 2:  # "0x" or empty
                result["summary"] = "No bytecode found (EOA or self-destructed contract)"
                return result

            # Remove 0x prefix
            bytecode = bytecode.replace("0x", "")

            # Scan for dangerous function selectors
            result["dangerous_functions"] = self._scan_selectors(bytecode)

            # Scan for dangerous opcodes
            result["dangerous_opcodes"] = self._scan_opcodes(bytecode)

            # Categorize findings by pattern type
            for func in result["dangerous_functions"]:
                category = func.get("category", "other")
                if category in result["patterns"]:
                    result["patterns"][category].append(func)

            # Check for specific dangerous patterns
            pattern_findings = self._detect_patterns(bytecode)
            result["patterns"].update(pattern_findings)

            # Calculate risk score
            result["risk_score"] = self._calculate_risk_score(result)
            result["risk_level"] = self._score_to_level(result["risk_score"])
            result["has_backdoors"] = result["risk_score"] >= 30

            # Generate summary
            result["summary"] = self._generate_summary(result)

            return result

        except Exception as e:
            logger.error(f"Backdoor scan failed for {address}: {e}")
            result["error"] = str(e)
            return result

    def _scan_selectors(self, bytecode: str) -> list[dict]:
        """
        Scan bytecode for dangerous function selectors.

        Args:
            bytecode: Contract bytecode as hex string (no 0x prefix)

        Returns:
            List of detected dangerous functions
        """
        findings = []

        for selector, info in DANGEROUS_SELECTORS.items():
            if selector.lower() in bytecode:
                finding = {
                    "selector": selector,
                    "name": info["name"],
                    "risk": info["risk"],
                    "category": info["category"],
                    "description": self._get_function_description(info["category"], info["name"])
                }
                findings.append(finding)

        return findings

    def _scan_opcodes(self, bytecode: str) -> list[dict]:
        """
        Scan bytecode for dangerous opcodes.

        Args:
            bytecode: Contract bytecode as hex string (no 0x prefix)

        Returns:
            List of detected dangerous opcodes
        """
        findings = []

        for opcode, info in DANGEROUS_OPCODES.items():
            # Count occurrences (simple pattern matching)
            # Note: This can have false positives as opcodes can appear in PUSH data
            count = bytecode.count(opcode)

            if count > 0:
                finding = {
                    "opcode": opcode,
                    "name": info["name"],
                    "risk": info["risk"],
                    "description": info["description"],
                    "occurrences": count
                }
                findings.append(finding)

        return findings

    def _detect_patterns(self, bytecode: str) -> dict[str, list]:
        """
        Detect specific dangerous patterns in bytecode.

        Args:
            bytecode: Contract bytecode as hex string

        Returns:
            Dict of pattern categories with findings
        """
        patterns = {}

        # Check for hidden owner patterns
        # Look for multiple owner-like storage patterns
        owner_selector = FUNCTION_SELECTORS.get("owner", "").lower()
        if owner_selector and bytecode.count(owner_selector) > 1:
            patterns["multiple_owner_refs"] = [{
                "finding": "Multiple owner references detected",
                "risk": "MEDIUM",
                "description": "Contract may have hidden owner mechanisms"
            }]

        # Check for emergency withdrawal pattern
        withdraw_patterns = ["3ccfd60b", "f3fef3a3", "51cff8d9"]  # Various withdraw selectors
        withdraw_count = sum(1 for p in withdraw_patterns if p in bytecode)
        if withdraw_count >= 2:
            patterns["multiple_withdrawals"] = [{
                "finding": "Multiple withdrawal functions detected",
                "risk": "MEDIUM",
                "description": "Contract has multiple withdrawal mechanisms"
            }]

        # Check for fee manipulation patterns
        fee_patterns = ["8c0b5e22", "c0246668", "8ee88c53", "af465a27", "b8c61130"]
        fee_count = sum(1 for p in fee_patterns if p in bytecode)
        if fee_count >= 2:
            patterns["fee_manipulation"] = [{
                "finding": f"Multiple fee functions detected ({fee_count})",
                "risk": "HIGH",
                "description": "Contract can manipulate fees in multiple ways"
            }]

        # Check for trading control patterns (honeypot indicators)
        trading_patterns = ["8f70ccf7", "a9e75723", "c9567bf9"]
        if any(p in bytecode for p in trading_patterns):
            patterns["trading_controls"] = [{
                "finding": "Trading control functions detected",
                "risk": "HIGH",
                "description": "Owner can enable/disable trading (potential honeypot)"
            }]

        # Check for hidden mint (mint without standard selector)
        # Look for SSTORE followed by common mint event signature
        mint_event_sig = "ddf252ad"  # Transfer event (used by mint)
        if mint_event_sig in bytecode:
            # Check if there are mint-like patterns without standard mint selector
            standard_mint = "40c10f19"
            if standard_mint not in bytecode:
                # Has Transfer event but no standard mint - could be hidden
                patterns["potential_hidden_mint"] = [{
                    "finding": "Transfer events without standard mint function",
                    "risk": "MEDIUM",
                    "description": "May have non-standard minting mechanism"
                }]

        return patterns

    def _get_function_description(self, category: str, name: str) -> str:
        """
        Get human-readable description for a dangerous function.

        Args:
            category: Function category
            name: Function name

        Returns:
            Description string
        """
        descriptions = {
            "mint": f"'{name}' can create new tokens, potentially inflating supply",
            "burn": f"'{name}' can destroy tokens from any address",
            "blacklist": f"'{name}' can block addresses from transacting",
            "whitelist": f"'{name}' restricts who can transact",
            "fees": f"'{name}' can change transaction fees (potential rug mechanism)",
            "trading": f"'{name}' controls trading ability (honeypot risk)",
            "withdrawal": f"'{name}' can extract funds from contract",
            "execution": f"'{name}' allows arbitrary code execution",
            "limits": f"'{name}' controls transaction/wallet limits",
        }
        return descriptions.get(category, f"Dangerous function: {name}")

    def _calculate_risk_score(self, result: dict) -> int:
        """
        Calculate overall risk score based on findings.

        Args:
            result: Scan results

        Returns:
            Risk score 0-100
        """
        score = 0

        # Score dangerous functions
        for func in result["dangerous_functions"]:
            risk = func.get("risk", "LOW")
            if risk == "CRITICAL":
                score += 25
            elif risk == "HIGH":
                score += 15
            elif risk == "MEDIUM":
                score += 8
            else:
                score += 3

        # Score dangerous opcodes
        for opcode in result["dangerous_opcodes"]:
            risk = opcode.get("risk", "LOW")
            occurrences = min(opcode.get("occurrences", 1), 3)  # Cap at 3
            if risk == "CRITICAL":
                score += 20 * occurrences
            elif risk == "HIGH":
                score += 10 * occurrences

        # Score pattern findings
        for category, findings in result["patterns"].items():
            for finding in findings:
                if isinstance(finding, dict):
                    risk = finding.get("risk", "LOW")
                    if risk == "HIGH":
                        score += 12
                    elif risk == "MEDIUM":
                        score += 6

        # Cap at 100
        return min(score, 100)

    def _score_to_level(self, score: int) -> str:
        """
        Convert numeric score to risk level.

        Args:
            score: Risk score 0-100

        Returns:
            Risk level string
        """
        if score >= 70:
            return "CRITICAL"
        elif score >= 50:
            return "HIGH"
        elif score >= 30:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_summary(self, result: dict) -> str:
        """
        Generate human-readable summary of findings.

        Args:
            result: Scan results

        Returns:
            Summary string
        """
        parts = []

        func_count = len(result["dangerous_functions"])
        opcode_count = len(result["dangerous_opcodes"])

        if func_count == 0 and opcode_count == 0:
            return "No dangerous patterns detected in contract bytecode."

        if func_count > 0:
            critical = sum(1 for f in result["dangerous_functions"] if f.get("risk") == "CRITICAL")
            high = sum(1 for f in result["dangerous_functions"] if f.get("risk") == "HIGH")

            if critical > 0:
                parts.append(f"{critical} CRITICAL-risk function(s)")
            if high > 0:
                parts.append(f"{high} HIGH-risk function(s)")
            if func_count > critical + high:
                parts.append(f"{func_count - critical - high} other risky function(s)")

        if opcode_count > 0:
            opcode_names = [o.get("name") for o in result["dangerous_opcodes"]]
            parts.append(f"Dangerous opcodes: {', '.join(opcode_names)}")

        # Add specific warnings
        if result["patterns"].get("trading_controls"):
            parts.append("⚠️ Trading can be disabled (honeypot risk)")

        if result["patterns"].get("fee_manipulation"):
            parts.append("⚠️ Fees can be manipulated")

        if any(f.get("category") == "blacklist" for f in result["dangerous_functions"]):
            parts.append("⚠️ Blacklist functionality exists")

        return ". ".join(parts) + "."

    def get_risk_explanation(self, risk_level: str) -> str:
        """
        Get detailed explanation of risk level.

        Args:
            risk_level: "LOW", "MEDIUM", "HIGH", or "CRITICAL"

        Returns:
            Risk explanation string
        """
        explanations = {
            "LOW": "Contract has minimal backdoor risk. No significant dangerous patterns detected.",
            "MEDIUM": "Contract has some concerning patterns that warrant caution. Review detected functions carefully.",
            "HIGH": "Contract has multiple dangerous capabilities. High risk of rug pull or user fund loss.",
            "CRITICAL": "Contract has severe backdoor risks. Strongly advise against interaction without thorough audit."
        }
        return explanations.get(risk_level, "Unknown risk level")


class HoneypotDetector:
    """
    Specialized detector for honeypot token patterns.

    A honeypot is a token where users can buy but cannot sell,
    or where sells are heavily taxed.
    """

    def __init__(self, w3: Web3):
        """
        Initialize honeypot detector.

        Args:
            w3: Web3 instance connected to an Ethereum node
        """
        self.w3 = w3

    async def check(self, address: str) -> dict[str, Any]:
        """
        Check if token exhibits honeypot characteristics.

        Args:
            address: Token contract address

        Returns:
            {
                "is_honeypot": bool,
                "confidence": str,  # "LOW", "MEDIUM", "HIGH"
                "indicators": list[dict],
                "recommendation": str
            }
        """
        address = self.w3.to_checksum_address(address)

        result = {
            "is_honeypot": False,
            "confidence": "LOW",
            "indicators": [],
            "recommendation": ""
        }

        try:
            bytecode = self.w3.eth.get_code(address).hex().lower()

            if len(bytecode) <= 2:
                result["recommendation"] = "No bytecode found"
                return result

            bytecode = bytecode.replace("0x", "")
            indicators = []

            # Indicator 1: Trading enable/disable functions
            trading_selectors = ["8f70ccf7", "a9e75723", "c9567bf9"]
            for sel in trading_selectors:
                if sel in bytecode:
                    indicators.append({
                        "type": "trading_control",
                        "severity": "HIGH",
                        "description": "Contract has trading enable/disable function"
                    })
                    break

            # Indicator 2: Blacklist functionality
            blacklist_selectors = ["f9f92be4", "0e136b19", "44337ea1"]
            for sel in blacklist_selectors:
                if sel in bytecode:
                    indicators.append({
                        "type": "blacklist",
                        "severity": "HIGH",
                        "description": "Contract can blacklist addresses from selling"
                    })
                    break

            # Indicator 3: Max transaction/wallet limits
            limit_selectors = ["e517f2b9", "f7739b5f"]
            for sel in limit_selectors:
                if sel in bytecode:
                    indicators.append({
                        "type": "limits",
                        "severity": "MEDIUM",
                        "description": "Contract has adjustable transaction/wallet limits"
                    })
                    break

            # Indicator 4: Dynamic fee functions
            fee_selectors = ["af465a27", "b8c61130"]  # setSellFee, setBuyFee
            for sel in fee_selectors:
                if sel in bytecode:
                    indicators.append({
                        "type": "dynamic_fees",
                        "severity": "CRITICAL",
                        "description": "Contract can set buy/sell fees (can be set to 100%)"
                    })
                    break

            # Indicator 5: Transfer restrictions in bytecode patterns
            # Look for patterns that check msg.sender against owner before allowing transfers
            transfer_selector = FUNCTION_SELECTORS.get("transfer", "").lower()
            owner_selector = FUNCTION_SELECTORS.get("owner", "").lower()

            if transfer_selector and owner_selector:
                # Both selectors present - check if they're close together (restriction pattern)
                t_pos = bytecode.find(transfer_selector)
                o_pos = bytecode.find(owner_selector)

                if t_pos > 0 and o_pos > 0 and abs(t_pos - o_pos) < 200:
                    indicators.append({
                        "type": "owner_transfer_check",
                        "severity": "MEDIUM",
                        "description": "Transfer function may check owner status"
                    })

            # Calculate honeypot likelihood
            result["indicators"] = indicators

            if not indicators:
                result["is_honeypot"] = False
                result["confidence"] = "LOW"
                result["recommendation"] = "No honeypot indicators detected. Always verify with small test transaction."
            else:
                high_severity = sum(1 for i in indicators if i.get("severity") in ["HIGH", "CRITICAL"])

                if high_severity >= 2:
                    result["is_honeypot"] = True
                    result["confidence"] = "HIGH"
                    result["recommendation"] = "HIGH RISK: Multiple honeypot indicators. Do not trade this token."
                elif high_severity == 1:
                    result["is_honeypot"] = True
                    result["confidence"] = "MEDIUM"
                    result["recommendation"] = "CAUTION: Honeypot indicators present. Trade with extreme caution."
                else:
                    result["is_honeypot"] = False
                    result["confidence"] = "LOW"
                    result["recommendation"] = "Low-risk indicators present. Verify with small test transaction."

            return result

        except Exception as e:
            logger.error(f"Honeypot check failed for {address}: {e}")
            result["error"] = str(e)
            result["recommendation"] = "Could not complete honeypot analysis"
            return result
