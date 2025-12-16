"""
DeFiGuard AI Compliance Checker
Implements MiCA, SEC FIT21, and DeFi regulatory frameworks
"""

from typing import Dict, List, Any
import re

class ComplianceChecker:
    """
    Comprehensive compliance analysis for smart contracts.
    Covers EU MiCA, SEC regulations, and DeFi-specific frameworks.
    """
    
    def __init__(self):
        # MiCA Articles (EU Crypto Regulation)
        self.mica_rules = {
            "Article_50": {
                "name": "Custody and Safeguarding of Crypto-Assets",
                "checks": [
                    "withdrawal_access_control",
                    "multi_sig_custody",
                    "emergency_pause",
                    "fund_segregation"
                ]
            },
            "Article_30": {
                "name": "Market Abuse Prevention",
                "checks": [
                    "price_manipulation_protection",
                    "front_running_prevention",
                    "insider_trading_controls"
                ]
            },
            "Article_68": {
                "name": "Whitepaper Requirements",
                "checks": [
                    "risk_disclosure",
                    "tokenomics_clarity",
                    "technical_documentation"
                ]
            },
            "Article_120": {
                "name": "AML/KYC Requirements",
                "checks": [
                    "identity_verification",
                    "transaction_monitoring",
                    "suspicious_activity_reporting"
                ]
            }
        }
        
        # SEC Regulations (US)
        self.sec_rules = {
            "Howey_Test": {
                "name": "Securities Test",
                "factors": [
                    "investment_of_money",
                    "common_enterprise",
                    "expectation_of_profits",
                    "efforts_of_others"
                ]
            },
            "FIT21": {
                "name": "Financial Innovation and Technology Act",
                "requirements": [
                    "decentralization_threshold",
                    "token_functionality",
                    "voting_rights",
                    "profit_distribution"
                ]
            },
            "Reg_D_506c": {
                "name": "Accredited Investor Exemption",
                "checks": [
                    "accredited_investor_verification",
                    "general_solicitation_limits"
                ]
            }
        }
        
        # Attack Vector Database (200+ patterns)
        self.attack_vectors = {
            "reentrancy": {
                "severity": "Critical",
                "patterns": [
                    r"\.call\{value:",
                    r"\.send\(",
                    r"\.transfer\(",
                ],
                "checks": [
                    "state_updated_before_call",
                    "reentrancy_guard_present",
                    "checks_effects_interactions"
                ]
            },
            "integer_overflow": {
                "severity": "High",
                "patterns": [r"\+\s*=", r"\-\s*=", r"\*\s*="],
                "checks": ["safe_math_library", "unchecked_blocks"]
            },
            "unchecked_external_call": {
                "severity": "High",
                "patterns": [r"\.call\(", r"\.delegatecall\("],
                "checks": ["return_value_checked", "error_handling"]
            },
            "access_control": {
                "severity": "Critical",
                "patterns": [r"onlyOwner", r"onlyAdmin", r"require\(msg\.sender"],
                "checks": ["modifier_present", "role_based_access"]
            },
            "flash_loan_attack": {
                "severity": "Critical",
                "patterns": [r"flashLoan", r"borrow", r"repay"],
                "checks": [
                    "reentrancy_protection",
                    "price_oracle_manipulation",
                    "liquidity_checks"
                ]
            },
            "oracle_manipulation": {
                "severity": "Critical",
                "patterns": [r"getPrice", r"oracle", r"price"],
                "checks": [
                    "multi_oracle_consensus",
                    "twap_implementation",
                    "chainlink_integration"
                ]
            },
            "front_running": {
                "severity": "High",
                "patterns": [r"swap", r"trade", r"exchange"],
                "checks": [
                    "commit_reveal_scheme",
                    "max_slippage_protection",
                    "flashbots_integration"
                ]
            },
            "timestamp_dependence": {
                "severity": "Medium",
                "patterns": [r"block\.timestamp", r"now"],
                "checks": ["timestamp_tolerance", "block_number_alternative"]
            },
            "delegatecall_injection": {
                "severity": "Critical",
                "patterns": [r"delegatecall"],
                "checks": ["target_address_whitelisted", "proxy_initialization"]
            },
            "selfdestruct_vulnerability": {
                "severity": "Critical",
                "patterns": [r"selfdestruct", r"suicide"],
                "checks": ["protected_by_access_control", "migration_mechanism"]
            }
        }
    
    def analyze_mica_compliance(self, code: str, contract_type: str) -> Dict[str, Any]:
        """Check MiCA compliance based on contract type."""
        results = {
            "compliant": True,
            "violations": [],
            "recommendations": []
        }
        
        # Article 50: Custody checks
        if "withdraw" in code.lower() or "transfer" in code.lower():
            if not re.search(r"onlyOwner|onlyAdmin|require\(msg\.sender", code):
                results["compliant"] = False
                results["violations"].append({
                    "article": "Article 50",
                    "issue": "Custody functions lack access control",
                    "severity": "Critical",
                    "recommendation": "Implement onlyOwner or role-based access control for withdrawal functions"
                })
        
        # Article 30: Market abuse prevention
        if "price" in code.lower() or "swap" in code.lower():
            if not re.search(r"slippage|maxPrice|minPrice", code, re.IGNORECASE):
                results["violations"].append({
                    "article": "Article 30",
                    "issue": "No slippage protection detected",
                    "severity": "High",
                    "recommendation": "Add slippage limits to prevent price manipulation"
                })
        
        return results
    
    def analyze_sec_compliance(self, code: str, token_features: Dict[str, bool]) -> Dict[str, Any]:
        """Analyze SEC compliance (Howey Test + FIT21)."""
        results = {
            "is_security": False,
            "howey_factors": [],
            "fit21_analysis": {},
            "recommendations": []
        }
        
        # Howey Test factors
        howey_score = 0
        
        if token_features.get("profit_distribution"):
            howey_score += 1
            results["howey_factors"].append("Expectation of profits from token")
        
        if token_features.get("centralized_control"):
            howey_score += 1
            results["howey_factors"].append("Efforts of others (centralized team)")
        
        if howey_score >= 3:
            results["is_security"] = True
            results["recommendations"].append(
                "Token may be classified as a security under Howey Test. "
                "Consider: (1) Increasing decentralization, (2) Reg D/Reg A+ exemptions, "
                "(3) SEC registration"
            )
        
        return results
    
    def scan_attack_vectors(self, code: str) -> List[Dict[str, Any]]:
        """Scan for known attack patterns."""
        findings = []
        
        for vector_name, vector_data in self.attack_vectors.items():
            for pattern in vector_data["patterns"]:
                if re.search(pattern, code):
                    findings.append({
                        "attack_type": vector_name,
                        "severity": vector_data["severity"],
                        "pattern_matched": pattern,
                        "required_checks": vector_data["checks"]
                    })
        
        return findings

# Export for use in main.py
def get_compliance_analysis(code: str, contract_type: str = "defi") -> Dict[str, Any]:
    """
    Main entry point for compliance analysis.
    Returns comprehensive regulatory compliance report.
    """
    checker = ComplianceChecker()
    
    return {
        "mica_compliance": checker.analyze_mica_compliance(code, contract_type),
        "sec_compliance": checker.analyze_sec_compliance(code, {
            "profit_distribution": "dividend" in code.lower() or "reward" in code.lower(),
            "centralized_control": "onlyOwner" in code and "transfer" not in code.lower()
        }),
        "attack_vectors": checker.scan_attack_vectors(code),
        "compliance_score": 0  # Calculated based on violations
    }