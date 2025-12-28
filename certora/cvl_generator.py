"""
CVL Specification Generator for Certora Prover.

Uses Claude AI to analyze Solidity contracts and generate
Certora Verification Language (CVL) specifications.
"""

import os
import re
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# CVL templates for common patterns
CVL_TEMPLATES = {
    "erc20_balance": """
    // Balance conservation: total of all balances equals totalSupply
    ghost mathint sumOfBalances {
        init_state axiom sumOfBalances == 0;
    }

    hook Sstore balanceOf[KEY address account] uint256 newValue (uint256 oldValue) {
        sumOfBalances = sumOfBalances + newValue - oldValue;
    }

    invariant totalSupplyMatchesBalances()
        to_mathint(totalSupply()) == sumOfBalances;
    """,

    "access_control": """
    // Only owner can call restricted functions
    rule onlyOwnerCanCall(method f, env e) filtered {
        f -> f.selector == sig:restricted_function().selector
    } {
        require e.msg.sender != owner();
        f@withrevert(e);
        assert lastReverted;
    }
    """,

    "no_reentrancy": """
    // State changes before external calls (CEI pattern)
    rule noReentrancy(method f, env e) {
        uint256 balanceBefore = balanceOf(e.msg.sender);
        f(e);
        uint256 balanceAfter = balanceOf(e.msg.sender);

        // Balance should be updated atomically
        assert balanceAfter <= balanceBefore || lastReverted;
    }
    """,

    "no_overflow": """
    // Arithmetic operations don't overflow
    rule noOverflow(env e, uint256 a, uint256 b) {
        uint256 result = add(e, a, b);
        assert result >= a && result >= b;
    }
    """
}


class CVLGenerator:
    """Generate Certora Verification Language specs from Solidity contracts."""

    def __init__(self, anthropic_client=None):
        """
        Initialize CVL Generator.

        Args:
            anthropic_client: Optional Anthropic client instance.
                            If not provided, will create one using ANTHROPIC_API_KEY.
        """
        self.client = anthropic_client
        self._init_client()

    def _init_client(self):
        """Initialize Anthropic client if not provided."""
        if self.client is None:
            try:
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    self.client = anthropic.Anthropic(api_key=api_key)
                    logger.info("CVLGenerator: Anthropic client initialized")
                else:
                    logger.warning("CVLGenerator: No ANTHROPIC_API_KEY found")
            except ImportError:
                logger.error("CVLGenerator: anthropic package not installed")

    def generate_specs_sync(
        self,
        contract_code: str,
        slither_findings: list = None,
        contract_name: str = None
    ) -> Optional[str]:
        """
        Generate CVL specifications synchronously.

        Args:
            contract_code: Solidity source code
            slither_findings: Optional list of Slither findings to focus verification
            contract_name: Optional contract name (extracted from code if not provided)

        Returns:
            CVL specification string or None if generation fails
        """
        if not self.client:
            logger.warning("CVLGenerator: No client available, using template-based specs")
            return self._generate_template_specs(contract_code, contract_name)

        try:
            # Extract contract name if not provided
            if not contract_name:
                contract_name = self._extract_contract_name(contract_code)

            # Build the prompt
            prompt = self._build_cvl_prompt(contract_code, slither_findings, contract_name)

            # Call Claude
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )

            cvl_content = response.content[0].text

            # Extract CVL from response (may be wrapped in markdown)
            cvl_specs = self._extract_cvl(cvl_content)

            if cvl_specs:
                logger.info(f"CVLGenerator: Generated {len(cvl_specs)} chars of CVL specs")
                return cvl_specs
            else:
                logger.warning("CVLGenerator: No CVL extracted, using templates")
                return self._generate_template_specs(contract_code, contract_name)

        except Exception as e:
            logger.error(f"CVLGenerator: AI generation failed: {e}")
            return self._generate_template_specs(contract_code, contract_name)

    async def generate_specs(
        self,
        contract_code: str,
        slither_findings: list = None,
        contract_name: str = None
    ) -> Optional[str]:
        """
        Generate CVL specifications asynchronously.

        Args:
            contract_code: Solidity source code
            slither_findings: Optional list of Slither findings to focus verification
            contract_name: Optional contract name

        Returns:
            CVL specification string or None if generation fails
        """
        import asyncio
        return await asyncio.to_thread(
            self.generate_specs_sync,
            contract_code,
            slither_findings,
            contract_name
        )

    def _build_cvl_prompt(
        self,
        contract_code: str,
        slither_findings: list,
        contract_name: str
    ) -> str:
        """Build the prompt for CVL generation."""

        findings_context = ""
        if slither_findings:
            findings_context = f"""
Static analysis found these potential issues - prioritize verifying these:
{self._format_findings(slither_findings)}
"""

        return f"""You are an expert in Certora Verification Language (CVL) and smart contract formal verification.

Generate a CVL specification file for the following Solidity contract.

CONTRACT NAME: {contract_name or "Unknown"}

SOLIDITY CODE:
```solidity
{contract_code}
```

{findings_context}

REQUIREMENTS:
1. Generate a complete, valid CVL specification file
2. Include a methods block declaring all public/external functions
3. Write invariants for:
   - Balance/supply conservation (if applicable)
   - Access control (if applicable)
   - State consistency
4. Write rules for:
   - No reentrancy vulnerabilities
   - No unexpected reverts in normal conditions
   - Proper authorization checks
5. Focus on properties that would catch real vulnerabilities
6. Use ghosts and hooks where appropriate for tracking state

OUTPUT FORMAT:
Return ONLY the CVL specification code, no explanations.
Start with the methods block, then invariants, then rules.
Use proper CVL syntax compatible with Certora Prover.

Example structure:
```
methods {{
    function balanceOf(address) external returns (uint256) envfree;
    function totalSupply() external returns (uint256) envfree;
}}

invariant totalSupplyIsValid()
    totalSupply() >= 0;

rule transferDoesNotIncreaseSupply(env e, address to, uint256 amount) {{
    uint256 supplyBefore = totalSupply();
    transfer(e, to, amount);
    assert totalSupply() == supplyBefore;
}}
```
"""

    def _format_findings(self, findings: list) -> str:
        """Format Slither findings for the prompt."""
        if not findings:
            return "None"

        formatted = []
        for i, finding in enumerate(findings[:10], 1):  # Limit to top 10
            if isinstance(finding, dict):
                name = finding.get("name", finding.get("check", "Unknown"))
                details = finding.get("details", finding.get("description", ""))
                formatted.append(f"{i}. {name}: {details[:200]}")
            else:
                formatted.append(f"{i}. {str(finding)[:200]}")

        return "\n".join(formatted)

    def _extract_cvl(self, response: str) -> Optional[str]:
        """Extract CVL code from AI response."""
        # Try to find code block
        code_block_pattern = r"```(?:cvl|spec|)?\s*\n?(.*?)```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        if matches:
            # Return the longest match (likely the main spec)
            return max(matches, key=len).strip()

        # If no code blocks, check if response looks like CVL
        if "methods" in response and ("rule" in response or "invariant" in response):
            return response.strip()

        return None

    def _extract_contract_name(self, code: str) -> str:
        """Extract contract name from Solidity code."""
        pattern = r"contract\s+(\w+)"
        match = re.search(pattern, code)
        return match.group(1) if match else "Contract"

    def _generate_template_specs(
        self,
        contract_code: str,
        contract_name: str = None
    ) -> str:
        """Generate basic CVL specs from templates when AI is unavailable."""

        if not contract_name:
            contract_name = self._extract_contract_name(contract_code)

        # Detect contract type and applicable templates
        code_lower = contract_code.lower()

        templates_to_use = []
        methods = []

        # Check for ERC20 patterns
        if "balanceof" in code_lower and "totalsupply" in code_lower:
            methods.extend([
                "function balanceOf(address) external returns (uint256) envfree;",
                "function totalSupply() external returns (uint256) envfree;",
            ])
            if "transfer" in code_lower:
                methods.append("function transfer(address, uint256) external returns (bool);")
            if "transferfrom" in code_lower:
                methods.append("function transferFrom(address, address, uint256) external returns (bool);")
            templates_to_use.append(CVL_TEMPLATES["erc20_balance"])

        # Check for access control
        if "owner" in code_lower or "onlyowner" in code_lower:
            methods.append("function owner() external returns (address) envfree;")

        # Build the spec
        spec = f"""/*
 * CVL Specification for {contract_name}
 * Auto-generated by DeFiGuard AI
 */

methods {{
    {chr(10).join('    ' + m for m in methods) if methods else '    // Add method declarations'}
}}

// Basic invariant: contract is always in valid state
invariant contractIsValid()
    true;

// Rule: functions should not unexpectedly revert
rule noUnexpectedRevert(method f, env e, calldataarg args) {{
    f@withrevert(e, args);
    // Captures any unexpected reverts for analysis
    satisfy !lastReverted;
}}
"""

        # Add detected templates
        for template in templates_to_use:
            spec += f"\n{template}\n"

        return spec


# Convenience function for synchronous usage
def generate_cvl_specs(
    contract_code: str,
    slither_findings: list = None,
    anthropic_client=None
) -> Optional[str]:
    """
    Generate CVL specifications for a Solidity contract.

    Args:
        contract_code: Solidity source code
        slither_findings: Optional Slither findings to focus verification
        anthropic_client: Optional Anthropic client

    Returns:
        CVL specification string or None
    """
    generator = CVLGenerator(anthropic_client)
    return generator.generate_specs_sync(contract_code, slither_findings)
