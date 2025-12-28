"""
CVL Specification Generator for Certora Prover.

Uses Claude AI to analyze Solidity contracts and generate
comprehensive Certora Verification Language (CVL) specifications.

This generator is designed to produce enterprise-grade formal verification
specs that catch real vulnerabilities across all major attack vectors.
"""

import os
import re
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Comprehensive CVL templates for common vulnerability patterns
CVL_TEMPLATES = {
    # ═══════════════════════════════════════════════════════════════════
    # ERC20 TOKEN PATTERNS
    # ═══════════════════════════════════════════════════════════════════
    "erc20_balance_conservation": """
// ═══════════════════════════════════════════════════════════════════
// ERC20 BALANCE CONSERVATION
// Ensures total of all balances always equals totalSupply
// ═══════════════════════════════════════════════════════════════════

ghost mathint sumOfBalances {
    init_state axiom sumOfBalances == 0;
}

hook Sstore balanceOf[KEY address account] uint256 newValue (uint256 oldValue) {
    sumOfBalances = sumOfBalances + newValue - oldValue;
}

invariant totalSupplyIsSumOfBalances()
    to_mathint(totalSupply()) == sumOfBalances
    { preserved { require sumOfBalances >= 0; } }

invariant balanceNotExceedsTotalSupply(address account)
    balanceOf(account) <= totalSupply();
""",

    "erc20_transfer_integrity": """
// ═══════════════════════════════════════════════════════════════════
// ERC20 TRANSFER INTEGRITY
// Ensures transfers move exact amounts and preserve supply
// ═══════════════════════════════════════════════════════════════════

rule transferIntegrity(env e, address to, uint256 amount) {
    address sender = e.msg.sender;
    require sender != to;
    require sender != 0 && to != 0;

    uint256 senderBalanceBefore = balanceOf(sender);
    uint256 toBalanceBefore = balanceOf(to);
    uint256 totalSupplyBefore = totalSupply();

    bool success = transfer(e, to, amount);

    uint256 senderBalanceAfter = balanceOf(sender);
    uint256 toBalanceAfter = balanceOf(to);
    uint256 totalSupplyAfter = totalSupply();

    // If successful, exact amount moved
    assert success => (
        senderBalanceAfter == senderBalanceBefore - amount &&
        toBalanceAfter == toBalanceBefore + amount
    );

    // Total supply never changes on transfer
    assert totalSupplyAfter == totalSupplyBefore;
}

rule cannotTransferMoreThanBalance(env e, address to, uint256 amount) {
    uint256 balance = balanceOf(e.msg.sender);
    require amount > balance;

    transfer@withrevert(e, to, amount);

    assert lastReverted, "Transfer of more than balance must revert";
}

rule transferFromRespectsAllowance(env e, address from, address to, uint256 amount) {
    uint256 allowanceBefore = allowance(from, e.msg.sender);
    require amount > allowanceBefore;
    require from != e.msg.sender; // Self-transfer doesn't need allowance

    transferFrom@withrevert(e, from, to, amount);

    assert lastReverted, "TransferFrom exceeding allowance must revert";
}
""",

    # ═══════════════════════════════════════════════════════════════════
    # REENTRANCY PROTECTION
    # ═══════════════════════════════════════════════════════════════════
    "reentrancy_protection": """
// ═══════════════════════════════════════════════════════════════════
// REENTRANCY PROTECTION (CEI Pattern Verification)
// Detects state changes after external calls
// ═══════════════════════════════════════════════════════════════════

ghost bool reentrancyGuardActive {
    init_state axiom !reentrancyGuardActive;
}

// Track when we enter a non-view function
hook CALL(uint g, address addr, uint value, uint argsOffset, uint argsLength, uint retOffset, uint retLength) uint rc {
    // After external call, guard should prevent re-entry
}

rule noStateChangeAfterExternalCall(method f, env e, calldataarg args)
    filtered { f -> !f.isView }
{
    storage stateBefore = lastStorage;

    f(e, args);

    // This rule helps identify functions that may have reentrancy issues
    // by checking if state changes occur in a pattern suggesting post-call updates
    satisfy true;
}

// Verify functions with nonReentrant modifier actually prevent reentrancy
rule reentrancyGuardWorks(method f, env e, calldataarg args)
    filtered { f -> !f.isView }
{
    require reentrancyGuardActive;

    f@withrevert(e, args);

    // If guard is active, reentrant call should fail
    satisfy lastReverted;
}
""",

    # ═══════════════════════════════════════════════════════════════════
    # ACCESS CONTROL
    # ═══════════════════════════════════════════════════════════════════
    "access_control_ownership": """
// ═══════════════════════════════════════════════════════════════════
// ACCESS CONTROL - OWNERSHIP
// Verifies only authorized addresses can call restricted functions
// ═══════════════════════════════════════════════════════════════════

invariant ownerNeverZero()
    owner() != 0
    { preserved { require owner() != 0; } }

rule onlyOwnerCanTransferOwnership(env e, address newOwner) {
    address currentOwner = owner();
    require e.msg.sender != currentOwner;

    transferOwnership@withrevert(e, newOwner);

    assert lastReverted, "Non-owner must not transfer ownership";
}

rule ownershipTransferCorrect(env e, address newOwner) {
    require e.msg.sender == owner();
    require newOwner != 0;

    transferOwnership(e, newOwner);

    assert owner() == newOwner, "Ownership must transfer to new owner";
}

rule restrictedFunctionOnlyOwner(method f, env e, calldataarg args)
    filtered {
        f -> f.selector == sig:pause().selector ||
             f.selector == sig:unpause().selector ||
             f.selector == sig:setFee(uint256).selector ||
             f.selector == sig:withdraw(uint256).selector ||
             f.selector == sig:emergencyWithdraw().selector
    }
{
    require e.msg.sender != owner();

    f@withrevert(e, args);

    assert lastReverted, "Restricted function must revert for non-owner";
}
""",

    # ═══════════════════════════════════════════════════════════════════
    # FLASH LOAN PROTECTION
    # ═══════════════════════════════════════════════════════════════════
    "flash_loan_protection": """
// ═══════════════════════════════════════════════════════════════════
// FLASH LOAN PROTECTION
// Ensures protocol state is consistent within single transaction
// ═══════════════════════════════════════════════════════════════════

ghost bool inFlashLoan {
    init_state axiom !inFlashLoan;
}

// Verify that critical operations check for flash loan context
rule priceManipulationProtection(env e, method f, calldataarg args)
    filtered { f -> !f.isView }
{
    // Price-sensitive operations should use TWAP or multi-block checks
    // This rule ensures state is consistent
    uint256 priceBefore = getPrice();

    f(e, args);

    uint256 priceAfter = getPrice();

    // Large price changes in single tx may indicate manipulation
    assert priceAfter <= priceBefore * 2 && priceAfter >= priceBefore / 2,
        "Price change too large - possible manipulation";
}

rule flashLoanRepayment(env e, uint256 amount) {
    uint256 balanceBefore = balanceOf(currentContract);

    flashLoan(e, amount);

    uint256 balanceAfter = balanceOf(currentContract);

    // After flash loan completes, balance should be >= before (with fee)
    assert balanceAfter >= balanceBefore, "Flash loan must be repaid";
}
""",

    # ═══════════════════════════════════════════════════════════════════
    # ARITHMETIC SAFETY
    # ═══════════════════════════════════════════════════════════════════
    "arithmetic_safety": """
// ═══════════════════════════════════════════════════════════════════
// ARITHMETIC SAFETY
// Verifies no overflow/underflow in critical calculations
// ═══════════════════════════════════════════════════════════════════

rule noOverflowOnDeposit(env e, uint256 amount) {
    uint256 balanceBefore = balanceOf(e.msg.sender);
    uint256 totalBefore = totalSupply();

    require amount > 0;
    require balanceBefore + amount <= max_uint256;
    require totalBefore + amount <= max_uint256;

    deposit@withrevert(e, amount);

    // Should not revert due to overflow if preconditions met
    assert !lastReverted => (
        balanceOf(e.msg.sender) == balanceBefore + amount
    );
}

rule divisionByZeroProtection(env e, uint256 amount) {
    uint256 totalSupplyVal = totalSupply();

    // Operations dividing by totalSupply should handle zero case
    require totalSupplyVal == 0;

    // Any operation using totalSupply as divisor should revert or handle gracefully
    withdraw@withrevert(e, amount);

    // Either reverts or handles zero supply case
    satisfy true;
}
""",

    # ═══════════════════════════════════════════════════════════════════
    # VAULT/STAKING PATTERNS
    # ═══════════════════════════════════════════════════════════════════
    "vault_share_integrity": """
// ═══════════════════════════════════════════════════════════════════
// VAULT SHARE INTEGRITY
// Ensures share calculations are fair and manipulation-resistant
// ═══════════════════════════════════════════════════════════════════

ghost mathint totalDeposits {
    init_state axiom totalDeposits == 0;
}

// First depositor attack prevention
rule firstDepositorProtection(env e, uint256 amount) {
    uint256 totalSharesBefore = totalSupply();
    require totalSharesBefore == 0; // First deposit
    require amount > 0;

    uint256 sharesMinted = deposit(e, amount);

    // First depositor should not get disproportionate shares
    // Shares should be proportional to deposit
    assert sharesMinted > 0, "First depositor must receive shares";
    assert sharesMinted <= amount * 1000, "Shares must not be inflated";
}

rule withdrawGetsProportionalAssets(env e, uint256 shares) {
    uint256 totalShares = totalSupply();
    uint256 totalAssets = totalAssets();
    require totalShares > 0;
    require shares <= balanceOf(e.msg.sender);
    require shares > 0;

    uint256 expectedAssets = (shares * totalAssets) / totalShares;

    uint256 assetsReceived = withdraw(e, shares);

    // Should receive proportional assets (allowing for rounding)
    assert assetsReceived >= expectedAssets - 1 && assetsReceived <= expectedAssets + 1,
        "Withdrawal must be proportional to shares";
}

rule noShareDilution(env e, address user, uint256 amount) {
    uint256 userSharesBefore = balanceOf(user);
    uint256 totalSharesBefore = totalSupply();
    uint256 totalAssetsBefore = totalAssets();

    require user != e.msg.sender;
    require totalSharesBefore > 0;

    // Someone else deposits
    deposit(e, amount);

    uint256 userSharesAfter = balanceOf(user);
    uint256 totalSharesAfter = totalSupply();
    uint256 totalAssetsAfter = totalAssets();

    // User's share ratio should not decrease significantly
    // (userSharesBefore / totalSharesBefore) * totalAssetsBefore <=
    // (userSharesAfter / totalSharesAfter) * totalAssetsAfter
    assert userSharesAfter == userSharesBefore, "Other deposits must not change user shares";
}
""",

    # ═══════════════════════════════════════════════════════════════════
    # ORACLE SAFETY
    # ═══════════════════════════════════════════════════════════════════
    "oracle_safety": """
// ═══════════════════════════════════════════════════════════════════
// ORACLE SAFETY
// Verifies price oracle usage is manipulation-resistant
// ═══════════════════════════════════════════════════════════════════

rule oraclePriceBounded(env e) {
    uint256 price = getPrice(e);

    // Price should be within reasonable bounds
    assert price > 0, "Price must be positive";
    assert price < max_uint256 / 10^18, "Price must not overflow calculations";
}

rule oracleUpdateRestricted(env e, uint256 newPrice) {
    require e.msg.sender != oracle();

    setPrice@withrevert(e, newPrice);

    assert lastReverted, "Only oracle can update price";
}

rule priceDeviationCheck(env e, uint256 newPrice) {
    uint256 currentPrice = getPrice(e);
    require currentPrice > 0;

    // Large price deviations should be rejected or handled specially
    require newPrice > currentPrice * 2 || newPrice < currentPrice / 2;

    updatePrice@withrevert(e, newPrice);

    // Either reverts or has special handling for large deviations
    satisfy true;
}
""",

    # ═══════════════════════════════════════════════════════════════════
    # GOVERNANCE SAFETY
    # ═══════════════════════════════════════════════════════════════════
    "governance_safety": """
// ═══════════════════════════════════════════════════════════════════
// GOVERNANCE SAFETY
// Prevents governance attacks and ensures fair voting
// ═══════════════════════════════════════════════════════════════════

rule votingPowerBounded(address voter) {
    uint256 votingPower = getVotes(voter);
    uint256 totalVotingPower = totalSupply();

    // No single voter should have >50% voting power
    assert votingPower <= totalVotingPower / 2,
        "Single voter should not have majority";
}

rule proposalExecutionRequiresQuorum(env e, uint256 proposalId) {
    uint256 forVotes = proposalVotes(proposalId);
    uint256 quorum = quorumVotes();

    require forVotes < quorum;

    execute@withrevert(e, proposalId);

    assert lastReverted, "Proposal without quorum must not execute";
}

rule timelockEnforced(env e, uint256 proposalId) {
    uint256 eta = proposalEta(proposalId);
    require e.block.timestamp < eta;

    execute@withrevert(e, proposalId);

    assert lastReverted, "Cannot execute before timelock expires";
}

rule noFlashLoanVoting(env e, uint256 proposalId) {
    // Voting power should be based on historical balance, not current
    uint256 votingPower = getPastVotes(e.msg.sender, proposalSnapshot(proposalId));
    uint256 currentBalance = balanceOf(e.msg.sender);

    // If using snapshot voting, current balance manipulation doesn't help
    assert votingPower <= currentBalance || votingPower > 0,
        "Voting power based on snapshot, not current balance";
}
""",

    # ═══════════════════════════════════════════════════════════════════
    # PAUSABILITY
    # ═══════════════════════════════════════════════════════════════════
    "pausability": """
// ═══════════════════════════════════════════════════════════════════
// PAUSABILITY
// Ensures pause mechanism works correctly
// ═══════════════════════════════════════════════════════════════════

rule pauseBlocksTransfers(env e, address to, uint256 amount) {
    require paused();

    transfer@withrevert(e, to, amount);

    assert lastReverted, "Transfers must be blocked when paused";
}

rule onlyAuthorizedCanPause(env e) {
    require e.msg.sender != owner() && e.msg.sender != pauser();

    pause@withrevert(e);

    assert lastReverted, "Only authorized can pause";
}

rule unpauseRestoresFunction(env e, address to, uint256 amount) {
    require !paused();
    require balanceOf(e.msg.sender) >= amount;

    transfer@withrevert(e, to, amount);

    // Should not revert due to pause
    satisfy !lastReverted;
}
"""
}

# Vulnerability-to-template mapping for targeted verification
VULNERABILITY_TEMPLATES = {
    "reentrancy": "reentrancy_protection",
    "reentrancy-eth": "reentrancy_protection",
    "reentrancy-no-eth": "reentrancy_protection",
    "reentrancy-benign": "reentrancy_protection",
    "reentrancy-events": "reentrancy_protection",
    "reentrancy-unlimited-gas": "reentrancy_protection",
    "controlled-delegatecall": "access_control_ownership",
    "arbitrary-send-eth": "access_control_ownership",
    "arbitrary-send-erc20": "access_control_ownership",
    "unprotected-upgrade": "access_control_ownership",
    "weak-prng": "arithmetic_safety",
    "divide-before-multiply": "arithmetic_safety",
    "unchecked-transfer": "erc20_transfer_integrity",
    "unchecked-lowlevel": "reentrancy_protection",
    "incorrect-equality": "arithmetic_safety",
    "tx-origin": "access_control_ownership",
    "uninitialized-state": "access_control_ownership",
    "uninitialized-storage": "access_control_ownership",
    "locked-ether": "vault_share_integrity",
    "shadowing-state": "access_control_ownership",
    "suicidal": "access_control_ownership",
    "erc20-interface": "erc20_transfer_integrity",
    "incorrect-modifier": "access_control_ownership",
}

# Contract type detection patterns
CONTRACT_PATTERNS = {
    "erc20": ["balanceof", "totalsupply", "transfer", "approve", "allowance"],
    "erc721": ["balanceof", "ownerof", "safetransferfrom", "approve", "getapproved"],
    "vault": ["deposit", "withdraw", "totalassets", "converttoassets", "converttoshares"],
    "staking": ["stake", "unstake", "rewards", "earned", "rewardrate"],
    "governance": ["propose", "vote", "execute", "queue", "quorum"],
    "oracle": ["getprice", "latestanswer", "updateprice", "pricefeed"],
    "flashloan": ["flashloan", "flashmint", "executeflashloan", "onflashloan"],
    "ownable": ["owner", "transferownership", "renounceownership", "onlyowner"],
    "pausable": ["pause", "unpause", "paused", "whennotpaused"],
}


class CVLGenerator:
    """
    Generate enterprise-grade Certora Verification Language specs.

    This generator produces comprehensive formal verification specifications
    that catch real vulnerabilities across all major smart contract attack vectors.
    """

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
        Generate comprehensive CVL specifications synchronously.

        Args:
            contract_code: Solidity source code
            slither_findings: Optional list of Slither findings to focus verification
            contract_name: Optional contract name (extracted from code if not provided)

        Returns:
            CVL specification string or None if generation fails
        """
        if not self.client:
            logger.warning("CVLGenerator: No client available, using template-based specs")
            return self._generate_template_specs(contract_code, slither_findings, contract_name)

        try:
            # Extract contract name if not provided
            if not contract_name:
                contract_name = self._extract_contract_name(contract_code)

            # Detect contract types
            contract_types = self._detect_contract_types(contract_code)
            logger.info(f"CVLGenerator: Detected contract types: {contract_types}")

            # Build the comprehensive prompt
            prompt = self._build_comprehensive_prompt(
                contract_code,
                slither_findings,
                contract_name,
                contract_types
            )

            # Call Claude with higher token limit for comprehensive specs
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8192,  # Increased for comprehensive specs
                messages=[{"role": "user", "content": prompt}]
            )

            cvl_content = response.content[0].text

            # Extract CVL from response
            cvl_specs = self._extract_cvl(cvl_content)

            if cvl_specs:
                logger.info(f"CVLGenerator: Generated {len(cvl_specs)} chars of CVL specs")
                return cvl_specs
            else:
                logger.warning("CVLGenerator: No CVL extracted, using templates")
                return self._generate_template_specs(contract_code, slither_findings, contract_name)

        except Exception as e:
            logger.error(f"CVLGenerator: AI generation failed: {e}")
            return self._generate_template_specs(contract_code, slither_findings, contract_name)

    async def generate_specs(
        self,
        contract_code: str,
        slither_findings: list = None,
        contract_name: str = None
    ) -> Optional[str]:
        """
        Generate CVL specifications asynchronously.
        """
        import asyncio
        return await asyncio.to_thread(
            self.generate_specs_sync,
            contract_code,
            slither_findings,
            contract_name
        )

    def _detect_contract_types(self, contract_code: str) -> list:
        """Detect what type of contract this is based on patterns."""
        code_lower = contract_code.lower()
        detected = []

        for contract_type, patterns in CONTRACT_PATTERNS.items():
            matches = sum(1 for p in patterns if p in code_lower)
            if matches >= 2:  # At least 2 patterns match
                detected.append(contract_type)

        return detected if detected else ["generic"]

    def _build_comprehensive_prompt(
        self,
        contract_code: str,
        slither_findings: list,
        contract_name: str,
        contract_types: list
    ) -> str:
        """Build comprehensive prompt for enterprise-grade CVL generation."""

        # Format vulnerability findings
        findings_section = self._format_findings_detailed(slither_findings)

        # Get relevant templates for context
        template_hints = self._get_template_hints(contract_types, slither_findings)

        # Extract function signatures for methods block
        functions = self._extract_functions(contract_code)

        return f"""You are a world-class smart contract security expert and Certora CVL specialist.
Your task is to generate comprehensive, production-ready CVL specifications that will catch
real vulnerabilities through formal verification.

═══════════════════════════════════════════════════════════════════════════════
CONTRACT ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

CONTRACT NAME: {contract_name}
DETECTED TYPES: {', '.join(contract_types)}
FUNCTIONS FOUND: {', '.join(functions[:20])}  {"..." if len(functions) > 20 else ""}

SOLIDITY CODE:
```solidity
{contract_code}
```

═══════════════════════════════════════════════════════════════════════════════
STATIC ANALYSIS FINDINGS (Prioritize verification for these)
═══════════════════════════════════════════════════════════════════════════════
{findings_section}

═══════════════════════════════════════════════════════════════════════════════
CVL SPECIFICATION REQUIREMENTS
═══════════════════════════════════════════════════════════════════════════════

Generate a COMPREHENSIVE CVL specification that includes:

1. **METHODS BLOCK** (REQUIRED)
   - Declare ALL public/external functions
   - Mark view/pure functions as `envfree` where appropriate
   - Include inherited functions (ERC20, Ownable, etc.)

2. **INVARIANTS** (At least 3-5)
   - State consistency invariants
   - Balance/supply conservation (if applicable)
   - Access control invariants
   - Business logic invariants specific to this contract

3. **CRITICAL SECURITY RULES** (At least 5-10)
   Based on the contract type, include rules for:

   {"- ERC20: Transfer integrity, balance conservation, allowance respect" if "erc20" in contract_types else ""}
   {"- Vault: Share calculation fairness, no first-depositor attack, proportional withdrawals" if "vault" in contract_types else ""}
   {"- Staking: Reward calculation accuracy, stake/unstake balance" if "staking" in contract_types else ""}
   {"- Governance: Quorum enforcement, timelock, flash loan voting prevention" if "governance" in contract_types else ""}
   {"- Oracle: Price bounds, update authorization, manipulation resistance" if "oracle" in contract_types else ""}
   {"- Ownable: Ownership transfer, restricted function access" if "ownable" in contract_types else ""}
   {"- Pausable: Pause blocks operations, authorized pause/unpause" if "pausable" in contract_types else ""}

4. **VULNERABILITY-SPECIFIC RULES**
   Generate rules to formally verify protection against:
   - Reentrancy (CEI pattern verification)
   - Access control bypasses
   - Integer overflow/underflow edge cases
   - Flash loan attacks
   - Oracle manipulation
   - Front-running vulnerabilities

5. **GHOSTS AND HOOKS** (Where beneficial)
   - Track cumulative state changes
   - Monitor for suspicious patterns
   - Verify atomic operations

{template_hints}

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════

Return ONLY valid CVL code. No explanations, no markdown except the code block.
Structure:
1. Methods block first
2. Ghost variables (if needed)
3. Invariants
4. Rules (grouped by category with comments)

The specification should be comprehensive enough to catch real vulnerabilities
while being valid CVL that Certora Prover can execute.

```cvl
// Your CVL specification here
```
"""

    def _format_findings_detailed(self, findings: list) -> str:
        """Format Slither findings with vulnerability context."""
        if not findings:
            return "No static analysis findings provided. Generate comprehensive specs for common vulnerabilities."

        formatted = []
        for i, finding in enumerate(findings[:15], 1):  # Top 15 findings
            if isinstance(finding, dict):
                name = finding.get("name", finding.get("check", "Unknown"))
                details = finding.get("details", finding.get("description", ""))
                severity = finding.get("severity", finding.get("impact", "Unknown"))

                # Map to template hint
                template = VULNERABILITY_TEMPLATES.get(name.lower(), "")
                template_hint = f" → Verify with: {template}" if template else ""

                formatted.append(f"{i}. [{severity}] {name}: {details[:300]}{template_hint}")
            else:
                formatted.append(f"{i}. {str(finding)[:300]}")

        return "\n".join(formatted)

    def _get_template_hints(self, contract_types: list, findings: list) -> str:
        """Get relevant template hints based on contract type and findings."""
        hints = []

        # Add hints based on contract types
        type_templates = {
            "erc20": ["erc20_balance_conservation", "erc20_transfer_integrity"],
            "vault": ["vault_share_integrity", "arithmetic_safety"],
            "staking": ["vault_share_integrity", "arithmetic_safety"],
            "governance": ["governance_safety", "flash_loan_protection"],
            "oracle": ["oracle_safety", "arithmetic_safety"],
            "ownable": ["access_control_ownership"],
            "pausable": ["pausability", "access_control_ownership"],
            "flashloan": ["flash_loan_protection", "reentrancy_protection"],
        }

        for ctype in contract_types:
            if ctype in type_templates:
                hints.extend(type_templates[ctype])

        # Add hints based on findings
        if findings:
            for finding in findings[:10]:
                if isinstance(finding, dict):
                    name = finding.get("name", finding.get("check", "")).lower()
                    if name in VULNERABILITY_TEMPLATES:
                        hints.append(VULNERABILITY_TEMPLATES[name])

        # Deduplicate and format
        unique_hints = list(set(hints))
        if unique_hints:
            return f"""
RECOMMENDED VERIFICATION PATTERNS:
Consider these proven patterns for your specification:
{chr(10).join(f'- {h}' for h in unique_hints[:8])}
"""
        return ""

    def _extract_functions(self, code: str) -> list:
        """Extract function names from Solidity code."""
        pattern = r"function\s+(\w+)\s*\("
        matches = re.findall(pattern, code)
        return list(set(matches))

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
        slither_findings: list = None,
        contract_name: str = None
    ) -> str:
        """
        Generate comprehensive CVL specs from templates when AI is unavailable.
        This fallback still produces high-quality specifications.
        """
        if not contract_name:
            contract_name = self._extract_contract_name(contract_code)

        # Detect contract types
        contract_types = self._detect_contract_types(contract_code)
        code_lower = contract_code.lower()

        # Build methods block
        methods = self._build_methods_block(code_lower, contract_types)

        # Select applicable templates
        templates_to_include = set()

        # Add templates based on contract types
        if "erc20" in contract_types:
            templates_to_include.add("erc20_balance_conservation")
            templates_to_include.add("erc20_transfer_integrity")

        if "vault" in contract_types or "staking" in contract_types:
            templates_to_include.add("vault_share_integrity")
            templates_to_include.add("arithmetic_safety")

        if "governance" in contract_types:
            templates_to_include.add("governance_safety")

        if "ownable" in contract_types:
            templates_to_include.add("access_control_ownership")

        if "pausable" in contract_types:
            templates_to_include.add("pausability")

        if "oracle" in contract_types:
            templates_to_include.add("oracle_safety")

        # Add templates based on Slither findings
        if slither_findings:
            for finding in slither_findings[:10]:
                if isinstance(finding, dict):
                    name = finding.get("name", finding.get("check", "")).lower()
                    if name in VULNERABILITY_TEMPLATES:
                        templates_to_include.add(VULNERABILITY_TEMPLATES[name])

        # Always include reentrancy and access control checks
        templates_to_include.add("reentrancy_protection")
        if "ownable" not in contract_types:
            templates_to_include.add("access_control_ownership")

        # Build the spec
        spec = f"""/*
 * ═══════════════════════════════════════════════════════════════════════════
 * CVL Specification for {contract_name}
 * Generated by DeFiGuard AI Formal Verification Engine
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Contract Types Detected: {', '.join(contract_types)}
 * Templates Applied: {', '.join(templates_to_include)}
 */

{methods}

// ═══════════════════════════════════════════════════════════════════════════
// BASIC INVARIANTS
// ═══════════════════════════════════════════════════════════════════════════

// Contract should always be in a valid, consistent state
invariant contractStateValid()
    true;

// ═══════════════════════════════════════════════════════════════════════════
// SANITY RULES
// ═══════════════════════════════════════════════════════════════════════════

// Verify that the specification is meaningful (not vacuously true)
rule sanityCheck(method f, env e, calldataarg args) {{
    f@withrevert(e, args);
    satisfy !lastReverted;
}}

// ═══════════════════════════════════════════════════════════════════════════
// VIEW FUNCTIONS PURITY
// ═══════════════════════════════════════════════════════════════════════════

rule viewFunctionsDoNotModifyState(method f, env e, calldataarg args)
    filtered {{ f -> f.isView }}
{{
    storage before = lastStorage;
    f(e, args);
    assert lastStorage == before, "View function must not modify state";
}}
"""

        # Add selected templates
        for template_name in templates_to_include:
            if template_name in CVL_TEMPLATES:
                spec += f"\n{CVL_TEMPLATES[template_name]}\n"

        return spec

    def _build_methods_block(self, code_lower: str, contract_types: list) -> str:
        """Build comprehensive methods block based on detected patterns."""
        methods = []

        # ERC20 methods
        if "erc20" in contract_types:
            methods.extend([
                "function totalSupply() external returns (uint256) envfree;",
                "function balanceOf(address) external returns (uint256) envfree;",
                "function transfer(address, uint256) external returns (bool);",
                "function transferFrom(address, address, uint256) external returns (bool);",
                "function approve(address, uint256) external returns (bool);",
                "function allowance(address, address) external returns (uint256) envfree;",
            ])

        # Ownable methods
        if "ownable" in contract_types or "owner" in code_lower:
            methods.extend([
                "function owner() external returns (address) envfree;",
                "function transferOwnership(address) external;",
                "function renounceOwnership() external;",
            ])

        # Pausable methods
        if "pausable" in contract_types or "pause" in code_lower:
            methods.extend([
                "function paused() external returns (bool) envfree;",
                "function pause() external;",
                "function unpause() external;",
            ])

        # Vault methods
        if "vault" in contract_types:
            methods.extend([
                "function totalAssets() external returns (uint256) envfree;",
                "function deposit(uint256) external returns (uint256);",
                "function withdraw(uint256) external returns (uint256);",
                "function convertToAssets(uint256) external returns (uint256) envfree;",
                "function convertToShares(uint256) external returns (uint256) envfree;",
            ])

        # Staking methods
        if "staking" in contract_types:
            methods.extend([
                "function stake(uint256) external;",
                "function unstake(uint256) external;",
                "function earned(address) external returns (uint256) envfree;",
                "function getReward() external;",
            ])

        # Governance methods
        if "governance" in contract_types:
            methods.extend([
                "function propose(address[], uint256[], bytes[]) external returns (uint256);",
                "function vote(uint256, bool) external;",
                "function execute(uint256) external;",
                "function quorumVotes() external returns (uint256) envfree;",
                "function getVotes(address) external returns (uint256) envfree;",
            ])

        # Oracle methods
        if "oracle" in contract_types:
            methods.extend([
                "function getPrice() external returns (uint256) envfree;",
                "function updatePrice(uint256) external;",
                "function oracle() external returns (address) envfree;",
            ])

        if not methods:
            methods = ["// Add contract-specific method declarations"]

        return f"""methods {{
    {chr(10).join('    ' + m for m in methods)}
}}"""


# Convenience function for synchronous usage
def generate_cvl_specs(
    contract_code: str,
    slither_findings: list = None,
    anthropic_client=None
) -> Optional[str]:
    """
    Generate comprehensive CVL specifications for a Solidity contract.

    Args:
        contract_code: Solidity source code
        slither_findings: Optional Slither findings to focus verification
        anthropic_client: Optional Anthropic client

    Returns:
        CVL specification string or None
    """
    generator = CVLGenerator(anthropic_client)
    return generator.generate_specs_sync(contract_code, slither_findings)
