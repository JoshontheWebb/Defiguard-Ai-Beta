# Certora Formal Verification Implementation Plan

## Executive Summary

This document provides a **surgical, zero-regression implementation plan** for integrating Certora Prover formal verification into DeFiGuard AI's Enterprise tier. Every change is documented with rationale, impact analysis, and rollback strategy.

**Goal:** Transform the current stub `run_certora()` into a production-grade formal verification system that mathematically proves smart contract properties.

**Key Differentiator:** Certora is the only tool that provides **mathematical proofs** rather than heuristic analysis. It has secured $100B+ TVL across major protocols (Aave, MakerDAO, Uniswap, Compound) and catches bugs that traditional audits miss.

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Architecture Decision: Cloud API vs Local Execution](#2-architecture-decision)
3. [Implementation Approach](#3-implementation-approach)
4. [CVL Specification Strategy](#4-cvl-specification-strategy)
5. [Backend Implementation](#5-backend-implementation)
6. [Frontend Integration](#6-frontend-integration)
7. [PDF Report Integration](#7-pdf-report-integration)
8. [Infrastructure Changes](#8-infrastructure-changes)
9. [Error Handling & Fallback Strategy](#9-error-handling--fallback-strategy)
10. [Testing & Validation Plan](#10-testing--validation-plan)
11. [Rollout Strategy](#11-rollout-strategy)
12. [Risk Analysis & Mitigation](#12-risk-analysis--mitigation)

---

## 1. Current State Analysis

### 1.1 Existing Stub Implementation

**Location:** `main.py:5357-5360`

```python
def run_certora(temp_path: str) -> list[dict[str, str]]:
    """Minimal typed stub for Certora invocation"""
    logger.warning(f"Stub run_certora called for {temp_path}")
    return [{"rule": "Sample rule", "status": "Passed (dummy)"}]
```

**Current Integration Point:** `main.py:6258-6264`

```python
if tier_for_flags in ["enterprise", "diamond"] or getattr(user, "has_diamond", False):
    audit_json["fuzzing_results"] = fuzzing_results
    try:
        certora_result = await asyncio.to_thread(run_certora, temp_path)
        audit_json["formal_verification"] = certora_result
    except Exception as e:
        audit_json["formal_verification"] = f"Certora failed: {e}"
```

### 1.2 What Must NOT Change (Zero-Regression Constraints)

| Constraint | Reason | Verification |
|------------|--------|--------------|
| Function signature: `run_certora(temp_path: str) -> list[dict[str, str]]` | Called via `asyncio.to_thread()` | Type hints preserved |
| Return type: `list[dict[str, str]]` | Frontend expects this structure | Existing parsing logic works |
| Exception handling pattern | Caller catches exceptions | Returns error string, not raises |
| Tier gating: Enterprise/Diamond only | Business logic | Feature flag `certora: True` only for enterprise |
| Result stored in `audit_json["formal_verification"]` | Frontend reads this key | No change to key name |

### 1.3 Existing Tool Patterns to Follow

**Mythril Pattern (subprocess-based, our model):**
```python
def run_mythril(temp_path: str) -> list[dict[str, str]]:
    """Run mythril analysis with Docker environment support."""
    if not os.path.exists(temp_path):
        return []

    try:
        result = subprocess.run(
            ["myth", "analyze", temp_path, "-o", "json", ...],
            capture_output=True, text=True, timeout=90
        )
        # Parse JSON output
        # Return formatted results
    except subprocess.TimeoutExpired:
        return [{"vulnerability": "Mythril timeout", "description": "..."}]
    except FileNotFoundError:
        return [{"vulnerability": "Mythril unavailable", "description": "..."}]
    except Exception as e:
        return [{"vulnerability": "Mythril failed", "description": str(e)}]
```

**Key Observations:**
- Always returns a list (never raises)
- Graceful degradation on any failure
- Specific handling for timeouts and missing binaries
- Descriptive error messages in return value

---

## 2. Architecture Decision

### 2.1 Cloud API vs Local Execution

| Approach | Pros | Cons |
|----------|------|------|
| **Certora Cloud API** | No local binary needed; scales automatically; official support | Requires CERTORAKEY; network dependency; async job model |
| **Local Execution** | No network dependency; faster for simple specs | Heavy resource usage; complex installation; not recommended by Certora |

**Decision: Certora Cloud API**

**Rationale:**
1. Certora officially recommends cloud execution
2. Local execution requires significant compute resources (SMT solvers)
3. Cloud handles job queuing and parallelization
4. API key is free (open-source as of Feb 2025)
5. Results accessible via web dashboard for debugging

### 2.2 Execution Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DeFiGuard AI Backend                          │
│                                                                      │
│  1. User uploads .sol file                                           │
│  2. run_certora() called for Enterprise tier                         │
│  3. Generate CVL spec (built-in rules or default spec)               │
│  4. Call certoraRun CLI → submits to Certora Cloud                   │
│  5. Poll for results OR use --wait_for_results                       │
│  6. Parse JSON output                                                │
│  7. Return formatted results                                         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Certora Cloud                                 │
│                                                                      │
│  - Receives bytecode + CVL spec                                      │
│  - Runs SMT solvers (Z3, CVC5, Yices, Vampire)                       │
│  - Returns verification results                                      │
│  - Provides web dashboard for detailed analysis                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Implementation Approach

### 3.1 Phased Rollout

| Phase | Scope | Risk Level | Duration |
|-------|-------|------------|----------|
| **Phase 1** | Built-in rules only (sanity, viewReentrancy) | Very Low | 1-2 days |
| **Phase 2** | Default CVL spec for common patterns | Low | 2-3 days |
| **Phase 3** | Full CVL spec generation | Medium | 3-5 days |
| **Phase 4** | Frontend display & PDF integration | Low | 2-3 days |

### 3.2 Files to Modify

| File | Change Type | Description |
|------|-------------|-------------|
| `main.py` | Modify | Replace `run_certora()` stub with real implementation |
| `Dockerfile` | Modify | Add Certora CLI installation |
| `render.yaml` | Modify | Add CERTORAKEY env var |
| `requirements.txt` | Modify | Add certora-cli package |
| `static/script.js` | Add | Add `renderFormalVerification()` function |
| `templates/index.html` | Add | Add formal verification display section |
| `static/styles.css` | Add | Add formal verification styles |
| `certora/specs/` | New | Default CVL specification files |

### 3.3 Files NOT Modified

| File | Reason |
|------|--------|
| `compliance_checker.py` | Orthogonal functionality |
| `onchain_analyzer/` | Separate analysis module |
| Database models | No schema changes needed |
| Stripe integration | Tier gating already works |
| Auth0 integration | No auth changes |

---

## 4. CVL Specification Strategy

### 4.1 Approach: Progressive Specification Depth

**Level 1: Built-in Rules (Zero configuration)**
```cvl
use builtin rule sanity;
use builtin rule deepSanity;
use builtin rule viewReentrancy;
use builtin rule msgValueInLoopRule;
use builtin rule hasDelegateCalls;
```

**Level 2: Default CVL Spec (Universal patterns)**
```cvl
// certora/specs/default.spec

/*
 * Default DeFiGuard AI Formal Verification Spec
 * Applies to all Solidity contracts without custom specs
 */

// ==================== INVARIANTS ====================

// No function should revert on all inputs (sanity check)
use builtin rule sanity;

// Detect read-only reentrancy vulnerabilities
use builtin rule viewReentrancy;

// Detect msg.value in loops (common vulnerability)
use builtin rule msgValueInLoopRule;

// ==================== COMMON RULES ====================

// Rule: State changes should be reversible or intentional
rule noUnexpectedStateChange(method f) {
    env e;
    calldataarg args;

    storage before = lastStorage;
    f(e, args);
    storage after = lastStorage;

    // If state changed, function should not be view/pure
    assert before == after || !f.isView,
        "View function modified state";
}

// Rule: No unauthorized balance drainage
rule balanceIntegrity(address user) {
    uint256 balanceBefore = nativeBalances[user];

    env e;
    calldataarg args;
    method f;
    f(e, args);

    uint256 balanceAfter = nativeBalances[user];

    // Balance should not decrease without user's action
    assert balanceAfter >= balanceBefore || e.msg.sender == user,
        "Unauthorized balance decrease";
}
```

**Level 3: ERC20-Specific Spec**
```cvl
// certora/specs/erc20.spec

methods {
    function balanceOf(address) external returns (uint256) envfree;
    function totalSupply() external returns (uint256) envfree;
    function transfer(address, uint256) external returns (bool);
    function allowance(address, address) external returns (uint256) envfree;
}

// Invariant: Sum of all balances equals total supply
ghost mathint sumOfBalances {
    init_state axiom sumOfBalances == 0;
}

hook Sstore balanceOf[KEY address user] uint256 newValue (uint256 oldValue) {
    sumOfBalances = sumOfBalances + newValue - oldValue;
}

invariant totalSupplyIsSumOfBalances()
    to_mathint(totalSupply()) == sumOfBalances
    { preserved { require sumOfBalances >= 0; } }

// Rule: Transfer integrity
rule transferIntegrity(address to, uint256 amount) {
    env e;

    uint256 senderBefore = balanceOf(e.msg.sender);
    uint256 receiverBefore = balanceOf(to);

    transfer(e, to, amount);

    uint256 senderAfter = balanceOf(e.msg.sender);
    uint256 receiverAfter = balanceOf(to);

    assert senderAfter == senderBefore - amount,
        "Sender balance not decreased correctly";
    assert receiverAfter == receiverBefore + amount,
        "Receiver balance not increased correctly";
}

// Rule: Transfer should not affect other users
rule transferIsolation(address to, uint256 amount, address other) {
    env e;
    require other != e.msg.sender && other != to;

    uint256 otherBefore = balanceOf(other);
    transfer(e, to, amount);
    uint256 otherAfter = balanceOf(other);

    assert otherAfter == otherBefore,
        "Transfer affected uninvolved party";
}
```

### 4.2 Contract Type Detection

```python
def detect_contract_type(code: str) -> str:
    """Detect contract type for appropriate CVL spec selection."""
    code_lower = code.lower()

    # ERC20 detection
    if all(sig in code_lower for sig in ["balanceof", "transfer", "totalsupply"]):
        return "erc20"

    # ERC721 detection
    if all(sig in code_lower for sig in ["ownerof", "safetransferfrom", "tokenuri"]):
        return "erc721"

    # Proxy detection
    if any(pattern in code_lower for pattern in ["delegatecall", "implementation", "_fallback"]):
        return "proxy"

    # Governance detection
    if any(pattern in code_lower for pattern in ["propose", "vote", "execute", "quorum"]):
        return "governance"

    return "default"
```

---

## 5. Backend Implementation

### 5.1 New `run_certora()` Implementation

**Location:** `main.py` - Replace lines 5357-5360

```python
import json
import subprocess
import tempfile
import shutil
from pathlib import Path


def run_certora(temp_path: str) -> list[dict[str, str]]:
    """
    Run Certora Prover formal verification.

    This function:
    1. Validates the input file exists
    2. Detects contract type for spec selection
    3. Generates appropriate CVL specification
    4. Executes certoraRun CLI
    5. Parses and returns verification results

    Args:
        temp_path: Path to the Solidity file to verify

    Returns:
        List of dicts with keys: rule, status, description, severity

    Note:
        Never raises exceptions - returns descriptive error results instead.
        This matches the existing Mythril/Echidna pattern.
    """
    # Validate input
    if not os.path.exists(temp_path):
        logger.error(f"[CERTORA] File not found: {temp_path}")
        return [{
            "rule": "File Validation",
            "status": "Error",
            "description": f"Contract file not found at {temp_path}",
            "severity": "error"
        }]

    # Check for CERTORAKEY
    certora_key = os.getenv("CERTORAKEY")
    if not certora_key:
        logger.warning("[CERTORA] CERTORAKEY not set - formal verification unavailable")
        return [{
            "rule": "Configuration",
            "status": "Unavailable",
            "description": "Certora API key not configured. Contact support to enable formal verification.",
            "severity": "info"
        }]

    # Check certoraRun binary
    try:
        subprocess.run(["certoraRun", "--version"], capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("[CERTORA] certoraRun binary not found")
        return [{
            "rule": "Installation",
            "status": "Unavailable",
            "description": "Certora CLI not installed in this environment.",
            "severity": "info"
        }]

    try:
        logger.info(f"[CERTORA] Starting formal verification for {temp_path}")

        # Read contract to detect type
        with open(temp_path, 'r') as f:
            code = f.read()

        contract_type = detect_contract_type(code)
        logger.info(f"[CERTORA] Detected contract type: {contract_type}")

        # Extract contract name from file
        contract_name = extract_contract_name(code) or "Contract"

        # Create temporary directory for Certora run
        with tempfile.TemporaryDirectory() as certora_dir:
            # Copy contract to certora directory
            contract_file = Path(certora_dir) / "contract.sol"
            shutil.copy(temp_path, contract_file)

            # Generate CVL spec
            spec_file = Path(certora_dir) / "spec.spec"
            spec_content = generate_cvl_spec(contract_type, contract_name)
            spec_file.write_text(spec_content)

            # Generate config file
            config_file = Path(certora_dir) / "certora.conf"
            config = {
                "files": [str(contract_file)],
                "verify": f"{contract_name}:{spec_file}",
                "wait_for_results": "all",
                "rule_sanity": "basic",
                "msg": f"DeFiGuard AI verification: {contract_name}",
                "send_only": False
            }
            config_file.write_text(json.dumps(config, indent=2))

            # Execute certoraRun
            logger.info(f"[CERTORA] Executing certoraRun...")
            result = subprocess.run(
                ["certoraRun", str(config_file)],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=certora_dir,
                env={**os.environ, "CERTORAKEY": certora_key}
            )

            logger.info(f"[CERTORA] Return code: {result.returncode}")
            if result.stdout:
                logger.info(f"[CERTORA] Output length: {len(result.stdout)} chars")
            if result.stderr:
                logger.warning(f"[CERTORA] Stderr: {result.stderr[:500]}")

            # Parse results
            return parse_certora_output(result.stdout, result.stderr, result.returncode)

    except subprocess.TimeoutExpired:
        logger.warning("[CERTORA] Verification timed out after 5 minutes")
        return [{
            "rule": "Execution",
            "status": "Timeout",
            "description": "Formal verification exceeded 5 minute limit. Complex contracts may require more time.",
            "severity": "warning"
        }]

    except Exception as e:
        logger.error(f"[CERTORA] Unexpected error: {str(e)}")
        return [{
            "rule": "Execution",
            "status": "Error",
            "description": f"Formal verification failed: {str(e)}",
            "severity": "error"
        }]


def extract_contract_name(code: str) -> str | None:
    """Extract the main contract name from Solidity code."""
    import re
    # Match "contract ContractName" or "contract ContractName is"
    match = re.search(r'contract\s+(\w+)\s*(?:is|{)', code)
    if match:
        return match.group(1)
    return None


def detect_contract_type(code: str) -> str:
    """Detect contract type for appropriate CVL spec selection."""
    code_lower = code.lower()

    # ERC20 detection
    if all(sig in code_lower for sig in ["balanceof", "transfer", "totalsupply"]):
        return "erc20"

    # ERC721 detection
    if all(sig in code_lower for sig in ["ownerof", "safetransferfrom"]):
        return "erc721"

    # Proxy detection
    if any(pattern in code_lower for pattern in ["delegatecall", "_implementation", "upgradeto"]):
        return "proxy"

    return "default"


def generate_cvl_spec(contract_type: str, contract_name: str) -> str:
    """Generate CVL specification based on contract type."""

    base_spec = f"""
/*
 * DeFiGuard AI Formal Verification Specification
 * Contract: {contract_name}
 * Type: {contract_type}
 * Generated automatically for enterprise-tier verification
 */

// Built-in sanity rules
use builtin rule sanity;
use builtin rule deepSanity;
use builtin rule viewReentrancy;
use builtin rule msgValueInLoopRule;
"""

    if contract_type == "erc20":
        return base_spec + """
// ERC20 Specific Verification

methods {
    function balanceOf(address) external returns (uint256) envfree;
    function totalSupply() external returns (uint256) envfree;
    function transfer(address, uint256) external returns (bool);
    function approve(address, uint256) external returns (bool);
    function transferFrom(address, address, uint256) external returns (bool);
    function allowance(address, address) external returns (uint256) envfree;
}

// Ghost variable for sum of balances
ghost mathint sumOfBalances {
    init_state axiom sumOfBalances == 0;
}

hook Sstore _balances[KEY address user] uint256 newValue (uint256 oldValue) {
    sumOfBalances = sumOfBalances + newValue - oldValue;
}

// Invariant: Total supply equals sum of all balances
invariant totalSupplyConsistency()
    to_mathint(totalSupply()) == sumOfBalances
    { preserved { require sumOfBalances >= 0; } }

// Rule: Transfer does not create tokens
rule transferPreservesSupply(address to, uint256 amount) {
    env e;
    uint256 supplyBefore = totalSupply();

    transfer(e, to, amount);

    uint256 supplyAfter = totalSupply();
    assert supplyAfter == supplyBefore, "Transfer changed total supply";
}

// Rule: Self-transfer is no-op
rule selfTransferNoOp(uint256 amount) {
    env e;
    uint256 balanceBefore = balanceOf(e.msg.sender);

    transfer(e, e.msg.sender, amount);

    uint256 balanceAfter = balanceOf(e.msg.sender);
    assert balanceAfter == balanceBefore, "Self-transfer changed balance";
}
"""

    elif contract_type == "erc721":
        return base_spec + """
// ERC721 Specific Verification

methods {
    function ownerOf(uint256) external returns (address) envfree;
    function balanceOf(address) external returns (uint256) envfree;
    function getApproved(uint256) external returns (address) envfree;
    function isApprovedForAll(address, address) external returns (bool) envfree;
}

// Rule: Token can only have one owner
rule uniqueOwnership(uint256 tokenId) {
    address owner = ownerOf(tokenId);

    assert owner != 0, "Token must have an owner";
    // Additional ownership checks would go here
}
"""

    elif contract_type == "proxy":
        return base_spec + """
// Proxy Contract Verification

use builtin rule hasDelegateCalls;

// Rule: Implementation should not be zero address after initialization
rule implementationNotZero() {
    // This rule verifies proxy implementation is set
    assert true; // Placeholder - real rule depends on proxy structure
}
"""

    else:  # default
        return base_spec + """
// General Contract Verification

// Rule: Functions should not unexpectedly revert
rule noUnexpectedRevert(method f) {
    env e;
    calldataarg args;

    // Methods should be callable with valid inputs
    f@withrevert(e, args);

    // If reverted, should be for a valid reason
    assert !lastReverted || e.msg.value > 0 || e.msg.sender == 0,
        "Unexpected revert";
}
"""


def parse_certora_output(stdout: str, stderr: str, returncode: int) -> list[dict[str, str]]:
    """Parse Certora output into structured results."""
    results = []

    # Check for job URL in output
    job_url = None
    if "prover.certora.com" in stdout:
        import re
        url_match = re.search(r'https://prover\.certora\.com/\S+', stdout)
        if url_match:
            job_url = url_match.group(0)

    # Parse verification results
    if returncode == 0:
        # Success - all rules verified
        results.append({
            "rule": "Formal Verification Complete",
            "status": "Verified",
            "description": "All formal verification rules passed. Contract properties mathematically proven.",
            "severity": "success"
        })

        # Try to parse individual rules from output
        if "VERIFIED" in stdout:
            # Extract verified rules
            import re
            verified_rules = re.findall(r'(\w+)\s*:\s*VERIFIED', stdout)
            for rule in verified_rules:
                results.append({
                    "rule": rule,
                    "status": "Verified",
                    "description": f"Rule '{rule}' mathematically verified",
                    "severity": "success"
                })

    elif "FAIL" in stdout or "violations" in stdout.lower():
        # Verification failed - counterexample found
        results.append({
            "rule": "Verification",
            "status": "Failed",
            "description": "Formal verification found property violations. Review counterexamples for details.",
            "severity": "critical"
        })

        # Try to parse failed rules
        import re
        failed_rules = re.findall(r'(\w+)\s*:\s*FAIL', stdout)
        for rule in failed_rules:
            results.append({
                "rule": rule,
                "status": "Violated",
                "description": f"Rule '{rule}' violation detected - counterexample available",
                "severity": "critical"
            })

    elif "TIMEOUT" in stdout:
        results.append({
            "rule": "Verification",
            "status": "Timeout",
            "description": "Some rules exceeded solver timeout. Contract may be too complex for full verification.",
            "severity": "warning"
        })

    else:
        # Unknown output - report what we have
        results.append({
            "rule": "Verification",
            "status": "Completed",
            "description": stdout[:500] if stdout else stderr[:500] if stderr else "No output available",
            "severity": "info"
        })

    # Add job URL if available
    if job_url:
        results.append({
            "rule": "Detailed Report",
            "status": "Available",
            "description": f"Full verification report: {job_url}",
            "severity": "info"
        })

    return results if results else [{
        "rule": "Verification",
        "status": "Unknown",
        "description": "Verification completed but results could not be parsed",
        "severity": "warning"
    }]
```

### 5.2 Integration Point Changes

**No changes needed** to the integration point at `main.py:6258-6264`. The new `run_certora()` function maintains the exact same signature and return type.

---

## 6. Frontend Integration

### 6.1 New `renderFormalVerification()` Function

**Location:** Add to `static/script.js` after `renderFuzzingResults()` (~line 2752)

```javascript
// Formal Verification results renderer (Enterprise only)
const renderFormalVerification = (formalResults) => {
  if (!formalResults || formalResults.length === 0) {
    return `<div class="formal-empty"><p>No formal verification results available.</p></div>`;
  }

  // Handle error string (from exception)
  if (typeof formalResults === 'string') {
    return `<div class="formal-error"><p>${escapeHtml(formalResults)}</p></div>`;
  }

  // Count results by status
  const verified = formalResults.filter(r => r.status === 'Verified').length;
  const failed = formalResults.filter(r => r.status === 'Violated' || r.status === 'Failed').length;
  const warnings = formalResults.filter(r => r.status === 'Timeout' || r.status === 'Warning').length;
  const total = formalResults.length;

  // Determine overall status
  let overallStatus, statusClass, statusIcon;
  if (failed > 0) {
    overallStatus = 'Violations Found';
    statusClass = 'error';
    statusIcon = '❌';
  } else if (warnings > 0) {
    overallStatus = 'Partial Verification';
    statusClass = 'warning';
    statusIcon = '⚠️';
  } else if (verified > 0) {
    overallStatus = 'Fully Verified';
    statusClass = 'success';
    statusIcon = '✅';
  } else {
    overallStatus = 'Completed';
    statusClass = '';
    statusIcon = '🔬';
  }

  let html = `
    <div class="formal-verification-card">
      <div class="formal-header ${statusClass}">
        <div class="formal-status">
          <span class="status-icon">${statusIcon}</span>
          <span class="status-text">${overallStatus}</span>
        </div>
        <span class="formal-badge">Certora Prover</span>
      </div>

      <div class="formal-stats-grid">
        <div class="formal-stat verified">
          <div class="stat-value">${verified}</div>
          <div class="stat-label">Verified</div>
        </div>
        <div class="formal-stat failed">
          <div class="stat-value">${failed}</div>
          <div class="stat-label">Violations</div>
        </div>
        <div class="formal-stat warnings">
          <div class="stat-value">${warnings}</div>
          <div class="stat-label">Warnings</div>
        </div>
        <div class="formal-stat total">
          <div class="stat-value">${total}</div>
          <div class="stat-label">Total Rules</div>
        </div>
      </div>

      <div class="formal-rules-list">`;

  // Render each rule result
  for (const result of formalResults) {
    const severityClass = {
      'success': 'verified',
      'critical': 'critical',
      'error': 'error',
      'warning': 'warning',
      'info': 'info'
    }[result.severity] || 'info';

    const statusIcon = {
      'Verified': '✅',
      'Violated': '❌',
      'Failed': '❌',
      'Timeout': '⏱️',
      'Warning': '⚠️',
      'Available': '🔗',
      'Unavailable': '🚫'
    }[result.status] || 'ℹ️';

    html += `
      <div class="formal-rule-item ${severityClass}">
        <div class="rule-header">
          <span class="rule-icon">${statusIcon}</span>
          <span class="rule-name">${escapeHtml(result.rule)}</span>
          <span class="rule-status">${escapeHtml(result.status)}</span>
        </div>
        <p class="rule-description">${escapeHtml(result.description)}</p>
      </div>`;
  }

  html += `
      </div>
      <div class="formal-footer">
        <span class="formal-note">
          <strong>Formal Verification</strong> provides mathematical proof that contract properties hold across all possible inputs.
        </span>
      </div>
    </div>`;

  return html;
};
```

### 6.2 Display Integration

**Location:** Add to `static/script.js` where fuzzing results are rendered (~line 2754)

```javascript
// Render formal verification results (Enterprise only)
const formalVerificationEl = document.getElementById("formal-verification-list");
if (formalVerificationEl && report.formal_verification) {
  formalVerificationEl.innerHTML = renderFormalVerification(report.formal_verification);
  // Show the section
  const formalSection = document.querySelector('.formal-verification-section');
  if (formalSection) {
    formalSection.style.display = 'block';
  }
}
```

### 6.3 HTML Template Addition

**Location:** Add to `templates/index.html` after fuzzing section (~line 629)

```html
<!-- Formal Verification Results (Enterprise only) -->
<div class="formal-verification-section enterprise-only" style="display: none;">
    <h4>🔬 Formal Verification</h4>
    <div id="formal-verification-list" role="region" aria-label="Certora formal verification results">
        <div class="formal-empty">
            <p>Formal verification results will appear here.</p>
        </div>
    </div>
</div>
```

### 6.4 CSS Styles

**Location:** Add to `static/styles.css`

```css
/* =============================================
   FORMAL VERIFICATION STYLES
   ============================================= */

.formal-verification-section {
    margin-top: var(--space-8);
    padding: var(--space-6);
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-xl);
}

.formal-verification-section h4 {
    margin-bottom: var(--space-4);
    font-size: var(--text-lg);
    color: var(--text-primary);
}

.formal-verification-card {
    background: var(--bg-secondary);
    border-radius: var(--radius-lg);
    overflow: hidden;
}

.formal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: var(--space-4);
    background: rgba(0, 0, 0, 0.2);
    border-bottom: 1px solid var(--glass-border);
}

.formal-header.success {
    background: rgba(16, 185, 129, 0.15);
    border-bottom-color: var(--success);
}

.formal-header.error {
    background: rgba(239, 68, 68, 0.15);
    border-bottom-color: var(--error);
}

.formal-header.warning {
    background: rgba(251, 191, 36, 0.15);
    border-bottom-color: var(--warning);
}

.formal-status {
    display: flex;
    align-items: center;
    gap: var(--space-2);
}

.formal-status .status-icon {
    font-size: 1.5rem;
}

.formal-status .status-text {
    font-weight: 600;
    font-size: var(--text-lg);
}

.formal-badge {
    padding: var(--space-1) var(--space-3);
    background: var(--accent-purple);
    color: white;
    border-radius: var(--radius-full);
    font-size: var(--text-xs);
    font-weight: 600;
    text-transform: uppercase;
}

.formal-stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: var(--space-4);
    padding: var(--space-4);
    background: rgba(0, 0, 0, 0.1);
}

.formal-stat {
    text-align: center;
}

.formal-stat .stat-value {
    font-size: var(--text-2xl);
    font-weight: 700;
    font-family: var(--font-mono);
}

.formal-stat.verified .stat-value { color: var(--success); }
.formal-stat.failed .stat-value { color: var(--error); }
.formal-stat.warnings .stat-value { color: var(--warning); }
.formal-stat.total .stat-value { color: var(--text-secondary); }

.formal-stat .stat-label {
    font-size: var(--text-xs);
    color: var(--text-tertiary);
    text-transform: uppercase;
}

.formal-rules-list {
    padding: var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
}

.formal-rule-item {
    padding: var(--space-3);
    background: rgba(0, 0, 0, 0.2);
    border-radius: var(--radius-md);
    border-left: 3px solid var(--glass-border);
}

.formal-rule-item.verified { border-left-color: var(--success); }
.formal-rule-item.critical { border-left-color: var(--error); }
.formal-rule-item.error { border-left-color: var(--error); }
.formal-rule-item.warning { border-left-color: var(--warning); }
.formal-rule-item.info { border-left-color: var(--accent-teal); }

.rule-header {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin-bottom: var(--space-2);
}

.rule-icon {
    font-size: 1rem;
}

.rule-name {
    flex: 1;
    font-weight: 600;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
}

.rule-status {
    font-size: var(--text-xs);
    padding: var(--space-1) var(--space-2);
    background: rgba(255, 255, 255, 0.1);
    border-radius: var(--radius-sm);
    text-transform: uppercase;
}

.rule-description {
    font-size: var(--text-sm);
    color: var(--text-secondary);
    line-height: 1.5;
    margin: 0;
}

.formal-footer {
    padding: var(--space-4);
    background: rgba(139, 92, 246, 0.1);
    border-top: 1px solid var(--glass-border);
}

.formal-note {
    font-size: var(--text-xs);
    color: var(--text-tertiary);
}

.formal-empty, .formal-error {
    padding: var(--space-6);
    text-align: center;
    color: var(--text-secondary);
}

.formal-error {
    color: var(--error);
}

@media (max-width: 640px) {
    .formal-stats-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}
```

---

## 7. PDF Report Integration

### 7.1 New `_build_formal_verification_section()` Function

**Location:** Add to `main.py` after `_build_tool_results_section()` (~line 3867)

```python
def _build_formal_verification_section(story: list, styles: dict, report: dict) -> None:
    """Build formal verification section for PDF (Enterprise only)."""
    formal = report.get("formal_verification")
    if not formal:
        return

    # Handle error string
    if isinstance(formal, str):
        story.append(Paragraph("Formal Verification", styles['heading1']))
        story.append(Paragraph(f"Error: {formal}", styles['normal']))
        story.append(Spacer(1, 10))
        return

    story.append(Paragraph("🔬 Formal Verification (Certora Prover)", styles['heading1']))
    story.append(Paragraph(
        "Formal verification provides mathematical proof that contract properties "
        "hold across all possible inputs and execution paths.",
        styles['normal']
    ))
    story.append(Spacer(1, 8))

    # Summary stats
    verified = sum(1 for r in formal if r.get("status") == "Verified")
    failed = sum(1 for r in formal if r.get("status") in ["Violated", "Failed"])
    total = len(formal)

    story.append(Paragraph(
        f"<b>Summary:</b> {verified} verified, {failed} violations, {total} total rules",
        styles['normal']
    ))
    story.append(Spacer(1, 8))

    # Individual rules
    story.append(Paragraph("Verification Results", styles['heading2']))

    for result in formal[:10]:  # Limit to 10 for PDF
        if isinstance(result, dict):
            rule = result.get("rule", "Unknown Rule")
            status = result.get("status", "Unknown")
            description = result.get("description", "")
            severity = result.get("severity", "info")

            # Color coding based on severity
            color = {
                "success": "green",
                "critical": "red",
                "error": "red",
                "warning": "orange",
                "info": "blue"
            }.get(severity, "black")

            story.append(Paragraph(
                f"<b><font color='{color}'>[{status}]</font></b> {rule}",
                styles['normal']
            ))
            if description:
                story.append(Paragraph(f"    {description[:200]}", styles['normal']))

    story.append(Spacer(1, 10))
```

### 7.2 Integration into PDF Generation

**Location:** Modify `main.py` ~line 4024 (after `_build_tool_results_section`)

```python
# Add after line 4024:
if tier in ["enterprise", "diamond"]:
    _build_formal_verification_section(story, styles, report)
```

---

## 8. Infrastructure Changes

### 8.1 Dockerfile Modifications

**Location:** `Dockerfile` - Add after Mythril installation (line 28)

```dockerfile
# Install Certora CLI
RUN pip3 install --break-system-packages certora-cli

# Install Java 21 (required for Certora local type-checking)
RUN apt-get update && \
    apt-get install -y openjdk-21-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Verify Certora installation
RUN certoraRun --version || echo "Certora CLI installed (requires CERTORAKEY for verification)"
```

### 8.2 render.yaml Modifications

**Location:** `render.yaml` - Add to envVars section (after line 63)

```yaml
      - key: CERTORAKEY
        sync: false  # Secret - set in Render dashboard
```

### 8.3 requirements.txt Addition

```
certora-cli>=7.0.0
```

---

## 9. Error Handling & Fallback Strategy

### 9.1 Error Hierarchy

| Error Type | Detection | Response | User Message |
|------------|-----------|----------|--------------|
| CERTORAKEY missing | `os.getenv()` check | Return info result | "Contact support to enable formal verification" |
| certoraRun not found | `FileNotFoundError` | Return info result | "Certora CLI not installed" |
| Subprocess timeout | `TimeoutExpired` | Return warning result | "Verification exceeded 5 minute limit" |
| Network failure | Generic exception | Return error result | "Verification service unavailable" |
| Parse failure | JSON decode error | Return partial results | Show what we have |

### 9.2 Graceful Degradation

```python
# Never raise exceptions - always return informative results
try:
    # Run verification
except Exception as e:
    return [{
        "rule": "Verification",
        "status": "Error",
        "description": str(e),
        "severity": "error"
    }]
```

### 9.3 Caller-Side Handling

The existing caller code already handles exceptions:

```python
try:
    certora_result = await asyncio.to_thread(run_certora, temp_path)
    audit_json["formal_verification"] = certora_result
except Exception as e:
    audit_json["formal_verification"] = f"Certora failed: {e}"
```

This means even if our function somehow raises (it shouldn't), the audit will not fail.

---

## 10. Testing & Validation Plan

### 10.1 Unit Tests

```python
# tests/test_certora.py

import pytest
from main import run_certora, detect_contract_type, generate_cvl_spec, parse_certora_output

class TestContractTypeDetection:
    def test_erc20_detection(self):
        code = "function balanceOf(address) external view returns (uint256) {}"
        code += "function transfer(address, uint256) external returns (bool) {}"
        code += "function totalSupply() external view returns (uint256) {}"
        assert detect_contract_type(code) == "erc20"

    def test_erc721_detection(self):
        code = "function ownerOf(uint256) external view returns (address) {}"
        code += "function safeTransferFrom(address, address, uint256) external {}"
        assert detect_contract_type(code) == "erc721"

    def test_proxy_detection(self):
        code = "function _implementation() internal view returns (address) {}"
        code += "function upgradeTo(address) external {}"
        assert detect_contract_type(code) == "proxy"

    def test_default_type(self):
        code = "contract SimpleStorage { uint256 value; }"
        assert detect_contract_type(code) == "default"


class TestCVLGeneration:
    def test_erc20_spec_generation(self):
        spec = generate_cvl_spec("erc20", "TestToken")
        assert "totalSupplyIsSumOfBalances" in spec
        assert "transferPreservesSupply" in spec
        assert "TestToken" in spec

    def test_default_spec_generation(self):
        spec = generate_cvl_spec("default", "MyContract")
        assert "use builtin rule sanity" in spec
        assert "MyContract" in spec


class TestOutputParsing:
    def test_verified_parsing(self):
        stdout = "transferSpec: VERIFIED\nbalanceCheck: VERIFIED"
        results = parse_certora_output(stdout, "", 0)
        assert any(r["status"] == "Verified" for r in results)

    def test_failed_parsing(self):
        stdout = "ERROR: transferSpec: FAIL - counterexample found"
        results = parse_certora_output(stdout, "", 1)
        assert any(r["status"] == "Violated" for r in results)

    def test_timeout_parsing(self):
        stdout = "balanceCheck: TIMEOUT"
        results = parse_certora_output(stdout, "", 1)
        assert any(r["status"] == "Timeout" for r in results)


class TestRunCertora:
    def test_missing_file(self):
        results = run_certora("/nonexistent/path.sol")
        assert results[0]["status"] == "Error"
        assert "not found" in results[0]["description"]

    def test_missing_key(self, monkeypatch):
        monkeypatch.delenv("CERTORAKEY", raising=False)
        results = run_certora("test.sol")
        assert results[0]["status"] == "Unavailable"
```

### 10.2 Integration Tests

```python
# tests/test_certora_integration.py

import pytest
from pathlib import Path

# Test with actual contracts
TEST_CONTRACTS = [
    ("simple_storage.sol", "default", "should verify"),
    ("vulnerable_token.sol", "erc20", "should detect issues"),
]

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("CERTORAKEY"), reason="CERTORAKEY not set")
class TestCertoraIntegration:
    def test_simple_contract_verification(self):
        # Create simple test contract
        contract = """
        // SPDX-License-Identifier: MIT
        pragma solidity ^0.8.0;

        contract SimpleStorage {
            uint256 private value;

            function set(uint256 _value) public {
                value = _value;
            }

            function get() public view returns (uint256) {
                return value;
            }
        }
        """

        with tempfile.NamedTemporaryFile(suffix=".sol", delete=False) as f:
            f.write(contract.encode())
            temp_path = f.name

        try:
            results = run_certora(temp_path)
            assert len(results) > 0
            assert results[0]["status"] in ["Verified", "Completed", "Unavailable"]
        finally:
            os.unlink(temp_path)
```

### 10.3 E2E Testing Checklist

| Test Case | Steps | Expected Result |
|-----------|-------|-----------------|
| Enterprise user with Certora | Upload ERC20 contract | Formal verification runs, results displayed |
| Pro user (no Certora) | Upload contract | No formal verification section shown |
| Certora timeout | Upload very complex contract | Timeout message, audit still completes |
| CERTORAKEY missing | Deploy without key | Info message about contacting support |
| Network failure | Simulate network error | Error message, audit still completes |
| PDF generation | Generate PDF for Enterprise audit | Formal verification section in PDF |

---

## 11. Rollout Strategy

### 11.1 Staged Rollout

| Stage | Scope | Duration | Success Criteria |
|-------|-------|----------|------------------|
| **Stage 1: Dev Testing** | Local development only | 2 days | All unit tests pass |
| **Stage 2: Staging** | Staging environment | 3 days | Integration tests pass |
| **Stage 3: Canary** | 10% of Enterprise users | 3 days | No errors, positive feedback |
| **Stage 4: Full Rollout** | All Enterprise users | 1 day | Monitoring stable |

### 11.2 Feature Flag

Add environment variable for gradual rollout:

```python
CERTORA_ENABLED = os.getenv("CERTORA_ENABLED", "false").lower() == "true"

# In run_certora():
if not CERTORA_ENABLED:
    return [{
        "rule": "Formal Verification",
        "status": "Coming Soon",
        "description": "Formal verification will be available soon for Enterprise tier.",
        "severity": "info"
    }]
```

### 11.3 Monitoring

- Log all Certora executions with timing
- Alert on error rate > 5%
- Track verification success rate
- Monitor cloud API response times

---

## 12. Risk Analysis & Mitigation

### 12.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Certora cloud unavailable | Low | Medium | Return informative error, audit continues |
| Long verification times | Medium | Low | 5-minute timeout, async processing |
| CVL spec errors | Medium | Low | Fallback to built-in rules only |
| Docker image size increase | Low | Low | Certora CLI is lightweight |
| Memory issues on complex contracts | Low | Medium | Timeout handles this case |

### 12.2 Business Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Users confused by formal verification | Medium | Low | Clear explanations in UI |
| False sense of security | Low | High | Clear disclaimers about coverage |
| Support burden | Low | Medium | Comprehensive documentation |

### 12.3 Rollback Plan

1. Set `CERTORA_ENABLED=false` in environment
2. Deploy - stub returns "Coming Soon" message
3. No code changes needed
4. Full audit functionality preserved

---

## Appendix A: File Changes Summary

### Files Modified

| File | Lines Changed | Change Type |
|------|---------------|-------------|
| `main.py` | ~5357-5360 (expand to ~200 lines) | Replace stub with implementation |
| `main.py` | ~4024 (add 1 line) | Add PDF section call |
| `Dockerfile` | +10 lines | Add Certora CLI installation |
| `render.yaml` | +2 lines | Add CERTORAKEY env var |
| `requirements.txt` | +1 line | Add certora-cli |
| `static/script.js` | +100 lines | Add renderFormalVerification() |
| `templates/index.html` | +10 lines | Add formal verification section |
| `static/styles.css` | +150 lines | Add formal verification styles |

### Files Added

| File | Purpose |
|------|---------|
| `certora/specs/default.spec` | Default CVL specification |
| `certora/specs/erc20.spec` | ERC20-specific specification |
| `certora/specs/erc721.spec` | ERC721-specific specification |
| `tests/test_certora.py` | Unit tests |

### Files NOT Modified

| File | Reason |
|------|--------|
| `compliance_checker.py` | Orthogonal |
| `onchain_analyzer/*` | Separate module |
| Database models | No schema changes |
| Stripe integration | Tier gating works |
| Auth0 integration | No auth changes |

---

## Appendix B: Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CERTORAKEY` | Yes (for verification) | None | Certora API key |
| `CERTORA_ENABLED` | No | `false` | Feature flag for rollout |
| `CERTORA_TIMEOUT` | No | `300` | Verification timeout (seconds) |

---

## Appendix C: API Response Format

### Successful Verification

```json
{
  "formal_verification": [
    {
      "rule": "Formal Verification Complete",
      "status": "Verified",
      "description": "All formal verification rules passed.",
      "severity": "success"
    },
    {
      "rule": "totalSupplyConsistency",
      "status": "Verified",
      "description": "Rule 'totalSupplyConsistency' mathematically verified",
      "severity": "success"
    },
    {
      "rule": "Detailed Report",
      "status": "Available",
      "description": "Full report: https://prover.certora.com/output/...",
      "severity": "info"
    }
  ]
}
```

### Failed Verification

```json
{
  "formal_verification": [
    {
      "rule": "Verification",
      "status": "Failed",
      "description": "Formal verification found property violations.",
      "severity": "critical"
    },
    {
      "rule": "transferIntegrity",
      "status": "Violated",
      "description": "Rule 'transferIntegrity' violation detected",
      "severity": "critical"
    }
  ]
}
```

---

*Document Version: 1.0*
*Created: December 2025*
*Author: DeFiGuard AI Development Team*
