# Certora Formal Verification Integration Plan

## Executive Summary

This document outlines a surgical, non-breaking implementation plan to integrate Certora formal verification into DeFiGuard AI's Enterprise tier. Every modification is designed to build upon existing architecture without affecting current functionality.

---

## Table of Contents

1. [Current Architecture Analysis](#1-current-architecture-analysis)
2. [Files Requiring Modification](#2-files-requiring-modification)
3. [Integration Architecture](#3-integration-architecture)
4. [Implementation Phases](#4-implementation-phases)
5. [Risk Mitigation](#5-risk-mitigation)
6. [Testing Strategy](#6-testing-strategy)
7. [Rollback Plan](#7-rollback-plan)

---

## 1. Current Architecture Analysis

### 1.1 Tool Integration Pattern

All security tools follow the same pattern in `main.py`:

```python
def run_<tool>(temp_path: str) -> list[dict[str, str]]:
    """Run <tool> analysis."""
    # 1. Validate file exists
    # 2. Execute subprocess/library call
    # 3. Parse output
    # 4. Return list of findings as dicts
    # 5. On error: return empty list or error dict (non-blocking)
```

**Key Insight**: Tools are isolated. One failing does NOT affect others.

### 1.2 Audit Orchestration Flow

```
/audit endpoint (line 5638)
    ↓
Phase 1: Slither (line 5952) → 10% progress
    ↓
Phase 2: Mythril (line 5966) → 30% progress
    ↓
Phase 3: Echidna (line 5980) → 50% progress [if fuzzing_enabled]
    ↓
[CERTORA INSERTION POINT] → 60% progress [if certora_enabled]
    ↓
Phase 4: AI Analysis (line 6077) → 70% progress
    ↓
Phase 5: On-chain (line 6001) → 80% progress [if contract_address]
    ↓
Phase 6: PDF Generation (line 6320) → 95% progress
    ↓
Response (line 6541)
```

### 1.3 Feature Flag System

Enterprise tier already has `certora: True` at line 2858:

```python
"enterprise": {
    ...
    "certora": True,  # Already defined!
    ...
}
```

**Key Insight**: No feature flag changes needed. Flag exists but function is stub.

### 1.4 Current Certora Stub

Location: `main.py:5357-5360`

```python
def run_certora(temp_path: str) -> list[dict[str, str]]:
    """Minimal typed stub for Certora invocation"""
    logger.warning(f"Stub run_certora called for {temp_path}")
    return [{"rule": "Sample rule", "status": "Passed (dummy)"}]
```

**Key Insight**: Stub exists with correct signature. We replace internals only.

---

## 2. Files Requiring Modification

### 2.1 Backend (Python)

| File | Lines | Change Type | Risk Level |
|------|-------|-------------|------------|
| `main.py` | 5357-5360 | Replace stub with real implementation | LOW |
| `main.py` | ~5995 | Add Certora phase after Echidna | LOW |
| `main.py` | 2407-2417 | Add `certora_results` to AUDIT_SCHEMA | LOW |
| `main.py` | 2423-2480 | Add Certora context to PROMPT_TEMPLATE | LOW |

**NO changes to:**
- Tier definitions (already has `certora: True`)
- Error handling patterns (copy existing)
- WebSocket/progress patterns (copy existing)
- PDF generation (add section using existing pattern)

### 2.2 Frontend (JavaScript/HTML)

| File | Lines | Change Type | Risk Level |
|------|-------|-------------|------------|
| `static/script.js` | 230-237 | Add 'certora' phase to phases object | LOW |
| `static/script.js` | 1685 | Update "Coming Soon" to active | LOW |
| `static/script.js` | 1726 | Update description text | LOW |
| `static/script.js` | ~2663 | Add certora results renderer (copy fuzzing pattern) | LOW |
| `templates/index.html` | ~630 | Add certora-placeholder div (copy fuzzing pattern) | LOW |

### 2.3 New Files (Additive Only)

| File | Purpose |
|------|---------|
| `cvl_generator.py` | AI-powered CVL specification generator |
| `certora_runner.py` | Certora cloud API integration |
| `cvl_templates/` | Template CVL specs for common patterns |

---

## 3. Integration Architecture

### 3.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXISTING AUDIT FLOW                          │
│  Slither → Mythril → Echidna → AI Analysis → Report            │
└─────────────────────────────────────────────────────────────────┘
                          ↓ (insert here)
┌─────────────────────────────────────────────────────────────────┐
│                    NEW CERTORA PHASE                            │
│                                                                 │
│  1. Check tier (certora feature flag)                          │
│  2. Generate CVL specs (Claude AI)                             │
│  3. Submit to Certora Cloud                                    │
│  4. Poll for results (with timeout)                            │
│  5. Parse verification results                                 │
│  6. Add to report context                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 CVL Generation Strategy

```python
# cvl_generator.py (NEW FILE)

class CVLGenerator:
    """Generate Certora Verification Language specs from Solidity contracts."""

    def __init__(self, anthropic_client):
        self.client = anthropic_client

    async def generate_specs(self, contract_code: str, slither_findings: list) -> str:
        """
        Generate CVL specifications using Claude.

        Uses Slither findings to focus verification on vulnerable areas.
        Returns CVL spec file content.
        """
        prompt = self._build_cvl_prompt(contract_code, slither_findings)

        response = await self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        return self._extract_cvl(response.content[0].text)

    def _build_cvl_prompt(self, code: str, findings: list) -> str:
        """Build prompt for CVL generation."""
        return f"""
        Generate Certora CVL specifications for this Solidity contract.

        Focus on verifying:
        1. Access control invariants
        2. Balance/supply conservation
        3. State transition validity
        4. Reentrancy protection
        5. Any issues from static analysis: {findings}

        Contract:
        {code}

        Return ONLY the CVL spec file content, no explanation.
        """
```

### 3.3 Certora Runner

```python
# certora_runner.py (NEW FILE)

import subprocess
import json
import tempfile
import os

class CertoraRunner:
    """Interface to Certora Prover cloud."""

    def __init__(self):
        self.api_key = os.getenv("CERTORAKEY")
        self.timeout = 600  # 10 minutes max

    async def run_verification(
        self,
        contract_path: str,
        spec_content: str
    ) -> dict:
        """
        Run Certora verification via cloud API.

        Returns:
            {
                "success": bool,
                "rules_verified": int,
                "rules_failed": int,
                "violations": [...],
                "job_url": str
            }
        """
        # Write spec to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.spec',
            delete=False
        ) as f:
            f.write(spec_content)
            spec_path = f.name

        try:
            # Build conf file
            conf = self._build_config(contract_path, spec_path)

            # Run certoraRun
            result = subprocess.run(
                ["certoraRun", conf, "--wait", "--json"],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            return self._parse_output(result.stdout)

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Verification timed out (10 min limit)",
                "rules_verified": 0,
                "rules_failed": 0,
                "violations": []
            }
        finally:
            os.unlink(spec_path)
```

### 3.4 Integration Point in main.py

```python
# Insert AFTER Echidna phase (line ~5995), BEFORE AI analysis

# Phase 3.5: Certora Formal Verification (Enterprise only)
certora_results = []
certora_enabled = usage_tracker.feature_flags.get(tier_for_flags, {}).get("certora", False)

if certora_enabled:
    await broadcast_audit_log(effective_username, "Running formal verification...")
    if job_id:
        await audit_queue.update_phase(job_id, "certora", 55)
        await notify_job_subscribers(job_id, {
            "status": "processing",
            "phase": "certora",
            "progress": 55
        })
    await asyncio.sleep(0)

    try:
        # Generate CVL specs using AI
        cvl_generator = CVLGenerator(anthropic_client)
        cvl_specs = await cvl_generator.generate_specs(
            code_content,
            slither_findings
        )

        # Run verification
        certora_runner = CertoraRunner()
        certora_results = await asyncio.to_thread(
            certora_runner.run_verification,
            temp_path,
            cvl_specs
        )

        await broadcast_audit_log(
            effective_username,
            f"Formal verification complete: {certora_results.get('rules_verified', 0)} rules verified"
        )
    except Exception as e:
        logger.error(f"Certora failed: {e}")
        certora_results = [{
            "status": "error",
            "message": str(e)
        }]

    await asyncio.sleep(0)
```

---

## 4. Implementation Phases

### Phase 1: Infrastructure Setup (No Code Changes)

**Duration**: 1-2 days

**Tasks**:
1. Obtain Certora API key (free registration)
2. Add `CERTORAKEY` to environment variables
3. Verify Certora CLI works in Docker environment
4. Test basic certoraRun with sample contract

**Verification**:
```bash
certoraRun --version
```

**Risk**: ZERO - No production code touched

---

### Phase 2: Core Module Development (New Files Only)

**Duration**: 3-5 days

**New Files**:
```
/home/user/Defiguard-Ai-Beta/
├── certora/
│   ├── __init__.py
│   ├── cvl_generator.py      # CVL spec generation
│   ├── certora_runner.py     # Prover integration
│   └── cvl_templates/
│       ├── erc20.spec        # ERC20 verification rules
│       ├── access_control.spec
│       ├── reentrancy.spec
│       └── common.spec
```

**Testing**:
- Unit tests for CVL generator
- Integration tests with Certora cloud
- Validate against known-vulnerable contracts

**Risk**: ZERO - New files only, not imported yet

---

### Phase 3: Backend Integration (Surgical Edits)

**Duration**: 2-3 days

**Modifications**:

#### 3.1 Replace Stub (main.py:5357-5360)

```python
# BEFORE (current stub)
def run_certora(temp_path: str) -> list[dict[str, str]]:
    """Minimal typed stub for Certora invocation"""
    logger.warning(f"Stub run_certora called for {temp_path}")
    return [{"rule": "Sample rule", "status": "Passed (dummy)"}]

# AFTER (real implementation)
def run_certora(temp_path: str, slither_findings: list = None) -> list[dict[str, Any]]:
    """Run Certora formal verification via cloud API."""
    from certora.certora_runner import CertoraRunner
    from certora.cvl_generator import CVLGenerator

    try:
        # Check if Certora is configured
        if not os.getenv("CERTORAKEY"):
            logger.warning("CERTORAKEY not configured, skipping formal verification")
            return [{"status": "skipped", "reason": "API key not configured"}]

        # Read contract
        with open(temp_path, 'r') as f:
            contract_code = f.read()

        # Generate CVL specs
        generator = CVLGenerator()
        cvl_specs = generator.generate_specs_sync(contract_code, slither_findings or [])

        if not cvl_specs:
            return [{"status": "skipped", "reason": "Could not generate specifications"}]

        # Run verification
        runner = CertoraRunner()
        results = runner.run_verification_sync(temp_path, cvl_specs)

        return results.get("violations", []) or [{"status": "verified", "message": "All rules passed"}]

    except Exception as e:
        logger.error(f"Certora verification failed: {e}")
        return [{"status": "error", "message": str(e)}]
```

#### 3.2 Add Certora Phase (insert after line ~5994)

```python
# Phase 3.5: Certora Formal Verification (Enterprise only)
certora_results = []
certora_enabled = usage_tracker.feature_flags.get(tier_for_flags, {}).get("certora", False)

if certora_enabled:
    await broadcast_audit_log(effective_username, "Running formal verification...")
    if job_id:
        await audit_queue.update_phase(job_id, "certora", 55)
        await notify_job_subscribers(job_id, {"status": "processing", "phase": "certora", "progress": 55})
    await asyncio.sleep(0)

    try:
        certora_results = await asyncio.to_thread(run_certora, temp_path, slither_findings)
    except Exception as e:
        logger.error(f"Certora failed: {e}")
        certora_results = [{"status": "error", "message": str(e)}]

    await asyncio.sleep(0)
    verified_count = sum(1 for r in certora_results if r.get("status") == "verified")
    await broadcast_audit_log(effective_username, f"Formal verification: {verified_count} properties verified")
```

#### 3.3 Add to Report Context (line ~5998)

```python
# Add certora_results to report
report["certora_results"] = certora_results
```

#### 3.4 Update PROMPT_TEMPLATE (line ~2436)

```python
# Add after FUZZING RESULTS line:
FORMAL_VERIFICATION: {certora_results}
```

**Testing**:
- Test with Enterprise tier user
- Test with non-Enterprise tier (should skip)
- Test with missing CERTORAKEY (should skip gracefully)
- Test with timeout (should not block audit)

**Risk**: LOW - Follows exact same pattern as Echidna

---

### Phase 4: Frontend Integration (Additive Only)

**Duration**: 1-2 days

#### 4.1 Add Phase to Progress (script.js:230-237)

```javascript
const phases = {
    'slither': { icon: '🔍', label: 'Running Slither Analysis', progress: 10 },
    'mythril': { icon: '🧠', label: 'Running Mythril Symbolic Analysis', progress: 25 },
    'echidna': { icon: '🧪', label: 'Running Echidna Fuzzing', progress: 40 },
    'certora': { icon: '🔒', label: 'Running Formal Verification', progress: 55 },  // NEW
    'grok': { icon: '🤖', label: 'Claude AI Analysis & Report Generation', progress: 60 },
    'finalizing': { icon: '✨', label: 'Finalizing Report', progress: 90 },
    'complete': { icon: '✅', label: 'Complete!', progress: 100 }
};
```

#### 4.2 Update Enterprise Features (script.js:1685)

```javascript
// BEFORE
"Formal Verification (Coming Soon)",

// AFTER
"Formal Verification (Certora Prover)",
```

#### 4.3 Add Results Display (script.js, new function near line 2754)

```javascript
const renderCertoraResults = (certoraResults) => {
    if (!certoraResults || certoraResults.length === 0) {
        return '<div class="certora-empty"><p>🔒 No formal verification results.</p></div>';
    }

    const verified = certoraResults.filter(r => r.status === 'verified').length;
    const failed = certoraResults.filter(r => r.status === 'violated').length;
    const skipped = certoraResults.filter(r => r.status === 'skipped').length;

    return `
        <div class="certora-status-card">
            <div class="certora-header">
                <span class="certora-icon">🔒</span>
                <h4>Formal Verification Results</h4>
            </div>
            <div class="certora-stats">
                <div class="stat verified">
                    <span class="stat-value">${verified}</span>
                    <span class="stat-label">Verified</span>
                </div>
                <div class="stat failed">
                    <span class="stat-value">${failed}</span>
                    <span class="stat-label">Violations</span>
                </div>
            </div>
            <div class="certora-rules">
                ${certoraResults.map(r => `
                    <div class="rule ${r.status}">
                        <span class="rule-icon">${r.status === 'verified' ? '✅' : r.status === 'violated' ? '❌' : '⏭️'}</span>
                        <span class="rule-name">${r.rule || r.message || 'Unknown rule'}</span>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
};
```

#### 4.4 Add HTML Placeholder (index.html, after line ~629)

```html
<!-- Formal Verification Results (Enterprise only) -->
<div class="certora-placeholder enterprise-only" style="display: none;">
    <h4>🔒 Formal Verification</h4>
    <div id="certora-list" role="region" aria-label="Certora formal verification results">
        <div class="certora-empty">
            <p>🔒 No formal verification results yet.</p>
        </div>
    </div>
</div>
```

**Risk**: LOW - Additive changes only, follows existing patterns

---

### Phase 5: Testing & Validation

**Duration**: 2-3 days

**Test Cases**:

| Test | Expected Result | Risk if Fails |
|------|-----------------|---------------|
| Free tier audit | Certora skipped, no UI shown | None |
| Starter tier audit | Certora skipped, no UI shown | None |
| Pro tier audit | Certora skipped, no UI shown | None |
| Enterprise audit (no key) | Certora skipped gracefully | None |
| Enterprise audit (valid key) | Certora runs, results shown | None |
| Certora timeout | Audit completes, Certora section shows timeout | None |
| Certora error | Audit completes, error logged | None |
| CVL generation fails | Audit completes, verification skipped | None |

**Regression Tests**:
- All existing tiers work exactly as before
- Slither/Mythril/Echidna unaffected
- PDF generation includes Certora section (Enterprise only)
- WebSocket progress updates work
- Queue system unaffected

---

## 5. Risk Mitigation

### 5.1 Isolation Strategy

| Component | Isolation Method |
|-----------|-----------------|
| CVL Generator | Separate module, imported only when needed |
| Certora Runner | Separate module, imported only when needed |
| Feature gate | Existing `certora` flag in tier config |
| Timeout | 600s hard limit, non-blocking |
| Errors | Caught and logged, audit continues |

### 5.2 Graceful Degradation

```python
# Always wrap Certora calls in try/except
try:
    certora_results = await run_certora(...)
except Exception as e:
    logger.error(f"Certora failed: {e}")
    certora_results = []  # Empty results, audit continues
```

### 5.3 Environment Safety

```python
# Check for CERTORAKEY before any Certora operations
if not os.getenv("CERTORAKEY"):
    return [{"status": "skipped", "reason": "Not configured"}]
```

### 5.4 Memory/CPU Safety

- Certora runs on their cloud, not your server
- Only CPU cost is CVL generation (Claude API call)
- Timeout prevents hanging

---

## 6. Testing Strategy

### 6.1 Unit Tests

```python
# test_cvl_generator.py
def test_generate_erc20_specs():
    generator = CVLGenerator()
    specs = generator.generate_specs_sync(ERC20_CODE, [])
    assert "balanceOf" in specs
    assert "totalSupply" in specs

def test_generate_with_findings():
    generator = CVLGenerator()
    findings = [{"name": "reentrancy", "details": "..."}]
    specs = generator.generate_specs_sync(VULNERABLE_CODE, findings)
    assert "reentrancy" in specs.lower()
```

### 6.2 Integration Tests

```python
# test_certora_integration.py
@pytest.mark.skipif(not os.getenv("CERTORAKEY"), reason="No Certora key")
def test_full_verification():
    runner = CertoraRunner()
    results = runner.run_verification_sync(
        "test_contracts/simple_erc20.sol",
        SIMPLE_CVL_SPEC
    )
    assert results["success"] is True
```

### 6.3 End-to-End Tests

```python
# test_audit_with_certora.py
async def test_enterprise_audit_includes_certora():
    response = await client.post("/audit", ...)
    assert "certora_results" in response.json()["report"]

async def test_free_tier_no_certora():
    response = await client.post("/audit", ...)
    assert "certora_results" not in response.json()["report"]
```

---

## 7. Rollback Plan

### 7.1 Instant Rollback

If issues arise, set environment variable:

```bash
CERTORA_DISABLED=true
```

Add check to run_certora:

```python
if os.getenv("CERTORA_DISABLED"):
    return [{"status": "disabled", "reason": "Temporarily disabled"}]
```

### 7.2 Code Rollback

All changes are in:
- `main.py` (3 small sections)
- `static/script.js` (4 small sections)
- `templates/index.html` (1 section)
- New `certora/` directory

To fully rollback:
```bash
git revert <commit-hash>
rm -rf certora/
```

### 7.3 Feature Flag Rollback

Change in `main.py:2858`:
```python
"certora": False,  # Disable without code changes
```

---

## Appendix A: File Change Summary

| File | Lines Changed | Type |
|------|---------------|------|
| main.py | ~50 lines | Modified |
| static/script.js | ~40 lines | Modified |
| templates/index.html | ~10 lines | Modified |
| certora/__init__.py | ~5 lines | New |
| certora/cvl_generator.py | ~150 lines | New |
| certora/certora_runner.py | ~120 lines | New |
| certora/cvl_templates/*.spec | ~200 lines | New |

**Total impact**: ~575 lines, 75% in new files

---

## Appendix B: Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| CERTORAKEY | Yes | Certora cloud API key (free) |
| CERTORA_DISABLED | No | Set to 'true' to disable |
| CERTORA_TIMEOUT | No | Override default 600s timeout |

---

## Appendix C: Dependencies

**Python**:
- No new dependencies (uses subprocess for certoraRun)
- Existing anthropic client for CVL generation

**System**:
- certoraRun CLI (install in Docker image)
- Java 19+ (required by Certora)
- solc (already installed)

**Docker Image Update**:
```dockerfile
# Add to Dockerfile
RUN pip install certora-cli
RUN apt-get install -y openjdk-19-jdk
```

---

## Approval Checklist

Before proceeding with implementation:

- [ ] Certora API key obtained
- [ ] Docker image update plan reviewed
- [ ] Test contracts prepared
- [ ] Rollback procedure understood
- [ ] Each phase signed off before next begins

---

*Document Version: 1.0*
*Created: December 2024*
*Author: Claude Code*
