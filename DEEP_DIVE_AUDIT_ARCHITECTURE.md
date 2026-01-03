# DeFiGuard AI - Comprehensive Audit System Architecture Deep Dive

**Date**: January 3, 2026
**Author**: Claude Code (Opus 4.5)
**Status**: Planning Document for Review

---

## Executive Summary

This document provides a comprehensive analysis of the DeFiGuard AI audit system architecture, covering:
1. Complete audit submission flow
2. API Key vs Audit Key terminology clarification
3. Notification and retrieval systems
4. **CRITICAL** persistence vulnerabilities identified
5. One-file-per-API-key policy analysis and implementation plan

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Key Terminology Clarification](#2-key-terminology-clarification)
3. [Audit Submission Flow](#3-audit-submission-flow)
4. [Notification System](#4-notification-system)
5. [Audit Retrieval System](#5-audit-retrieval-system)
6. [Background Persistence Analysis](#6-background-persistence-analysis)
7. [One-File-Per-Key Policy](#7-one-file-per-key-policy)
8. [Recommended Changes](#8-recommended-changes)
9. [Implementation Roadmap](#9-implementation-roadmap)

---

## 1. System Architecture Overview

### High-Level Flow

```
User Submit
    │
    ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   /audit or      │───▶│   AuditQueue     │───▶│  process_audit   │
│   /audit/submit  │    │   (in-memory)    │    │  _queue()        │
└──────────────────┘    └──────────────────┘    └──────────────────┘
         │                      │                        │
         ▼                      ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   AuditResult    │    │   Priority       │    │   Claude AI +    │
│   (database)     │    │   Processing     │    │   Tool Analysis  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
         │                                               │
         ▼                                               ▼
┌──────────────────┐                            ┌──────────────────┐
│   WebSocket      │◀───────────────────────────│   Results +      │
│   Real-time      │                            │   PDF Generation │
└──────────────────┘                            └──────────────────┘
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| AuditQueue | main.py:267-510 | In-memory priority queue |
| AuditResult | main.py:1072-1156 | Database persistence model |
| APIKey | main.py:993-1070 | User API key management |
| process_audit_queue | main.py:7569-7743 | Background job processor |
| WebSocket | main.py (multiple) | Real-time updates |

---

## 2. Key Terminology Clarification

### ⚠️ CRITICAL DISTINCTION: API Key ≠ Audit Key

These are **completely separate systems** that serve different purposes:

| Feature | API Key | Audit Key |
|---------|---------|-----------|
| **Purpose** | Authentication & organization | Access to specific audit result |
| **Model** | APIKey table | AuditResult.audit_key field |
| **Format** | `secrets.token_urlsafe(32)` | `dga_` + `secrets.token_urlsafe(32)` |
| **Example** | `7KQw3x...` (raw token) | `dga_7KQw3x...` |
| **Authentication** | Requires user login | No auth required |
| **Scope** | Multiple audits | Single audit |
| **Visibility** | Pro/Enterprise only | All users |
| **Created when** | User generates in dashboard | Audit is submitted |
| **Used for** | Project organization | Retrieving results |

### User-Facing Terminology Recommendations

**Current Confusion**: UI uses "API Key" in multiple contexts

**Recommended Clarification**:
- **API Key**: "Project Key" or "Organization Key" - for organizing audits
- **Audit Key**: "Access Key" or "Result Key" - for accessing specific audit

### Code References

```python
# API Key Generation (main.py:2117)
new_key = APIKey(
    key=secrets.token_urlsafe(32),  # The secret credential
    user_id=user.id,
    label=label
)

# Audit Key Generation (main.py:1159-1162)
def generate_audit_key() -> str:
    return f"dga_{secrets.token_urlsafe(32)}"  # Unique per audit
```

---

## 3. Audit Submission Flow

### Two Entry Points

**Endpoint 1: `/audit/submit` (Queue-based)**
- Location: main.py:7769-7969
- For: Long-running audits with queue management
- Returns: job_id + audit_key immediately
- Processing: Background via `process_audit_queue()`

**Endpoint 2: `/audit` (Direct)**
- Location: main.py:8274+
- For: Immediate processing (bypasses queue when possible)
- Returns: audit_key + results
- Processing: Synchronous or queued based on load

### Submission Sequence

```
1. User uploads .sol file
   ↓
2. Frontend calls /audit?username=X&api_key_id=Y
   ↓
3. Backend validates:
   - File type (.sol, .vy, .yul)
   - File size (< 500KB)
   - User tier and limits
   - API key ownership (if provided)
   ↓
4. Generate audit_key (dga_xxx)
   ↓
5. Create AuditResult in database (status=queued)
   ↓
6. Add to AuditQueue (in-memory)
   ↓
7. Return audit_key to user
   ↓
8. process_audit_queue() picks up job
   ↓
9. Run analysis tools (Slither, Mythril, AI)
   ↓
10. Update AuditResult (status=completed)
    ↓
11. Send WebSocket notification
    ↓
12. Send email (if configured)
```

### Priority Queue Order

| Tier | Priority | Processing Order |
|------|----------|------------------|
| Enterprise | 0 | First |
| Pro | 1 | Second |
| Starter | 2 | Third |
| Free | 3 | Last |

---

## 4. Notification System

### Email Notifications

**Location**: main.py (SMTP configuration)

**Triggers**:
1. Audit completed successfully
2. Audit failed with error

**Email Content**:
- Contract name
- Status (completed/failed)
- Link with audit_key for retrieval
- Summary metrics (if completed)

**Configuration** (environment variables):
- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USER`
- `SMTP_PASS`

### WebSocket Real-time Updates

**Endpoints**:
- `/ws/job/{job_id}` - Per-job status updates
- `/ws-audit-log` - Global audit activity stream

**Update Types**:
```python
# Status updates sent during processing
{
    "type": "status",
    "job_id": "xxx",
    "status": "processing",
    "phase": "slither",  # or "mythril", "ai_analysis"
    "progress": 45
}

# Completion notification
{
    "type": "complete",
    "job_id": "xxx",
    "audit_key": "dga_xxx",
    "risk_score": 65,
    "issues_count": 12
}
```

---

## 5. Audit Retrieval System

### Retrieval Endpoint

**Location**: main.py:8005-8084
**Endpoint**: `GET /audit/retrieve/{audit_key}`
**Authentication**: NONE (audit_key serves as credential)

### Response by Status

**Queued**:
```json
{
    "status": "queued",
    "queue_position": 3,
    "estimated_wait_seconds": 120
}
```

**Processing**:
```json
{
    "status": "processing",
    "current_phase": "mythril",
    "progress": 65
}
```

**Completed**:
```json
{
    "status": "completed",
    "completed_at": "2026-01-03T...",
    "risk_score": 45,
    "issues_count": 8,
    "report": { /* full results */ },
    "pdf_url": "/api/reports/audit_report_xxx.pdf"
}
```

### Frontend Retrieval

**Function**: `window.retrieveAuditByKey(auditKey)`
**Location**: script.js:241-298

**UI Access**: "🔑 Retrieve Previous Audit" button
**Location**: templates/index.html:286

---

## 6. Background Persistence Analysis

### 🚨 CRITICAL VULNERABILITIES IDENTIFIED

The current architecture has **severe data loss vulnerabilities** that must be addressed:

### Issue 1: No Server Restart Recovery (CRITICAL)

**Problem**: When server restarts, all queued/processing jobs are lost permanently.

**Current State**:
- `AuditQueue` is pure in-memory (main.py:267-507)
- No recovery logic in `lifespan()` startup (main.py:1619-1676)
- Queue initialized empty on restart

**Impact**:
- AuditResult stays "queued" in database forever
- Job never processes
- User sees "queued" but it will never complete

**Proof of Vulnerability**:
```python
# main.py:1664 - Queue starts empty
processor_task = asyncio.create_task(process_audit_queue())
# NO CODE TO LOAD PENDING JOBS FROM DATABASE
```

### Issue 2: File Content Not Persisted (CRITICAL)

**Problem**: Uploaded file content exists ONLY in memory.

**Current State**:
```python
# AuditJob (in-memory) - main.py:256
file_content: bytes  # This is volatile!

# AuditResult (database) - main.py:1072-1156
# NO file_content field!
```

**Impact**:
- Server restart = file content lost
- Cannot retry interrupted audits
- User must re-upload

### Issue 3: Orphaned Processing Jobs (CRITICAL)

**Problem**: Jobs stuck in "processing" state are never recovered.

**Current State**:
- No timeout mechanism for long-running jobs
- `cleanup_old_jobs()` only cleans COMPLETED/FAILED (main.py:495-506)
- No detection of stale "processing" jobs

**Impact**:
- Resources stuck forever
- User cannot retry
- Memory leak over time

### Issue 4: Race Condition in Status Updates

**Problem**: Status updates split between in-memory and database.

**Sequence** (main.py:7588-7643):
```
1. Remove job from queue (in-memory)
2. Update AuditResult.status = "processing" (database)
3. [VULNERABILITY WINDOW - crash here = orphaned job]
4. Run audit_contract() (30-300 seconds)
5. Update AuditResult.status = "completed" (database)
```

**Impact**: Crash during step 3-4 creates unrecoverable orphan.

### Summary of Persistence Gaps

| Issue | Severity | Type | Impact |
|-------|----------|------|--------|
| No restart recovery | CRITICAL | Data Loss | All pending jobs lost |
| No file persistence | CRITICAL | Data Loss | Cannot retry audits |
| Orphaned processing jobs | CRITICAL | Resource Leak | Jobs never complete |
| Race condition | CRITICAL | Inconsistency | Job state corruption |
| No timeout detection | HIGH | Deadlock | Stuck jobs |

---

## 7. One-File-Per-Key Policy

### User Requirement

> "1 file per key so the user can audit the same file multiple times just not different files with the same key"

### Interpretation

```
ALLOWED:
  - api_key_1: TokenA.sol (audit #1, time=t1)
  - api_key_1: TokenA.sol (audit #2, time=t2) ← Re-audit same file ✓
  - api_key_2: TokenA.sol (audit #3)          ← Different key ✓

NOT ALLOWED:
  - api_key_1: TokenA.sol (audit #1)
  - api_key_1: TokenB.sol (audit #2)          ← Different file, same key ✗
```

### File Identification Method

**Recommended**: Use `contract_hash` (SHA256 of content)

**Rationale**:
- Already computed and indexed
- Content-based (not filename-based)
- Allows re-audits of exact same content
- Works for all file types

### Current State

No constraint exists. Users can currently:
- Audit different files with same API key (violation of policy)
- Audit same file multiple times with same key (desired behavior)

### Implementation Approach

**Phase 1: Soft Validation (Warning)**
```python
# Add to /audit/submit and /audit endpoints
if validated_api_key_id is not None:
    existing = db.query(AuditResult).filter(
        AuditResult.api_key_id == validated_api_key_id,
        AuditResult.contract_hash != contract_hash,  # Different file
        AuditResult.status == "completed"
    ).first()

    if existing:
        # Return warning but allow (for now)
        logger.warning(f"Different file for same API key")
```

**Phase 2: Hard Enforcement**
```python
# Add to validation
if existing:
    raise HTTPException(
        status_code=409,
        detail=f"This API key is assigned to '{existing.contract_name}'. "
               "Use a different key for different files."
    )
```

### Pros of Policy

✓ Cleaner organization (one file = one key)
✓ Enables per-file billing
✓ Enterprise-standard pattern
✓ Prevents duplicate audits

### Cons of Policy

✗ May confuse users initially
✗ Requires clear error messaging
✗ Need UI updates to explain

---

## 8. Recommended Changes

### Priority 1: CRITICAL Persistence Fixes

#### Fix 1.1: Add File Content Persistence

```python
# Add to AuditResult model (main.py:1072)
file_content: Mapped[Optional[bytes]] = mapped_column(
    LargeBinary,
    nullable=True,
    doc="Persisted file content for recovery"
)
```

#### Fix 1.2: Startup Recovery

```python
# Add to lifespan() after queue initialization (main.py:1664)
async def recover_pending_jobs():
    """Recover jobs that were pending when server crashed."""
    db = SessionLocal()
    try:
        pending = db.query(AuditResult).filter(
            AuditResult.status.in_(["queued", "processing"]),
            AuditResult.file_content.isnot(None)
        ).all()

        for result in pending:
            # Reset processing jobs to queued
            if result.status == "processing":
                result.status = "queued"
                result.started_at = None

            # Re-add to queue
            job = AuditJob(
                job_id=result.job_id or str(uuid.uuid4()),
                file_content=result.file_content,
                filename=result.contract_name,
                tier=result.user_tier,
                user_id=result.user_id
            )
            await audit_queue.submit(job)

        db.commit()
        logger.info(f"[RECOVERY] Recovered {len(pending)} pending jobs")
    finally:
        db.close()
```

#### Fix 1.3: Timeout Detection

```python
# Add background task
async def detect_stale_jobs():
    """Detect and recover jobs stuck in processing."""
    while True:
        await asyncio.sleep(300)  # Check every 5 minutes
        db = SessionLocal()
        try:
            timeout = datetime.utcnow() - timedelta(minutes=30)
            stale = db.query(AuditResult).filter(
                AuditResult.status == "processing",
                AuditResult.started_at < timeout
            ).all()

            for result in stale:
                result.status = "failed"
                result.error_message = "Processing timeout - job took too long"
                result.completed_at = datetime.utcnow()
                logger.warning(f"[TIMEOUT] Job {result.job_id} timed out")

            db.commit()
        finally:
            db.close()
```

### Priority 2: One-File-Per-Key Policy

#### Fix 2.1: Add Validation

```python
# Add to /audit/submit (main.py:7835) and /audit (main.py:8338)
if validated_api_key_id is not None:
    # Check for different files under same key
    different_file = db.query(AuditResult).filter(
        AuditResult.api_key_id == validated_api_key_id,
        AuditResult.contract_hash != contract_hash,
        AuditResult.status == "completed"
    ).first()

    if different_file:
        raise HTTPException(
            status_code=409,
            detail=f"API key already assigned to '{different_file.contract_name}'. "
                   "Each API key can only audit one file. Create a new key for different files."
        )
```

#### Fix 2.2: Update Frontend

```javascript
// Add to script.js - handle 409 Conflict
if (response.status === 409) {
    const error = await response.json();
    showToast(error.detail, 'error');
    // Show link to create new key
    showCreateKeyPrompt();
    return;
}
```

### Priority 3: UI Terminology Fixes

#### Fix 3.1: Clarify Key Types

**templates/index.html changes**:
- Change "API Key" to "Project Key" in organization context
- Keep "Audit Key" or use "Access Key" for result retrieval
- Add tooltips explaining the difference

---

## 9. Implementation Roadmap

### Phase 1: Critical Persistence Fixes (Week 1)

| Task | Files | Lines | Risk |
|------|-------|-------|------|
| Add file_content to AuditResult | main.py | 1072-1156 | Medium |
| Add startup recovery | main.py | 1619-1676 | Medium |
| Add timeout detection | main.py | new function | Low |
| Database migration | alembic | new migration | Medium |

### Phase 2: One-File-Per-Key Policy (Week 2)

| Task | Files | Lines | Risk |
|------|-------|-------|------|
| Add soft validation (warning) | main.py | 7835, 8338 | Low |
| Add frontend 409 handling | script.js | new | Low |
| Monitor violations | logging | - | None |

### Phase 3: Hard Enforcement (Week 3)

| Task | Files | Lines | Risk |
|------|-------|-------|------|
| Change to HTTPException | main.py | 7835, 8338 | Low |
| Add DB unique constraint | alembic | new migration | Medium |
| Update UI messaging | index.html, script.js | multiple | Low |

### Phase 4: UI Polish (Week 4)

| Task | Files | Lines | Risk |
|------|-------|-------|------|
| Terminology clarification | index.html | multiple | Low |
| Add tooltips | index.html, script.js | multiple | Low |
| Update help text | various | multiple | Low |

---

## Appendix A: File References

| Component | File | Lines |
|-----------|------|-------|
| AuditQueue class | main.py | 267-510 |
| AuditResult model | main.py | 1072-1156 |
| APIKey model | main.py | 993-1070 |
| /audit/submit endpoint | main.py | 7769-7969 |
| /audit endpoint | main.py | 8274+ |
| /audit/retrieve endpoint | main.py | 8005-8084 |
| process_audit_queue | main.py | 7569-7743 |
| lifespan (startup) | main.py | 1619-1676 |
| Frontend retrieval | script.js | 241-407 |
| Results rendering | script.js | 3046-3406 |

---

## Appendix B: Testing Checklist

### Persistence Tests

- [ ] Server restart recovers queued jobs
- [ ] Server restart recovers processing jobs (reset to queued)
- [ ] Timeout detection marks stale jobs as failed
- [ ] File content persisted and recoverable

### One-File-Per-Key Tests

- [ ] Same file + same key = allowed (re-audit)
- [ ] Different file + same key = rejected (409)
- [ ] Same file + different key = allowed
- [ ] No API key (NULL) = no restriction

### UI Tests

- [ ] 409 error displays correctly
- [ ] Create new key prompt works
- [ ] Terminology is clear and consistent

---

**Document Status**: Ready for Review
**Next Steps**: Await approval before implementation
