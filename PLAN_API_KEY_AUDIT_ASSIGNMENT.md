# Implementation Plan: API Key Assignment to Audits

## Executive Summary

Enable users to assign audits to specific API keys for organized project/client management.
This follows enterprise patterns (AWS, GCP, Stripe) where resources can be scoped to specific credentials.

---

## Current State Analysis

### Data Models

**APIKey** (main.py:996-1010)
```
id, user_id, key, label, created_at, last_used_at, is_active
```

**AuditResult** (main.py:1072-1146)
```
id, user_id, audit_key, contract_name, contract_hash, status, ...
```

**Current Relationship**: Both link to User via `user_id`, but no link between them.

---

## Proposed Changes

### Phase 1: Database Schema

**Add to AuditResult model:**
```python
# Optional API key assignment (for Pro/Enterprise users)
api_key_id: Mapped[Optional[int]] = mapped_column(
    Integer,
    ForeignKey('api_keys.id', ondelete='SET NULL'),  # Keep audit if key deleted
    nullable=True,
    index=True
)

# Relationship
api_key = relationship("APIKey", backref="audits")
```

**Why `ondelete='SET NULL'`**: If an API key is revoked, audits should remain accessible (just unassigned).

---

### Phase 2: Backend API Changes

**1. Update `/audit/submit` endpoint** (main.py:7750)
- Add optional `api_key_id` query parameter
- Validate that the API key belongs to the user
- Store the `api_key_id` in AuditResult

**2. Update `/api/keys` response** (main.py:1906)
- Add `audit_count` field showing how many audits are assigned to each key

**3. New endpoint: `/api/keys/{key_id}/audits`**
- List all audits assigned to a specific API key
- Enables filtering by project/client

**4. Update `/api/user/audits`** (main.py:8034)
- Add optional `api_key_id` filter parameter
- Add `api_key_label` to response

---

### Phase 3: Frontend UI Changes

**1. Audit Form Enhancement** (templates/index.html)
- Add API key selector dropdown (Pro/Enterprise only)
- Same pattern as existing tier/contract dropdowns

**2. Settings Modal Enhancement** (script.js)
- Show audit count per API key
- Add "View Audits" button per key

**3. Audit History View**
- Filter by API key
- Show assigned key label in list

---

## Implementation Steps

### Step 1: Database Schema (main.py)
```python
# Line ~1143 in AuditResult class, add:
api_key_id: Mapped[Optional[int]] = mapped_column(
    Integer,
    ForeignKey('api_keys.id', ondelete='SET NULL'),
    nullable=True,
    index=True
)
api_key = relationship("APIKey", backref="audits")
```

### Step 2: Backend - Update /audit/submit
```python
# Add parameter:
api_key_id: int = Query(None)

# Add validation (after user auth):
if api_key_id:
    api_key = db.query(APIKey).filter(
        APIKey.id == api_key_id,
        APIKey.user_id == user.id,
        APIKey.is_active == True
    ).first()
    if not api_key:
        raise HTTPException(status_code=400, detail="Invalid API key")

# Add to AuditResult creation:
audit_result = AuditResult(
    ...existing fields...,
    api_key_id=api_key_id if api_key_id else None
)
```

### Step 3: Backend - Update /api/keys
```python
# In list_api_keys, add audit_count:
for key in keys:
    audit_count = db.query(AuditResult).filter(
        AuditResult.api_key_id == key.id
    ).count()

keys_data = [{
    ...existing fields...,
    "audit_count": audit_count
} for key in keys]
```

### Step 4: Frontend - API Key Selector
```html
<!-- After contract_address field in index.html -->
<div class="form-group pro-enterprise-only api-key-selector" style="display: none;">
    <label for="api_key_select">
        Assign to API Key (Optional)
        <span class="badge">Pro/Enterprise</span>
    </label>
    <select id="api_key_select" name="api_key_id">
        <option value="">-- No Assignment --</option>
        <!-- Populated dynamically -->
    </select>
    <small>Organize audits by project or client</small>
</div>
```

### Step 5: Frontend - Populate Selector
```javascript
// In fetchTierData or similar:
if (tier === 'pro' || tier === 'enterprise') {
    const keysResponse = await fetch('/api/keys', { credentials: 'include' });
    const keysData = await keysResponse.json();

    const selector = document.getElementById('api_key_select');
    if (selector && keysData.keys) {
        selector.innerHTML = '<option value="">-- No Assignment --</option>';
        keysData.keys.forEach(key => {
            const option = document.createElement('option');
            option.value = key.id;
            option.textContent = `${key.label} (${key.audit_count} audits)`;
            selector.appendChild(option);
        });
    }
}
```

### Step 6: Frontend - Include in Form Submission
```javascript
// In handleSubmit:
const apiKeySelect = document.getElementById('api_key_select');
const apiKeyId = apiKeySelect?.value;

// Add to URL if set:
let url = `/audit?username=${encodeURIComponent(username)}`;
if (apiKeyId) {
    url += `&api_key_id=${apiKeyId}`;
}
```

---

## Security Considerations

1. **Authorization**: API key must belong to the submitting user
2. **Null Safety**: api_key_id is nullable (guest audits, unassigned)
3. **Cascade**: SET NULL on delete (audits preserved if key revoked)
4. **XSS Prevention**: Escape key labels in UI
5. **Rate Limiting**: Existing rate limits apply

---

## Testing Checklist

- [ ] Pro user can see API key selector
- [ ] Enterprise user can see API key selector
- [ ] Free/Starter users do NOT see selector
- [ ] Selecting a key assigns audit correctly
- [ ] "No Assignment" option works
- [ ] Invalid api_key_id returns 400
- [ ] API key revocation doesn't delete audits
- [ ] Audit history shows assigned key
- [ ] API key list shows audit count

---

## Rollback Plan

If issues arise:
1. The `api_key_id` column is nullable - existing audits unaffected
2. Frontend selector can be hidden via CSS
3. Backend ignores the parameter if not provided

---

## Files to Modify

| File | Changes |
|------|---------|
| `main.py` | AuditResult model, /audit/submit, /api/keys |
| `templates/index.html` | API key selector dropdown |
| `static/script.js` | Populate selector, include in submit |

---

## Estimated Changes

- **main.py**: ~50 lines
- **index.html**: ~15 lines
- **script.js**: ~40 lines
- **Total**: ~105 lines of focused, surgical changes
