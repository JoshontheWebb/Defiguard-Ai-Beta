// ---------------------------------------------------------------------
// GLOBAL DEBUG HELPER – controlled via DEBUG_MODE flag
// Set to false for production to silence debug logs
// ---------------------------------------------------------------------
const DEBUG_MODE = false;  // Set to true for development debugging
const log = (label, ...args) => {
    if (DEBUG_MODE) console.log(`[${label}]`, ...args, `time=${new Date().toISOString()}`);
};
// Debug wrapper for console.log calls - only logs in debug mode
const debugLog = (...args) => { if (DEBUG_MODE) console.log(...args); };

// ---------------------------------------------------------------------
// RESOURCE CLEANUP MANAGER - Prevents memory leaks from intervals/WebSockets
// ---------------------------------------------------------------------
const ResourceManager = {
    intervals: new Set(),
    websockets: new Set(),

    // Track an interval for cleanup
    addInterval(intervalId) {
        this.intervals.add(intervalId);
        return intervalId;
    },

    // Remove a tracked interval
    removeInterval(intervalId) {
        if (intervalId) {
            clearInterval(intervalId);
            this.intervals.delete(intervalId);
        }
    },

    // Track a WebSocket for cleanup
    addWebSocket(ws) {
        this.websockets.add(ws);
        return ws;
    },

    // Remove a tracked WebSocket
    removeWebSocket(ws) {
        if (ws) {
            try {
                if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
                    ws.close();
                }
            } catch (e) {
                debugLog('[ResourceManager] Error closing WebSocket:', e);
            }
            this.websockets.delete(ws);
        }
    },

    // Clean up all tracked resources
    cleanup() {
        debugLog('[ResourceManager] Cleaning up resources...');

        // Clear all intervals
        this.intervals.forEach(intervalId => {
            clearInterval(intervalId);
        });
        this.intervals.clear();

        // Close all WebSockets
        this.websockets.forEach(ws => {
            try {
                if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
                    ws.close();
                }
            } catch (e) {
                debugLog('[ResourceManager] Error closing WebSocket:', e);
            }
        });
        this.websockets.clear();

        // Also clean up any global intervals
        if (window.authTierInterval) {
            clearInterval(window.authTierInterval);
            window.authTierInterval = null;
        }
        if (window.authCheckInterval) {
            clearInterval(window.authCheckInterval);
            window.authCheckInterval = null;
        }

        debugLog('[ResourceManager] Cleanup complete');
    }
};

// Clean up resources when page is being unloaded
window.addEventListener('beforeunload', () => {
    ResourceManager.cleanup();
});

// Also handle visibility changes (mobile tab switching)
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') {
        // Page is hidden - clean up WebSockets to prevent stale connections
        ResourceManager.websockets.forEach(ws => {
            try {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.close();
                }
            } catch (e) {
                // Ignore errors during visibility change
            }
        });
    }
});

// ---------------------------------------------------------------------
// AUDIT KEY HELPERS - For persistent audit access
// ---------------------------------------------------------------------

// Copy audit key to clipboard
window.copyAuditKey = async function(auditKey) {
    try {
        await navigator.clipboard.writeText(auditKey);
        // Show success feedback
        const copyBtn = document.querySelector('.audit-key-copy');
        if (copyBtn) {
            const originalText = copyBtn.textContent;
            copyBtn.textContent = '✓';
            copyBtn.style.color = 'var(--accent-teal)';
            setTimeout(() => {
                copyBtn.textContent = originalText;
                copyBtn.style.color = '';
            }, 2000);
        }
        ToastNotification.show('Audit key copied to clipboard!', 'success');
    } catch (err) {
        console.error('Failed to copy audit key:', err);
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = auditKey;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        ToastNotification.show('Audit key copied!', 'success');
    }
};

// Retrieve audit results by key
window.retrieveAuditByKey = async function(auditKey) {
    if (!auditKey || !auditKey.startsWith('dga_')) {
        ToastNotification.show('Invalid audit key format. Keys start with "dga_"', 'error');
        return null;
    }

    try {
        const response = await fetch(`/audit/retrieve/${auditKey}`);
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || 'Failed to retrieve audit');
        }

        const data = await response.json();
        debugLog('[AUDIT_RETRIEVE]', data);

        // Display results based on status
        if (data.status === 'completed' && data.report) {
            // Render the full report
            window.renderAuditResults(data.report);
            ToastNotification.show('Audit results loaded!', 'success');
        } else if (data.status === 'processing') {
            ToastNotification.show(`Audit in progress: ${data.current_phase || 'analyzing'}...`, 'info');
        } else if (data.status === 'queued') {
            ToastNotification.show(`Audit queued at position ${data.queue_position || '?'}`, 'info');
        } else if (data.status === 'failed') {
            ToastNotification.show(`Audit failed: ${data.error || 'Unknown error'}`, 'error');
        }

        return data;
    } catch (err) {
        console.error('[AUDIT_RETRIEVE] Error:', err);
        ToastNotification.show(err.message, 'error');
        return null;
    }
};

// Render audit results in the UI
window.renderAuditResults = function(report) {
    // Find or create results container
    let resultsSection = document.querySelector('.results-section');
    if (!resultsSection) {
        resultsSection = document.createElement('section');
        resultsSection.className = 'results-section';
        const auditSection = document.getElementById('audit');
        if (auditSection) {
            auditSection.insertAdjacentElement('afterend', resultsSection);
        } else {
            document.body.appendChild(resultsSection);
        }
    }

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });

    // Dispatch event for existing result rendering logic
    window.dispatchEvent(new CustomEvent('auditComplete', { detail: report }));
};

// Show the audit key retrieval modal
window.showAuditKeyRetrievalModal = function() {
    // Check if modal already exists
    let modal = document.getElementById('audit-key-modal');
    if (modal) {
        modal.style.display = 'block';
        return;
    }

    // Create modal
    modal = document.createElement('div');
    modal.id = 'audit-key-modal';
    modal.className = 'modal-backdrop';
    modal.innerHTML = `
        <div class="modal-content" style="max-width: 500px;">
            <div class="modal-header">
                <h3>🔑 Retrieve Audit Results</h3>
                <button class="modal-close" onclick="document.getElementById('audit-key-modal').style.display='none'">×</button>
            </div>
            <div class="modal-body">
                <p style="color: var(--text-secondary); margin-bottom: var(--space-4);">
                    Enter your audit access key to retrieve results from a previous audit.
                </p>
                <div class="form-group">
                    <label for="retrieve-audit-key">Audit Key</label>
                    <input type="text" id="retrieve-audit-key" placeholder="dga_..."
                           style="font-family: monospace; width: 100%;">
                </div>
                <button id="retrieve-audit-btn" class="btn btn-primary" style="width: 100%; margin-top: var(--space-4);">
                    🔍 Retrieve Results
                </button>

                <div id="recent-audit-keys" style="margin-top: var(--space-6);"></div>
            </div>
        </div>
    `;

    document.body.appendChild(modal);

    // Load recent keys from localStorage
    try {
        const savedKeys = JSON.parse(localStorage.getItem('auditKeys') || '[]');
        if (savedKeys.length > 0) {
            const recentContainer = document.getElementById('recent-audit-keys');

            // Create elements safely using DOM methods to prevent XSS
            const heading = document.createElement('h4');
            heading.style.cssText = 'color: var(--text-secondary); margin-bottom: var(--space-2);';
            heading.textContent = 'Recent Audits';

            const listContainer = document.createElement('div');
            listContainer.className = 'recent-keys-list';

            savedKeys.forEach(({ key, timestamp }) => {
                // Validate audit key format before displaying
                if (typeof key !== 'string' || !key.startsWith('dga_')) {
                    return; // Skip invalid keys
                }

                const btn = document.createElement('button');
                btn.className = 'recent-key-btn';
                btn.dataset.auditKey = key; // Store key in data attribute (safe)

                const codeEl = document.createElement('code');
                codeEl.textContent = key.substring(0, 20) + '...'; // Safe text content

                const timeEl = document.createElement('span');
                timeEl.className = 'recent-key-time';
                timeEl.textContent = new Date(timestamp).toLocaleDateString();

                btn.appendChild(codeEl);
                btn.appendChild(timeEl);

                // Safe event listener - no inline JavaScript
                btn.addEventListener('click', () => {
                    const auditKey = btn.dataset.auditKey;
                    document.getElementById('retrieve-audit-key').value = auditKey;
                    window.retrieveAuditByKey(auditKey);
                    document.getElementById('audit-key-modal').style.display = 'none';
                });

                listContainer.appendChild(btn);
            });

            recentContainer.appendChild(heading);
            recentContainer.appendChild(listContainer);
        }
    } catch (e) {
        console.warn('Could not load recent audit keys:', e);
    }

    // Handle retrieve button click
    document.getElementById('retrieve-audit-btn').addEventListener('click', async () => {
        const key = document.getElementById('retrieve-audit-key').value.trim();
        if (key) {
            modal.style.display = 'none';
            await window.retrieveAuditByKey(key);
        }
    });

    // Handle Enter key
    document.getElementById('retrieve-audit-key').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            document.getElementById('retrieve-audit-btn').click();
        }
    });

    // Close on backdrop click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });
};

// ---------------------------------------------------------------------
// TOAST NOTIFICATION SYSTEM - For background job notifications
// ---------------------------------------------------------------------
const ToastNotification = {
    container: null,

    init() {
        if (this.container) return;

        // Create toast container if it doesn't exist
        this.container = document.createElement('div');
        this.container.id = 'toast-container';
        this.container.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10000;
            display: flex;
            flex-direction: column;
            gap: 10px;
            max-width: 400px;
        `;
        document.body.appendChild(this.container);
    },

    show(message, type = 'info', duration = 8000) {
        this.init();

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.style.cssText = `
            padding: 16px 20px;
            border-radius: 8px;
            color: white;
            font-size: 14px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            animation: slideIn 0.3s ease-out;
            cursor: pointer;
            display: flex;
            align-items: flex-start;
            gap: 12px;
            background: ${type === 'success' ? 'linear-gradient(135deg, #27ae60, #2ecc71)' :
                         type === 'error' ? 'linear-gradient(135deg, #c0392b, #e74c3c)' :
                         type === 'warning' ? 'linear-gradient(135deg, #d35400, #e67e22)' :
                         'linear-gradient(135deg, #2980b9, #3498db)'};
        `;

        const icon = type === 'success' ? '✅' :
                     type === 'error' ? '❌' :
                     type === 'warning' ? '⚠️' : '🔒';

        toast.innerHTML = `
            <span style="font-size: 20px;">${icon}</span>
            <div style="flex: 1;">
                ${message}
            </div>
            <button style="background: none; border: none; color: white; cursor: pointer; font-size: 18px; padding: 0; opacity: 0.7;" onclick="this.parentElement.remove()">×</button>
        `;

        toast.addEventListener('click', (e) => {
            if (e.target.tagName !== 'BUTTON') {
                toast.remove();
            }
        });

        this.container.appendChild(toast);

        // Auto-remove after duration
        if (duration > 0) {
            setTimeout(() => toast.remove(), duration);
        }

        return toast;
    },

    success(message, duration) { return this.show(message, 'success', duration); },
    error(message, duration) { return this.show(message, 'error', duration); },
    warning(message, duration) { return this.show(message, 'warning', duration); },
    info(message, duration) { return this.show(message, 'info', duration); }
};

// Add CSS animation for toasts
const toastStyles = document.createElement('style');
toastStyles.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
`;
document.head.appendChild(toastStyles);

// ---------------------------------------------------------------------
// CERTORA NOTIFICATION CHECKER - Polls for completed background jobs
// ---------------------------------------------------------------------
const CertoraNotificationChecker = {
    checkInterval: null,
    isChecking: false,

    async checkNotifications() {
        if (this.isChecking) return;
        this.isChecking = true;

        try {
            const response = await fetch('/api/certora/notifications', {
                credentials: 'include'
            });

            if (!response.ok) {
                this.isChecking = false;
                return;
            }

            const data = await response.json();
            const { notifications } = data;

            if (notifications && notifications.length > 0) {
                // Show notifications for each completed job
                for (const job of notifications) {
                    const hasViolations = job.rules_violated > 0;
                    const type = hasViolations ? 'warning' : 'success';
                    const message = `
                        <strong>Certora Verification Complete!</strong><br>
                        <span style="opacity: 0.9;">${job.contract_name}</span><br>
                        <span style="margin-top: 4px; display: inline-block;">
                            ✓ ${job.rules_verified} verified
                            ${hasViolations ? `| ⚠️ ${job.rules_violated} violations` : ''}
                        </span>
                    `;
                    ToastNotification.show(message, type, 15000);
                }

                // Dismiss the notifications
                const jobIds = notifications.map(n => n.job_id);
                await this.dismissNotifications(jobIds);
            }
        } catch (error) {
            debugLog('[CERTORA_NOTIFY] Error checking notifications:', error);
        }

        this.isChecking = false;
    },

    async dismissNotifications(jobIds) {
        try {
            // Get CSRF token
            const csrfResponse = await fetch('/csrf-token', { credentials: 'include' });
            const { csrf_token } = await csrfResponse.json();

            await fetch('/api/certora/notifications/dismiss', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrf_token
                },
                credentials: 'include',
                body: JSON.stringify({ job_ids: jobIds })
            });
        } catch (error) {
            debugLog('[CERTORA_NOTIFY] Error dismissing notifications:', error);
        }
    },

    start(intervalMs = 60000) {
        // Stop any existing interval first
        this.stop();

        // Check immediately on start
        this.checkNotifications();

        // Then check periodically - track with ResourceManager
        this.checkInterval = ResourceManager.addInterval(setInterval(() => {
            this.checkNotifications();
        }, intervalMs));

        debugLog('[CERTORA_NOTIFY] Notification checker started');
    },

    stop() {
        if (this.checkInterval) {
            ResourceManager.removeInterval(this.checkInterval);
            this.checkInterval = null;
        }
    }
};

// Start notification checker when page loads (for Enterprise/Diamond users)
document.addEventListener('DOMContentLoaded', () => {
    // Start checking after a short delay to not impact initial page load
    setTimeout(() => {
        CertoraNotificationChecker.start(60000); // Check every 60 seconds
    }, 5000);
});

// ---------------------------------------------------------------------
// AUDIT QUEUE TRACKER - Real-time queue position and status updates
// ---------------------------------------------------------------------
class AuditQueueTracker {
    constructor() {
        this.jobId = null;
        this.auditKey = null;
        this.ws = null;
        this.onUpdate = null;
        this.onComplete = null;
        this.onError = null;
        this.pollInterval = null;
    }

    async submitAudit(formData, csrfToken) {
        try {
            const response = await fetch('/audit/submit', {
                method: 'POST',
                headers: { 'X-CSRFToken': csrfToken },
                body: formData,
                credentials: 'include'
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || 'Failed to submit audit');
            }

            const data = await response.json();
            this.jobId = data.job_id;
            this.auditKey = data.audit_key;

            log('QUEUE', `Audit submitted: job_id=${this.jobId}, audit_key=${this.auditKey}, position=${data.position}`);

            // Show audit key to user
            if (this.auditKey) {
                this.showAuditKey(this.auditKey);
            }

            // Start WebSocket connection for real-time updates
            this.connectWebSocket();

            // Fallback polling in case WebSocket fails
            this.startPolling();

            return data;
        } catch (error) {
            log('QUEUE_ERROR', error.message);
            throw error;
        }
    }

    showAuditKey(auditKey) {
        // Validate audit key format for security
        if (typeof auditKey !== 'string' || !auditKey.startsWith('dga_') || auditKey.length > 64) {
            console.error('Invalid audit key format');
            return;
        }

        // Create and show the audit key notification
        const existingNotif = document.getElementById('audit-key-notification');
        if (existingNotif) existingNotif.remove();

        // Build notification using safe DOM methods (no innerHTML with user data)
        const notification = document.createElement('div');
        notification.id = 'audit-key-notification';
        notification.className = 'audit-key-notification';

        const card = document.createElement('div');
        card.className = 'audit-key-card';

        // Header
        const header = document.createElement('div');
        header.className = 'audit-key-header';

        const icon = document.createElement('span');
        icon.className = 'audit-key-icon';
        icon.textContent = '🔑';

        const title = document.createElement('h4');
        title.textContent = 'Your Audit Access Key';

        const closeBtn = document.createElement('button');
        closeBtn.className = 'audit-key-close';
        closeBtn.textContent = '×';
        closeBtn.addEventListener('click', () => notification.remove());

        header.appendChild(icon);
        header.appendChild(title);
        header.appendChild(closeBtn);

        // Body
        const body = document.createElement('div');
        body.className = 'audit-key-body';

        const info = document.createElement('p');
        info.className = 'audit-key-info';
        info.textContent = 'Save this key to access your results anytime, even after leaving this page:';

        const keyValue = document.createElement('div');
        keyValue.className = 'audit-key-value';

        const keyDisplay = document.createElement('code');
        keyDisplay.id = 'audit-key-display';
        keyDisplay.textContent = auditKey; // Safe: textContent escapes HTML

        const copyBtn = document.createElement('button');
        copyBtn.className = 'audit-key-copy';
        copyBtn.title = 'Copy to clipboard';
        copyBtn.textContent = '📋';
        copyBtn.dataset.auditKey = auditKey; // Store in data attribute
        copyBtn.addEventListener('click', () => window.copyAuditKey(copyBtn.dataset.auditKey));

        keyValue.appendChild(keyDisplay);
        keyValue.appendChild(copyBtn);

        const hint = document.createElement('p');
        hint.className = 'audit-key-hint';
        hint.textContent = '📧 Pro/Enterprise users will also receive an email when the audit completes.';

        body.appendChild(info);
        body.appendChild(keyValue);
        body.appendChild(hint);

        card.appendChild(header);
        card.appendChild(body);
        notification.appendChild(card);

        // Insert after the audit form
        const auditForm = document.getElementById('audit-form');
        if (auditForm) {
            auditForm.insertAdjacentElement('afterend', notification);
        } else {
            document.body.appendChild(notification);
        }

        // Store in localStorage for persistence (only if valid format)
        try {
            const savedKeys = JSON.parse(localStorage.getItem('auditKeys') || '[]');
            savedKeys.unshift({ key: auditKey, timestamp: Date.now() });
            // Keep only last 10 keys
            localStorage.setItem('auditKeys', JSON.stringify(savedKeys.slice(0, 10)));
        } catch (e) {
            console.warn('Could not save audit key to localStorage:', e);
        }
    }
    
    connectWebSocket() {
        if (!this.jobId) return;

        // Clean up existing connection first
        this.disconnectWebSocket();

        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws/job/${this.jobId}`;

        try {
            this.ws = new WebSocket(wsUrl);
            ResourceManager.addWebSocket(this.ws);

            this.ws.onopen = () => {
                log('QUEUE_WS', 'Connected to job status WebSocket');
                // Stop polling since WebSocket is working
                this.stopPolling();
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleStatusUpdate(data);
                } catch (e) {
                    log('QUEUE_WS', 'Failed to parse message:', e);
                }
            };

            this.ws.onerror = (error) => {
                log('QUEUE_WS', 'WebSocket error, falling back to polling');
                this.startPolling();
            };

            this.ws.onclose = () => {
                log('QUEUE_WS', 'WebSocket closed');
                ResourceManager.removeWebSocket(this.ws);
            };

            // Keep-alive ping every 25 seconds - track with ResourceManager
            this.pingInterval = ResourceManager.addInterval(setInterval(() => {
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send('ping');
                }
            }, 25000));

        } catch (e) {
            log('QUEUE_WS', 'Failed to connect WebSocket:', e);
            this.startPolling();
        }
    }

    disconnectWebSocket() {
        if (this.pingInterval) {
            ResourceManager.removeInterval(this.pingInterval);
            this.pingInterval = null;
        }
        if (this.ws) {
            ResourceManager.removeWebSocket(this.ws);
            this.ws = null;
        }
    }

    startPolling() {
        if (this.pollInterval) return; // Already polling

        this.pollInterval = ResourceManager.addInterval(setInterval(async () => {
            if (!this.jobId) return;

            try {
                const response = await fetch(`/audit/status/${this.jobId}`);
                if (response.ok) {
                    const data = await response.json();
                    this.handleStatusUpdate(data);
                }
            } catch (e) {
                log('QUEUE_POLL', 'Polling error:', e);
            }
        }, 3000)); // Poll every 3 seconds
    }

    stopPolling() {
        if (this.pollInterval) {
            ResourceManager.removeInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }
    
    handleStatusUpdate(data) {
        log('QUEUE_STATUS', `Status: ${data.status}, Position: ${data.position}, Phase: ${data.current_phase}`);
        
        // Call user-provided update handler
        if (this.onUpdate) {
            this.onUpdate(data);
        }
        
        switch (data.status) {
            case 'queued':
                this.showQueuePosition(data);
                break;
            case 'processing':
                this.showProcessing(data);
                break;
            case 'completed':
                this.stopPolling();
                this.disconnect();
                if (this.onComplete) {
                    this.onComplete(data.result);
                }
                break;
            case 'failed':
                this.stopPolling();
                this.disconnect();
                if (this.onError) {
                    this.onError(data.error);
                }
                break;
        }
    }
    
    showQueuePosition(data) {
        const queueUI = document.getElementById('queue-status');
        if (!queueUI) return;
        
        // Get user's tier from sidebar or default to free
        const userTier = document.getElementById('sidebar-tier-name')?.textContent?.toLowerCase() || 'free';
        const isFreeTier = userTier === 'free';
        const isStarterTier = userTier === 'starter';
        
        // Build users ahead breakdown if available
        let usersAheadHtml = '';
        if (data.users_ahead) {
            const ahead = data.users_ahead;
            const parts = [];
            if (ahead.enterprise > 0) parts.push(`${ahead.enterprise} Enterprise`);
            if (ahead.pro > 0) parts.push(`${ahead.pro} Pro`);
            if (ahead.starter > 0) parts.push(`${ahead.starter} Starter`);
            if (ahead.free > 0) parts.push(`${ahead.free} Free`);
            if (parts.length > 0) {
                usersAheadHtml = `<div class="queue-breakdown">${parts.join(' • ')}</div>`;
            }
        }
        
        // Upgrade prompt for Free/Starter users
        let upgradeHtml = '';
        if (isFreeTier) {
            upgradeHtml = `
                <div class="queue-upgrade-prompt">
                    <span class="upgrade-icon">⚡</span>
                    <span>Pro & Enterprise audits are prioritized.</span>
                    <a href="#tier-select" class="upgrade-link" onclick="document.getElementById('tier-select').scrollIntoView({behavior: 'smooth'})">Upgrade to skip ahead</a>
                </div>
            `;
        } else if (isStarterTier) {
            upgradeHtml = `
                <div class="queue-upgrade-prompt starter">
                    <span class="upgrade-icon">⚡</span>
                    <span>Pro & Enterprise audits get priority.</span>
                    <a href="#tier-select" class="upgrade-link" onclick="document.getElementById('tier-select').scrollIntoView({behavior: 'smooth'})">Upgrade to Pro</a>
                </div>
            `;
        }
        
        queueUI.innerHTML = `
            <div class="queue-card">
                <div class="queue-icon">⏳</div>
                <div class="queue-info">
                    <h3>In Queue</h3>
                    <div class="queue-position">
                        <span class="position-number">${data.position}</span>
                        <span class="position-label">${data.position === 1 ? "You're next!" : 'position in line'}</span>
                    </div>
                    ${usersAheadHtml}
                    <div class="queue-stats">
                        ${data.queue_length || data.position} in queue • ${data.processing_count || 1} processing
                    </div>
                    ${upgradeHtml}
                </div>
                <div class="queue-animation">
                    <div class="pulse"></div>
                </div>
            </div>
        `;
        queueUI.style.display = 'block';
    }
    
    showProcessing(data) {
        const queueUI = document.getElementById('queue-status');
        if (!queueUI) return;
        
        const phases = {
            'slither': { icon: '🔍', label: 'Running Slither Analysis', progress: 10 },
            'mythril': { icon: '🧠', label: 'Running Mythril Symbolic Analysis', progress: 25 },
            'echidna': { icon: '🧪', label: 'Running Echidna Fuzzing', progress: 40 },
            'certora': { icon: '🔒', label: 'Running Formal Verification', progress: 55 },
            'grok': { icon: '🤖', label: 'Claude AI Analysis & Report Generation', progress: 65 },
            'finalizing': { icon: '✨', label: 'Finalizing Report', progress: 90 },
            'complete': { icon: '✅', label: 'Complete!', progress: 100 }
        };
        
        const currentPhase = phases[data.current_phase] || { icon: '⚡', label: 'Processing...', progress: data.progress_percent || 50 };
        
        queueUI.innerHTML = `
            <div class="processing-card">
                <div class="processing-icon">${currentPhase.icon}</div>
                <div class="processing-info">
                    <h3>${currentPhase.label}</h3>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${currentPhase.progress}%"></div>
                    </div>
                    <div class="progress-text">${currentPhase.progress}% complete</div>
                </div>
            </div>
        `;
        queueUI.style.display = 'block';
    }
    
    hideQueueUI() {
        const queueUI = document.getElementById('queue-status');
        if (queueUI) {
            queueUI.style.display = 'none';
        }
    }
    
    disconnect() {
        this.stopPolling();
        this.disconnectWebSocket();
        this.jobId = null;
    }
}

// Global queue tracker instance
const queueTracker = new AuditQueueTracker();

// ---------------------------------------------------------------------
// WALLET CONNECTION MANAGER - Connect MetaMask/WalletConnect, import contracts
// ---------------------------------------------------------------------
class WalletManager {
    constructor() {
        this.provider = null;
        this.signer = null;
        this.address = null;
        this.contracts = [];
        this.isConnecting = false;

        // EIP-6963 detected wallets
        this.detectedWallets = [];

        // WalletConnect provider reference
        this.wcProvider = null;

        // Etherscan API for fetching deployed contracts
        this.etherscanApiKey = null;
        this.etherscanBaseUrl = 'https://api.etherscan.io/api';
    }

    async init() {
        // Setup EIP-6963 wallet detection
        this.setupEIP6963Detection();

        // Check if already connected (from previous session)
        const savedAddress = localStorage.getItem('walletAddress');
        const savedProviderType = localStorage.getItem('walletProviderType');

        if (savedAddress) {
            try {
                // Try to reconnect based on saved provider type
                if (savedProviderType === 'injected' && window.ethereum) {
                    const accounts = await window.ethereum.request({ method: 'eth_accounts' });
                    if (accounts.length > 0 && accounts[0].toLowerCase() === savedAddress.toLowerCase()) {
                        await this.connectWithProvider(window.ethereum, true);
                    }
                }
            } catch (e) {
                log('WALLET', 'Failed to restore session:', e);
                localStorage.removeItem('walletAddress');
                localStorage.removeItem('walletProviderType');
            }
        }

        // Listen for account changes on injected provider
        if (window.ethereum) {
            window.ethereum.on('accountsChanged', (accounts) => {
                if (accounts.length === 0) {
                    this.disconnect();
                } else if (accounts[0].toLowerCase() !== this.address?.toLowerCase()) {
                    this.connectWithProvider(window.ethereum, false);
                }
            });

            window.ethereum.on('chainChanged', () => {
                window.location.reload();
            });
        }

        this.bindEvents();
        this.updateWalletTags();
    }

    setupEIP6963Detection() {
        // EIP-6963: Multi Injected Provider Discovery
        this.detectedWallets = [];

        const handleAnnouncement = (event) => {
            const { info, provider } = event.detail;
            log('WALLET', `EIP-6963 detected: ${info.name}`);

            // Avoid duplicates
            if (!this.detectedWallets.find(w => w.info.uuid === info.uuid)) {
                this.detectedWallets.push({ info, provider });
                this.renderDetectedWallets();
            }
        };

        window.addEventListener('eip6963:announceProvider', handleAnnouncement);

        // Request providers to announce themselves
        window.dispatchEvent(new Event('eip6963:requestProvider'));
    }

    renderDetectedWallets() {
        const container = document.getElementById('detected-wallets-list');
        const section = document.getElementById('detected-wallets-section');

        if (!container || !section) return;

        if (this.detectedWallets.length === 0) {
            section.style.display = 'none';
            return;
        }

        section.style.display = 'block';
        container.innerHTML = '';

        this.detectedWallets.forEach(wallet => {
            const btn = document.createElement('button');
            btn.className = 'wallet-option detected';
            // Escape wallet name to prevent XSS from malicious wallet providers
            const safeName = escapeHtml(wallet.info.name);
            const safeIcon = escapeHtml(wallet.info.icon);
            btn.innerHTML = `
                <img src="${safeIcon}" alt="${safeName}" class="wallet-icon">
                <span class="wallet-name">${safeName}</span>
                <span class="wallet-tag installed-tag">Detected</span>
            `;
            btn.addEventListener('click', () => this.connectWithEIP6963Wallet(wallet));
            container.appendChild(btn);
        });
    }

    updateWalletTags() {
        // Update MetaMask tag
        const metamaskTag = document.getElementById('metamask-tag');
        if (metamaskTag) {
            if (window.ethereum?.isMetaMask) {
                metamaskTag.textContent = 'Installed';
                metamaskTag.className = 'wallet-tag installed-tag';
            }
        }

        // Update Coinbase tag
        const coinbaseTag = document.getElementById('coinbase-tag');
        if (coinbaseTag) {
            if (window.ethereum?.isCoinbaseWallet || window.coinbaseWalletExtension) {
                coinbaseTag.textContent = 'Installed';
                coinbaseTag.className = 'wallet-tag installed-tag';
            }
        }

        // Show no wallet message if needed
        const noWalletMsg = document.getElementById('no-wallet-message');
        const popularSection = document.getElementById('popular-wallets-section');
        if (noWalletMsg && popularSection) {
            if (!window.ethereum && this.detectedWallets.length === 0) {
                noWalletMsg.style.display = 'block';
            } else {
                noWalletMsg.style.display = 'none';
            }
        }
    }

    bindEvents() {
        const connectBtn = document.getElementById('wallet-connect');
        const disconnectBtn = document.getElementById('wallet-disconnect');
        const contractSelect = document.getElementById('deployed-contracts');
        const importBtn = document.getElementById('import-contract-btn');

        // Main connect button opens modal
        if (connectBtn) {
            connectBtn.addEventListener('click', () => this.openWalletModal());
        }

        if (disconnectBtn) {
            disconnectBtn.addEventListener('click', () => this.disconnect());
        }

        if (contractSelect) {
            contractSelect.addEventListener('change', (e) => {
                if (importBtn) {
                    importBtn.disabled = !e.target.value;
                }
            });
        }

        if (importBtn) {
            importBtn.addEventListener('click', () => this.importSelectedContract());
        }

        // Wallet modal events
        this.bindModalEvents();
    }

    bindModalEvents() {
        const modalBackdrop = document.getElementById('wallet-modal-backdrop');
        const modal = document.getElementById('wallet-modal');
        const closeBtn = document.getElementById('wallet-modal-close');
        const walletConnectBack = document.getElementById('walletconnect-back');

        // Close modal handlers
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.closeWalletModal());
        }
        if (modalBackdrop) {
            modalBackdrop.addEventListener('click', () => this.closeWalletModal());
        }

        // WalletConnect back button
        if (walletConnectBack) {
            walletConnectBack.addEventListener('click', () => this.showWalletList());
        }

        // Wallet option clicks
        document.querySelectorAll('.wallet-option[data-wallet]').forEach(btn => {
            btn.addEventListener('click', () => {
                const walletType = btn.dataset.wallet;
                this.handleWalletSelection(walletType, btn);
            });
        });
    }

    openWalletModal() {
        const modal = document.getElementById('wallet-modal');
        const backdrop = document.getElementById('wallet-modal-backdrop');

        if (modal) modal.style.display = 'block';
        if (backdrop) backdrop.style.display = 'block';

        // Refresh detected wallets
        this.updateWalletTags();
        this.showWalletList();
    }

    closeWalletModal() {
        const modal = document.getElementById('wallet-modal');
        const backdrop = document.getElementById('wallet-modal-backdrop');

        if (modal) modal.style.display = 'none';
        if (backdrop) backdrop.style.display = 'none';

        // Reset any connecting states
        document.querySelectorAll('.wallet-option.connecting').forEach(el => {
            el.classList.remove('connecting');
        });
    }

    showWalletList() {
        const walletListSections = document.querySelectorAll('#detected-wallets-section, #popular-wallets-section, #no-wallet-message');
        const qrSection = document.getElementById('walletconnect-qr-section');

        walletListSections.forEach(el => {
            if (el) el.style.display = '';
        });
        if (qrSection) qrSection.style.display = 'none';

        // Re-check visibility
        this.updateWalletTags();
        if (this.detectedWallets.length > 0) {
            document.getElementById('detected-wallets-section').style.display = 'block';
        }
    }

    async handleWalletSelection(walletType, buttonEl) {
        if (this.isConnecting) return;

        log('WALLET', `Selected wallet type: ${walletType}`);

        switch (walletType) {
            case 'metamask':
                await this.connectMetaMask(buttonEl);
                break;
            case 'coinbase':
                await this.connectCoinbase(buttonEl);
                break;
            case 'walletconnect':
                await this.showWalletConnectQR();
                break;
            default:
                log('WALLET', `Unknown wallet type: ${walletType}`);
        }
    }

    async connectWithEIP6963Wallet(wallet) {
        if (this.isConnecting) return;
        this.isConnecting = true;

        log('WALLET', `Connecting via EIP-6963: ${wallet.info.name}`);

        try {
            await wallet.provider.request({ method: 'eth_requestAccounts' });
            await this.connectWithProvider(wallet.provider, false);
            localStorage.setItem('walletProviderType', 'eip6963');
            localStorage.setItem('walletProviderUuid', wallet.info.uuid);
            this.closeWalletModal();
        } catch (error) {
            log('WALLET', 'EIP-6963 connection failed:', error);
            this.showError(error.message || 'Failed to connect wallet');
        } finally {
            this.isConnecting = false;
        }
    }

    async connectMetaMask(buttonEl) {
        if (this.isConnecting) return;
        this.isConnecting = true;

        if (buttonEl) buttonEl.classList.add('connecting');

        try {
            // Check if MetaMask is available
            const provider = window.ethereum?.providers?.find(p => p.isMetaMask) ||
                           (window.ethereum?.isMetaMask ? window.ethereum : null);

            if (!provider) {
                // Redirect to MetaMask install or deep link on mobile
                const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
                if (isMobile) {
                    // Deep link to MetaMask app
                    const currentUrl = encodeURIComponent(window.location.href);
                    window.location.href = `https://metamask.app.link/dapp/${window.location.host}${window.location.pathname}`;
                    return;
                } else {
                    window.open('https://metamask.io/download/', '_blank');
                    throw new Error('Please install MetaMask and refresh the page');
                }
            }

            await provider.request({ method: 'eth_requestAccounts' });
            await this.connectWithProvider(provider, false);
            localStorage.setItem('walletProviderType', 'injected');
            this.closeWalletModal();

        } catch (error) {
            log('WALLET', 'MetaMask connection failed:', error);
            if (error.code !== 4001) { // User rejected
                this.showError(error.message || 'Failed to connect MetaMask');
            }
        } finally {
            this.isConnecting = false;
            if (buttonEl) buttonEl.classList.remove('connecting');
        }
    }

    async connectCoinbase(buttonEl) {
        if (this.isConnecting) return;
        this.isConnecting = true;

        if (buttonEl) buttonEl.classList.add('connecting');

        try {
            // Check if Coinbase Wallet is available
            const provider = window.ethereum?.providers?.find(p => p.isCoinbaseWallet) ||
                           window.coinbaseWalletExtension ||
                           (window.ethereum?.isCoinbaseWallet ? window.ethereum : null);

            if (!provider) {
                // Redirect to Coinbase Wallet
                const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
                if (isMobile) {
                    const currentUrl = encodeURIComponent(window.location.href);
                    window.location.href = `https://go.cb-w.com/dapp?cb_url=${currentUrl}`;
                    return;
                } else {
                    window.open('https://www.coinbase.com/wallet/downloads', '_blank');
                    throw new Error('Please install Coinbase Wallet and refresh the page');
                }
            }

            await provider.request({ method: 'eth_requestAccounts' });
            await this.connectWithProvider(provider, false);
            localStorage.setItem('walletProviderType', 'injected');
            this.closeWalletModal();

        } catch (error) {
            log('WALLET', 'Coinbase connection failed:', error);
            if (error.code !== 4001) {
                this.showError(error.message || 'Failed to connect Coinbase Wallet');
            }
        } finally {
            this.isConnecting = false;
            if (buttonEl) buttonEl.classList.remove('connecting');
        }
    }

    async showWalletConnectQR() {
        const walletListSections = document.querySelectorAll('#detected-wallets-section, #popular-wallets-section, #no-wallet-message');
        const qrSection = document.getElementById('walletconnect-qr-section');
        const qrContainer = document.getElementById('walletconnect-qr-container');

        walletListSections.forEach(el => {
            if (el) el.style.display = 'none';
        });
        if (qrSection) qrSection.style.display = 'block';

        // Show loading state
        if (qrContainer) {
            qrContainer.innerHTML = '<div class="qr-loading">Initializing WalletConnect...</div>';
        }

        try {
            // Check if WalletConnect provider is available
            if (!window.WalletConnectProvider) {
                throw new Error('WalletConnect not loaded yet. Please wait a moment and try again.');
            }

            log('WALLET', 'Initializing WalletConnect...');

            // Initialize WalletConnect EthereumProvider
            const provider = await window.WalletConnectProvider.init({
                projectId: window.WALLETCONNECT_PROJECT_ID,
                metadata: window.WALLETCONNECT_METADATA,
                showQrModal: false, // We'll show our own QR
                chains: [1], // Ethereum mainnet
                optionalChains: [8453, 42161, 137, 10], // Base, Arbitrum, Polygon, Optimism
                methods: ['eth_sendTransaction', 'eth_signTransaction', 'eth_sign', 'personal_sign', 'eth_signTypedData'],
                events: ['chainChanged', 'accountsChanged', 'disconnect']
            });

            // Listen for display_uri event to show QR code
            provider.on('display_uri', async (uri) => {
                log('WALLET', 'WalletConnect URI received');

                if (qrContainer) {
                    // Create QR code canvas
                    qrContainer.innerHTML = `
                        <div style="text-align: center;">
                            <canvas id="walletconnect-qr-canvas" style="max-width: 280px; margin: 0 auto;"></canvas>
                            <p style="margin-top: var(--space-3); font-size: var(--text-sm); color: var(--text-secondary);">
                                Scan with your mobile wallet
                            </p>
                            <p style="margin-top: var(--space-2); font-size: var(--text-xs); color: var(--text-tertiary);">
                                MetaMask, Trust, Rainbow, Coinbase & 300+ wallets
                            </p>
                        </div>
                    `;

                    // Generate QR code
                    const canvas = document.getElementById('walletconnect-qr-canvas');
                    if (canvas && window.QRCode) {
                        await window.QRCode.toCanvas(canvas, uri, {
                            width: 280,
                            margin: 2,
                            color: {
                                dark: '#000000',
                                light: '#ffffff'
                            }
                        });
                    }
                }
            });

            // Listen for connection
            provider.on('connect', async () => {
                log('WALLET', 'WalletConnect connected!');

                // Store provider reference
                this.wcProvider = provider;

                // Connect with ethers
                await this.connectWithProvider(provider);

                // Close modal on success
                this.closeWalletModal();
            });

            // Listen for disconnect
            provider.on('disconnect', () => {
                log('WALLET', 'WalletConnect disconnected');
                this.disconnect();
            });

            // Enable the provider (triggers QR code display)
            await provider.connect();

        } catch (error) {
            log('WALLET', 'WalletConnect error:', error);

            if (qrContainer) {
                qrContainer.innerHTML = `
                    <div style="text-align: center; padding: var(--space-4);">
                        <div style="font-size: 2rem; margin-bottom: var(--space-3);">⚠️</div>
                        <p style="color: var(--text-secondary); margin-bottom: var(--space-4);">
                            ${error.message || 'Failed to initialize WalletConnect'}
                        </p>
                        <p style="font-size: var(--text-sm); color: var(--text-tertiary); margin-bottom: var(--space-4);">
                            <strong>Alternative:</strong> Open this page in your wallet app's browser
                        </p>
                        <div style="background: var(--glass-bg); padding: var(--space-3); border-radius: var(--radius-md); font-size: var(--text-xs); word-break: break-all; color: var(--text-tertiary);">
                            ${window.location.origin}
                        </div>
                        <button onclick="window.walletManager.showWalletConnectQR()" class="btn btn-secondary" style="margin-top: var(--space-4);">
                            Try Again
                        </button>
                    </div>
                `;
            }
        }
    }

    // Legacy connect method (kept for backwards compatibility)
    async connect() {
        this.openWalletModal();
    }

    async connectWithProvider(ethereumProvider, isReconnect = false) {
        try {
            // Use ethers.js v6 BrowserProvider
            this.provider = new ethers.BrowserProvider(ethereumProvider);
            this.signer = await this.provider.getSigner();
            this.address = await this.signer.getAddress();

            log('WALLET', `Connected: ${this.address}`);

            // Save to localStorage for session persistence
            localStorage.setItem('walletAddress', this.address);

            // Verify ownership with signature (only on fresh connect, not reconnect)
            if (!isReconnect) {
                const verified = await this.verifyOwnership();
                if (!verified) {
                    throw new Error('Wallet verification failed');
                }
            }

            // Update UI
            this.updateUI(true);

            // Fetch deployed contracts
            await this.fetchDeployedContracts();

        } catch (error) {
            log('WALLET', 'Provider connection failed:', error);
            throw error;
        }
    }

    async fetchCsrfToken() {
        try {
            const response = await fetch(`/csrf-token?_=${Date.now()}`, {
                method: 'GET',
                credentials: 'include'
            });
            if (!response.ok) throw new Error('CSRF fetch failed');
            const data = await response.json();
            return data.csrf_token || null;
        } catch (e) {
            log('WALLET', 'CSRF token fetch failed:', e);
            return null;
        }
    }

    async verifyOwnership() {
        try {
            const message = `DeFiGuard AI Wallet Verification\n\nTimestamp: ${Date.now()}\n\nSign this message to prove you own this wallet. This does not cost any gas.`;
            const signature = await this.signer.signMessage(message);

            // Fetch fresh CSRF token
            const csrfToken = await this.fetchCsrfToken();
            if (!csrfToken) {
                log('WALLET', 'No CSRF token available');
                return true; // Continue without backend save
            }

            // Send to backend to verify and save
            const response = await fetch('/api/wallet/connect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken
                },
                credentials: 'include',
                body: JSON.stringify({
                    address: this.address,
                    message: message,
                    signature: signature
                })
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.detail || 'Verification failed');
            }

            const data = await response.json();
            log('WALLET', 'Verification successful:', data);

            // Store Etherscan API key if provided
            if (data.etherscan_api_key) {
                this.etherscanApiKey = data.etherscan_api_key;
            }

            return true;

        } catch (error) {
            log('WALLET', 'Verification error:', error);
            // Don't block on signature rejection - just continue without saving to backend
            if (error.code === 4001 || error.message?.includes('rejected')) {
                log('WALLET', 'User rejected signature - continuing without backend save');
                return true; // Allow continuing without backend verification
            }
            return false;
        }
    }

    async fetchDeployedContracts() {
        const countHint = document.getElementById('wallet-contract-count');
        const selectorWrapper = document.getElementById('contract-selector-wrapper');
        const select = document.getElementById('deployed-contracts');

        if (countHint) countHint.textContent = 'Loading your contracts...';

        try {
            // Fetch from backend (which proxies to Etherscan)
            const response = await fetch(`/api/wallet/contracts?address=${this.address}`, {
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error('Failed to fetch contracts');
            }

            const data = await response.json();
            this.contracts = data.contracts || [];

            log('WALLET', `Found ${this.contracts.length} contracts`);

            // Update UI
            if (select) {
                select.innerHTML = '<option value="">-- Select a contract to import --</option>';

                this.contracts.forEach(contract => {
                    const option = document.createElement('option');
                    option.value = contract.address;
                    option.textContent = `${contract.name || 'Unknown'} (${this.formatAddress(contract.address)})`;
                    option.dataset.name = contract.name || '';
                    select.appendChild(option);
                });
            }

            if (this.contracts.length > 0) {
                if (selectorWrapper) selectorWrapper.style.display = 'block';
                if (countHint) countHint.textContent = `Found ${this.contracts.length} verified contract${this.contracts.length !== 1 ? 's' : ''}`;
            } else {
                if (selectorWrapper) selectorWrapper.style.display = 'none';
                if (countHint) countHint.textContent = 'No verified contracts found for this address';
            }

        } catch (error) {
            log('WALLET', 'Failed to fetch contracts:', error);
            if (countHint) countHint.textContent = 'Could not load contracts';
            if (selectorWrapper) selectorWrapper.style.display = 'none';
        }
    }

    async importSelectedContract() {
        const select = document.getElementById('deployed-contracts');
        const importBtn = document.getElementById('import-contract-btn');
        const contractAddress = select?.value;

        if (!contractAddress) return;

        if (importBtn) {
            importBtn.textContent = '⏳ Importing...';
            importBtn.disabled = true;
        }

        try {
            // Fetch source code from backend
            const response = await fetch(`/api/wallet/contract-source?address=${contractAddress}`, {
                credentials: 'include'
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.detail || 'Failed to fetch source code');
            }

            const data = await response.json();

            if (!data.source_code) {
                throw new Error('No verified source code found for this contract');
            }

            // Populate the code editor
            const codeInput = document.getElementById('code');
            if (codeInput) {
                codeInput.value = data.source_code;
                log('WALLET', `Imported ${data.contract_name || 'contract'} source code (${data.source_code.length} chars)`);

                // Show success message
                this.showSuccess(`Imported ${data.contract_name || 'contract'} successfully!`);

                // Also set contract address if that field exists
                const addressInput = document.getElementById('contract_address');
                if (addressInput) {
                    addressInput.value = contractAddress;
                }
            }

        } catch (error) {
            log('WALLET', 'Import failed:', error);
            this.showError(error.message || 'Failed to import contract');
        } finally {
            if (importBtn) {
                importBtn.textContent = '📥 Import Contract Code';
                importBtn.disabled = false;
            }
        }
    }

    async disconnect() {
        // Disconnect WalletConnect if active
        if (this.wcProvider) {
            try {
                await this.wcProvider.disconnect();
            } catch (e) {
                log('WALLET', 'WalletConnect disconnect error:', e);
            }
            this.wcProvider = null;
        }

        this.provider = null;
        this.signer = null;
        this.address = null;
        this.contracts = [];

        localStorage.removeItem('walletAddress');
        localStorage.removeItem('walletProviderType');

        this.updateUI(false);
        log('WALLET', 'Disconnected');
    }

    updateUI(connected) {
        const notConnectedEl = document.getElementById('wallet-not-connected');
        const connectedEl = document.getElementById('wallet-connected');
        const addressDisplay = document.getElementById('wallet-address-display');
        const selectorWrapper = document.getElementById('contract-selector-wrapper');

        if (connected && this.address) {
            if (notConnectedEl) notConnectedEl.style.display = 'none';
            if (connectedEl) connectedEl.style.display = 'block';
            if (addressDisplay) addressDisplay.textContent = this.formatAddress(this.address);
        } else {
            if (notConnectedEl) notConnectedEl.style.display = 'block';
            if (connectedEl) connectedEl.style.display = 'none';
            if (selectorWrapper) selectorWrapper.style.display = 'none';

            // Reset connect button
            const connectBtn = document.getElementById('wallet-connect');
            if (connectBtn) {
                connectBtn.textContent = '🔗 Connect Wallet';
                connectBtn.disabled = false;
            }
        }
    }

    formatAddress(address) {
        if (!address) return '';
        return `${address.slice(0, 6)}...${address.slice(-4)}`;
    }

    showError(message) {
        // Use existing notification system if available
        if (typeof showNotification === 'function') {
            showNotification(message, 'error');
        } else {
            alert(message);
        }
    }

    showSuccess(message) {
        if (typeof showNotification === 'function') {
            showNotification(message, 'success');
        }
    }
}

// Global wallet manager instance
const walletManager = new WalletManager();

// ---------------------------------------------------------------------
// UTILITY FUNCTIONS – Must be at top so they're available everywhere
// ---------------------------------------------------------------------

/**
 * Escape HTML special characters to prevent XSS
 */
function escapeHtml(text) {
    if (text === null || text === undefined) {
        return '';
    }
    
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    
    return String(text).replace(/[&<>"']/g, m => map[m]);
}

/**
 * Format code with syntax highlighting (simple version)
 */
function formatCode(code) {
    if (!code) return '';
    return `<pre><code>${escapeHtml(code)}</code></pre>`;
}

// ---------------------------------------------------------------------
// Section1: DOM Handling
// ---------------------------------------------------------------------
function waitForDOM(selectors, callback, maxAttempts = 20, interval = 300) {
    let attempts = 0;
    const elements = {};

    const check = () => {
        let allFound = true;
        const missing = [];

        for (const [key, selector] of Object.entries(selectors)) {
            const el = document.querySelector(selector);
            elements[key] = el;
            if (!el) {
                allFound = false;
                missing.push(key);
            }
        }

        if (allFound) {
            debugLog('[DEBUG] All DOM elements found – initializing');
            callback(elements);
        } else if (attempts < maxAttempts) {
            attempts++;
            debugLog(`[DEBUG] Waiting for DOM elements, attempt ${attempts}/${maxAttempts}, missing: ${missing.join(', ')}`);
            setTimeout(check, interval);
        } else {
            // *** FALLBACK: Proceed even if some elements are missing ***
            console.warn('[WARN] DOM elements not fully loaded after max attempts. Proceeding with partial init. Missing:', missing);
            callback(elements);
        }
    };

    check();
}

document.addEventListener("DOMContentLoaded", () => {
  // Section2: CSRF Token Management – fresh token for every POST (no cache, no stale risk, Stripe always works)
  const fetchCsrfToken = async () => {
    try {
      const response = await fetch(`/csrf-token?_=${Date.now()}`, {
        method: "GET",
        credentials: "include",
      });
      if (!response.ok) throw new Error("CSRF fetch failed");
      const data = await response.json();
      if (!data.csrf_token) throw new Error("Empty token");
      debugLog(`[DEBUG] Fresh CSRF token fetched: ${data.csrf_token.substring(0, 10)}...`);
      return data.csrf_token;
    } catch (err) {
      console.error("[ERROR] CSRF token fetch failed:", err);
      return null;
    }
  };

  const withCsrfToken = async (fetchFn) => {
    const token = await fetchCsrfToken();
    if (!token) {
      const usageWarning = document.querySelector(".usage-warning");
      if (usageWarning) {
        usageWarning.textContent = "Secure connection failed – please refresh the page";
        usageWarning.classList.add("error");
      }
      return;
    }
    return fetchFn(token);
  };

  // Initialize wallet connection manager
  walletManager.init().catch(e => console.warn('[WALLET] Init failed:', e));

  // Fetch signed-in proof from /me (enhances with sub/provider/logged_in)
  const fetchUsername = async () => {
    try {
      const resp = await fetch('/me', { credentials: 'include' });
      if (resp.ok) {
        const data = await resp.json();
        return data.logged_in ? { username: data.username, sub: data.sub, provider: data.provider } : null;
      }
    } catch (e) {
      console.debug('[AUTH] Failed to fetch /me');
    }
    return null;
  };

  // Section3: DOM Initialization
  waitForDOM(
    {
      auditForm: ".audit-section form",
      loading: ".loading",
      resultsDiv: "#results",
      riskScoreSpan: "#risk-score",
      issuesBody: "#issues-body",
      predictionsList: "#predictions-list",
      recommendationsList: "#recommendations-list",
      fuzzingList: "#fuzzing-list",
      remediationRoadmap: "#remediation-roadmap",
      usageWarning: ".usage-warning",
      sidebarTierName: "#sidebar-tier-name",
      sidebarTierUsage: "#sidebar-tier-usage",
      sidebarTierFeatures: "#sidebar-tier-features",
      tierDescription: "#tier-description",
      sizeLimit: "#size-limit",
      features: "#features",
      upgradeLink: "#upgrade-link",
      tierSelect: "#tier-select",
      tierSwitchButton: "#tier-switch",
      contractAddressInput: "#contract_address",
      facetWell: "#facet-preview",
      downloadReportButton: "#download-report",
      pdfDownloadButton: "#pdf-download",
      mintNftButton: "#mint-nft",
      customReportInput: "#custom_report",
      hamburger: "#hamburger",
      sidebar: "#sidebar",
      mainContent: ".main-content",
      logoutSidebar: "#logout-sidebar",
      authStatus: "#auth-status",
      auditLog: "#audit-log",
      sidebarSettingsLink: "#sidebar-settings-link",
      settingsModal: "#settings-modal",
      settingsModalBackdrop: "#settings-modal-backdrop",
      settingsModalClose: "#settings-modal-close",
      modalUsername: "#modal-username",
      modalEmail: "#modal-email",
      modalTier: "#modal-tier",
      modalMemberSince: "#modal-member-since",
      modalAuditsUsed: "#modal-audits-used",
      modalAuditsRemaining: "#modal-audits-remaining",
      modalSizeLimit: "#modal-size-limit",
      modalUsageProgress: "#modal-usage-progress",
      modalUsageText: "#modal-usage-text",
      modalApiSection: "#modal-api-section",
      apiKeyCountDisplay: "#api-key-count-display",
      apiKeysTableBody: "#api-keys-table-body",
      createApiKeyButton: "#create-api-key",
      createKeyModal: "#create-key-modal",
      createKeyModalBackdrop: "#create-key-modal-backdrop",
      createKeyModalClose: "#create-key-modal-close",
      newKeyLabelInput: "#new-key-label",
      createKeyConfirm: "#create-key-confirm",
      createKeyCancel: "#create-key-cancel",
      modalUpgradeButton: "#modal-upgrade",
      modalLogoutButton: "#modal-logout",
      // Certora Jobs section
      modalCertoraSection: "#modal-certora-section",
      certoraJobsTableBody: "#certora-jobs-table-body",
      certoraUploadBtn: "#certora-upload-btn",
      certoraUploadInput: "#certora-upload-input"
    },
    async (els) => {
      const {
        auditForm, loading, resultsDiv, riskScoreSpan, issuesBody, predictionsList,
        recommendationsList, fuzzingList, remediationRoadmap, usageWarning, sidebarTierName,
        sidebarTierUsage, sidebarTierFeatures, tierDescription, sizeLimit, features, 
        upgradeLink, tierSelect, tierSwitchButton, contractAddressInput, facetWell, 
        downloadReportButton, pdfDownloadButton, mintNftButton, customReportInput, apiKeySpan, 
        hamburger, sidebar, mainContent, logoutSidebar, authStatus, auditLog
      } = els;

      // Real-time audit log
      const logMessage = (msg) => {
        console.log(`[AUDIT] ${msg}`);
        if (auditLog) {
          const entry = document.createElement("div");
          entry.textContent = `[${new Date().toISOString()}] ${msg}`;
          auditLog.appendChild(entry);
          auditLog.scrollTop = auditLog.scrollHeight;
          if (auditLog.children.length > 100) auditLog.removeChild(auditLog.firstChild);
          if (auditLog.style.display === "none") auditLog.style.display = "block";
        }
      };

      // WebSocket audit log (single instance) - tracked by ResourceManager for cleanup
      const wsUsername = (await fetchUsername())?.username || "guest";
      const auditLogWs = new WebSocket(`${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws-audit-log?username=${encodeURIComponent(wsUsername)}`);
      ResourceManager.addWebSocket(auditLogWs);
      auditLogWs.onopen = () => logMessage("Connected to audit log");
      auditLogWs.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          if (data.type === "audit_log") logMessage(data.message);
        } catch (_) {}
      };
      auditLogWs.onerror = () => logMessage("WebSocket error");
      auditLogWs.onclose = () => {
        logMessage("Disconnected from audit log");
        ResourceManager.removeWebSocket(auditLogWs);
      };

      // Hamburger Menu
      if (hamburger && sidebar && mainContent) {
        hamburger.addEventListener("click", () => {
          sidebar.classList.toggle("open");
          hamburger.classList.toggle("open");
          document.body.classList.toggle('sidebar-open');
          // Use empty string when closed to let CSS handle centering
          mainContent.style.marginLeft = sidebar.classList.contains("open") ? "270px" : "";
        });
        hamburger.setAttribute("tabindex", "0");
        hamburger.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            hamburger.click();
          }
        });
      }
      
      // Section6: Authentication – instantly show real username + provider
      const updateAuthStatus = async () => {
        const user = await fetchUsername();  // Returns { username, sub, provider } from /me
        console.log(
          `[DEBUG] updateAuthStatus: user=${JSON.stringify(user) || 'null'}, time=${new Date().toISOString()}`
        );
        if (!authStatus) {
          console.error("[ERROR] #auth-status not found in DOM");
          return;
        }
        if (user && user.username) {
          const displayProvider = user.provider && user.provider !== "unknown" 
            ? user.provider 
            : (user.sub?.includes('|') ? user.sub.split('|')[0] : "auth0");
          authStatus.innerHTML = `Signed in as <strong>${escapeHtml(user.username)}</strong> <small>(${escapeHtml(displayProvider)})</small>`;
          localStorage.setItem('userSub', user.sub);
          sidebar.classList.add("logged-in");
        } else {
          authStatus.innerHTML = '<a href="/auth">Sign In / Create Account</a>';
          localStorage.removeItem('userSub');
          sidebar.classList.remove("logged-in");
        }
      };

      window.addEventListener("storage", () => {
        console.log(
          "[DEBUG] Storage event detected, re-running updateAuthStatus"
        );
        updateAuthStatus();
      });

      setTimeout(() => {
        console.log(
          `[DEBUG] Extended auth check after load, time=${new Date().toISOString()}`
        );
        updateAuthStatus();
      }, 5000);

      // Logout handler - with null check
      if (logoutSidebar) {
        logoutSidebar.addEventListener("click", (e) => {
          e.preventDefault();
          debugLog(`[DEBUG] Logout initiated, time=${new Date().toISOString()}`);
          window.location.href = "/logout";
        });
      } else {
        console.error("[ERROR] Logout button (#logout-sidebar) not found in DOM");
      }

      // Section7: Payment Handling
      const handlePostPaymentRedirect = async () => {
        const urlParams = new URLSearchParams(window.location.search);
        const sessionId = urlParams.get("session_id");
        const tier = urlParams.get("tier");
        const tempId = urlParams.get("temp_id");
        const username =
          urlParams.get("username") || localStorage.getItem("username");
        const upgradeStatus = urlParams.get("upgrade");
        const message = urlParams.get("message");
        console.log(
          `[DEBUG] Handling post-payment redirect: session_id=${sessionId}, tier=${tier}, temp_id=${tempId}, username=${username}, upgrade=${upgradeStatus}, message=${message}, time=${new Date().toISOString()}`
        );

        // NEW: Direct success from backend (no session_id needed)
        if (upgradeStatus === "success") {
          usageWarning.textContent =
            message || `Tier upgrade to ${tier} completed`;
          usageWarning.classList.add("success");
          console.log(
            `[DEBUG] Direct upgrade success: tier=${tier}, time=${new Date().toISOString()}`
          );
          window.history.replaceState({}, document.title, "/ui");
          await fetchTierData();
          await updateAuthStatus();
          return;
        }

        // NEW: Direct failure
        if (upgradeStatus === "failed" || upgradeStatus === "error") {
          usageWarning.textContent = message || "Tier upgrade failed";
          usageWarning.classList.add("error");
          console.log(
            `[DEBUG] Direct upgrade failed: message=${message}, time=${new Date().toISOString()}`
          );
          window.history.replaceState({}, document.title, "/ui");
          return;
        }

        // ORIGINAL: Legacy flow with session_id (Enterprise pending or old tier)
        if (sessionId && username) {
          try {
            let endpoint = "";
            let query = "";
            if (tempId) {
              endpoint = "/complete-enterprise-audit";
              query = `session_id=${encodeURIComponent(
                sessionId
              )}&temp_id=${encodeURIComponent(
                tempId
              )}&username=${encodeURIComponent(username)}`;
            } else if (tier) {
              endpoint = "/complete-tier-checkout";
              query = `session_id=${encodeURIComponent(
                sessionId
              )}&tier=${encodeURIComponent(
                tier
              )}&username=${encodeURIComponent(
                username
              )}`;
            } else {
              console.error(
                `[ERROR] Invalid post-payment redirect: missing tier or temp_id, time=${new Date().toISOString()}`
              );
              usageWarning.textContent =
                "Error: Invalid payment redirect parameters";
              usageWarning.classList.add("error");
              return;
            }
            console.log(
              `[DEBUG] Fetching ${endpoint}?${query}, time=${new Date().toISOString()}`
            );
            const response = await fetch(`${endpoint}?${query}`, {
              method: "GET",
              headers: {
                Accept: "application/json",
                "Cache-Control": "no-cache",
              },
              credentials: "include",
            });
            if (!response.ok) {
              const errorData = await response.json().catch(() => ({}));
              throw new Error(
                errorData.detail ||
                  `Failed to complete ${
                    tempId ? "Enterprise audit" : "tier upgrade"
                  }`
              );
            }
            localStorage.setItem("username", username);
            usageWarning.textContent = `Successfully completed ${
              tempId ? "Enterprise audit" : "tier upgrade"
            }`;
            usageWarning.classList.add("success");
            console.log(
              `[DEBUG] Post-payment completed: endpoint=${endpoint}, time=${new Date().toISOString()}`
            );
            await fetchTierData();
            await updateAuthStatus();
            window.history.replaceState({}, document.title, "/ui");
          } catch (error) {
            console.error(
              `[ERROR] Post-payment redirect error: ${
                error.message
              }, endpoint=${endpoint}, time=${new Date().toISOString()}`
            );
            usageWarning.textContent = `Error completing ${
              tempId ? "Enterprise audit" : "tier upgrade"
            }: ${error.message}`;
            usageWarning.classList.add("error");
            if (
              error.message.includes("User not found") ||
              error.message.includes("Please login")
            ) {
              console.log(
                `[DEBUG] Redirecting to /auth due to user not found, time=${new Date().toISOString()}`
              );
              window.location.href = "/auth?redirect_reason=post_payment";
            }
          }
        } else {
          console.warn(
            `[DEBUG] No post-payment redirect params found: session_id=${sessionId}, username=${username}, time=${new Date().toISOString()}`
          );
        }
      };

      // Section8: Facet Preview
      const fetchFacetPreview = async (
        contractAddress,
        attempt = 1,
        maxAttempts = 3
      ) => {
        facetWell.textContent = "";
        const loadingDiv = document.createElement("div");
        loadingDiv.className = "facet-loading";
        loadingDiv.setAttribute("aria-live", "polite");
        loadingDiv.innerHTML = `
                <div class="spinner"></div>
                <p>Loading facet preview...</p>
            `;
        facetWell.appendChild(loadingDiv);
        try {
          const user = await fetchUsername();
          const username = user?.username || "";
          const response = await fetch(
            `/facets/${contractAddress}?username=${encodeURIComponent(
              username
            )}&_=${Date.now()}`,
            {
              method: "GET",
              headers: {
                Accept: "application/json",
                "Cache-Control": "no-cache",
              },
              credentials: "include",
            }
          );
          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || "Failed to fetch facet data");
          }
          const data = await response.json();
          facetWell.textContent = "";
          if (data.facets.length === 0) {
            facetWell.textContent = "No facets found for this contract.";
            return;
          }
          const table = document.createElement("table");
          table.className = "table is-striped is-fullwidth";
          table.setAttribute("role", "table");
          table.setAttribute("aria-describedby", "facet-desc");
          table.innerHTML = `
                    <thead>
                        <tr>
                            <th>Facet Address</th>
                            <th>Function Selectors</th>
                            <th>Functions</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.facets
                          .map(
                            (facet) => `
                            <tr tabindex="0">
                                <td>${facet.facetAddress}</td>
                                <td>${facet.functionSelectors.join(", ")}</td>
                                <td>${facet.functions.join(", ")}</td>
                            </tr>
                        `
                          )
                          .join("")}
                    </tbody>
                `;
          const heading = document.createElement("h3");
          heading.className = "title is-4";
          heading.setAttribute("aria-label", "Enterprise Facet Preview");
          heading.textContent = "Enterprise Facet Preview";
          const desc = document.createElement("small");
          desc.id = "facet-desc";
          desc.textContent =
            "Table of facet addresses and function selectors for Enterprise Pattern contracts.";
          facetWell.appendChild(heading);
          facetWell.appendChild(table);
          facetWell.appendChild(desc);
          if (data.is_preview) {
            const watermark = document.createElement("p");
            watermark.className = "has-text-warning";
            watermark.textContent =
              "Pro Tier Preview – Upgrade to Enterprise add-on for Full Audit";
            facetWell.appendChild(watermark);
          }
          console.log(
            `[DEBUG] Facet preview loaded for address: ${contractAddress}, is_preview=${
              data.is_preview
            }, time=${new Date().toISOString()}`
          );
        } catch (error) {
          console.error(
            `Facet preview error (attempt ${attempt}/${maxAttempts}): ${error.message}`
          );
          if (
            attempt < maxAttempts &&
            !error.message.includes("Pro or Enterprise tier")
          ) {
            console.log(`Retrying facet fetch in 1s...`);
            setTimeout(
              () =>
                fetchFacetPreview(contractAddress, attempt + 1, maxAttempts),
              1000
            );
          } else {
            facetWell.textContent = `Error loading facet preview: ${error.message}`;
            facetWell.className = "has-text-danger";
            facetWell.setAttribute("aria-live", "assertive");
            if (error.message.includes("Pro or Enterprise tier")) {
              facetWell.innerHTML = `<p class="has-text-warning" aria-live="assertive">Enterprise Pattern facet preview requires Pro tier or Enterprise add-on. <a href="/upgrade">Upgrade now</a></p>`;
            }
          }
        }
      };

      contractAddressInput?.addEventListener("input", (e) => {
        const address = e.target.value.trim();
        if (address && address.match(/^0x[a-fA-F0-9]{40}$/)) {
          fetchFacetPreview(address);
        } else {
          facetWell.textContent = "";
        }
      });

      // Section9: Tier Management
      const fetchTierData = async () => {
        try {
          const user = await fetchUsername();
          const username = user?.username || "";
          const url = username
            ? `/tier?username=${encodeURIComponent(username)}`
            : "/tier";
          const response = await fetch(url, {
            headers: { Accept: "application/json" },
            credentials: "include",
          });
          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || "Failed to fetch tier data");
          }
          const data = await response.json();
          const {
            tier,
            size_limit,
            feature_flags,
            api_key,
            audit_count,
            audit_limit,
            } = data;
          auditCount = audit_count;
          auditLimit = audit_limit;
          // ✅ Handle logged out state
          if (data.tier === "logged_out" || data.logged_in === false) {
            debugLog("[DEBUG] User is logged out, showing logged-out UI");
            
            // Update sidebar to show "Not Signed In"
            sidebarTierName.textContent = "Not Signed In";
            sidebarTierUsage.textContent = "—";
            sidebarTierFeatures.innerHTML = '<div class="feature-item">Sign in to access features</div>';
            
            // Update main tier description
            tierDescription.textContent = "Sign in to start auditing smart contracts";
            sizeLimit.textContent = "Max file size: N/A";
            features.textContent = "Features: Sign in required";
            
            // Show warning message
            usageWarning.textContent = "Please sign in to audit smart contracts";
            usageWarning.classList.remove("error", "success");
            usageWarning.classList.add("info");
            usageWarning.style.display = "block";
            
            // Disable audit button
            const auditButton = document.querySelector("#audit-submit") || auditForm?.querySelector('button[type="submit"]');
            if (auditButton) {
              auditButton.disabled = true;
              auditButton.textContent = "Sign in to Audit";
              auditButton.style.opacity = "0.6";
              auditButton.style.cursor = "not-allowed";
            }
            
            // Hide upgrade link
            if (upgradeLink) upgradeLink.style.display = "none";
            
            // Hide API key section
            const apiKeySection = document.getElementById("api-key");
            if (apiKeySection) apiKeySection.style.display = "none";
            
            debugLog("[DEBUG] Logged-out UI state applied");
            return; // Exit early, don't run normal tier display code
          }
          
          // Update sidebar tier info card
          const tierNameCap = tier.charAt(0).toUpperCase() + tier.slice(1);
          const tierIcons = {
            free: "🆓",
            starter: "🚀",
            pro: "⭐",
            enterprise: "💎"
          };
          
          // Update tier icon
          const tierIconEl = document.querySelector(".tier-icon");
          if (tierIconEl) {
            tierIconEl.textContent = tierIcons[tier] || "💎";
          }
          
          sidebarTierName.textContent = tierNameCap;
          
          if (audit_limit === 9999 || size_limit === "Unlimited") {
            sidebarTierUsage.textContent = "Unlimited audits";
          } else {
            const remaining = auditLimit - auditCount;
            sidebarTierUsage.textContent = `${remaining} audits remaining (${auditCount}/${auditLimit} used)`;
          }
          
          // Build features list for sidebar - Premium tier descriptions
          const featuresList = [];
          const tierDisplayNames = {
            "enterprise": "Enterprise",
            "pro": "Team",
            "starter": "Developer",
            "free": "Free Trial"
          };
          const displayName = tierDisplayNames[tier] || tierNameCap;

          if (tier === "enterprise") {
            featuresList.push(
              "Unlimited audits & file size",
              "Slither + Mythril + Echidna",
              "Formal Verification (Coming Soon)",
              "Multi-AI consensus verification",
              "White-label reports",
              "Unlimited API keys",
              "Team accounts & permissions",
              "Dedicated account manager"
            );
          } else if (tier === "pro") {
            featuresList.push(
              "Unlimited audits",
              "Slither + Mythril + Claude AI",
              "Echidna fuzzing engine",
              "On-chain data analysis",
              "API access (3 keys)",
              "Priority queue & support"
            );
          } else if (tier === "starter") {
            featuresList.push(
              "25 audits/month",
              "Slither + Claude AI analysis",
              "Full vulnerability detection",
              "AI-powered fix recommendations",
              "PDF security reports",
              "MiCA + SEC FIT21 compliance"
            );
          } else {
            featuresList.push(
              "1 audit/month",
              "Slither static analysis",
              "Top 3 critical issues",
              "Basic risk score"
            );
          }

          sidebarTierFeatures.innerHTML = featuresList
            .map(f => `<div class="feature-item">✓ ${f}</div>`)
            .join("");

          // Update main tier description with professional copy
          tierDescription.textContent = `${displayName} Plan: ${
            tier === "enterprise"
              ? "Protocol-grade security with Slither, Mythril & Echidna. Formal Verification coming soon. White-label reports, unlimited API, dedicated support."
              : tier === "pro"
              ? "Unlimited audits with full security stack. Fuzzing, on-chain analysis, API access, priority support."
              : tier === "starter"
              ? `25 audits/month with AI-powered analysis & compliance scoring. (${auditCount}/${auditLimit} used)`
              : `Trial plan: 1 audit/month with basic analysis. (${auditCount}/${auditLimit} used)`
          }`;
          
          sizeLimit.textContent = `Max file size: ${size_limit}`;
          features.textContent = `Features: ${featuresList.join(", ")}`;
          usageWarning.textContent =
            tier === "free" || tier === "starter"
              ? `${
                  tier.charAt(0).toUpperCase() + tier.slice(1)
                } tier: ${auditCount}/${auditLimit} audits remaining`
              : "";
          usageWarning.classList.remove("error");
          upgradeLink.style.display = tier !== "enterprise" ? "inline-block" : "none";
          maxFileSize =
            size_limit === "Unlimited"
              ? Infinity
              : parseFloat(size_limit.replace("MB", "")) * 1024 * 1024;
          document.querySelector(
            "#file-help"
          ).textContent = `Max size: ${size_limit}. Ensure code is valid Solidity.`;
          document
            .querySelectorAll(".pro-enterprise-only")
            .forEach(
              (el) =>
                (el.style.display =
                  tier === "pro" || tier === "enterprise" ? "block" : "none")
            );
          customReportInput.style.display =
            tier === "pro" || tier === "enterprise" ? "block" : "none";
          downloadReportButton.style.display = feature_flags.reports
            ? "block"
            : "none";
          
          // PDF download button (Starter/Pro/Enterprise)
          if (pdfDownloadButton) {
            const hasPdfExport = feature_flags.pdf_export && feature_flags.pdf_export !== false;
            pdfDownloadButton.style.display = hasPdfExport ? "block" : "none";
            
            // Update button text based on PDF type
            if (feature_flags.pdf_export === "whitelabel") {
              pdfDownloadButton.textContent = "📑 Download White-Label PDF";
            } else if (feature_flags.pdf_export === "branded") {
              pdfDownloadButton.textContent = "📑 Download Branded PDF";
            } else if (feature_flags.pdf_export === "plain") {
              pdfDownloadButton.textContent = "📑 Download PDF Report";
            }
          }
          
          // NFT minting button (Enterprise only)
          if (mintNftButton) {
            mintNftButton.style.display = feature_flags.nft_rewards ? "block" : "none";
          }
          
          document
            .querySelectorAll(".enterprise-only")
            .forEach(
              (el) => (el.style.display = tier === "enterprise" ? "block" : "none")
            );
          const remediationPlaceholder = document.querySelector(".remediation-placeholder");
          if (remediationPlaceholder) remediationPlaceholder.style.display =
            tier === "enterprise" ? "block" : "none";
          
          const fuzzingPlaceholder = document.querySelector(".fuzzing-placeholder");
          if (fuzzingPlaceholder) fuzzingPlaceholder.style.display =
            feature_flags.fuzzing ? "block" : "none";
          
          const prioritySupport = document.querySelector(".priority-support");
          if (prioritySupport) prioritySupport.style.display =
            feature_flags.priority_support ? "block" : "none";
          debugLog(`[DEBUG] Tier data fetched: tier=${tier}, auditCount=${auditCount}, auditLimit=${auditLimit}, time=${new Date().toISOString()}`);
        } catch (error) {
          console.error("Tier fetch error:", error);
          usageWarning.textContent = `Error fetching tier data: ${error.message}`;
          usageWarning.classList.add("error");
        }
      };

      tierSwitchButton?.addEventListener("click", () => {
        withCsrfToken(async (token) => {
          if (!token) {
            usageWarning.textContent = "Unable to establish secure connection.";
            usageWarning.classList.add("error");
            console.error(
              `[ERROR] No CSRF token for tier switch, time=${new Date().toISOString()}`
            );
            return;
          }
          const selectedTier = tierSelect?.value;
          if (!selectedTier) {
            console.error(
              `[ERROR] tierSelect element not found, time=${new Date().toISOString()}`
            );
            usageWarning.textContent = "Error: Tier selection unavailable";
            usageWarning.classList.add("error");
            return;
          }
          if (!["starter", "pro", "enterprise"].includes(selectedTier)) {
            console.error(
              `[ERROR] Invalid tier selected: ${selectedTier}, time=${new Date().toISOString()}`
            );
            usageWarning.textContent = `Error: Invalid tier '${selectedTier}'. Choose starter, pro, or enterprise`;
            usageWarning.classList.add("error");
            return;
          }
          const user = await fetchUsername();
          if (!user?.username) {
            console.error("[ERROR] No username found, redirecting to /auth");
            window.location.href = "/auth";
            return;
          }
          const username = user.username;
          const effectiveTier = selectedTier;
          console.log(
            `[DEBUG] Initiating tier switch: username=${username}, tier=${effectiveTier}, time=${new Date().toISOString()}`
          );
          try {
            const requestBody = JSON.stringify({
              username: username,
              tier: effectiveTier
            });
            console.log(
              `[DEBUG] Sending /create-tier-checkout request with body: ${requestBody}, time=${new Date().toISOString()}`
            );
            const response = await fetch("/create-tier-checkout", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": token,
                Accept: "application/json",
              },
              credentials: "include",
              body: requestBody,
            });
            console.log(
              `[DEBUG] /create-tier-checkout response status: ${
                response.status
              }, ok: ${response.ok}, headers: ${JSON.stringify([
                ...response.headers,
              ])}, time=${new Date().toISOString()}`
            );
            if (!response.ok) {
              const errorData = await response.json().catch(() => ({}));
              console.error(
                `[ERROR] /create-tier-checkout failed: status=${
                  response.status
                }, detail=${
                  errorData.detail || "Unknown error"
                }, response_body=${JSON.stringify(
                  errorData
                )}, time=${new Date().toISOString()}`
              );
              throw new Error(
                errorData.detail ||
                  `Failed to initiate tier upgrade: ${response.status}`
              );
            }
            const data = await response.json();
            console.log(
              `[DEBUG] Stripe checkout session created: session_url=${
                data.session_url
              }, time=${new Date().toISOString()}`
            );
            window.location.href = data.session_url;
          } catch (error) {
            console.error(
              `[ERROR] Tier switch error: ${
                error.message
              }, time=${new Date().toISOString()}`
            );
            usageWarning.textContent = `Error initiating tier upgrade: ${error.message}`;
            usageWarning.classList.add("error");
          }
        });
      });

      // Section10: Enterprise Audit (query button directly as it's optional)
      const enterpriseAuditButton = document.querySelector("#enterprise-audit");
      enterpriseAuditButton?.addEventListener("click", () => {
        withCsrfToken(async (token) => {
          if (!token) {
            usageWarning.textContent = "Unable to establish secure connection.";
            usageWarning.classList.add("error");
            console.error(
              `[ERROR] No CSRF token for enterprise audit, time=${new Date().toISOString()}`
            );
            return;
          }
          const fileInput = document.querySelector("#file");
          const file = fileInput.files[0];
          if (!file) {
            usageWarning.textContent = "Please select a file for Enterprise audit";
            usageWarning.classList.add("error");
            console.error(
              `[ERROR] No file selected for enterprise audit, time=${new Date().toISOString()}`
            );
            return;
          }
          const user = await fetchUsername();
          if (!user?.username) {
            console.error(
              `[ERROR] No username found, redirecting to /auth, time=${new Date().toISOString()}`
            );
            window.location.href = "/auth";
            return;
          }
          const username = user.username;

          const formData = new FormData();
          formData.append("file", file);
          try {
            console.log(
              `[DEBUG] Sending /enterprise-audit request for username=${username}, time=${new Date().toISOString()}`
            );
            const response = await fetch(
              `/enterprise-audit?username=${encodeURIComponent(username)}`,
              {
                method: "POST",
                headers: { 
                  "X-CSRFToken": token
                },
                body: formData,
                credentials: "include",
              }
            );
            console.log(
              `[DEBUG] /enterprise-audit response status: ${
                response.status
              }, ok: ${response.ok}, headers: ${JSON.stringify([
                ...response.headers,
              ])}, time=${new Date().toISOString()}`
            );
            if (!response.ok) {
              const errorData = await response.json().catch(() => ({}));
              console.error(
                `[ERROR] /enterprise-audit failed: status=${
                  response.status
                }, detail=${
                  errorData.detail || "Unknown error"
                }, response_body=${JSON.stringify(
                  errorData
                )}, time=${new Date().toISOString()}`
              );
              throw new Error(
                errorData.detail || "Enterprise audit request failed"
              );
            }
            const data = await response.json();
            console.log(
              `[DEBUG] Redirecting to Stripe for Enterprise audit, session_url=${
                data.session_url
              }, time=${new Date().toISOString()}`
            );
            window.location.href = data.session_url;
          } catch (error) {
            console.error(
              `[ERROR] Enterprise audit error: ${
                error.message
              }, time=${new Date().toISOString()}`
            );
            usageWarning.textContent = `Error initiating Enterprise audit: ${error.message}`;
            usageWarning.classList.add("error");
          }
        });
      });

      // Section11: Audit Handling – ENHANCED FOR PRO/ENTERPRISE TIERS
      const handleAuditResponse = (data) => {
      // Store issues globally for modal access
      window.currentAuditIssues = data.report.issues || [];
      // Only update tier if backend provides it, otherwise keep existing tier from fetchTierData()
      if (data.tier) {
        window.currentAuditTier = data.tier;
        debugLog('[DEBUG] Tier from audit response:', data.tier);
      } else {
        // Fallback: Get tier from already-loaded tier data (from sidebar or initial fetch)
        const sidebarTier = document.getElementById('sidebar-tier-name')?.textContent?.toLowerCase() || 'free';
        window.currentAuditTier = window.currentAuditTier || sidebarTier;
        debugLog('[DEBUG] Tier from sidebar fallback:', window.currentAuditTier);
      } // Store tier info
      debugLog('[DEBUG] Audit response received:', {
        tier: data.tier,
        currentAuditTier: window.currentAuditTier,
        issuesCount: data.report?.issues?.length || 0,
        fullDataStructure: JSON.stringify(data, null, 2).substring(0, 500) // First 500 chars
      }); // Store tier info
  
      const report = data.report || {};
      const overageCost = data.overage_cost || null;
        
        // Display risk score
        riskScoreSpan.textContent = report.risk_score || "N/A";

        
        // Add risk score color coding
        const riskScore = parseFloat(report.risk_score);
        if (riskScore >= 80) {
          riskScoreSpan.style.color = "var(--error)";
        } else if (riskScore >= 50) {
          riskScoreSpan.style.color = "var(--warning)";
        } else {
          riskScoreSpan.style.color = "var(--success)";
        }
        
        // Executive summary (if provided)
        const execSummaryEl = document.getElementById("executive-summary");
        if (execSummaryEl && report.executive_summary) {
          execSummaryEl.textContent = report.executive_summary;
          execSummaryEl.style.display = "block";
        }
        
        // Severity counts
        const severityCountsEl = document.getElementById("severity-counts");
        if (severityCountsEl) {
          severityCountsEl.innerHTML = `
            <span class="severity-badge critical">Critical: ${report.critical_count || 0}</span>
            <span class="severity-badge high">High: ${report.high_count || 0}</span>
            <span class="severity-badge medium">Medium: ${report.medium_count || 0}</span>
            <span class="severity-badge low">Low: ${report.low_count || 0}</span>
          `;
          severityCountsEl.style.display = "block";
        }

        // ON-CHAIN ANALYSIS SECTION (Pro/Enterprise)
        const onchainSection = document.getElementById("onchain-analysis");
        if (onchainSection && data.onchain_analysis) {
          const onchain = data.onchain_analysis;
          log('ONCHAIN', 'Displaying on-chain analysis:', onchain);

          // Proxy Detection
          const proxyContent = document.getElementById("onchain-proxy-content");
          if (proxyContent && onchain.proxy) {
            const proxy = onchain.proxy;
            const isProxy = proxy.is_proxy;
            const statusClass = isProxy ? (proxy.upgrade_risk === 'HIGH' || proxy.upgrade_risk === 'CRITICAL' ? 'danger' : 'warning') : 'safe';

            proxyContent.innerHTML = `
              <div class="onchain-status">
                <span class="onchain-status-indicator ${statusClass}"></span>
                <span class="onchain-status-text">${isProxy ? (proxy.proxy_type || 'Unknown') + ' Proxy' : 'Not a Proxy'}</span>
              </div>
              ${isProxy ? `
                <div class="onchain-detail">
                  <div class="onchain-detail-row">
                    <span class="onchain-detail-label">Upgrade Risk:</span>
                    <span class="onchain-detail-value">${proxy.upgrade_risk || 'N/A'}</span>
                  </div>
                  ${proxy.implementation ? `
                    <div class="onchain-detail-row">
                      <span class="onchain-detail-label">Implementation:</span>
                    </div>
                    <div class="onchain-address">${proxy.implementation}</div>
                  ` : ''}
                </div>
              ` : '<div class="onchain-detail">Contract code is immutable</div>'}
            `;
          }

          // Storage/Ownership
          const storageContent = document.getElementById("onchain-storage-content");
          if (storageContent && onchain.storage) {
            const storage = onchain.storage;
            const centralRisk = storage.centralization_risk || 'LOW';
            const statusClass = centralRisk === 'HIGH' || centralRisk === 'CRITICAL' ? 'danger' : (centralRisk === 'MEDIUM' ? 'warning' : 'safe');

            storageContent.innerHTML = `
              <div class="onchain-status">
                <span class="onchain-status-indicator ${statusClass}"></span>
                <span class="onchain-status-text">Centralization: ${centralRisk}</span>
              </div>
              <div class="onchain-detail">
                ${storage.owner ? `
                  <div class="onchain-detail-row">
                    <span class="onchain-detail-label">Owner:</span>
                  </div>
                  <div class="onchain-address">${storage.owner}</div>
                ` : '<div class="onchain-detail-row"><span class="onchain-detail-label">No owner detected</span></div>'}
                ${storage.is_pausable !== undefined ? `
                  <div class="onchain-detail-row">
                    <span class="onchain-detail-label">Pausable:</span>
                    <span class="onchain-detail-value">${storage.is_pausable ? (storage.is_paused ? '⏸️ PAUSED' : '✅ Active') : 'No'}</span>
                  </div>
                ` : ''}
                ${storage.eth_balance ? `
                  <div class="onchain-detail-row">
                    <span class="onchain-detail-label">ETH Balance:</span>
                    <span class="onchain-detail-value">${parseFloat(storage.eth_balance).toFixed(4)} ETH</span>
                  </div>
                ` : ''}
              </div>
            `;
          }

          // Backdoor Detection
          const backdoorsContent = document.getElementById("onchain-backdoors-content");
          if (backdoorsContent && onchain.backdoors) {
            const backdoors = onchain.backdoors;
            const hasBackdoors = backdoors.has_backdoors;
            const riskLevel = backdoors.risk_level || 'LOW';
            const statusClass = riskLevel === 'CRITICAL' || riskLevel === 'HIGH' ? 'danger' : (riskLevel === 'MEDIUM' ? 'warning' : 'safe');

            backdoorsContent.innerHTML = `
              <div class="onchain-status">
                <span class="onchain-status-indicator ${statusClass}"></span>
                <span class="onchain-status-text">${hasBackdoors ? riskLevel + ' Risk Detected' : 'No Backdoors Found'}</span>
              </div>
              ${backdoors.dangerous_functions && backdoors.dangerous_functions.length > 0 ? `
                <ul class="onchain-danger-list">
                  ${backdoors.dangerous_functions.slice(0, 3).map(fn => `
                    <li class="onchain-danger-item">
                      <code>${fn.name || fn.selector}</code>
                      <span>(${fn.category || 'unknown'})</span>
                    </li>
                  `).join('')}
                  ${backdoors.dangerous_functions.length > 3 ? `
                    <li class="onchain-danger-item" style="color: var(--text-tertiary);">
                      +${backdoors.dangerous_functions.length - 3} more...
                    </li>
                  ` : ''}
                </ul>
              ` : '<div class="onchain-detail">No dangerous functions detected</div>'}
            `;
          }

          // Honeypot Detection
          const honeypotContent = document.getElementById("onchain-honeypot-content");
          if (honeypotContent && onchain.honeypot) {
            const honeypot = onchain.honeypot;
            const isHoneypot = honeypot.is_honeypot;
            const confidence = honeypot.confidence || 'LOW';
            const statusClass = isHoneypot ? (confidence === 'HIGH' ? 'danger' : 'warning') : 'safe';

            honeypotContent.innerHTML = `
              <div class="onchain-status">
                <span class="onchain-status-indicator ${statusClass}"></span>
                <span class="onchain-status-text">${isHoneypot ? '⚠️ Honeypot (' + confidence + ')' : '✅ Not a Honeypot'}</span>
              </div>
              <div class="onchain-detail">
                ${honeypot.recommendation ? '<p style="margin: 0; font-size: 0.75rem;">' + honeypot.recommendation + '</p>' : ''}
                ${honeypot.indicators && honeypot.indicators.length > 0 ? `
                  <ul class="onchain-danger-list">
                    ${honeypot.indicators.slice(0, 2).map(ind => `
                      <li class="onchain-danger-item">${ind.description || ind.type}</li>
                    `).join('')}
                  </ul>
                ` : ''}
              </div>
            `;
          }

          // Overall Risk
          const overallRiskEl = document.getElementById("onchain-overall-risk");
          if (overallRiskEl && onchain.overall_risk) {
            const overall = onchain.overall_risk;
            const level = (overall.level || 'LOW').toLowerCase();

            overallRiskEl.innerHTML = `
              <div class="onchain-risk-left">
                <span class="onchain-risk-badge ${level}">${overall.level || 'N/A'}</span>
                <span class="onchain-risk-score">${overall.score || 0}/100</span>
              </div>
              <div class="onchain-risk-summary">
                ${overall.summary || 'On-chain analysis complete.'}
              </div>
            `;
          }

          // Show the section
          onchainSection.style.display = "block";
          log('ONCHAIN', 'On-chain section displayed');
        } else if (onchainSection) {
          onchainSection.style.display = "none";
        }

        // FREE TIER UPGRADE PROMPT
        if (report.upgrade_prompt) {
          const upgradePromptEl = document.getElementById("upgrade-prompt");
          if (upgradePromptEl) {
            upgradePromptEl.innerHTML = `
              <div class="upgrade-banner">
                <div class="upgrade-icon">🔒</div>
                <div class="upgrade-content">
                  <h3>More Issues Detected!</h3>
                  <p>${report.upgrade_prompt}</p>
                  <a href="#tier-select" class="btn btn-primary" onclick="document.getElementById('tier-select').scrollIntoView({behavior: 'smooth'})">
                    Upgrade to See All Issues & Get Fix Recommendations
                  </a>
                </div>
              </div>
            `;
            upgradePromptEl.style.display = "block";
          }
        }
        
        // ISSUES TABLE - Enhanced for Pro+ tiers
        issuesBody.innerHTML = report.issues.length === 0 
          ? '<tr><td colspan="8">No issues found.</td></tr>'
          : report.issues.map((issue, index) => {
          // Add null safety checks for all fields
          const severity = (issue.severity || "unknown").toLowerCase();
          const severityDisplay = issue.severity || "Unknown";
          const type = issue.type || "Unknown Issue";
          const description = issue.description || "No description available";
          const fix = issue.fix || "No fix recommendation available";
          const isProven = issue.proven === true;
          const source = issue.source || "";

          const hasProFeatures = issue.line_number || issue.function_name || issue.vulnerable_code;

          return `
            <tr class="issue-row ${hasProFeatures ? 'expandable' : ''} ${isProven ? 'proven-issue' : ''}" data-issue-id="${index}">
              <td class="severity-cell">
                <span class="severity-badge ${severity}">${severityDisplay}</span>
                ${isProven ? '<span class="proven-badge" title="Mathematically proven by Certora formal verification">PROVEN</span>' : ''}
              </td>
              <td><strong>${type}</strong>${source ? `<br><small class="issue-source">${escapeHtml(source)}</small>` : ''}</td>
              <td>${description}</td>
              <td class="fix-cell">${fix}</td>
                  ${hasProFeatures ? `
                    <td class="expand-cell">
                      ${window.currentAuditTier === 'enterprise' ? `
                        <button class="expand-btn" onclick="showIssueModal(${index})" style="background: var(--gradient-accent-reverse);">
                          🔬 Full Analysis
                        </button>
                      ` : `
                        <button class="expand-btn" onclick="toggleIssueDetails(${index})">
                          <span class="expand-icon">▼</span> Details
                        </button>
                      `}
                    </td>
                  ` : '<td></td>'}
                </tr>
                ${hasProFeatures ? `
                  <tr class="issue-details" id="issue-details-${index}" style="display: none;">
                    <td colspan="5">
                      <div class="issue-details-content">
                        
                        ${issue.line_number ? `
                          <div class="detail-section">
                            <strong>📍 Location:</strong> 
                            Line ${issue.line_number}${issue.function_name ? ` in <code>${issue.function_name}()</code>` : ''}
                          </div>
                        ` : ''}
                        
                        ${issue.vulnerable_code ? `
                          <div class="detail-section">
                            <strong>🔍 Vulnerable Code:</strong>
                            <pre class="code-snippet"><code>${escapeHtml(issue.vulnerable_code)}</code></pre>
                          </div>
                        ` : ''}
                        
                        ${issue.exploit_scenario ? `
                          <div class="detail-section exploit-scenario">
                            <strong>⚠️ Exploit Scenario:</strong>
                            <p>${issue.exploit_scenario}</p>
                          </div>
                        ` : ''}
                        
                        ${issue.estimated_impact ? `
                          <div class="detail-section impact-estimate">
                            <strong>💰 Estimated Impact:</strong> ${issue.estimated_impact}
                          </div>
                        ` : ''}
                        
                        ${issue.code_fix ? `
                          <div class="detail-section code-fix">
                            <strong>✅ Recommended Fix:</strong>
                            <div class="code-diff">
                              <div class="diff-before">
                                <strong>Before:</strong>
                                <pre><code>${escapeHtml(issue.code_fix.before)}</code></pre>
                              </div>
                              <div class="diff-after">
                                <strong>After:</strong>
                                <pre><code>${escapeHtml(issue.code_fix.after)}</code></pre>
                              </div>
                            </div>
                            ${issue.code_fix.explanation ? `<p class="fix-explanation">${issue.code_fix.explanation}</p>` : ''}
                          </div>
                        ` : ''}
                        
                        ${issue.alternatives && issue.alternatives.length > 0 ? `
                          <div class="detail-section alternatives">
                            <strong>🔄 Alternative Fixes:</strong>
                            ${issue.alternatives.map((alt, altIndex) => `
                              <div class="alternative-item">
                                <strong>Option ${altIndex + 1}: ${alt.approach}</strong>
                                <p><strong>Pros:</strong> ${alt.pros}</p>
                                <p><strong>Cons:</strong> ${alt.cons}</p>
                                <p><strong>Gas Impact:</strong> ${alt.gas_impact}</p>
                              </div>
                            `).join('')}
                          </div>
                        ` : ''}
                        
                        ${issue.proof_of_concept ? `
                          <div class="detail-section poc">
                            <strong>🎯 Proof of Concept (Enterprise):</strong>
                            <pre class="poc-code"><code>${escapeHtml(issue.proof_of_concept)}</code></pre>
                          </div>
                        ` : ''}
                        
                        ${issue.references && issue.references.length > 0 ? `
                          <div class="detail-section references">
                            <strong>📚 References:</strong>
                            <ul>
                              ${issue.references.map(ref => `
                                <li><a href="${ref.url}" target="_blank" rel="noopener">${ref.title}</a></li>
                              `).join('')}
                            </ul>
                          </div>
                        ` : ''}
                        
                      </div>
                    </td>
                  </tr>
                ` : ''}
              `;
            }).join('');
            
        // Helper function to escape HTML (attach to window for inline onclick handlers)
        window.escapeHtml = (text) => {
          const div = document.createElement('div');
          div.textContent = text;
          return div.innerHTML;
        };
        
        // Helper function to toggle issue details (attach to window for inline onclick handlers)
        window.toggleIssueDetails = (index) => {
          const detailsRow = document.getElementById(`issue-details-${index}`);
          const expandBtn = document.querySelector(`[onclick="toggleIssueDetails(${index})"]`);
          const expandIcon = expandBtn?.querySelector('.expand-icon');
          
          if (detailsRow.style.display === 'none') {
            detailsRow.style.display = 'table-row';
            if (expandIcon) expandIcon.textContent = '▲';
            if (expandBtn) expandBtn.innerHTML = '<span class="expand-icon">▲</span> Hide';
          } else {
            detailsRow.style.display = 'none';
            if (expandIcon) expandIcon.textContent = '▼';
            if (expandBtn) expandBtn.innerHTML = '<span class="expand-icon">▼</span> Details';
          }
        };
        // Show issue details in premium modal (Enterprise only)
window.showIssueModal = (index) => {
  const issue = window.currentAuditIssues[index];
  if (!issue) {
    console.error('[ERROR] Issue not found:', index);
    return;
  }
  
  // Create modal if it doesn't exist
  let modal = document.getElementById('issue-details-modal');
  let backdrop = document.getElementById('issue-details-modal-backdrop');
  
  if (!modal) {
    // Create modal HTML
    backdrop = document.createElement('div');
    backdrop.id = 'issue-details-modal-backdrop';
    backdrop.className = 'modal-backdrop';
    backdrop.style.display = 'none';
    document.body.appendChild(backdrop);
    
    modal = document.createElement('div');
    modal.id = 'issue-details-modal';
    modal.innerHTML = `
      <div class="modal-header">
        <h2 id="modal-issue-title">Issue Details</h2>
        <div style="display: flex; gap: var(--space-3); align-items: center;">
          <button class="btn btn-secondary btn-sm" id="copy-all-modal-content" style="padding: var(--space-2) var(--space-4);">
            📋 Copy All
          </button>
          <button class="modal-close" id="issue-modal-close">&times;</button>
        </div>
      </div>
      <div class="modal-body" id="modal-issue-body">
        <!-- Content populated dynamically -->
      </div>
    `;
    document.body.appendChild(modal);
// Copy All button handler
document.getElementById('copy-all-modal-content').addEventListener('click', () => {
  const modalBody = document.getElementById('modal-issue-body');
  
  // Get all text content from the modal, formatted nicely
  let copyText = `${issue.type} (${issue.severity})\n`;
  copyText += '='.repeat(50) + '\n\n';
  
  if (issue.description) {
    copyText += `DESCRIPTION:\n${issue.description}\n\n`;
  }
  
  if (issue.line_number || issue.function_name) {
    copyText += `LOCATION:\n`;
    copyText += `Line: ${issue.line_number || 'N/A'}\n`;
    if (issue.function_name) copyText += `Function: ${issue.function_name}\n`;
    copyText += '\n';
  }
  
  if (issue.vulnerable_code) {
    copyText += `VULNERABLE CODE:\n${issue.vulnerable_code}\n\n`;
  }
  
  if (issue.exploit_scenario) {
    copyText += `EXPLOIT SCENARIO:\n${issue.exploit_scenario}\n\n`;
  }
  
  if (issue.estimated_impact) {
    copyText += `ESTIMATED IMPACT:\n${issue.estimated_impact}\n\n`;
  }
  
  if (issue.code_fix) {
    copyText += `RECOMMENDED FIX:\n`;
    copyText += `Before:\n${issue.code_fix.before}\n\n`;
    copyText += `After:\n${issue.code_fix.after}\n\n`;
    if (issue.code_fix.explanation) {
      copyText += `Explanation: ${issue.code_fix.explanation}\n\n`;
    }
  }
  
  if (issue.alternatives && issue.alternatives.length > 0) {
    copyText += `ALTERNATIVE FIXES:\n`;
    issue.alternatives.forEach((alt, idx) => {
      copyText += `\nOption ${idx + 1}: ${alt.approach}\n`;
      copyText += `Pros: ${alt.pros}\n`;
      copyText += `Cons: ${alt.cons}\n`;
      copyText += `Gas Impact: ${alt.gas_impact}\n`;
    });
    copyText += '\n';
  }
  
  if (issue.proof_of_concept) {
    copyText += `PROOF OF CONCEPT:\n${issue.proof_of_concept}\n\n`;
  }
  
  if (issue.references && issue.references.length > 0) {
    copyText += `REFERENCES:\n`;
    issue.references.forEach(ref => {
      copyText += `- ${ref.title}: ${ref.url}\n`;
    });
  }
  
  // Copy to clipboard
  navigator.clipboard.writeText(copyText).then(() => {
    const copyBtn = document.getElementById('copy-all-modal-content');
    const originalText = copyBtn.innerHTML;
    copyBtn.innerHTML = '✅ Copied!';
    copyBtn.style.background = 'var(--success)';
    
    setTimeout(() => {
      copyBtn.innerHTML = originalText;
      copyBtn.style.background = '';
    }, 2000);
  }).catch(err => {
    console.error('[ERROR] Failed to copy:', err);
    alert('Failed to copy content');
  });
});
    // Close button handler
    document.getElementById('issue-modal-close').addEventListener('click', () => {
      modal.style.display = 'none';
      backdrop.style.display = 'none';
      document.body.style.overflow = '';
    });
    
    // Backdrop click to close
    backdrop.addEventListener('click', () => {
      modal.style.display = 'none';
      backdrop.style.display = 'none';
      document.body.style.overflow = '';
    });
    
    // Escape key to close
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && modal.style.display === 'block') {
        modal.style.display = 'none';
        backdrop.style.display = 'none';
        document.body.style.overflow = '';
      }
    });
  }
  
  // Populate modal with issue data
  const modalBody = document.getElementById('modal-issue-body');
  const modalTitle = document.getElementById('modal-issue-title');
  
  modalTitle.textContent = `${issue.type} (${issue.severity})`;
  
  let modalContent = `
    <section>
      <h3>📝 Description</h3>
      <p>${issue.description}</p>
    </section>
  `;
  
  if (issue.line_number || issue.function_name) {
    modalContent += `
      <section>
        <h3>📍 Location</h3>
        <p><strong>Line:</strong> ${issue.line_number || 'N/A'}</p>
        ${issue.function_name ? `<p><strong>Function:</strong> <code>${issue.function_name}</code></p>` : ''}
      </section>
    `;
  }
  
  if (issue.vulnerable_code) {
    modalContent += `
      <section>
        <h3>🔍 Vulnerable Code</h3>
        <pre><code>${escapeHtml(issue.vulnerable_code)}</code></pre>
      </section>
    `;
  }
  
  if (issue.exploit_scenario) {
    modalContent += `
      <section>
        <h3>⚠️ Exploit Scenario</h3>
        <p>${issue.exploit_scenario}</p>
      </section>
    `;
  }
  
  if (issue.estimated_impact) {
    modalContent += `
      <section>
        <h3>💰 Estimated Impact</h3>
        <p>${issue.estimated_impact}</p>
      </section>
    `;
  }
  
  if (issue.code_fix) {
    modalContent += `
      <section>
        <h3>✅ Recommended Fix</h3>
        <div class="code-comparison">
          <div>
            <h4>Before:</h4>
            <pre><code>${escapeHtml(issue.code_fix.before)}</code></pre>
          </div>
          <div>
            <h4>After:</h4>
            <pre><code>${escapeHtml(issue.code_fix.after)}</code></pre>
          </div>
        </div>
        ${issue.code_fix.explanation ? `<p style="margin-top: var(--space-4); color: var(--text-secondary);">${issue.code_fix.explanation}</p>` : ''}
      </section>
    `;
  }
  
  if (issue.alternatives && issue.alternatives.length > 0) {
    modalContent += `
      <section>
        <h3>🔄 Alternative Fixes</h3>
        ${issue.alternatives.map((alt, altIndex) => `
          <div class="alternative-fix">
            <h4>Option ${altIndex + 1}: ${alt.approach}</h4>
            <p><strong>Pros:</strong> ${alt.pros}</p>
            <p><strong>Cons:</strong> ${alt.cons}</p>
            <p><strong>Gas Impact:</strong> <code>${alt.gas_impact}</code></p>
          </div>
        `).join('')}
      </section>
    `;
  }
  
  if (issue.proof_of_concept) {
    modalContent += `
      <section class="poc">
        <h3>🎯 Proof of Concept</h3>
        <pre><code>${escapeHtml(issue.proof_of_concept)}</code></pre>
      </section>
    `;
  }
  
  if (issue.references && issue.references.length > 0) {
    modalContent += `
      <section>
        <h3>📚 References</h3>
        <ul>
          ${issue.references.map(ref => `
            <li><a href="${ref.url}" target="_blank" rel="noopener">${ref.title}</a></li>
          `).join('')}
        </ul>
      </section>
    `;
  }
  
  modalBody.innerHTML = modalContent;
  
  // Show modal
  modal.style.display = 'block';
  backdrop.style.display = 'block';
  document.body.style.overflow = 'hidden';
  
  debugLog(`[DEBUG] Modal opened for issue ${index}`);
};
        predictionsList.innerHTML = report.predictions.length === 0 
          ? "<li>No predictions available.</li>"
          : report.predictions.map(p => `<li tabindex="0">Scenario: ${p.scenario} | Impact: ${p.impact}</li>`).join('');
          
        // RECOMMENDATIONS - Categorized by urgency
        if (report.recommendations && typeof report.recommendations === 'object') {
          // New format with immediate/short_term/long_term
          let recHtml = '';
          
          if (report.recommendations.immediate && report.recommendations.immediate.length > 0) {
            recHtml += '<h4 class="rec-category critical">🚨 Immediate (Fix Before Deploy)</h4><ul>';
            recHtml += report.recommendations.immediate.map(r => `<li tabindex="0">${r}</li>`).join('');
            recHtml += '</ul>';
          }
          
          if (report.recommendations.short_term && report.recommendations.short_term.length > 0) {
            recHtml += '<h4 class="rec-category warning">⚠️ Short-Term (Next 7 Days)</h4><ul>';
            recHtml += report.recommendations.short_term.map(r => `<li tabindex="0">${r}</li>`).join('');
            recHtml += '</ul>';
          }
          
          if (report.recommendations.long_term && report.recommendations.long_term.length > 0) {
            recHtml += '<h4 class="rec-category info">💡 Long-Term (Future Improvements)</h4><ul>';
            recHtml += report.recommendations.long_term.map(r => `<li tabindex="0">${r}</li>`).join('');
            recHtml += '</ul>';
          }
          
          recommendationsList.innerHTML = recHtml || '<li>No recommendations available.</li>';
        } else if (Array.isArray(report.recommendations)) {
          // Legacy format - array of strings
          recommendationsList.innerHTML = report.recommendations.length === 0 
            ? "<li>No recommendations available.</li>"
            : report.recommendations.map(r => `<li tabindex="0">${r}</li>`).join('');
        } else {
          recommendationsList.innerHTML = '<li>No recommendations available.</li>';
        }
        
        // Fuzzing results renderer
        const renderFuzzingResults = (fuzzingResults) => {
          if (!fuzzingResults || fuzzingResults.length === 0) {
            return `<div class="fuzzing-empty"><p>🧪 No fuzzing results available.</p></div>`;
          }
          
          const firstResult = fuzzingResults[0];
          const parsed = firstResult.parsed;
          
          if (!parsed) {
            return fuzzingResults.map(r => 
              `<div class="fuzzing-legacy-item"><strong>${escapeHtml(r.vulnerability)}</strong><p>${escapeHtml(r.description)}</p></div>`
            ).join('');
          }
          
          const statusIcon = {'success': '✅', 'complete': '✅', 'issues_found': '⚠️', 'error': '❌', 'timeout': '⏱️'}[parsed.status] || '🧪';
          const statusClass = {'success': 'success', 'complete': 'success', 'issues_found': 'warning', 'error': 'error'}[parsed.status] || '';
          
          const formatGas = (gas) => {
            if (!gas) return 'N/A';
            if (gas >= 1000000) return `${(gas / 1000000).toFixed(1)}M`;
            if (gas >= 1000) return `${(gas / 1000).toFixed(0)}K`;
            return gas.toString();
          };
          
          let html = `
            <div class="fuzzing-results-card">
              <div class="fuzzing-header ${statusClass}">
                <div class="fuzzing-status">
                  <span class="status-icon">${statusIcon}</span>
                  <span class="status-text">${parsed.execution_summary || 'Fuzzing Complete'}</span>
                </div>
                ${parsed.contract_name ? `<span class="contract-badge">${escapeHtml(parsed.contract_name)}</span>` : ''}
              </div>
              
              <div class="fuzzing-stats-grid">
                <div class="fuzzing-stat">
                  <div class="stat-value">${parsed.tests_passed || 0}/${parsed.tests_total || 0}</div>
                  <div class="stat-label">Tests Passed</div>
                </div>
                <div class="fuzzing-stat">
                  <div class="stat-value">${(parsed.fuzzing_iterations || 0).toLocaleString()}</div>
                  <div class="stat-label">Iterations</div>
                </div>
                <div class="fuzzing-stat">
                  <div class="stat-value">${parsed.coverage?.instructions || 0}</div>
                  <div class="stat-label">Instructions</div>
                </div>
                <div class="fuzzing-stat">
                  <div class="stat-value">${formatGas(parsed.gas_per_second)}</div>
                  <div class="stat-label">Gas/sec</div>
                </div>
              </div>`;
          
          if (parsed.function_tests && parsed.function_tests.length > 0) {
            html += `
              <div class="fuzzing-tests-section">
                <h5>Function Tests</h5>
                <div class="function-tests-list">
                  ${parsed.function_tests.map(test => `
                    <div class="function-test-item ${test.passed ? 'passed' : 'failed'}">
                      <span class="test-icon">${test.icon}</span>
                      <code class="test-name">${escapeHtml(test.function)}</code>
                      <span class="test-status">${test.status}</span>
                    </div>
                  `).join('')}
                </div>
              </div>`;
          }
          
          if (parsed.coverage && (parsed.coverage.corpus_size > 0 || parsed.coverage.codehashes > 0)) {
            html += `
              <div class="fuzzing-coverage-section">
                <h5>Coverage Details</h5>
                <div class="coverage-grid">
                  ${parsed.coverage.corpus_size > 0 ? `<div class="coverage-item"><span class="coverage-label">Corpus:</span><span class="coverage-value">${parsed.coverage.corpus_size}</span></div>` : ''}
                  ${parsed.coverage.codehashes > 0 ? `<div class="coverage-item"><span class="coverage-label">Codehashes:</span><span class="coverage-value">${parsed.coverage.codehashes}</span></div>` : ''}
                </div>
              </div>`;
          }
          
          if (parsed.compile_time || parsed.slither_time) {
            html += `<div class="fuzzing-timing">
              ${parsed.compile_time ? `<span>⚡ Compile: ${parsed.compile_time.toFixed(1)}s</span>` : ''}
              ${parsed.slither_time ? `<span>🔍 Slither: ${parsed.slither_time.toFixed(1)}s</span>` : ''}
            </div>`;
          }
          
          html += `</div>`;
          return html;
        };
        
        fuzzingList.innerHTML = renderFuzzingResults(report.fuzzing_results);

        // Certora formal verification results renderer
        const renderCertoraResults = (certoraResults) => {
          if (!certoraResults || certoraResults.length === 0) {
            return `<div class="certora-empty"><p>🔒 No formal verification results available.</p></div>`;
          }

          const verified = certoraResults.filter(r => r.status === 'verified').length;
          const violated = certoraResults.filter(r => r.status === 'violated' || r.status === 'issues_found').length;
          const skipped = certoraResults.filter(r => r.status === 'skipped').length;
          const pending = certoraResults.filter(r => r.status === 'pending').length;
          const errors = certoraResults.filter(r => r.status === 'error' || r.status === 'incomplete' || r.status === 'timeout').length;

          // Determine overall status and messaging
          let overallStatus, statusIcon, statusText;
          if (pending > 0 && verified === 0 && violated === 0) {
            // Verification in progress on Certora cloud
            overallStatus = 'pending';
            statusIcon = '🔄';
            statusText = 'Verification In Progress';
          } else if (violated > 0) {
            overallStatus = 'warning';
            statusIcon = '⚠️';
            statusText = `Found ${violated} Issue${violated > 1 ? 's' : ''} Requiring Review`;
          } else if (errors > 0) {
            overallStatus = 'info';
            statusIcon = '🔄';
            statusText = 'Verification Incomplete';
          } else if (verified > 0) {
            overallStatus = 'success';
            statusIcon = '✅';
            statusText = `${verified} Propert${verified > 1 ? 'ies' : 'y'} Verified`;
          } else {
            overallStatus = 'info';
            statusIcon = '🔒';
            statusText = 'Formal Verification Complete';
          }

          // Categorize verification results for better display
          const ruleCategories = {
            stateIntegrity: certoraResults.filter(r => r.rule && (
              r.rule.toLowerCase().includes('state') ||
              r.rule.toLowerCase().includes('revert') ||
              r.rule.toLowerCase().includes('view') ||
              r.rule.toLowerCase().includes('readonly')
            )).length,
            transferSafety: certoraResults.filter(r => r.rule && (
              r.rule.toLowerCase().includes('transfer') ||
              r.rule.toLowerCase().includes('balance') ||
              r.rule.toLowerCase().includes('supply')
            )).length,
            sanityChecks: certoraResults.filter(r => r.rule && (
              r.rule.toLowerCase().includes('sanity') ||
              r.rule.toLowerCase().includes('vacuous') ||
              r.rule.toLowerCase().includes('envfree')
            )).length
          };

          // Calculate what percentage of contract was formally verified
          const totalRules = certoraResults.length;
          const verificationCoverage = totalRules > 0 ? Math.round((verified / totalRules) * 100) : 0;

          let html = `
            <div class="certora-results-card">
              <div class="certora-header ${overallStatus}">
                <div class="certora-status">
                  <span class="status-icon">${statusIcon}</span>
                  <span class="status-text">${statusText}</span>
                </div>
                <div class="certora-powered-by" style="font-size: 0.75em; opacity: 0.7;">Powered by Certora Prover</div>
              </div>

              <div class="certora-stats-grid">
                <div class="certora-stat verified">
                  <div class="stat-value">${verified}</div>
                  <div class="stat-label">Verified</div>
                </div>
                <div class="certora-stat violated">
                  <div class="stat-value">${violated}</div>
                  <div class="stat-label">Issues Found</div>
                </div>
                ${pending > 0 ? `
                <div class="certora-stat pending">
                  <div class="stat-value">${pending}</div>
                  <div class="stat-label">In Progress</div>
                </div>
                ` : `
                <div class="certora-stat skipped">
                  <div class="stat-value">${skipped + errors}</div>
                  <div class="stat-label">Skipped/Errors</div>
                </div>
                `}
              </div>

              <div class="certora-rules-list">
          `;

          certoraResults.forEach(result => {
            // Map status to icon
            let ruleIcon, ruleClass;
            switch(result.status) {
              case 'verified':
                ruleIcon = '✅';
                ruleClass = 'verified';
                break;
              case 'violated':
              case 'issues_found':
                ruleIcon = '❌';
                ruleClass = 'violated';
                break;
              case 'timeout':
                ruleIcon = '⏱️';
                ruleClass = 'timeout';
                break;
              case 'error':
              case 'incomplete':
                ruleIcon = '⚠️';
                ruleClass = 'error';
                break;
              case 'skipped':
                ruleIcon = '⏭️';
                ruleClass = 'skipped';
                break;
              case 'pending':
                ruleIcon = '🔄';
                ruleClass = 'pending';
                break;
              default:
                ruleIcon = '🔍';
                ruleClass = 'info';
            }

            // Get description, falling back to reason
            const description = result.description || result.reason || 'No additional details available';

            // For violations, show severity and actionable fix
            const isViolation = result.status === 'violated' || result.status === 'issues_found';
            const severity = result.severity || 'HIGH';
            const category = result.category || 'Formal Verification';
            const fix = result.fix || '';
            const isProven = result.proven === true;

            html += `
              <div class="certora-rule ${ruleClass}">
                <span class="rule-icon">${ruleIcon}</span>
                <div class="rule-info">
                  <div class="rule-header">
                    <span class="rule-name">${escapeHtml(result.rule || 'Verification Check')}</span>
                    ${isViolation ? `
                      <span class="severity-badge ${severity.toLowerCase()}">${severity}</span>
                      ${isProven ? '<span class="proven-badge" title="Mathematically proven by formal verification">PROVEN</span>' : ''}
                    ` : ''}
                  </div>
                  ${isViolation && category ? `<span class="rule-category">${escapeHtml(category)}</span>` : ''}
                  <span class="rule-description">${escapeHtml(description)}</span>
                  ${isViolation && fix ? `
                    <div class="rule-fix">
                      <strong>🔧 Recommended Fix:</strong> ${escapeHtml(fix)}
                    </div>
                  ` : ''}
                </div>
              </div>
            `;
          });

          html += `</div></div>`;
          return html;
        };

        // Render Certora results (Enterprise only)
        const certoraList = document.getElementById("certora-list");
        if (certoraList && report.certora_results) {
          certoraList.innerHTML = renderCertoraResults(report.certora_results);
        }

        // CODE QUALITY METRICS (if available)
        const codeQualityEl = document.getElementById("code-quality-metrics");
        if (codeQualityEl && report.code_quality_metrics) {
          const metrics = report.code_quality_metrics;
          codeQualityEl.innerHTML = `
            <div class="metrics-grid">
              <div class="metric-card">
                <div class="metric-value">${metrics.lines_of_code || 'N/A'}</div>
                <div class="metric-label">Lines of Code</div>
              </div>
              <div class="metric-card">
                <div class="metric-value">${metrics.functions_count || 'N/A'}</div>
                <div class="metric-label">Functions</div>
              </div>
              <div class="metric-card">
                <div class="metric-value">${metrics.complexity_score ? metrics.complexity_score.toFixed(1) : 'N/A'}</div>
                <div class="metric-label">Complexity Score</div>
              </div>
            </div>
          `;
          codeQualityEl.style.display = "block";
        }
        
        // Format remediation roadmap as numbered list
        if (remediationRoadmap && report.remediation_roadmap) {
          const roadmapText = report.remediation_roadmap;
          // Parse numbered items (1. ... 2. ... 3. ...)
          const steps = roadmapText.split(/\d+\.\s+/).filter(step => step.trim());
          
          if (steps.length > 0) {
            remediationRoadmap.innerHTML = steps
              .map((step, index) => `
                <li tabindex="0">
                  <span class="step-number">${index + 1}</span>
                  <span class="step-content">${step.trim()}</span>
                </li>
              `)
              .join('');
          } else {
            // Fallback if parsing fails
            remediationRoadmap.innerHTML = `<li tabindex="0">${roadmapText}</li>`;
          }
        }
        
        // Store PDF path if available (support both old and new field names)
        if (data.pdf_report_url) {
          window.currentAuditPdfUrl = data.pdf_report_url;
          window.currentAuditPdfPath = data.pdf_report_url;  // Backwards compat
          debugLog(`[DEBUG] PDF available: ${data.pdf_report_url}`);
        } else if (data.compliance_pdf) {
          window.currentAuditPdfPath = data.compliance_pdf;
          debugLog(`[DEBUG] PDF generated (legacy): ${data.compliance_pdf}`);
        }

        // Update mobile view based on tier
        const currentTier = data.tier || window.currentAuditTier || 'free';
        const hasPdfAccess = ['starter', 'beginner', 'pro', 'enterprise', 'diamond'].includes(currentTier);

        const mobilePaidView = document.getElementById('mobile-paid-view');
        const mobileFreeView = document.getElementById('mobile-free-view');

        // Severity counts for both views
        const criticalCount = report.critical_count || 0;
        const highCount = report.high_count || 0;
        const mediumCount = report.medium_count || 0;
        const lowCount = report.low_count || 0;

        const severityHtml = `
          ${criticalCount > 0 ? `<span class="mobile-severity-badge critical">⚠️ ${criticalCount} Critical</span>` : ''}
          ${highCount > 0 ? `<span class="mobile-severity-badge high">🔴 ${highCount} High</span>` : ''}
          ${mediumCount > 0 ? `<span class="mobile-severity-badge medium">🟡 ${mediumCount} Medium</span>` : ''}
          ${lowCount > 0 ? `<span class="mobile-severity-badge low">🟢 ${lowCount} Low</span>` : ''}
          ${(criticalCount + highCount + mediumCount + lowCount) === 0 ? '<span class="mobile-severity-badge low">✅ No issues found</span>' : ''}
        `;

        if (hasPdfAccess) {
          // PAID tier: Show PDF download view with tool results
          if (mobilePaidView) mobilePaidView.style.display = 'block';
          if (mobileFreeView) mobileFreeView.style.display = 'none';

          const paidSeveritySummary = document.getElementById('mobile-severity-summary-paid');
          if (paidSeveritySummary) {
            // Build comprehensive tool results summary for mobile
            let toolsHtml = severityHtml;

            // Add tools that ran
            const toolsRan = [];
            if (report.slither_time || (report.issues && report.issues.length > 0)) toolsRan.push('🔍 Slither');
            if (report.mythril_results && report.mythril_results.length > 0) toolsRan.push('🧠 Mythril');
            if (report.fuzzing_results && report.fuzzing_results.length > 0) toolsRan.push('🧪 Echidna');
            if (report.certora_results && report.certora_results.length > 0) toolsRan.push('🔒 Certora');
            if (report.onchain_analysis) toolsRan.push('⛓️ On-Chain');

            if (toolsRan.length > 0) {
              toolsHtml += `<div class="mobile-tools-summary" style="margin-top: 12px; padding: 8px; background: rgba(var(--accent-teal-rgb), 0.1); border-radius: 8px; font-size: 0.85em;">
                <strong>Tools Used:</strong> ${toolsRan.join(' • ')}
              </div>`;
            }

            // Add Certora summary if available
            if (report.certora_results && report.certora_results.length > 0) {
              const certoraVerified = report.certora_results.filter(r => r.status === 'verified').length;
              const certoraViolated = report.certora_results.filter(r => r.status === 'violated' || r.status === 'issues_found').length;
              const certoraPending = report.certora_results.filter(r => r.status === 'pending').length;
              const certoraStatus = certoraPending > 0 && certoraVerified === 0 ? '🔄 In Progress' : certoraViolated > 0 ? '⚠️ Issues Found' : certoraVerified > 0 ? '✅ Verified' : '🔍 Complete';

              toolsHtml += `<div class="mobile-certora-summary" style="margin-top: 8px; padding: 8px; background: rgba(var(--accent-purple-rgb), 0.1); border-radius: 8px; font-size: 0.85em;">
                <strong>🔒 Formal Verification:</strong> ${certoraStatus}
                ${certoraVerified > 0 ? `<br><span style="color: var(--green);">✓ ${certoraVerified} properties proven</span>` : ''}
                ${certoraViolated > 0 ? `<br><span style="color: var(--red);">✗ ${certoraViolated} issues to review</span>` : ''}
                ${certoraPending > 0 ? `<br><span style="color: var(--accent-purple);">⏳ Analysis running on Certora cloud</span>` : ''}
              </div>`;
            }

            // Add Mythril summary if available
            if (report.mythril_results && report.mythril_results.length > 0) {
              const mythrilCount = report.mythril_results.length;
              const mythrilHasIssues = report.mythril_results.some(r =>
                r.vulnerability && !r.vulnerability.toLowerCase().includes('no issues')
              );
              const mythrilStatus = mythrilHasIssues ? '⚠️ Issues Found' : '✅ No Issues';

              toolsHtml += `<div class="mobile-mythril-summary" style="margin-top: 8px; padding: 8px; background: rgba(var(--accent-blue-rgb), 0.1); border-radius: 8px; font-size: 0.85em;">
                <strong>🧠 Mythril:</strong> ${mythrilStatus}
                ${mythrilHasIssues ? `<br>${mythrilCount} potential issue${mythrilCount > 1 ? 's' : ''} found` : ''}
              </div>`;
            }

            // Add fuzzing summary if available
            if (report.fuzzing_results && report.fuzzing_results.length > 0) {
              const fuzzResult = report.fuzzing_results[0];
              const fuzzParsed = fuzzResult.parsed || {};
              const fuzzStatus = fuzzParsed.tests_passed === fuzzParsed.tests_total ? '✅ All Passed' : '⚠️ Issues Found';

              toolsHtml += `<div class="mobile-fuzz-summary" style="margin-top: 8px; padding: 8px; background: rgba(var(--accent-orange-rgb), 0.1); border-radius: 8px; font-size: 0.85em;">
                <strong>🧪 Fuzzing:</strong> ${fuzzStatus}
                ${fuzzParsed.tests_total ? `<br>${fuzzParsed.tests_passed || 0}/${fuzzParsed.tests_total} tests passed` : ''}
              </div>`;
            }

            paidSeveritySummary.innerHTML = toolsHtml;
          }

        } else {
          // FREE tier: Show issues list + upgrade CTA
          if (mobilePaidView) mobilePaidView.style.display = 'none';
          if (mobileFreeView) mobileFreeView.style.display = 'block';

          const freeSeveritySummary = document.getElementById('mobile-severity-summary-free');
          if (freeSeveritySummary) freeSeveritySummary.innerHTML = severityHtml;

          // Populate issues list for free users
          const mobileIssuesList = document.getElementById('mobile-issues-list');
          if (mobileIssuesList && report.issues) {
            const issues = report.issues.slice(0, 5); // Show max 5 issues on mobile
            const severityIcon = { critical: '⚠️', high: '🔴', medium: '🟡', low: '🟢' };

            let issuesHtml = issues.map((issue, index) => {
              const severity = (issue.severity || 'medium').toLowerCase();
              const desc = issue.description || '';
              const needsExpand = desc.length > 80; // Show expand hint if description is long
              return `
                <div class="mobile-issue-item ${severity}" data-issue-index="${index}" onclick="this.classList.toggle('expanded')">
                  <span class="mobile-issue-severity">${severityIcon[severity] || '🔵'}</span>
                  <div class="mobile-issue-content">
                    <div class="mobile-issue-type">${issue.type || issue.title || 'Issue'}</div>
                    <div class="mobile-issue-desc">${desc}</div>
                    ${needsExpand ? '<div class="mobile-issue-expand-hint">tap to expand</div>' : ''}
                  </div>
                </div>
              `;
            }).join('');

            if (report.issues.length > 5) {
              issuesHtml += `<div class="mobile-issues-more">+ ${report.issues.length - 5} more issues (view on desktop)</div>`;
            }

            if (report.issues.length === 0) {
              issuesHtml = '<div class="mobile-issues-more">No issues found - your contract looks secure!</div>';
            }

            mobileIssuesList.innerHTML = issuesHtml;
          }
        }

        debugLog(`[DEBUG] Mobile view: tier=${currentTier}, hasPdfAccess=${hasPdfAccess}`);

        if (overageCost) {
          usageWarning.textContent = `Enterprise audit completed with $${overageCost.toFixed(2)} overage charged.`;
          usageWarning.classList.add("success");
        }
        loading.classList.remove("show");
        if (resultsDiv) {
          resultsDiv.classList.add("show");
          resultsDiv.focus();
          resultsDiv.scrollIntoView({ behavior: "smooth" });
        }
        logMessage(`Audit complete – risk score ${report.risk_score}`);
      };

      const handleSubmit = (e) => {
        e.preventDefault();
        withCsrfToken(async (token) => {
          loading.classList.add("show");
          if (resultsDiv) resultsDiv.classList.remove("show");
          usageWarning.textContent = "";
          usageWarning.classList.remove("error", "success");

          const file = auditForm.querySelector("#file")?.files[0];
          if (!file) {
            loading.classList.remove("show");
            usageWarning.textContent = "Please select a file";
            usageWarning.classList.add("error");
            return;
          }

          const username = (await fetchUsername())?.username || "guest";
          const formData = new FormData(auditForm);

          try {
            logMessage("Submitting audit...");
            
            // Submit to /audit - it will either run immediately or queue
            const response = await fetch(`/audit?username=${encodeURIComponent(username)}`, {
              method: 'POST',
              headers: { 'X-CSRFToken': token },
              body: formData,
              credentials: 'include'
            });
            
            if (!response.ok) {
              const errorData = await response.json().catch(() => ({}));
              throw new Error(errorData.detail || 'Audit request failed');
            }
            
            const data = await response.json();
            
            // Check if audit was queued or ran immediately
            if (data.queued) {
              // QUEUED: Server at capacity, use queue tracker for real-time updates
              logMessage(`Server busy - queued at position ${data.position}`);
              
              queueTracker.jobId = data.job_id;
              
              queueTracker.onUpdate = (status) => {
                logMessage(`Queue status: ${status.status} - ${status.current_phase || 'waiting'}`);
              };
              
              queueTracker.onComplete = async (result) => {
                logMessage("Audit complete!");
                queueTracker.hideQueueUI();
                loading.classList.remove("show");
                if (result.tier) window.currentAuditTier = result.tier;
                handleAuditResponse(result);
                
                // Update sidebar with fresh count from response
                if (result.audit_count !== undefined && result.audit_limit !== undefined) {
                  const remaining = result.audit_limit === 9999 ? 'Unlimited' : (result.audit_limit - result.audit_count);
                  if (sidebarTierUsage) {
                    sidebarTierUsage.textContent = result.audit_limit === 9999 
                      ? "Unlimited audits" 
                      : `${remaining} audits remaining (${result.audit_count}/${result.audit_limit} used)`;
                  }
                  log('USAGE', `Updated from queue: ${result.audit_count}/${result.audit_limit} audits used`);
                }
                
                await fetchTierData();
              };
              
              queueTracker.onError = (error) => {
                logMessage(`Audit failed: ${error}`);
                queueTracker.hideQueueUI();
                loading.classList.remove("show");
                usageWarning.textContent = error || "Audit failed";
                usageWarning.classList.add("error");
              };
              
              queueTracker.showQueuePosition(data);
              queueTracker.connectWebSocket();
              queueTracker.startPolling();
              
            } else if (data.session_url) {
              // UPGRADE REDIRECT: File too large or usage limit
              logMessage("Redirecting to Stripe for upgrade");
              window.location.href = data.session_url;
              
            } else {
              // IMMEDIATE: Audit ran and completed
              logMessage("Audit complete!");
              loading.classList.remove("show");
              if (data.tier) window.currentAuditTier = data.tier;
              handleAuditResponse(data);
              
              // Update sidebar with fresh count from response (faster than full fetch)
              if (data.audit_count !== undefined && data.audit_limit !== undefined) {
                const remaining = data.audit_limit === 9999 ? 'Unlimited' : (data.audit_limit - data.audit_count);
                if (sidebarTierUsage) {
                  sidebarTierUsage.textContent = data.audit_limit === 9999 
                    ? "Unlimited audits" 
                    : `${remaining} audits remaining (${data.audit_count}/${data.audit_limit} used)`;
                }
                log('USAGE', `Updated: ${data.audit_count}/${data.audit_limit} audits used`);
              }
              
              // Also do full tier refresh to sync everything
              await fetchTierData();
            }
            
          } catch (err) {
            console.error(err);
            queueTracker.hideQueueUI();
            loading.classList.remove("show");
            usageWarning.textContent = err.message || "Audit error";
            usageWarning.classList.add("error");
          }
        });
      };

      if (auditForm) {
        auditForm.addEventListener("submit", handleSubmit);
        debugLog("[DEBUG] Audit submit handler attached successfully");
      } else {
        console.error("[ERROR] auditForm not found – submit handler NOT attached");
      }

      // Section12: Report Download
      downloadReportButton?.addEventListener("click", () => {
        const reportData = {
          risk_score: riskScoreSpan.textContent,
          issues: Array.from(issuesBody.querySelectorAll("tr"))
            .filter(row => row.cells && row.cells.length >= 4)
            .map((row) => ({
              type: row.cells[0].textContent,
              severity: row.cells[1].textContent,
              description: row.cells[2].textContent,
              fix: row.cells[3].textContent,
            })),
          predictions: Array.from(predictionsList.querySelectorAll("li")).map(
            (li) => ({
              scenario: li.textContent
                .split(" | ")[0]
                .replace("Scenario: ", ""),
              impact: li.textContent.split(" | ")[1].replace("Impact: ", ""),
            })
          ),
          recommendations: Array.from(
            recommendationsList.querySelectorAll("li")
          ).map((li) => li.textContent),
          fuzzing_results: Array.from(fuzzingList.querySelectorAll("li")).map(
            (li) => ({
              vulnerability: li.textContent
                .split(" | ")[0]
                .replace("Vulnerability: ", ""),
              description: li.textContent
                .split(" | ")[1]
                .replace("Description: ", ""),
            })
          ),
          remediation_roadmap: remediationRoadmap?.textContent || null,
        };
        const blob = new Blob([JSON.stringify(reportData, null, 2)], {
          type: "application/json",
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `DeFiGuard_Audit_Report_${new Date().toISOString()}.json`;
        a.click();
        URL.revokeObjectURL(url);
        debugLog("[DEBUG] Report downloaded");
      });

      // PDF Download Button
      pdfDownloadButton?.addEventListener("click", async () => {
        // Support both new URL format and legacy path format
        const pdfUrl = window.currentAuditPdfUrl || window.currentAuditPdfPath;
        const originalText = pdfDownloadButton.textContent;

        debugLog(`[DEBUG] PDF Download clicked. URL: ${pdfUrl}`);

        if (!pdfUrl) {
          usageWarning.textContent = "No PDF available. Please run an audit first.";
          usageWarning.classList.add("error");
          alert("No PDF available. Please run an audit first, then download.");
          return;
        }

        try {
          // Show loading state
          pdfDownloadButton.textContent = "⏳ Downloading...";
          pdfDownloadButton.disabled = true;

          // Determine the correct fetch URL
          let fetchUrl;
          if (pdfUrl.startsWith('/api/reports/')) {
            // New format: already a URL path
            fetchUrl = pdfUrl;
          } else {
            // Legacy format: extract filename and use new endpoint
            const filename = pdfUrl.split('/').pop();
            fetchUrl = `/api/reports/${filename}`;
          }

          debugLog(`[DEBUG] Fetching PDF from: ${fetchUrl}`);
          const response = await fetch(fetchUrl, {
            credentials: "include"
          });

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP ${response.status}: PDF not found`);
          }

          const blob = await response.blob();

          if (blob.size === 0) {
            throw new Error("PDF file is empty");
          }

          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          // Tier-aware fallback filename (server provides actual name)
          const userTier = document.getElementById('sidebar-tier-name')?.textContent?.toLowerCase() || 'free';
          const fallbackName = userTier === 'enterprise' || userTier === 'diamond'
            ? `Security_Audit_Report_${Date.now()}.pdf`  // White-label: no DeFiGuard branding
            : `DeFiGuard_Report_${Date.now()}.pdf`;
          a.download = pdfUrl.split('/').pop() || fallbackName;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);

          debugLog("[DEBUG] PDF downloaded successfully");
          pdfDownloadButton.textContent = "✅ Downloaded!";
          setTimeout(() => {
            pdfDownloadButton.textContent = originalText;
            pdfDownloadButton.disabled = false;
          }, 2000);
        } catch (error) {
          console.error("[ERROR] PDF download failed:", error);
          usageWarning.textContent = `PDF download failed: ${error.message}`;
          usageWarning.classList.add("error");
          alert(`PDF download failed: ${error.message}`);
          pdfDownloadButton.textContent = originalText;
          pdfDownloadButton.disabled = false;
        }
      });

      // Mobile PDF Download Button (same logic as desktop)
      const mobilePdfDownload = document.getElementById('mobile-pdf-download');
      mobilePdfDownload?.addEventListener("click", async () => {
        const pdfUrl = window.currentAuditPdfUrl || window.currentAuditPdfPath;

        debugLog(`[DEBUG] Mobile PDF Download clicked. URL: ${pdfUrl}`);

        if (!pdfUrl) {
          alert("No PDF available yet. Please run an audit first.");
          return;
        }

        try {
          let fetchUrl;
          if (pdfUrl.startsWith('/api/reports/')) {
            fetchUrl = pdfUrl;
          } else {
            const filename = pdfUrl.split('/').pop();
            fetchUrl = `/api/reports/${filename}`;
          }

          debugLog(`[DEBUG] Mobile: Fetching PDF from: ${fetchUrl}`);

          // Show loading state
          mobilePdfDownload.textContent = "⏳ Downloading...";
          mobilePdfDownload.disabled = true;

          const response = await fetch(fetchUrl, {
            credentials: "include"
          });

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP ${response.status}: PDF not found`);
          }

          const blob = await response.blob();

          if (blob.size === 0) {
            throw new Error("PDF file is empty");
          }

          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          // Tier-aware fallback filename (server provides actual name)
          const userTier = document.getElementById('sidebar-tier-name')?.textContent?.toLowerCase() || 'free';
          const fallbackName = userTier === 'enterprise' || userTier === 'diamond'
            ? `Security_Audit_Report_${Date.now()}.pdf`  // White-label: no DeFiGuard branding
            : `DeFiGuard_Report_${Date.now()}.pdf`;
          a.download = pdfUrl.split('/').pop() || fallbackName;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);

          // Show success briefly
          mobilePdfDownload.textContent = "✅ Downloaded!";
          debugLog("[DEBUG] Mobile PDF downloaded successfully");
          setTimeout(() => {
            mobilePdfDownload.textContent = "📄 Download Full PDF Report";
            mobilePdfDownload.disabled = false;
          }, 2000);
        } catch (error) {
          console.error("[ERROR] Mobile PDF download failed:", error);
          mobilePdfDownload.textContent = "📄 Download Full PDF Report";
          mobilePdfDownload.disabled = false;
          alert(`PDF download failed: ${error.message}`);
        }
      });

      // NFT Minting Button
      mintNftButton?.addEventListener("click", async () => {
        withCsrfToken(async (token) => {
          if (!token) {
            usageWarning.textContent = "Unable to establish secure connection.";
            usageWarning.classList.add("error");
            return;
          }
          
          try {
            const user = await fetchUsername();
            const username = user?.username;
            
            if (!username) {
              usageWarning.textContent = "Please log in to mint NFT rewards.";
              usageWarning.classList.add("error");
              return;
            }
            
            usageWarning.textContent = "Minting NFT reward...";
            usageWarning.classList.remove("error");
            
            const response = await fetch(`/mint-nft?username=${encodeURIComponent(username)}`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                "X-CSRF-Token": token,
              },
              credentials: "include",
            });
            
            if (!response.ok) {
              const errorData = await response.json().catch(() => ({}));
              throw new Error(errorData.detail || "NFT minting failed");
            }
            
            const data = await response.json();
            usageWarning.textContent = `🎉 ${data.message} - NFT ID: ${data.nft_id}`;
            usageWarning.classList.add("success");
            debugLog(`[DEBUG] NFT minted: ${data.nft_id}`);
          } catch (error) {
            console.error("[ERROR] NFT minting failed:", error);
            usageWarning.textContent = `NFT minting failed: ${error.message}`;
            usageWarning.classList.add("error");
          }
        });
      });

      // ============================================================================
      // SETTINGS MODAL - Account management
      // ============================================================================
      
      const {
        sidebarSettingsLink, settingsModal, settingsModalBackdrop, settingsModalClose,
        modalUsername, modalEmail, modalTier, modalMemberSince, modalAuditsUsed,
        modalAuditsRemaining, modalSizeLimit, modalUsageProgress, modalUsageText,
        modalApiSection, apiKeyCountDisplay, apiKeysTableBody, createApiKeyButton,
        createKeyModal, createKeyModalBackdrop, createKeyModalClose,
        newKeyLabelInput, createKeyConfirm, createKeyCancel,
        modalUpgradeButton, modalLogoutButton,
        modalCertoraSection, certoraJobsTableBody, certoraUploadBtn, certoraUploadInput
      } = els;

      // Function to populate modal with user data
      const populateSettingsModal = async () => {
        try {
          // Fetch user info
          const user = await fetchUsername();
          const username = user?.username || "Not signed in";
          
          // Fetch tier data
          const tierUrl = user?.username 
            ? `/tier?username=${encodeURIComponent(user.username)}`
            : "/tier";
          const tierResponse = await fetch(tierUrl, {
            headers: { Accept: "application/json" },
            credentials: "include"
          });
          
          if (!tierResponse.ok) throw new Error("Failed to fetch tier data");
          
          const tierData = await tierResponse.json();
          const { tier, size_limit, api_key, audit_count, audit_limit } = tierData;
          
          // Populate account info
          if (modalUsername) modalUsername.textContent = username;
          if (modalEmail) modalEmail.textContent = user?.sub?.split('|')[1] || "N/A";
          
          // Set tier badge with styling
          if (modalTier) {
            const tierNameCap = tier.charAt(0).toUpperCase() + tier.slice(1);
            modalTier.textContent = tierNameCap;
            modalTier.className = `badge badge-${tier}`;
          }
          
          // Member since (you can add actual date from backend)
          if (modalMemberSince) {
            modalMemberSince.textContent = "December 2024"; // TODO: Get from backend
          }
          
          // Usage statistics
          if (modalAuditsUsed) modalAuditsUsed.textContent = audit_count;
          
          if (modalAuditsRemaining) {
            if (audit_limit === 9999) {
              modalAuditsRemaining.textContent = "Unlimited";
            } else {
              modalAuditsRemaining.textContent = audit_limit - audit_count;
            }
          }
          
          if (modalSizeLimit) modalSizeLimit.textContent = size_limit;
          
          // Usage progress bar
          if (modalUsageProgress && modalUsageText) {
            if (audit_limit === 9999) {
              modalUsageProgress.style.width = "100%";
              modalUsageText.textContent = "Unlimited usage";
            } else {
              const percentage = (audit_count / audit_limit) * 100;
              modalUsageProgress.style.width = `${percentage}%`;
              modalUsageText.textContent = `${percentage.toFixed(0)}% used`;
            }
          }
          
          // API Key section (Pro/Enterprise only)
          if (modalApiSection) {
            if (tier === "pro" || tier === "enterprise") {
              modalApiSection.style.display = "block";
              await loadApiKeys(); // Load all keys from API
            } else {
              modalApiSection.style.display = "none";
            }
          }

          // Certora Jobs section (Enterprise/Diamond only)
          if (modalCertoraSection) {
            const hasCertoraAccess = tier === "enterprise" || tierData.has_diamond;
            if (hasCertoraAccess) {
              modalCertoraSection.style.display = "block";
              await loadCertoraJobs(); // Load Certora job history
            } else {
              modalCertoraSection.style.display = "none";
            }
          }

          debugLog("[DEBUG] Settings modal populated successfully");
        } catch (error) {
          console.error("[ERROR] Failed to populate settings modal:", error);
        }
      };

      // Load and display all API keys
      const loadApiKeys = async () => {
        try {
          const response = await fetch("/api/keys", {
            credentials: "include"
          });
          
          if (!response.ok) throw new Error("Failed to load API keys");
          
          const data = await response.json();
          const { keys, active_count, max_keys, tier } = data;
          
          // Update count display
          if (apiKeyCountDisplay) {
            if (max_keys === null) {
              apiKeyCountDisplay.textContent = `Active Keys: ${active_count} (Unlimited - Enterprise)`;
            } else {
              apiKeyCountDisplay.textContent = `Active Keys: ${active_count}/${max_keys} (Pro Tier)`;
            }
          }
          
          // Update table
          if (apiKeysTableBody) {
            if (keys.length === 0) {
              apiKeysTableBody.innerHTML = `
                <tr>
                  <td colspan="5" style="padding: var(--space-6); text-align: center; color: var(--text-tertiary);">
                    No API keys yet. Create your first key to get started!
                  </td>
                </tr>
              `;
            } else {
              apiKeysTableBody.innerHTML = keys.map(key => `
                <tr style="border-bottom: 1px solid var(--glass-border);">
                  <td style="padding: var(--space-3); font-weight: 600; color: var(--text-primary);">
                    ${key.label}
                  </td>
                  <td style="padding: var(--space-3);">
                    <code style="font-family: 'JetBrains Mono', monospace; font-size: var(--text-sm); color: var(--accent-teal);">
                      ${key.key.substring(0, 8)}...${key.key.substring(key.key.length - 4)}
                    </code>
                  </td>
                  <td style="padding: var(--space-3); font-size: var(--text-sm); color: var(--text-tertiary);">
                    ${new Date(key.created_at).toLocaleDateString()}
                  </td>
                  <td style="padding: var(--space-3); font-size: var(--text-sm); color: var(--text-tertiary);">
                    ${key.last_used_at ? new Date(key.last_used_at).toLocaleDateString() : 'Never'}
                  </td>
                  <td style="padding: var(--space-3); text-align: right;">
                    <button class="copy-key-btn btn btn-sm" data-key="${key.key}" style="margin-right: var(--space-2);">
                      📋 Copy
                    </button>
                    <button class="revoke-key-btn btn btn-secondary btn-sm" data-key-id="${key.id}" data-key-label="${key.label}">
                      🗑️ Revoke
                    </button>
                  </td>
                </tr>
              `).join('');
              
              // Copy button handlers
              document.querySelectorAll('.copy-key-btn').forEach(btn => {
                btn.addEventListener('click', async (e) => {
                  const key = e.target.dataset.key;
                  try {
                    await navigator.clipboard.writeText(key);
                    e.target.textContent = "✅ Copied!";
                    setTimeout(() => { e.target.textContent = "📋 Copy"; }, 2000);
                  } catch (err) {
                    console.error("[ERROR] Failed to copy:", err);
                    alert("Failed to copy API key");
                  }
                });
              });
              
              // Revoke button handlers
              document.querySelectorAll('.revoke-key-btn').forEach(btn => {
                btn.addEventListener('click', async (e) => {
                  const keyId = e.target.dataset.keyId;
                  const keyLabel = e.target.dataset.keyLabel;
                  
                  if (!confirm(`Revoke "${keyLabel}"? This cannot be undone and the key will stop working immediately.`)) {
                    return;
                  }
                  
                  try {
                    await withCsrfToken(async (csrfToken) => {
                      const response = await fetch(`/api/keys/${keyId}`, {
                        method: "DELETE",
                        headers: { "X-CSRFToken": csrfToken },
                        credentials: "include"
                      });
                      
                      if (!response.ok) throw new Error("Failed to revoke key");
                      
                      alert(`✅ "${keyLabel}" revoked successfully`);
                      await loadApiKeys(); // Reload
                    });
                  } catch (err) {
                    console.error("[ERROR] Failed to revoke:", err);
                    alert("Failed to revoke API key");
                  }
                });
              });
            }
          }
          
          debugLog("[DEBUG] API keys loaded successfully");
        } catch (error) {
          console.error("[ERROR] Failed to load API keys:", error);
          if (apiKeysTableBody) {
            apiKeysTableBody.innerHTML = `
              <tr>
                <td colspan="5" style="padding: var(--space-6); text-align: center; color: var(--error);">
                  Failed to load API keys. Please try again.
                </td>
              </tr>
            `;
          }
        }
      };

      // ============================================================================
      // CERTORA JOBS MANAGEMENT (Enterprise/Diamond only)
      // ============================================================================

      // Load and display Certora verification jobs
      const loadCertoraJobs = async () => {
        try {
          const response = await fetch("/api/certora/jobs", {
            credentials: "include"
          });

          if (!response.ok) throw new Error("Failed to load Certora jobs");

          const data = await response.json();
          const { jobs } = data;

          // Update table
          if (certoraJobsTableBody) {
            if (!jobs || jobs.length === 0) {
              certoraJobsTableBody.innerHTML = `
                <tr>
                  <td colspan="5" style="padding: var(--space-6); text-align: center; color: var(--text-tertiary);">
                    No verification jobs yet. Upload a contract to start!
                  </td>
                </tr>
              `;
            } else {
              certoraJobsTableBody.innerHTML = jobs.map(job => {
                // Status badge styling
                const statusStyles = {
                  'completed': 'background: rgba(39, 174, 96, 0.2); color: var(--green);',
                  'running': 'background: rgba(155, 89, 182, 0.2); color: var(--accent-purple);',
                  'pending': 'background: rgba(241, 196, 15, 0.2); color: #f1c40f;',
                  'error': 'background: rgba(231, 76, 60, 0.2); color: var(--red);'
                };
                const statusStyle = statusStyles[job.status] || statusStyles['pending'];

                // Results display
                let resultsHtml = '—';
                if (job.status === 'completed') {
                  resultsHtml = `
                    <span style="color: var(--green);">✓ ${job.rules_verified}</span> /
                    <span style="color: var(--red);">✗ ${job.rules_violated}</span>
                  `;
                } else if (job.status === 'running' || job.status === 'pending') {
                  resultsHtml = '<span style="color: var(--accent-purple);">⏳ Running...</span>';
                }

                // Truncate job ID for display
                const shortJobId = job.job_id.length > 12
                  ? job.job_id.substring(0, 8) + '...'
                  : job.job_id;

                return `
                  <tr style="border-bottom: 1px solid var(--glass-border);">
                    <td style="padding: var(--space-3); font-weight: 600; color: var(--text-primary); max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                      ${job.contract_name}
                    </td>
                    <td style="padding: var(--space-3);">
                      <span style="display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: var(--text-xs); text-transform: uppercase; ${statusStyle}">
                        ${job.status}
                      </span>
                    </td>
                    <td style="padding: var(--space-3); text-align: center; font-size: var(--text-sm);">
                      ${resultsHtml}
                    </td>
                    <td style="padding: var(--space-3);">
                      <a href="${job.job_url}" target="_blank" rel="noopener noreferrer"
                         style="font-family: 'JetBrains Mono', monospace; font-size: var(--text-sm); color: var(--accent-teal); text-decoration: none;"
                         title="View on Certora Dashboard: ${job.job_id}">
                        ${shortJobId} ↗
                      </a>
                    </td>
                    <td style="padding: var(--space-3); text-align: right;">
                      <button class="copy-job-id-btn btn btn-sm" data-job-id="${job.job_id}" style="margin-right: var(--space-2);">
                        📋
                      </button>
                      ${job.status === 'running' || job.status === 'pending'
                        ? `<button class="poll-job-btn btn btn-sm" data-job-id="${job.job_id}" style="margin-right: var(--space-2);">🔄</button>`
                        : ''}
                      <button class="delete-job-btn btn btn-secondary btn-sm" data-job-id="${job.job_id}" data-contract="${job.contract_name}">
                        🗑️
                      </button>
                    </td>
                  </tr>
                `;
              }).join('');

              // Copy Job ID button handlers
              document.querySelectorAll('.copy-job-id-btn').forEach(btn => {
                btn.addEventListener('click', async (e) => {
                  const jobId = e.target.closest('button').dataset.jobId;
                  try {
                    await navigator.clipboard.writeText(jobId);
                    e.target.closest('button').textContent = "✅";
                    setTimeout(() => { e.target.closest('button').textContent = "📋"; }, 2000);
                  } catch (err) {
                    console.error("[ERROR] Failed to copy:", err);
                  }
                });
              });

              // Poll button handlers (for pending/running jobs)
              document.querySelectorAll('.poll-job-btn').forEach(btn => {
                btn.addEventListener('click', async (e) => {
                  const jobId = e.target.closest('button').dataset.jobId;
                  const button = e.target.closest('button');
                  button.disabled = true;
                  button.textContent = "⏳";

                  try {
                    await withCsrfToken(async (csrfToken) => {
                      const response = await fetch(`/api/certora/poll/${jobId}`, {
                        method: "POST",
                        headers: { "X-CSRFToken": csrfToken },
                        credentials: "include"
                      });

                      if (!response.ok) throw new Error("Failed to poll job");

                      const data = await response.json();
                      if (data.status === 'completed') {
                        alert(`✅ Verification complete!\\n\\nVerified: ${data.rules_verified}\\nViolations: ${data.rules_violated}`);
                      } else if (data.status === 'error') {
                        alert(`❌ Verification failed: ${data.message}`);
                      } else {
                        alert(`⏳ ${data.message}`);
                      }

                      await loadCertoraJobs(); // Reload to update status
                    });
                  } catch (err) {
                    console.error("[ERROR] Failed to poll:", err);
                    button.disabled = false;
                    button.textContent = "🔄";
                  }
                });
              });

              // Delete button handlers
              document.querySelectorAll('.delete-job-btn').forEach(btn => {
                btn.addEventListener('click', async (e) => {
                  const jobId = e.target.closest('button').dataset.jobId;
                  const contractName = e.target.closest('button').dataset.contract;

                  if (!confirm(`Delete cached verification for "${contractName}"?\\n\\nThis will force a fresh verification on your next audit.`)) {
                    return;
                  }

                  try {
                    await withCsrfToken(async (csrfToken) => {
                      const response = await fetch(`/api/certora/job/${jobId}`, {
                        method: "DELETE",
                        headers: { "X-CSRFToken": csrfToken },
                        credentials: "include"
                      });

                      if (!response.ok) throw new Error("Failed to delete job");

                      await loadCertoraJobs(); // Reload
                    });
                  } catch (err) {
                    console.error("[ERROR] Failed to delete:", err);
                    alert("Failed to delete Certora job");
                  }
                });
              });
            }
          }

          debugLog("[DEBUG] Certora jobs loaded successfully");
        } catch (error) {
          console.error("[ERROR] Failed to load Certora jobs:", error);
          if (certoraJobsTableBody) {
            certoraJobsTableBody.innerHTML = `
              <tr>
                <td colspan="5" style="padding: var(--space-6); text-align: center; color: var(--error);">
                  Failed to load Certora jobs. Please try again.
                </td>
              </tr>
            `;
          }
        }
      };

      // Certora Upload Button - trigger file input
      certoraUploadBtn?.addEventListener("click", () => {
        certoraUploadInput?.click();
      });

      // Handle Certora file upload
      certoraUploadInput?.addEventListener("change", async (e) => {
        const file = e.target.files?.[0];
        if (!file) return;

        // Reset input so same file can be selected again
        e.target.value = '';

        if (!file.name.endsWith('.sol')) {
          alert("Please upload a Solidity (.sol) file");
          return;
        }

        try {
          certoraUploadBtn.disabled = true;
          certoraUploadBtn.textContent = "⏳ Starting verification...";

          const formData = new FormData();
          formData.append('file', file);

          await withCsrfToken(async (csrfToken) => {
            const response = await fetch("/api/certora/start", {
              method: "POST",
              headers: { "X-CSRFToken": csrfToken },
              credentials: "include",
              body: formData
            });

            if (!response.ok) {
              const error = await response.json();
              throw new Error(error.detail || "Failed to start verification");
            }

            const data = await response.json();

            if (data.status === 'started') {
              alert(`🔒 Verification Started!\\n\\nContract: ${file.name}\\nJob ID: ${data.job_id}\\n\\nVerification typically takes 2-5 minutes. Results will be cached and used in your next audit.\\n\\nYou can track progress at:\\n${data.job_url}`);
            } else if (data.status === 'already_running') {
              alert(`⏳ Verification Already Running\\n\\nJob ID: ${data.job_id}\\n\\n${data.message}`);
            } else if (data.status === 'cached') {
              alert(`✅ Verification Already Complete!\\n\\nVerified: ${data.rules_verified}\\nViolations: ${data.rules_violated}\\n\\nCompleted: ${data.completed_at}\\n\\n${data.message}`);
            }

            // Reload jobs list
            await loadCertoraJobs();

            certoraUploadBtn.disabled = false;
            certoraUploadBtn.textContent = "🔒 Verify Contract Now";
          });

        } catch (error) {
          console.error("[ERROR] Failed to start Certora verification:", error);
          alert(`❌ Failed to start verification: ${error.message}`);
          certoraUploadBtn.disabled = false;
          certoraUploadBtn.textContent = "🔒 Verify Contract Now";
        }
      });

      // Open modal helper function
      const openSettingsModal = async () => {
        debugLog("[DEBUG] Opening settings modal");
        await populateSettingsModal();
        settingsModal?.classList.add("active");
        settingsModalBackdrop?.classList.add("active");
        document.body.style.overflow = "hidden"; // Prevent background scroll
      };

      // Open modal from sidebar settings link
      sidebarSettingsLink?.addEventListener("click", (e) => {
        e.preventDefault();
        openSettingsModal();
      });

      // Close modal
      const closeSettingsModal = () => {
        debugLog("[DEBUG] Closing settings modal");
        settingsModal?.classList.remove("active");
        settingsModalBackdrop?.classList.remove("active");
        document.body.style.overflow = ""; // Restore scroll
      };

      settingsModalClose?.addEventListener("click", closeSettingsModal);
      settingsModalBackdrop?.addEventListener("click", closeSettingsModal);

      // Keyboard shortcuts for settings modal
      document.addEventListener("keydown", (e) => {
        // Cmd+, or Ctrl+, opens settings modal
        if ((e.metaKey || e.ctrlKey) && e.key === ",") {
          e.preventDefault();
          if (!settingsModal?.classList.contains("active")) {
            openSettingsModal();
          }
        }
        
        // Escape key closes settings modal
        if (e.key === "Escape" && settingsModal?.classList.contains("active")) {
          closeSettingsModal();
        }
      });

      // Create API Key Button
      createApiKeyButton?.addEventListener("click", () => {
        if (createKeyModal && createKeyModalBackdrop) {
          createKeyModal.style.display = "block";
          createKeyModalBackdrop.style.display = "block";
          document.body.style.overflow = "hidden";
          if (newKeyLabelInput) {
            newKeyLabelInput.value = "";
            newKeyLabelInput.focus();
          }
        }
      });

      // Close Create Key Modal
      const closeCreateKeyModal = () => {
        if (createKeyModal && createKeyModalBackdrop) {
          createKeyModal.style.display = "none";
          createKeyModalBackdrop.style.display = "none";
          document.body.style.overflow = "auto";
        }
      };

      createKeyModalClose?.addEventListener("click", closeCreateKeyModal);
      createKeyCancel?.addEventListener("click", closeCreateKeyModal);
      createKeyModalBackdrop?.addEventListener("click", closeCreateKeyModal);

      // Confirm Create Key
      createKeyConfirm?.addEventListener("click", async () => {
        const label = newKeyLabelInput?.value.trim();
        
        if (!label) {
          alert("Please enter a label for your API key");
          return;
        }
        
        try {
          createKeyConfirm.disabled = true;
          createKeyConfirm.textContent = "Creating...";
          
          await withCsrfToken(async (csrfToken) => {
            const response = await fetch("/api/keys/create", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": csrfToken
              },
              credentials: "include",
              body: JSON.stringify({ label })
            });
            
            if (!response.ok) {
              const error = await response.json();
              throw new Error(error.detail || "Failed to create API key");
            }
            
            const data = await response.json();
            
            closeCreateKeyModal();
            
            // Show the full key ONCE
            alert(`✅ API Key Created Successfully!

Label: ${data.label}
Key: ${data.api_key}

⚠️ IMPORTANT: This is the only time you'll see the full key. Copy it now!`);
            
            // Reload keys table
            await loadApiKeys();
            
            createKeyConfirm.disabled = false;
            createKeyConfirm.textContent = "Create Key";
          });
          
        } catch (error) {
          console.error("[ERROR] Failed to create API key:", error);
          alert(`❌ Failed to create API key: ${error.message}`);
          createKeyConfirm.disabled = false;
          createKeyConfirm.textContent = "Create Key";
        }
      });

      // Upgrade button in modal
      modalUpgradeButton?.addEventListener("click", () => {
        closeSettingsModal();
        // Scroll to pricing section
        document.querySelector("#tier-select")?.scrollIntoView({ behavior: "smooth" });
      });

      // Logout button in modal
      if (modalLogoutButton) {
        modalLogoutButton.addEventListener("click", (e) => {
          e.preventDefault();
          debugLog(`[DEBUG] Logout initiated from settings modal, time=${new Date().toISOString()}`);
          closeSettingsModal();
          window.location.href = "/logout";
        });
      }

      // ============================================================================
      // THEME SYSTEM - Color theme management
      // ============================================================================
      
      const themeSelect = document.getElementById("theme-select");
      
      // Load saved theme on page load
      const loadTheme = () => {
        const savedTheme = localStorage.getItem("userTheme") || "default";
        document.documentElement.setAttribute("data-theme", savedTheme);
        if (themeSelect) themeSelect.value = savedTheme;
        debugLog(`[DEBUG] Theme loaded: ${savedTheme}`);
      };
      
      // Apply theme
      const applyTheme = (theme) => {
        document.documentElement.setAttribute("data-theme", theme);
        localStorage.setItem("userTheme", theme);
        debugLog(`[DEBUG] Theme applied: ${theme}`);
      };
      
      // Theme change handler
      themeSelect?.addEventListener("change", (e) => {
        const selectedTheme = e.target.value;
        applyTheme(selectedTheme);
      });
      
      // Initialize theme on page load
      loadTheme();

      // Section13: Header Scroll Behavior
      let lastScrollY = 0;
      let ticking = false;

      const updateHeader = () => {
        const header = document.querySelector("header");
        if (!header) return;

        const currentScrollY = window.scrollY;

        if (currentScrollY > 30) {
          if (currentScrollY > lastScrollY) {
            // Scrolling down — hide
            header.classList.add("scrolled");
          } else {
            // Scrolling up — show
            header.classList.remove("scrolled");
          }
        } else {
          // Near top — always show
          header.classList.remove("scrolled");
        }

        lastScrollY = currentScrollY;
        ticking = false;
      };

      window.addEventListener("scroll", () => {
        if (!ticking) {
          requestAnimationFrame(updateHeader);
          ticking = true;
        }
      }, { passive: true });

      // ============================================================================
      // LEGAL ACCEPTANCE SYSTEM
      // ============================================================================
      
      const legalModal = document.getElementById("legal-modal");
      const legalModalBackdrop = document.getElementById("legal-modal-backdrop");
      const legalAcceptButton = document.getElementById("legal-accept");
      const legalDeclineButton = document.getElementById("legal-decline");
      const acceptTermsCheckbox = document.getElementById("accept-terms-checkbox");
      const acceptPrivacyCheckbox = document.getElementById("accept-privacy-checkbox");

      // Check if user needs to accept legal documents
      const checkLegalAcceptance = async () => {
        try {
          const response = await fetch("/legal/status", {
            credentials: "include"
          });
          
          if (!response.ok) {
            console.error("[LEGAL] Failed to check acceptance status");
            return;
          }
          
          const data = await response.json();
          
          if (data.needs_acceptance && !data.is_guest) {
            // Show legal modal
            showLegalModal();
          } else {
            console.log("[LEGAL] User has accepted current terms");
          }
        } catch (error) {
          console.error("[LEGAL] Error checking acceptance:", error);
        }
      };

      const showLegalModal = () => {
        if (legalModal && legalModalBackdrop) {
          legalModal.style.display = "block";
          legalModalBackdrop.style.display = "block";
          document.body.style.overflow = "hidden";
          console.log("[LEGAL] Legal acceptance modal shown");
        }
      };

      const closeLegalModal = () => {
        if (legalModal && legalModalBackdrop) {
          legalModal.style.display = "none";
          legalModalBackdrop.style.display = "none";
          document.body.style.overflow = "";
          console.log("[LEGAL] Legal acceptance modal closed");
        }
      };

      // Enable accept button only when both checkboxes are checked
      const updateAcceptButton = () => {
        if (legalAcceptButton && acceptTermsCheckbox && acceptPrivacyCheckbox) {
          const bothChecked = acceptTermsCheckbox.checked && acceptPrivacyCheckbox.checked;
          legalAcceptButton.disabled = !bothChecked;
          legalAcceptButton.style.opacity = bothChecked ? "1" : "0.5";
          legalAcceptButton.style.cursor = bothChecked ? "pointer" : "not-allowed";
        }
      };

      // Checkbox change handlers
      acceptTermsCheckbox?.addEventListener("change", updateAcceptButton);
      acceptPrivacyCheckbox?.addEventListener("change", updateAcceptButton);

      // Accept button handler
      legalAcceptButton?.addEventListener("click", async () => {
        if (legalAcceptButton.disabled) return;
        
        try {
          legalAcceptButton.disabled = true;
          legalAcceptButton.textContent = "Accepting...";
          
          await withCsrfToken(async (csrfToken) => {
            const response = await fetch("/legal/accept", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": csrfToken
              },
              credentials: "include",
              body: JSON.stringify({
                accepted_terms: acceptTermsCheckbox.checked,
                accepted_privacy: acceptPrivacyCheckbox.checked
              })
            });
            
            if (!response.ok) {
              const error = await response.json();
              throw new Error(error.detail || "Failed to record acceptance");
            }
            
            console.log("[LEGAL] Acceptance recorded successfully");
            closeLegalModal();
            
            // Show success message
            if (usageWarning) {
              usageWarning.textContent = "✅ Thank you for accepting our terms. You can now use DeFiGuard AI!";
              usageWarning.classList.add("success");
              usageWarning.style.display = "block";
              setTimeout(() => {
                usageWarning.style.display = "none";
              }, 5000);
            }
          });
          
        } catch (error) {
          console.error("[LEGAL] Acceptance failed:", error);
          alert(`Failed to record acceptance: ${error.message}`);
          legalAcceptButton.disabled = false;
          legalAcceptButton.textContent = "I Accept";
        }
      });

      // Decline button handler
      legalDeclineButton?.addEventListener("click", () => {
        if (confirm("You must accept our Terms of Service and Privacy Policy to use DeFiGuard AI. Would you like to sign out?")) {
          window.location.href = "/logout";
        }
      });

      // Check legal acceptance on page load (after auth)
      setTimeout(() => {
        checkLegalAcceptance();
      }, 1000);

      // Initial tier data fetch
      await fetchTierData(); 
      await updateAuthStatus();
      
      // ← NEW: Calm, efficient auth+tier refresh every 30 seconds (no more spam)
      if (window.authTierInterval) clearInterval(window.authTierInterval);
      window.authTierInterval = setInterval(async () => {
        await updateAuthStatus();
        await fetchTierData();
      }, 30000);
      
      handlePostPaymentRedirect();
    } // Closing brace for waitForDOM callback
  ); // Closing parenthesis for waitForDOM function call

  // FINAL DUPLICATE AUTH CHECK KILLER – NO MORE MULTIPLES EVER
  if (window.authCheckInterval) {
    clearInterval(window.authCheckInterval);
    debugLog("[DEBUG] Cleared duplicate auth check interval");
  }
  window.authCheckInterval = null;
}); // ← THIS CLOSES document.addEventListener("DOMContentLoaded", () => {