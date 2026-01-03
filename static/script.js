// ---------------------------------------------------------------------
// GLOBAL DEBUG HELPER – controlled via DEBUG_MODE flag
// Set to false for production to silence debug logs
// ---------------------------------------------------------------------
const DEBUG_MODE = false;  // Set to true for debugging
const log = (label, ...args) => {
    if (DEBUG_MODE) console.log(`[${label}]`, ...args, `time=${new Date().toISOString()}`);
};
// Debug wrapper for console.log calls - only logs in debug mode
const debugLog = (...args) => { if (DEBUG_MODE) console.log(...args); };

// ---------------------------------------------------------------------
// ADVANCED DEBUG TRACER - Comprehensive function tracing for debugging
// This helps identify exactly where UI initialization fails
// ---------------------------------------------------------------------
const DebugTracer = {
    enabled: false,  // Set to true for comprehensive function tracing
    traces: [],
    startTime: Date.now(),

    // Log function entry with timing
    enter(funcName, context = {}) {
        const entry = {
            type: 'ENTER',
            func: funcName,
            time: Date.now() - this.startTime,
            timestamp: new Date().toISOString(),
            context: JSON.stringify(context).substring(0, 200)
        };
        this.traces.push(entry);
        if (this.enabled) {
            console.log(`%c▶ ENTER ${funcName}`, 'color: #00ff00; font-weight: bold',
                `+${entry.time}ms`, context);
        }
        return entry.time;
    },

    // Log function exit with timing
    exit(funcName, startTime, result = null) {
        const duration = (Date.now() - this.startTime) - startTime;
        const entry = {
            type: 'EXIT',
            func: funcName,
            time: Date.now() - this.startTime,
            duration: duration,
            timestamp: new Date().toISOString()
        };
        this.traces.push(entry);
        if (this.enabled) {
            const color = duration > 500 ? '#ff0000' : duration > 100 ? '#ffaa00' : '#00ff00';
            console.log(`%c◀ EXIT ${funcName}`, `color: ${color}; font-weight: bold`,
                `took ${duration}ms`, result ? `result: ${JSON.stringify(result).substring(0, 100)}` : '');
        }
    },

    // Log an error with stack trace
    error(funcName, error) {
        const entry = {
            type: 'ERROR',
            func: funcName,
            time: Date.now() - this.startTime,
            error: error.message,
            stack: error.stack
        };
        this.traces.push(entry);
        console.error(`%c✖ ERROR in ${funcName}`, 'color: #ff0000; font-weight: bold', error);
    },

    // Log element availability check
    checkElements(elements) {
        const results = {};
        for (const [name, el] of Object.entries(elements)) {
            results[name] = el ? '✓' : '✗ MISSING';
        }
        console.log('%c🔍 Element Check:', 'color: #00aaff; font-weight: bold', results);

        // Log any missing elements as warnings
        const missing = Object.entries(results).filter(([_, v]) => v.includes('MISSING'));
        if (missing.length > 0) {
            console.warn('%c⚠ MISSING ELEMENTS:', 'color: #ffaa00; font-weight: bold',
                missing.map(([k]) => k).join(', '));
        }
        return results;
    },

    // Log state snapshot
    snapshot(label, state) {
        console.log(`%c📷 SNAPSHOT [${label}]`, 'color: #aa00ff; font-weight: bold', state);
    },

    // Generate debug report
    report() {
        console.log('%c═══════════════ DEBUG REPORT ═══════════════', 'color: #ff00ff; font-weight: bold');
        console.log('Total traces:', this.traces.length);
        console.log('Errors:', this.traces.filter(t => t.type === 'ERROR').length);

        // Find slow operations (> 500ms)
        const slow = this.traces.filter(t => t.type === 'EXIT' && t.duration > 500);
        if (slow.length > 0) {
            console.warn('%c⏱ SLOW OPERATIONS:', 'color: #ff0000', slow.map(s => `${s.func}: ${s.duration}ms`));
        }

        // Log full trace
        console.table(this.traces);
        return this.traces;
    }
};

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
// AUDIT STATE PERSISTENCE - Survives page navigation and browser close
// Stores audit log messages and current audit state in sessionStorage
// ---------------------------------------------------------------------
const AuditStateManager = {
    STORAGE_KEY: 'defiguard_audit_state',
    MAX_LOG_ENTRIES: 100,

    // Get current state
    getState() {
        try {
            const stored = sessionStorage.getItem(this.STORAGE_KEY);
            return stored ? JSON.parse(stored) : { logs: [], currentAuditKey: null, lastUpdate: null };
        } catch (e) {
            return { logs: [], currentAuditKey: null, lastUpdate: null };
        }
    },

    // Save state
    saveState(state) {
        try {
            // Trim logs to max entries
            if (state.logs && state.logs.length > this.MAX_LOG_ENTRIES) {
                state.logs = state.logs.slice(-this.MAX_LOG_ENTRIES);
            }
            state.lastUpdate = Date.now();
            sessionStorage.setItem(this.STORAGE_KEY, JSON.stringify(state));
        } catch (e) {
            debugLog('[AuditState] Error saving state:', e);
        }
    },

    // Add a log message
    addLog(message) {
        const state = this.getState();
        state.logs.push({
            time: new Date().toISOString(),
            message: message
        });
        this.saveState(state);
    },

    // Set current audit key (so we can resume tracking)
    setCurrentAudit(auditKey) {
        const state = this.getState();
        state.currentAuditKey = auditKey;
        this.saveState(state);
    },

    // Clear current audit (when complete or failed)
    clearCurrentAudit() {
        const state = this.getState();
        state.currentAuditKey = null;
        state.logs = []; // Clear logs when audit completes
        this.saveState(state);
    },

    // Get logs for display
    getLogs() {
        return this.getState().logs;
    },

    // Check if there's an ongoing audit
    hasOngoingAudit() {
        const state = this.getState();
        return !!state.currentAuditKey;
    },

    // Get current audit key
    getCurrentAuditKey() {
        return this.getState().currentAuditKey;
    }
};

// ---------------------------------------------------------------------
// HAMBURGER MENU MANAGER - Ensures mobile menu works after navigation
// Handles iOS Safari bfcache (back-forward cache) issues
// ---------------------------------------------------------------------
const HamburgerManager = {
    initialized: false,

    // Initialize or re-initialize hamburger menu
    init() {
        const hamburger = document.getElementById('hamburger');
        const sidebar = document.getElementById('sidebar');
        const mainContent = document.querySelector('.main-content');

        if (!hamburger || !sidebar || !mainContent) {
            debugLog('[Hamburger] Elements not found, will retry');
            return false;
        }

        // Check if already initialized
        if (hamburger._hamburgerInitialized) {
            debugLog('[Hamburger] Already initialized');
            return true;
        }

        // Add click handler
        hamburger.addEventListener('click', () => {
            sidebar.classList.toggle('open');
            hamburger.classList.toggle('open');
            document.body.classList.toggle('sidebar-open');
            mainContent.style.marginLeft = sidebar.classList.contains('open') ? '270px' : '';
        });

        // Add keyboard support
        hamburger.setAttribute('tabindex', '0');
        hamburger.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                hamburger.click();
            }
        });

        hamburger._hamburgerInitialized = true;
        this.initialized = true;
        debugLog('[Hamburger] Initialized successfully');
        return true;
    },

    // Retry initialization with delay (for iOS timing issues)
    initWithRetry(maxRetries = 5, delay = 100) {
        let attempts = 0;
        const tryInit = () => {
            attempts++;
            if (this.init()) {
                return;
            }
            if (attempts < maxRetries) {
                setTimeout(tryInit, delay * attempts); // Exponential backoff
            } else {
                console.warn('[Hamburger] Failed to initialize after', maxRetries, 'attempts');
            }
        };
        tryInit();
    }
};

// Handle iOS Safari bfcache - re-initialize UI when returning from external page
window.addEventListener('pageshow', (event) => {
    if (event.persisted) {
        // Page was restored from bfcache (common on iOS after Stripe redirect)
        debugLog('[PageShow] Page restored from bfcache, re-initializing UI');
        HamburgerManager.initWithRetry();

        // Also trigger a visibility event to reconnect WebSockets
        document.dispatchEvent(new Event('visibilitychange'));
    }
});

// Also handle focus events (user returns to tab)
window.addEventListener('focus', () => {
    // Ensure hamburger is initialized when window regains focus
    if (!HamburgerManager.initialized) {
        HamburgerManager.initWithRetry();
    }
});

// ---------------------------------------------------------------------
// WEBSOCKET RECONNECTION MANAGER - Auto-reconnects audit log WebSocket
// ---------------------------------------------------------------------
const WebSocketManager = {
    ws: null,
    wsUrl: null,
    reconnectAttempts: 0,
    maxReconnectAttempts: 10,
    reconnectDelay: 1000,
    isConnecting: false,
    onMessageCallback: null,
    onConnectCallback: null,
    onDisconnectCallback: null,

    // Connect to WebSocket with auto-reconnect
    connect(url, callbacks = {}) {
        this.wsUrl = url;
        this.onMessageCallback = callbacks.onMessage || null;
        this.onConnectCallback = callbacks.onConnect || null;
        this.onDisconnectCallback = callbacks.onDisconnect || null;
        this._doConnect();
    },

    _doConnect() {
        if (this.isConnecting || !this.wsUrl) return;

        this.isConnecting = true;
        debugLog('[WS] Connecting to:', this.wsUrl);

        try {
            this.ws = new WebSocket(this.wsUrl);
            ResourceManager.addWebSocket(this.ws);

            this.ws.onopen = () => {
                debugLog('[WS] Connected');
                this.isConnecting = false;
                this.reconnectAttempts = 0;
                if (this.onConnectCallback) this.onConnectCallback();
            };

            this.ws.onmessage = (event) => {
                if (this.onMessageCallback) this.onMessageCallback(event);
            };

            this.ws.onerror = (error) => {
                debugLog('[WS] Error:', error);
                this.isConnecting = false;
            };

            this.ws.onclose = () => {
                debugLog('[WS] Disconnected');
                this.isConnecting = false;
                ResourceManager.removeWebSocket(this.ws);
                this.ws = null;
                if (this.onDisconnectCallback) this.onDisconnectCallback();
                this._scheduleReconnect();
            };
        } catch (e) {
            debugLog('[WS] Connection error:', e);
            this.isConnecting = false;
            this._scheduleReconnect();
        }
    },

    _scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            debugLog('[WS] Max reconnect attempts reached');
            return;
        }

        // Only reconnect if page is visible
        if (document.visibilityState !== 'visible') {
            debugLog('[WS] Page not visible, skipping reconnect');
            return;
        }

        this.reconnectAttempts++;
        const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), 30000);
        debugLog(`[WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

        setTimeout(() => this._doConnect(), delay);
    },

    // Force reconnect (e.g., when page becomes visible)
    reconnect() {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            this.reconnectAttempts = 0;
            this._doConnect();
        }
    },

    // Check if connected
    isConnected() {
        return this.ws && this.ws.readyState === WebSocket.OPEN;
    },

    // Disconnect
    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
};

// Reconnect WebSocket when page becomes visible again
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
        // Page is visible again - try to reconnect if needed
        setTimeout(() => {
            if (!WebSocketManager.isConnected() && WebSocketManager.wsUrl) {
                debugLog('[WS] Page visible, attempting reconnect');
                WebSocketManager.reconnect();
            }
        }, 500);
    }
});

// ---------------------------------------------------------------------
// RESILIENT FETCH - Retry with exponential backoff for all API calls
// This ensures network hiccups don't break the app
// ---------------------------------------------------------------------
const fetchWithRetry = async (url, options = {}, maxRetries = 3, baseDelay = 1000) => {
    let lastError;

    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            const response = await fetch(url, {
                credentials: 'include',
                ...options
            });

            // Don't retry on client errors (4xx) except 429 (rate limit)
            if (response.status >= 400 && response.status < 500 && response.status !== 429) {
                return response;
            }

            // Retry on server errors (5xx) or rate limiting
            if (response.status >= 500 || response.status === 429) {
                lastError = new Error(`HTTP ${response.status}`);
                if (attempt < maxRetries - 1) {
                    const delay = baseDelay * Math.pow(2, attempt);
                    debugLog(`[Fetch] Retry ${attempt + 1}/${maxRetries} for ${url} after ${delay}ms`);
                    await new Promise(r => setTimeout(r, delay));
                    continue;
                }
            }

            return response;
        } catch (err) {
            lastError = err;
            if (attempt < maxRetries - 1) {
                const delay = baseDelay * Math.pow(2, attempt);
                debugLog(`[Fetch] Network error, retry ${attempt + 1}/${maxRetries} for ${url} after ${delay}ms`);
                await new Promise(r => setTimeout(r, delay));
            }
        }
    }

    throw lastError;
};

// ---------------------------------------------------------------------
// TIER CACHE - Cache tier/usage data with TTL to reduce API calls
// ---------------------------------------------------------------------
const TierCache = {
    data: null,
    timestamp: 0,
    TTL: 30000, // 30 seconds
    pendingRequest: null, // Deduplication

    async get(forceRefresh = false) {
        const now = Date.now();

        // Return cached if fresh
        if (!forceRefresh && this.data && (now - this.timestamp) < this.TTL) {
            debugLog('[TierCache] Returning cached tier data');
            return this.data;
        }

        // Deduplicate concurrent requests
        if (this.pendingRequest) {
            debugLog('[TierCache] Waiting for pending request');
            return this.pendingRequest;
        }

        // Fetch fresh data
        this.pendingRequest = this._fetch();
        try {
            const result = await this.pendingRequest;
            return result;
        } finally {
            this.pendingRequest = null;
        }
    },

    async _fetch() {
        try {
            const response = await fetchWithRetry('/api/tier');
            if (response.ok) {
                this.data = await response.json();
                this.timestamp = Date.now();
                debugLog('[TierCache] Cached fresh tier data');
                return this.data;
            }
        } catch (e) {
            debugLog('[TierCache] Error fetching tier:', e);
        }
        return this.data; // Return stale data on error
    },

    invalidate() {
        this.timestamp = 0;
        debugLog('[TierCache] Cache invalidated');
    }
};

// ---------------------------------------------------------------------
// FORM STATE MANAGER - Persist form data across page navigation
// ---------------------------------------------------------------------
const FormStateManager = {
    STORAGE_KEY: 'defiguard_form_state',

    // Save form field value
    saveField(fieldName, value) {
        try {
            const state = this.getState();
            state[fieldName] = value;
            state.lastUpdate = Date.now();
            sessionStorage.setItem(this.STORAGE_KEY, JSON.stringify(state));
        } catch (e) {
            debugLog('[FormState] Error saving:', e);
        }
    },

    // Get all saved state
    getState() {
        try {
            const stored = sessionStorage.getItem(this.STORAGE_KEY);
            return stored ? JSON.parse(stored) : {};
        } catch (e) {
            return {};
        }
    },

    // Get specific field
    getField(fieldName) {
        return this.getState()[fieldName];
    },

    // Clear after successful submission
    clear() {
        try {
            sessionStorage.removeItem(this.STORAGE_KEY);
            debugLog('[FormState] Cleared');
        } catch (e) {}
    },

    // Restore form fields from saved state
    restore() {
        const state = this.getState();
        if (!state.lastUpdate) return;

        // Only restore if less than 30 minutes old
        if (Date.now() - state.lastUpdate > 30 * 60 * 1000) {
            this.clear();
            return;
        }

        // Restore contract address
        if (state.contractAddress) {
            const contractInput = document.getElementById('contract_address');
            if (contractInput && !contractInput.value) {
                contractInput.value = state.contractAddress;
                debugLog('[FormState] Restored contract address');
            }
        }

        // Restore custom report input
        if (state.customReport) {
            const customInput = document.getElementById('custom_report');
            if (customInput && !customInput.value) {
                customInput.value = state.customReport;
                debugLog('[FormState] Restored custom report');
            }
        }

        // Notify user of restored data
        if (state.contractAddress || state.customReport || state.fileName) {
            ToastNotification.show('Form data restored from previous session', 'info', 3000);
        }
    }
};

// ---------------------------------------------------------------------
// LOADING MANAGER - Ensure loading states never get stuck
// ---------------------------------------------------------------------
const LoadingManager = {
    activeLoaders: new Map(),
    DEFAULT_TIMEOUT: 5 * 60 * 1000, // 5 minutes max

    // Show loading with auto-timeout
    show(element, timeoutMs = this.DEFAULT_TIMEOUT) {
        if (!element) return;

        element.classList.add('show');

        // Set timeout to auto-hide
        const timeoutId = setTimeout(() => {
            this.hide(element);
            console.warn('[Loading] Auto-hidden after timeout');
            ToastNotification.show('Operation timed out. Please try again.', 'warning');
        }, timeoutMs);

        this.activeLoaders.set(element, timeoutId);
    },

    // Hide loading and clear timeout
    hide(element) {
        if (!element) return;

        element.classList.remove('show');

        const timeoutId = this.activeLoaders.get(element);
        if (timeoutId) {
            clearTimeout(timeoutId);
            this.activeLoaders.delete(element);
        }
    },

    // Clear all loading states (emergency cleanup)
    clearAll() {
        this.activeLoaders.forEach((timeoutId, element) => {
            clearTimeout(timeoutId);
            element.classList.remove('show');
        });
        this.activeLoaders.clear();
    }
};

// Clean up loading states on page unload
window.addEventListener('beforeunload', () => {
    LoadingManager.clearAll();
});

// ---------------------------------------------------------------------
// POLLING MANAGER - Handle polling with visibility awareness and cleanup
// ---------------------------------------------------------------------
const PollingManager = {
    polls: new Map(),

    // Start a poll with auto-cleanup
    start(name, callback, intervalMs, options = {}) {
        const {
            maxDuration = 30 * 60 * 1000, // 30 min default
            pauseWhenHidden = true,
            maxConsecutiveErrors = 10
        } = options;

        // Clear existing poll with same name
        this.stop(name);

        const state = {
            intervalId: null,
            startTime: Date.now(),
            consecutiveErrors: 0,
            isPaused: false
        };

        const wrappedCallback = async () => {
            // Check duration limit
            if (Date.now() - state.startTime > maxDuration) {
                debugLog(`[Polling] ${name} exceeded max duration, stopping`);
                this.stop(name);
                return;
            }

            // Skip if paused
            if (state.isPaused) return;

            try {
                await callback();
                state.consecutiveErrors = 0;
            } catch (e) {
                state.consecutiveErrors++;
                debugLog(`[Polling] ${name} error (${state.consecutiveErrors}/${maxConsecutiveErrors}):`, e);

                if (state.consecutiveErrors >= maxConsecutiveErrors) {
                    debugLog(`[Polling] ${name} too many errors, stopping`);
                    this.stop(name);
                }
            }
        };

        state.intervalId = setInterval(wrappedCallback, intervalMs);
        ResourceManager.addInterval(state.intervalId);

        // Handle visibility
        if (pauseWhenHidden) {
            const visibilityHandler = () => {
                state.isPaused = document.visibilityState === 'hidden';
                debugLog(`[Polling] ${name} ${state.isPaused ? 'paused' : 'resumed'}`);
            };
            document.addEventListener('visibilitychange', visibilityHandler);
            state.visibilityHandler = visibilityHandler;
        }

        this.polls.set(name, state);

        // Run immediately
        wrappedCallback();

        return name;
    },

    stop(name) {
        const state = this.polls.get(name);
        if (state) {
            if (state.intervalId) {
                clearInterval(state.intervalId);
                ResourceManager.removeInterval(state.intervalId);
            }
            if (state.visibilityHandler) {
                document.removeEventListener('visibilitychange', state.visibilityHandler);
            }
            this.polls.delete(name);
            debugLog(`[Polling] ${name} stopped`);
        }
    },

    stopAll() {
        this.polls.forEach((state, name) => this.stop(name));
    }
};

// Clean up polls on page unload
window.addEventListener('beforeunload', () => {
    PollingManager.stopAll();
});

// ---------------------------------------------------------------------
// ACCESS KEY HELPERS - For persistent audit retrieval
// Access Keys (dga_xxx) are unique identifiers for retrieving specific audit results
// Different from Project Keys which organize audits by client/project
// ---------------------------------------------------------------------

// Copy access key to clipboard
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
        ToastNotification.show('Access key copied to clipboard!', 'success');
    } catch (err) {
        console.error('Failed to copy access key:', err);
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = auditKey;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        ToastNotification.show('Access key copied!', 'success');
    }
};

// Retrieve audit results by access key
window.retrieveAuditByKey = async function(auditKey) {
    if (!auditKey || !auditKey.startsWith('dga_')) {
        ToastNotification.show('Invalid access key format. Keys start with "dga_"', 'error');
        return null;
    }

    try {
        ToastNotification.show('Retrieving audit...', 'info');
        const response = await fetchWithRetry(`/audit/retrieve/${auditKey}`, {}, 3, 1000);
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || 'Failed to retrieve audit');
        }

        const data = await response.json();
        console.log('[AUDIT_RETRIEVE] Retrieved data:', data);

        // Display results based on status
        if (data.status === 'completed' && data.report) {
            // Build the same structure that handleAuditResponse expects
            const auditData = {
                report: data.report,
                risk_score: data.risk_score || data.report.risk_score,
                tier: data.user_tier,
                audit_key: data.audit_key,
                pdf_url: data.pdf_url
            };

            console.log('[AUDIT_RETRIEVE] Calling handleAuditResponse with:', auditData);

            // Use a custom event to trigger the audit response handler
            // This ensures handleAuditResponse (defined inside DOMContentLoaded) can receive it
            window.dispatchEvent(new CustomEvent('retrievedAuditComplete', { detail: auditData }));

            ToastNotification.show('Audit results loaded successfully!', 'success');

            // Scroll to results section
            const resultsSection = document.querySelector('.results-section');
            if (resultsSection) {
                resultsSection.scrollIntoView({ behavior: 'smooth' });
            }
        } else if (data.status === 'processing') {
            ToastNotification.show(`Audit in progress: ${data.current_phase || 'analyzing'}...`, 'info');
        } else if (data.status === 'queued') {
            ToastNotification.show(`Audit queued at position ${data.queue_position || '?'}`, 'info');
        } else if (data.status === 'failed') {
            ToastNotification.show(`Audit failed: ${data.error || 'Unknown error'}`, 'error');
        } else {
            ToastNotification.show(`Audit status: ${data.status}`, 'info');
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

// Show the access key retrieval modal
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
                    Enter your <strong>Access Key</strong> to retrieve results from a previous audit.
                    Access keys start with <code>dga_</code> and are provided when you submit an audit.
                </p>
                <div class="form-group">
                    <label for="retrieve-audit-key">Access Key</label>
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

    // Load recent keys from sessionStorage (more secure than localStorage - clears on tab close)
    try {
        const savedKeys = JSON.parse(sessionStorage.getItem('auditKeys') || '[]');
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
                ${escapeHtml(message)}
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
            const response = await fetchWithRetry('/api/certora/notifications', {}, 2, 1000);

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
            // Get CSRF token with retry
            const csrfResponse = await fetchWithRetry('/csrf-token', {}, 2, 500);
            const { csrf_token } = await csrfResponse.json();

            await fetchWithRetry('/api/certora/notifications/dismiss', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrf_token
                },
                body: JSON.stringify({ job_ids: jobIds })
            }, 2, 1000);
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
        this.wsToken = null;
        this.onUpdate = null;
        this.onComplete = null;
        this.onError = null;
        this.pollInterval = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000; // Start with 1 second
        this.isCompleted = false; // Track if job finished
    }

    async submitAudit(formData, csrfToken) {
        try {
            const response = await fetchWithRetry('/audit/submit', {
                method: 'POST',
                headers: { 'X-CSRFToken': csrfToken },
                body: formData
            }, 2, 2000); // 2 retries with 2s delay for uploads

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

            // Start WebSocket connection for real-time updates (async, with auth token)
            await this.connectWebSocket();

            // Fallback polling in case WebSocket fails (started inside connectWebSocket if needed)
            // Only start polling if WebSocket didn't connect
            if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
                this.startPolling();
            }

            return data;
        } catch (error) {
            log('QUEUE_ERROR', error.message);
            throw error;
        }
    }

    showAuditKey(auditKey) {
        // Validate access key format for security
        if (typeof auditKey !== 'string' || !auditKey.startsWith('dga_') || auditKey.length > 64) {
            console.error('Invalid access key format');
            return;
        }

        // Create and show the access key notification
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
        title.textContent = 'Your Access Key';

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
        info.textContent = 'Save this Access Key to retrieve your audit results anytime:';

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

        // Store in sessionStorage for security (clears on tab close, not vulnerable to XSS persistence)
        try {
            const savedKeys = JSON.parse(sessionStorage.getItem('auditKeys') || '[]');
            savedKeys.unshift({ key: auditKey, timestamp: Date.now() });
            // Keep only last 10 keys
            sessionStorage.setItem('auditKeys', JSON.stringify(savedKeys.slice(0, 10)));
        } catch (e) {
            console.warn('Could not save audit key to sessionStorage:', e);
        }
    }
    
    async connectWebSocket() {
        if (!this.jobId) return;

        // Clean up existing connection first
        this.disconnectWebSocket();

        // Get WebSocket authentication token
        let wsToken = "";
        try {
            const tokenResponse = await fetchWithRetry("/api/ws-token", {}, 2, 500);
            if (tokenResponse.ok) {
                const tokenData = await tokenResponse.json();
                wsToken = tokenData.token || "";
            }
        } catch (e) {
            log('QUEUE_WS', 'Failed to get WS token, will use polling fallback');
        }

        if (!wsToken) {
            log('QUEUE_WS', 'No WS token available, using polling');
            this.startPolling();
            return;
        }

        // Store token for reconnection
        this.wsToken = wsToken;

        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws/job/${this.jobId}?token=${encodeURIComponent(wsToken)}`;

        try {
            this.ws = new WebSocket(wsUrl);
            ResourceManager.addWebSocket(this.ws);

            this.ws.onopen = () => {
                log('QUEUE_WS', 'Connected to job status WebSocket');
                // Reset reconnection state on successful connection
                this.reconnectAttempts = 0;
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
                log('QUEUE_WS', 'WebSocket error');
                // Don't immediately fall back to polling - let onclose handle reconnection
            };

            this.ws.onclose = (event) => {
                log('QUEUE_WS', `WebSocket closed (code=${event.code})`);
                ResourceManager.removeWebSocket(this.ws);
                this.ws = null;

                // Don't reconnect if job is completed or we've exceeded max attempts
                if (this.isCompleted) {
                    log('QUEUE_WS', 'Job completed, no reconnection needed');
                    return;
                }

                // Reconnect with exponential backoff
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
                    this.reconnectAttempts++;
                    log('QUEUE_WS', `Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

                    setTimeout(async () => {
                        if (!this.isCompleted && this.jobId) {
                            await this.connectWebSocket();
                        }
                    }, delay);
                } else {
                    log('QUEUE_WS', 'Max reconnection attempts reached, falling back to polling');
                    this.startPolling();
                }
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

        let consecutiveErrors = 0;
        const maxErrors = 10;
        let pollDelay = 3000; // Start at 3 seconds

        const doPoll = async () => {
            if (!this.jobId || this.isCompleted) {
                this.stopPolling();
                return;
            }

            // Pause polling when page is hidden to save resources
            if (document.visibilityState === 'hidden') {
                return;
            }

            try {
                const response = await fetchWithRetry(`/audit/status/${this.jobId}`, {}, 2, 1000);
                if (response.ok) {
                    const data = await response.json();
                    this.handleStatusUpdate(data);
                    consecutiveErrors = 0;
                    pollDelay = 3000; // Reset to normal speed
                } else {
                    throw new Error(`HTTP ${response.status}`);
                }
            } catch (e) {
                consecutiveErrors++;
                log('QUEUE_POLL', `Polling error (${consecutiveErrors}/${maxErrors}):`, e);

                if (consecutiveErrors >= maxErrors) {
                    log('QUEUE_POLL', 'Too many errors, stopping polling');
                    this.stopPolling();
                    if (this.onError) {
                        this.onError('Lost connection to server. Please refresh the page.');
                    }
                    return;
                }

                // Exponential backoff on errors
                pollDelay = Math.min(pollDelay * 1.5, 15000);
            }
        };

        // Use PollingManager for better control
        PollingManager.start('queue-status', doPoll, 3000, {
            maxDuration: 60 * 60 * 1000, // 1 hour max
            pauseWhenHidden: true,
            maxConsecutiveErrors: 10
        });
        this.pollInterval = 'queue-status'; // Store name instead of ID
    }

    stopPolling() {
        if (this.pollInterval) {
            if (typeof this.pollInterval === 'string') {
                PollingManager.stop(this.pollInterval);
            } else {
                ResourceManager.removeInterval(this.pollInterval);
            }
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
                this.isCompleted = true; // Prevent reconnection attempts
                this.stopPolling();
                this.disconnect();
                if (this.onComplete) {
                    this.onComplete(data.result);
                }
                break;
            case 'failed':
                this.isCompleted = true; // Prevent reconnection attempts
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
        
        // Upgrade prompt for Free/Starter users - Psychological triggers: Loss aversion, urgency, social proof
        let upgradeHtml = '';
        if (isFreeTier) {
            // Calculate potential wait based on position
            const estimatedWait = data.position > 1 ? Math.ceil(data.position * 2.5) : 1;
            upgradeHtml = `
                <div class="queue-upgrade-prompt" style="background: linear-gradient(135deg, rgba(231, 76, 60, 0.1), rgba(155, 89, 182, 0.1)); border: 1px solid var(--accent-purple);">
                    <div style="display: flex; align-items: center; gap: var(--space-2); margin-bottom: var(--space-2);">
                        <span class="upgrade-icon">⏱️</span>
                        <span style="font-weight: 600; color: var(--text-primary);">Estimated wait: ~${estimatedWait} minutes</span>
                    </div>
                    <p style="font-size: var(--text-sm); color: var(--text-secondary); margin: 0 0 var(--space-2) 0;">
                        <strong style="color: var(--accent-purple);">Team ($349/mo)</strong> and <strong>Enterprise</strong> audits are processed first.
                        Skip the line and get deeper analysis with Mythril + Echidna fuzzing.
                    </p>
                    <a href="#tier-select" class="upgrade-link" style="display: inline-block; padding: 8px 16px; background: var(--accent-purple); color: white; border-radius: 6px; text-decoration: none; font-weight: 600;" onclick="document.getElementById('tier-select').scrollIntoView({behavior: 'smooth'})">
                        ⚡ Upgrade & Skip Ahead
                    </a>
                </div>
            `;
        } else if (isStarterTier) {
            upgradeHtml = `
                <div class="queue-upgrade-prompt starter" style="background: linear-gradient(135deg, rgba(52, 152, 219, 0.1), rgba(155, 89, 182, 0.1)); border: 1px solid var(--accent-teal);">
                    <div style="display: flex; align-items: center; gap: var(--space-2); margin-bottom: var(--space-2);">
                        <span class="upgrade-icon">🔓</span>
                        <span style="font-weight: 600; color: var(--text-primary);">Unlock Priority Processing</span>
                    </div>
                    <p style="font-size: var(--text-sm); color: var(--text-secondary); margin: 0 0 var(--space-2) 0;">
                        <strong style="color: var(--green);">Team plan</strong> includes priority queue, Mythril deep analysis,
                        Echidna fuzzing, on-chain detection, and API access for CI/CD integration.
                    </p>
                    <a href="#tier-select" class="upgrade-link" style="display: inline-block; padding: 8px 16px; background: var(--green); color: white; border-radius: 6px; text-decoration: none; font-weight: 600;" onclick="document.getElementById('tier-select').scrollIntoView({behavior: 'smooth'})">
                        🚀 Upgrade to Team - $349/mo
                    </a>
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
            'starting': { icon: '🚀', label: 'Starting Audit...', progress: 5 },
            'slither': { icon: '🔍', label: 'Running Slither Analysis', progress: 10 },
            'mythril': { icon: '🧠', label: 'Running Mythril Symbolic Analysis', progress: 30 },
            'echidna': { icon: '🧪', label: 'Running Echidna Fuzzing', progress: 50 },
            'certora': { icon: '🔒', label: 'Running Formal Verification', progress: 55 },
            'ai_analysis': { icon: '🤖', label: 'Claude AI Analysis & Report Generation', progress: 70 },
            'grok': { icon: '🤖', label: 'Claude AI Analysis & Report Generation', progress: 70 }, // Legacy alias
            'finalizing': { icon: '✨', label: 'Finalizing Report', progress: 95 },
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
                            ${escapeHtml(error.message || 'Failed to initialize WalletConnect')}
                        </p>
                        <p style="font-size: var(--text-sm); color: var(--text-tertiary); margin-bottom: var(--space-4);">
                            <strong>Alternative:</strong> Open this page in your wallet app's browser
                        </p>
                        <div style="background: var(--glass-bg); padding: var(--space-3); border-radius: var(--radius-md); font-size: var(--text-xs); word-break: break-all; color: var(--text-tertiary);">
                            ${escapeHtml(window.location.origin)}
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
            const response = await fetchWithRetry(`/csrf-token?_=${Date.now()}`, {
                method: 'GET'
            }, 2, 500);
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
            const response = await fetchWithRetry('/api/wallet/connect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken
                },
                body: JSON.stringify({
                    address: this.address,
                    message: message,
                    signature: signature
                })
            }, 2, 1000);

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
            const response = await fetchWithRetry(`/api/wallet/contracts?address=${this.address}`, {}, 2, 1000);

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
            const response = await fetchWithRetry(`/api/wallet/contract-source?address=${contractAddress}`, {}, 2, 1000);

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
      const response = await fetchWithRetry(`/csrf-token?_=${Date.now()}`, {
        method: "GET"
      }, 2, 500);
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
  // Cached to prevent repeated API calls (expires after 30 seconds)
  let usernameCache = null;
  let usernameCacheTime = 0;
  const USERNAME_CACHE_TTL = 30000; // 30 seconds

  const fetchUsername = async (forceRefresh = false) => {
    const now = Date.now();

    // Return cached if fresh
    if (!forceRefresh && usernameCache !== undefined && (now - usernameCacheTime) < USERNAME_CACHE_TTL) {
      return usernameCache;
    }

    try {
      const resp = await fetchWithRetry('/me', {}, 2, 1000);
      if (resp.ok) {
        const data = await resp.json();
        usernameCache = data.logged_in ? {
          username: data.username,
          sub: data.sub,
          provider: data.provider,
          member_since: data.member_since
        } : null;
        usernameCacheTime = now;
        return usernameCache;
      }
    } catch (e) {
      console.debug('[AUTH] Failed to fetch /me');
    }
    usernameCache = null;
    usernameCacheTime = now;
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
      modalLogoutButton: "#modal-logout"
      // Note: Certora elements (modal-certora-section, certora-jobs-table-body, etc.)
      // are fetched dynamically when needed since they're only in enterprise/diamond tier HTML
    },
    async (els) => {
      const _waitForDOMStart = DebugTracer.enter('waitForDOM_callback', { url: window.location.href });

      const {
        auditForm, loading, resultsDiv, riskScoreSpan, issuesBody, predictionsList,
        recommendationsList, fuzzingList, remediationRoadmap, usageWarning, sidebarTierName,
        sidebarTierUsage, sidebarTierFeatures, tierDescription, sizeLimit, features,
        upgradeLink, tierSelect, tierSwitchButton, contractAddressInput, facetWell,
        downloadReportButton, pdfDownloadButton, mintNftButton, customReportInput, apiKeySpan,
        hamburger, sidebar, mainContent, logoutSidebar, authStatus, auditLog
      } = els;

      // DEBUG: Check critical element availability
      DebugTracer.checkElements({
        hamburger, sidebar, mainContent, authStatus, usageWarning,
        sidebarTierName, sidebarTierUsage, auditForm, logoutSidebar
      });

      // DEBUG: Snapshot URL state for post-payment detection
      const urlParams = new URLSearchParams(window.location.search);
      DebugTracer.snapshot('URL_PARAMS', {
        upgrade: urlParams.get('upgrade'),
        session_id: urlParams.get('session_id'),
        tier: urlParams.get('tier'),
        message: urlParams.get('message'),
        fullSearch: window.location.search
      });

      // =====================================================
      // HAMBURGER MENU - Initialize FIRST before async operations
      // This ensures navigation works even if other init fails
      // =====================================================
      const _hamburgerStart = DebugTracer.enter('hamburger_init');
      try {
        if (hamburger && sidebar && mainContent) {
          // Check if event listeners are already attached (prevent duplicates)
          // Use same flag as HamburgerManager to prevent duplicate event listeners
          if (!hamburger._hamburgerInitialized) {
            hamburger.addEventListener("click", () => {
              DebugTracer.snapshot('hamburger_click', {
                sidebarOpen: sidebar.classList.contains('open'),
                hamburgerOpen: hamburger.classList.contains('open')
              });
              sidebar.classList.toggle("open");
              hamburger.classList.toggle("open");
              document.body.classList.toggle('sidebar-open');
              mainContent.style.marginLeft = sidebar.classList.contains("open") ? "270px" : "";
            });
            hamburger.setAttribute("tabindex", "0");
            hamburger.addEventListener("keydown", (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                hamburger.click();
              }
            });
            hamburger._hamburgerInitialized = true;
            HamburgerManager.initialized = true;
            DebugTracer.exit('hamburger_init', _hamburgerStart, { success: true });
          } else {
            DebugTracer.exit('hamburger_init', _hamburgerStart, { skipped: 'already_initialized' });
          }
          debugLog('[INIT] Hamburger menu initialized');
        } else {
          DebugTracer.exit('hamburger_init', _hamburgerStart, {
            failed: true,
            hamburger: !!hamburger,
            sidebar: !!sidebar,
            mainContent: !!mainContent
          });
          console.warn('[INIT] Hamburger elements missing:', {
            hamburger: !!hamburger,
            sidebar: !!sidebar,
            mainContent: !!mainContent
          });
        }
      } catch (e) {
        DebugTracer.error('hamburger_init', e);
        console.error('[INIT] Hamburger init error:', e);
      }

      // Real-time audit log with persistence
      const logMessage = (msg, persist = true) => {
        console.log(`[AUDIT] ${msg}`);

        // Persist to sessionStorage so logs survive page navigation
        if (persist) {
          AuditStateManager.addLog(msg);
        }

        if (auditLog) {
          const entry = document.createElement("div");
          entry.textContent = `[${new Date().toISOString()}] ${msg}`;
          auditLog.appendChild(entry);
          auditLog.scrollTop = auditLog.scrollHeight;
          if (auditLog.children.length > 100) auditLog.removeChild(auditLog.firstChild);
          if (auditLog.style.display === "none") auditLog.style.display = "block";
        }
      };

      // Restore previous log messages on page load (e.g., after navigation)
      const restoreLogs = () => {
        const savedLogs = AuditStateManager.getLogs();
        if (savedLogs.length > 0 && auditLog) {
          debugLog(`[AUDIT] Restoring ${savedLogs.length} saved log messages`);
          savedLogs.forEach(logEntry => {
            const entry = document.createElement("div");
            entry.textContent = `[${logEntry.time}] ${logEntry.message}`;
            entry.style.opacity = '0.7'; // Slightly faded to indicate restored
            auditLog.appendChild(entry);
          });
          auditLog.scrollTop = auditLog.scrollHeight;
          if (auditLog.style.display === "none") auditLog.style.display = "block";

          // Add separator to show where restored logs end
          const separator = document.createElement("div");
          separator.textContent = "--- Session resumed ---";
          separator.style.textAlign = "center";
          separator.style.color = "var(--accent-teal)";
          separator.style.borderTop = "1px solid var(--accent-teal)";
          separator.style.marginTop = "4px";
          separator.style.paddingTop = "4px";
          auditLog.appendChild(separator);
        }
      };

      // Restore logs on page load
      restoreLogs();

      // Check for ongoing audits from SERVER (works across all devices)
      // This queries /api/user/audits and finds any in-progress audits
      const checkOngoingAudits = async () => {
        try {
          const response = await fetchWithRetry('/api/user/audits?limit=5', {}, 2, 1000);
          if (!response.ok) {
            debugLog('[AUDIT] Could not fetch user audits - user may not be logged in');
            return;
          }

          const data = await response.json();
          const audits = data.audits || [];

          // Find any audits that are still in progress
          const inProgress = audits.filter(a =>
            a.status === 'processing' || a.status === 'queued'
          );

          if (inProgress.length > 0) {
            // Show all in-progress audits
            inProgress.forEach(audit => {
              logMessage(`📋 In-progress audit: ${audit.contract_name || 'Contract'} (${audit.status})`, false);
            });

            // Store the most recent one for tracking
            const mostRecent = inProgress[0];
            AuditStateManager.setCurrentAudit(mostRecent.audit_key);

            // Start polling for updates on the most recent
            startAuditPolling(mostRecent.audit_key);
          }

          // Also check for recently completed audits that user might have missed
          const recentlyCompleted = audits.filter(a => {
            if (a.status !== 'completed' || !a.completed_at) return false;
            const completedTime = new Date(a.completed_at).getTime();
            const fiveMinutesAgo = Date.now() - (5 * 60 * 1000);
            return completedTime > fiveMinutesAgo;
          });

          if (recentlyCompleted.length > 0 && !inProgress.length) {
            const recent = recentlyCompleted[0];
            logMessage(`✅ Recent audit completed: ${recent.contract_name || 'Contract'} - Score: ${recent.risk_score}`, false);

            // Offer to load the results
            ToastNotification.show(
              `Your audit "${recent.contract_name || 'Contract'}" completed! Click to view results.`,
              'success',
              8000
            );
          }

        } catch (e) {
          debugLog('[AUDIT] Error checking ongoing audits:', e);
        }
      };

      // Poll for audit status updates (server-side state)
      let auditPollInterval = null;
      const startAuditPolling = (auditKey) => {
        if (auditPollInterval) {
          clearInterval(auditPollInterval);
        }

        const pollAudit = async () => {
          try {
            const response = await fetchWithRetry(`/audit/retrieve/${auditKey}`, {}, 2, 1000);
            if (!response.ok) return;

            const data = await response.json();
            debugLog(`[AUDIT] Poll status: ${data.status}`);

            if (data.status === 'completed') {
              clearInterval(auditPollInterval);
              auditPollInterval = null;
              logMessage('✅ Audit completed!');
              AuditStateManager.clearCurrentAudit();

              // Trigger UI update with results
              window.dispatchEvent(new CustomEvent('retrievedAuditComplete', { detail: {
                report: data.report,
                risk_score: data.risk_score,
                tier: data.user_tier,
                audit_key: data.audit_key,
                pdf_url: data.pdf_url
              }}));

              ToastNotification.show('Your audit is complete! Results are now available.', 'success');

            } else if (data.status === 'failed') {
              clearInterval(auditPollInterval);
              auditPollInterval = null;
              logMessage(`❌ Audit failed: ${data.error || 'Unknown error'}`);
              AuditStateManager.clearCurrentAudit();

            } else if (data.status === 'processing') {
              // Update with current phase
              if (data.current_phase) {
                logMessage(`🔄 ${data.current_phase}`, false);
              }
            }
          } catch (e) {
            debugLog('[AUDIT] Poll error:', e);
          }
        };

        // Poll every 10 seconds
        auditPollInterval = setInterval(pollAudit, 10000);
        ResourceManager.addInterval(auditPollInterval);

        // Also poll immediately
        pollAudit();
      };

      // Check ongoing audits after a short delay (let UI initialize first)
      setTimeout(checkOngoingAudits, 1500);

      // WebSocket audit log with auto-reconnect
      // Security: Only connect if we have a valid token (server requires authentication)
      let wsToken = "";
      try {
        const tokenResponse = await fetchWithRetry("/api/ws-token", {}, 2, 500);
        if (tokenResponse.ok) {
          const tokenData = await tokenResponse.json();
          wsToken = tokenData.token || "";
        }
      } catch (e) {
        debugLog("[AUDIT] Could not fetch WS token - user may not be logged in");
      }

      // Use WebSocketManager for auto-reconnect capability
      if (wsToken) {
        const wsUrl = `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws-audit-log?token=${encodeURIComponent(wsToken)}`;

        WebSocketManager.connect(wsUrl, {
          onConnect: () => logMessage("Connected to audit log"),
          onMessage: (e) => {
            try {
              const data = JSON.parse(e.data);
              if (data.type === "audit_log") logMessage(data.message);
            } catch (_) {}
          },
          onDisconnect: () => logMessage("Disconnected from audit log (will auto-reconnect)")
        });
      } else {
        debugLog("[AUDIT] Skipping WebSocket connection - no auth token available");
      }

      // Section6: Authentication – instantly show real username + provider
      const updateAuthStatus = async () => {
        const _authStart = DebugTracer.enter('updateAuthStatus');
        try {
          const user = await fetchUsername();  // Returns { username, sub, provider } from /me
          DebugTracer.snapshot('updateAuthStatus_user', { user: user?.username || 'null' });
          console.log(
            `[DEBUG] updateAuthStatus: user=${JSON.stringify(user) || 'null'}, time=${new Date().toISOString()}`
          );
          if (!authStatus) {
            console.error("[ERROR] #auth-status not found in DOM");
            DebugTracer.exit('updateAuthStatus', _authStart, { error: 'authStatus_element_missing' });
            return;
          }
          if (user && user.username) {
            const displayProvider = user.provider && user.provider !== "unknown"
              ? user.provider
              : (user.sub?.includes('|') ? user.sub.split('|')[0] : "auth0");
            authStatus.innerHTML = `Signed in as <strong>${escapeHtml(user.username)}</strong> <small>(${escapeHtml(displayProvider)})</small>`;
            localStorage.setItem('userSub', user.sub);
            if (sidebar) sidebar.classList.add("logged-in");
            DebugTracer.exit('updateAuthStatus', _authStart, { loggedIn: true, user: user.username });
          } else {
            authStatus.innerHTML = '<a href="/auth">Sign In / Create Account</a>';
            localStorage.removeItem('userSub');
            if (sidebar) sidebar.classList.remove("logged-in");
            DebugTracer.exit('updateAuthStatus', _authStart, { loggedIn: false });
          }
        } catch (err) {
          DebugTracer.error('updateAuthStatus', err);
          DebugTracer.exit('updateAuthStatus', _authStart, { error: err.message });
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
        const _paymentStart = DebugTracer.enter('handlePostPaymentRedirect');
        const urlParams = new URLSearchParams(window.location.search);
        const sessionId = urlParams.get("session_id");
        const tier = urlParams.get("tier");
        const tempId = urlParams.get("temp_id");
        const username =
          urlParams.get("username") || localStorage.getItem("username");
        const upgradeStatus = urlParams.get("upgrade");
        const message = urlParams.get("message");

        DebugTracer.snapshot('payment_params', {
          sessionId, tier, tempId, username, upgradeStatus, message
        });

        console.log(
          `[DEBUG] Handling post-payment redirect: session_id=${sessionId}, tier=${tier}, temp_id=${tempId}, username=${username}, upgrade=${upgradeStatus}, message=${message}, time=${new Date().toISOString()}`
        );

        // NEW: Direct success from backend (no session_id needed)
        if (upgradeStatus === "success") {
          DebugTracer.snapshot('payment_branch', { branch: 'success' });
          console.log(
            `[PAYMENT] ✅ Upgrade success detected: tier=${tier}, time=${new Date().toISOString()}`
          );

          // Show immediate success feedback (with null check)
          if (usageWarning) {
            usageWarning.textContent = message || `🎉 Tier upgrade completed!`;
            usageWarning.classList.remove("error", "info");
            usageWarning.classList.add("success");
            usageWarning.style.display = "block";
          }

          // Clean URL immediately for good UX
          window.history.replaceState({}, document.title, "/ui");

          // Small delay to ensure backend DB transaction is fully committed
          // This prevents race conditions where tier isn't updated yet
          await new Promise(resolve => setTimeout(resolve, 300));

          // Fetch fresh tier data to update UI
          console.log("[PAYMENT] Refreshing tier data after upgrade...");
          try {
            await fetchTierData();
            await updateAuthStatus();

            // If tier was provided in URL, update sidebar immediately
            if (tier) {
              const sidebarTierNameEl = document.getElementById("sidebar-tier-name");
              if (sidebarTierNameEl) {
                const tierNameCap = tier.charAt(0).toUpperCase() + tier.slice(1);
                sidebarTierNameEl.textContent = tierNameCap;
              }
            }
          } catch (fetchErr) {
            console.error("[PAYMENT] Error refreshing tier data:", fetchErr);
          }

          console.log("[PAYMENT] ✅ UI fully refreshed with new tier");
          DebugTracer.exit('handlePostPaymentRedirect', _paymentStart, { result: 'success' });
          return;
        }

        // Handle upgrade cancellation (user closed Stripe checkout)
        if (upgradeStatus === "cancel") {
          DebugTracer.snapshot('payment_branch', { branch: 'cancel' });
          console.log(`[PAYMENT] Upgrade cancelled by user, time=${new Date().toISOString()}`);
          if (usageWarning) {
            usageWarning.textContent = "Upgrade cancelled. You can try again anytime.";
            usageWarning.classList.remove("success", "error");
            usageWarning.classList.add("info");
            usageWarning.style.display = "block";
          }
          window.history.replaceState({}, document.title, "/ui");
          try {
            await fetchTierData();
            await updateAuthStatus();
          } catch (e) {
            console.error("[PAYMENT] Error refreshing after cancel:", e);
            DebugTracer.error('handlePostPaymentRedirect_cancel', e);
          }
          DebugTracer.exit('handlePostPaymentRedirect', _paymentStart, { result: 'cancel' });
          return;
        }

        // Handle upgrade failure
        if (upgradeStatus === "failed" || upgradeStatus === "error") {
          DebugTracer.snapshot('payment_branch', { branch: 'failed' });
          console.log(`[PAYMENT] ❌ Upgrade failed: message=${message}, time=${new Date().toISOString()}`);
          if (usageWarning) {
            usageWarning.textContent = message || "Tier upgrade failed. Please try again or contact support.";
            usageWarning.classList.remove("success", "info");
            usageWarning.classList.add("error");
            usageWarning.style.display = "block";
          }
          window.history.replaceState({}, document.title, "/ui");
          // Still fetch tier data so UI is in sync
          try {
            await fetchTierData();
            await updateAuthStatus();
          } catch (e) {
            console.error("[PAYMENT] Error refreshing after failure:", e);
            DebugTracer.error('handlePostPaymentRedirect_failed', e);
          }
          DebugTracer.exit('handlePostPaymentRedirect', _paymentStart, { result: 'failed' });
          return;
        }

        // ORIGINAL: Legacy flow with session_id (Enterprise pending or old tier)
        if (sessionId && username) {
          DebugTracer.snapshot('payment_branch', { branch: 'legacy_session', sessionId, username });
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
              if (usageWarning) {
                usageWarning.textContent = "Error: Invalid payment redirect parameters";
                usageWarning.classList.add("error");
              }
              return;
            }
            console.log(
              `[DEBUG] Fetching ${endpoint}?${query}, time=${new Date().toISOString()}`
            );
            const response = await fetchWithRetry(`${endpoint}?${query}`, {
              method: "GET",
              headers: {
                Accept: "application/json",
                "Cache-Control": "no-cache",
              }
            }, 3, 2000);
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

            // Show immediate success feedback
            const successMsg = tempId ? "Enterprise audit completed!" : "🎉 Tier upgrade completed!";
            if (usageWarning) {
              usageWarning.textContent = successMsg;
              usageWarning.classList.remove("error", "info");
              usageWarning.classList.add("success");
              usageWarning.style.display = "block";
            }

            console.log(
              `[PAYMENT] ✅ Post-payment completed: endpoint=${endpoint}, time=${new Date().toISOString()}`
            );

            // Clean URL immediately
            window.history.replaceState({}, document.title, "/ui");

            // Small delay to ensure DB is fully committed
            await new Promise(resolve => setTimeout(resolve, 300));

            // Refresh all UI data
            console.log("[PAYMENT] Refreshing tier data after checkout completion...");
            try {
              await fetchTierData();
              await updateAuthStatus();
            } catch (refreshErr) {
              console.error("[PAYMENT] Error refreshing after checkout:", refreshErr);
            }

            console.log("[PAYMENT] ✅ UI fully refreshed");
            DebugTracer.exit('handlePostPaymentRedirect', _paymentStart, { result: 'legacy_success' });
          } catch (error) {
            DebugTracer.error('handlePostPaymentRedirect_legacy', error);
            console.error(
              `[ERROR] Post-payment redirect error: ${
                error.message
              }, endpoint=${endpoint}, time=${new Date().toISOString()}`
            );
            if (usageWarning) {
              usageWarning.textContent = `Error completing ${
                tempId ? "Enterprise audit" : "tier upgrade"
              }: ${error.message}`;
              usageWarning.classList.add("error");
            }
            if (
              error.message.includes("User not found") ||
              error.message.includes("Please login")
            ) {
              console.log(
                `[DEBUG] Redirecting to /auth due to user not found, time=${new Date().toISOString()}`
              );
              DebugTracer.exit('handlePostPaymentRedirect', _paymentStart, { result: 'redirect_to_auth' });
              window.location.href = "/auth?redirect_reason=post_payment";
              return;
            }
            DebugTracer.exit('handlePostPaymentRedirect', _paymentStart, { result: 'legacy_error' });
          }
        } else {
          DebugTracer.snapshot('payment_branch', { branch: 'no_params' });
          console.warn(
            `[DEBUG] No post-payment redirect params found: session_id=${sessionId}, username=${username}, time=${new Date().toISOString()}`
          );
          DebugTracer.exit('handlePostPaymentRedirect', _paymentStart, { result: 'no_params' });
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
          const response = await fetchWithRetry(
            `/facets/${contractAddress}?username=${encodeURIComponent(
              username
            )}&_=${Date.now()}`,
            {
              method: "GET",
              headers: {
                Accept: "application/json",
                "Cache-Control": "no-cache",
              }
            }, 2, 1000
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
                                <td>${escapeHtml(facet.facetAddress || '')}</td>
                                <td>${escapeHtml((facet.functionSelectors || []).join(", "))}</td>
                                <td>${escapeHtml((facet.functions || []).join(", "))}</td>
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

        // Persist form state for recovery after navigation
        FormStateManager.saveField('contractAddress', address);

        if (address && address.match(/^0x[a-fA-F0-9]{40}$/)) {
          fetchFacetPreview(address);
        } else {
          facetWell.textContent = "";
        }
      });

      // Persist custom report input as user types
      // Note: customReportInput already destructured from els, use the element directly
      const customReportEl = document.getElementById('custom_report');
      customReportEl?.addEventListener('input', (e) => {
        FormStateManager.saveField('customReport', e.target.value);
      });

      // Persist file selection info
      const fileInput = document.getElementById('file');
      fileInput?.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
          FormStateManager.saveField('fileName', file.name);
          FormStateManager.saveField('fileSize', file.size);
        }
      });

      // Restore form state on page load
      FormStateManager.restore();

      // ============================================================================
      // API KEY SELECTOR - Populate dropdown for audit assignment
      // ============================================================================

      /**
       * Populates the API key selector dropdown for Pro/Enterprise users.
       * Called when tier data is loaded and user has Pro/Enterprise tier.
       */
      const populateApiKeySelector = async () => {
        const selector = document.getElementById('api_key_select');
        if (!selector) return;

        try {
          const response = await fetchWithRetry('/api/keys', {
            headers: { 'Accept': 'application/json' }
          }, 2, 1000);

          if (!response.ok) {
            // User might not have API key access yet - hide selector silently
            const selectorGroup = document.querySelector('.api-key-selector-group');
            if (selectorGroup) selectorGroup.style.display = 'none';
            return;
          }

          const data = await response.json();
          const { keys } = data;

          // Clear and rebuild options
          selector.innerHTML = '<option value="">-- No Assignment (Personal) --</option>';

          if (keys && keys.length > 0) {
            keys.forEach(key => {
              const option = document.createElement('option');
              option.value = key.id;
              option.textContent = `${key.label} (${key.audit_count || 0} audits)`;
              selector.appendChild(option);
            });

            // Show "Create new key" link handler
            const createKeyLink = document.getElementById('create-key-from-audit');
            if (createKeyLink) {
              createKeyLink.addEventListener('click', (e) => {
                e.preventDefault();
                // Open settings modal to API keys section
                const settingsLink = document.getElementById('sidebar-settings-link');
                if (settingsLink) settingsLink.click();
              });
            }

            debugLog('[API_KEY_SELECTOR] Populated with', keys.length, 'keys');
          } else {
            // No keys yet - show helpful message
            selector.innerHTML = '<option value="">No API keys yet - create one in Settings</option>';
            debugLog('[API_KEY_SELECTOR] No keys available');
          }

        } catch (error) {
          console.error('[API_KEY_SELECTOR] Failed to load API keys:', error);
          // Hide selector on error
          const selectorGroup = document.querySelector('.api-key-selector-group');
          if (selectorGroup) selectorGroup.style.display = 'none';
        }
      };

      // Section9: Tier Management
      // Cache to prevent redundant API calls
      let tierDataCache = null;
      let tierDataCacheTime = 0;
      const TIER_CACHE_TTL = 30000; // 30 seconds

      const fetchTierData = async (forceRefresh = false) => {
        const _tierStart = DebugTracer.enter('fetchTierData');

        // Return cached data if fresh
        const now = Date.now();
        if (!forceRefresh && tierDataCache && (now - tierDataCacheTime) < TIER_CACHE_TTL) {
          debugLog('[TierData] Returning cached tier data');
          DebugTracer.exit('fetchTierData', _tierStart, { cached: true });
          return tierDataCache;
        }

        try {
          const user = await fetchUsername();
          DebugTracer.snapshot('fetchTierData_user', { user: user?.username || 'null' });
          const username = user?.username || "";
          const url = username
            ? `/tier?username=${encodeURIComponent(username)}`
            : "/tier";

          // Use fetchWithRetry for resilience
          const response = await fetchWithRetry(url, {
            headers: { Accept: "application/json" },
          }, 3, 1000);

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || "Failed to fetch tier data");
          }
          const data = await response.json();

          // Cache the response
          tierDataCache = data;
          tierDataCacheTime = now;
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

            // Update sidebar to show "Not Signed In" (with null checks)
            if (sidebarTierName) sidebarTierName.textContent = "Not Signed In";
            if (sidebarTierUsage) sidebarTierUsage.textContent = "—";
            if (sidebarTierFeatures) sidebarTierFeatures.innerHTML = '<div class="feature-item">Sign in to access features</div>';

            // Update main tier description (with null checks)
            if (tierDescription) tierDescription.textContent = "Sign in to start auditing smart contracts";
            if (sizeLimit) sizeLimit.textContent = "Max file size: N/A";
            if (features) features.textContent = "Features: Sign in required";

            // Show warning message (with null checks)
            if (usageWarning) {
              usageWarning.textContent = "Please sign in to audit smart contracts";
              usageWarning.classList.remove("error", "success");
              usageWarning.classList.add("info");
              usageWarning.style.display = "block";
            }
            
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

          if (sidebarTierName) sidebarTierName.textContent = tierNameCap;

          if (sidebarTierUsage) {
            if (audit_limit === 9999 || size_limit === "Unlimited") {
              sidebarTierUsage.textContent = "Unlimited audits";
            } else {
              const remaining = auditLimit - auditCount;
              sidebarTierUsage.textContent = `${remaining} audits remaining (${auditCount}/${auditLimit} used)`;
            }
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
              "API access (5 keys)",
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

          if (sidebarTierFeatures) {
            sidebarTierFeatures.innerHTML = featuresList
              .map(f => `<div class="feature-item">✓ ${f}</div>`)
              .join("");
          }

          // Update main tier description with professional copy
          if (tierDescription) {
            tierDescription.textContent = `${displayName} Plan: ${
              tier === "enterprise"
                ? "Protocol-grade security with Slither, Mythril & Echidna. Formal Verification coming soon. White-label reports, unlimited API, dedicated support."
                : tier === "pro"
                ? "Unlimited audits with full security stack. Fuzzing, on-chain analysis, API access, priority support."
                : tier === "starter"
                ? `25 audits/month with AI-powered analysis & compliance scoring. (${auditCount}/${auditLimit} used)`
                : `Trial plan: 1 audit/month with basic analysis. (${auditCount}/${auditLimit} used)`
            }`;
          }

          if (sizeLimit) sizeLimit.textContent = `Max file size: ${size_limit}`;
          if (features) features.textContent = `Features: ${featuresList.join(", ")}`;

          if (usageWarning) {
            usageWarning.textContent =
              tier === "free" || tier === "starter"
                ? `${tier.charAt(0).toUpperCase() + tier.slice(1)} tier: ${auditCount}/${auditLimit} audits remaining`
                : "";
            usageWarning.classList.remove("error");
          }

          if (upgradeLink) upgradeLink.style.display = tier !== "enterprise" ? "inline-block" : "none";

          // =========================================================================
          // TIER-AWARE UI CLEANING: Hide irrelevant upgrade prompts for paid users
          // Psychological principle: Reduce cognitive load, show only relevant options
          // =========================================================================
          const pricingTable = document.getElementById("pricing-table");
          const upgradePrompt = document.querySelector(".upgrade-prompt");
          const tierSelector = document.getElementById("tier-select");
          const tierSwitchBtn = document.getElementById("tier-switch");

          // Enterprise users: Hide ALL upgrade prompts (they're at the top!)
          if (tier === "enterprise") {
            if (pricingTable) pricingTable.style.display = "none";
            if (tierSelector) tierSelector.parentElement.style.display = "none";
            if (tierSwitchBtn) tierSwitchBtn.style.display = "none";
            // Show appreciation message instead
            if (upgradePrompt) {
              upgradePrompt.innerHTML = `
                <div class="enterprise-appreciation" style="text-align: center; padding: var(--space-6);">
                  <h3 style="color: var(--accent-purple);">💼 Enterprise Plan Active</h3>
                  <p style="color: var(--text-secondary); margin-top: var(--space-3);">
                    You have access to our complete security suite: Slither, Mythril, Echidna,
                    formal verification, multi-AI consensus, and white-label reports.
                  </p>
                  <p style="color: var(--green); font-weight: 600; margin-top: var(--space-2);">
                    ✓ Priority Queue Position: <strong>1st</strong> | ✓ Unlimited Audits | ✓ Dedicated Support
                  </p>
                </div>
              `;
            }
          }
          // Pro users: Only show Enterprise upgrade (not Developer)
          else if (tier === "pro") {
            if (pricingTable) {
              // Hide Free, Developer, and Team rows - only show Enterprise
              const rows = pricingTable.querySelectorAll("tbody tr");
              rows.forEach((row, idx) => {
                // Keep only Enterprise row (last one, idx=3)
                row.style.display = idx === 3 ? "table-row" : "none";
              });
            }
            if (tierSelector) {
              // Only show Enterprise option - use disabled+hidden for cross-browser support
              Array.from(tierSelector.options).forEach(opt => {
                if (opt.value === "enterprise") {
                  opt.disabled = false;
                  opt.hidden = false;
                  opt.style.display = "block";
                  opt.selected = true;
                } else {
                  opt.disabled = true;
                  opt.hidden = true;
                  opt.style.display = "none";
                }
              });
            }
            // Update header messaging
            const promptHeader = upgradePrompt?.querySelector("h3");
            if (promptHeader) {
              promptHeader.innerHTML = "🚀 Unlock Protocol-Grade Security";
            }
          }
          // Starter users: Show Pro and Enterprise (not Free)
          else if (tier === "starter") {
            if (pricingTable) {
              const rows = pricingTable.querySelectorAll("tbody tr");
              rows.forEach((row, idx) => {
                // Hide Free row (first one) and Developer row (second one - they have it)
                row.style.display = idx <= 1 ? "none" : "table-row";
              });
            }
            if (tierSelector) {
              // Hide starter option (they already have it) and select pro as default
              Array.from(tierSelector.options).forEach(opt => {
                if (opt.value === "starter") {
                  opt.disabled = true;
                  opt.hidden = true;
                  opt.style.display = "none";
                } else {
                  opt.disabled = false;
                  opt.hidden = false;
                  opt.style.display = "block";
                }
                // Default to pro for starter users
                if (opt.value === "pro") opt.selected = true;
              });
            }
          }
          // Free users: Show all options with value anchoring
          else {
            // Add value anchoring message for free users
            const valueAnchor = document.getElementById("value-anchor-message");
            if (!valueAnchor && upgradePrompt) {
              const anchorDiv = document.createElement("div");
              anchorDiv.id = "value-anchor-message";
              anchorDiv.className = "value-anchor";
              anchorDiv.innerHTML = `
                <div style="background: linear-gradient(135deg, rgba(155, 89, 182, 0.15), rgba(52, 152, 219, 0.15));
                            border-radius: 12px; padding: var(--space-4); margin-bottom: var(--space-4);
                            border: 1px solid var(--accent-purple);">
                  <p style="font-size: var(--text-lg); font-weight: 600; color: var(--text-primary); margin: 0;">
                    💡 <strong>Traditional audits cost $15,000 - $70,000+</strong>
                  </p>
                  <p style="color: var(--text-secondary); margin-top: var(--space-2); font-size: var(--text-sm);">
                    Get the same vulnerability detection for <strong style="color: var(--green);">99.6% less</strong>.
                    Our AI-powered platform uses the exact same tools (Slither, Mythril, Echidna) trusted by
                    CertiK, Trail of Bits, and ConsenSys Diligence.
                  </p>
                </div>
              `;
              upgradePrompt.insertBefore(anchorDiv, upgradePrompt.firstChild);
            }
          }

          maxFileSize =
            size_limit === "Unlimited"
              ? Infinity
              : parseFloat(size_limit.replace("MB", "")) * 1024 * 1024;

          const fileHelpEl = document.querySelector("#file-help");
          if (fileHelpEl) fileHelpEl.textContent = `Max size: ${size_limit}. Ensure code is valid Solidity.`;

          document
            .querySelectorAll(".pro-enterprise-only")
            .forEach(
              (el) =>
                (el.style.display =
                  tier === "pro" || tier === "enterprise" ? "block" : "none")
            );

          // Populate API key selector for Pro/Enterprise users
          if (tier === "pro" || tier === "enterprise") {
            populateApiKeySelector();
          }

          if (customReportInput) {
            customReportInput.style.display =
              tier === "pro" || tier === "enterprise" ? "block" : "none";
          }

          if (downloadReportButton) {
            downloadReportButton.style.display = feature_flags.reports
              ? "block"
              : "none";
          }
          
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
          DebugTracer.exit('fetchTierData', _tierStart, { tier, auditCount, auditLimit });
        } catch (error) {
          DebugTracer.error('fetchTierData', error);
          console.error("Tier fetch error:", error);
          if (usageWarning) {
            usageWarning.textContent = `Error fetching tier data: ${error.message}`;
            usageWarning.classList.add("error");
          }
          DebugTracer.exit('fetchTierData', _tierStart, { error: error.message });
        }
      };

      tierSwitchButton?.addEventListener("click", () => {
        // Show loading state immediately
        const originalButtonText = tierSwitchButton.textContent;
        tierSwitchButton.textContent = "⏳ Redirecting to checkout...";
        tierSwitchButton.disabled = true;

        withCsrfToken(async (token) => {
          try {
            if (!token) {
              throw new Error("Unable to establish secure connection. Please refresh the page.");
            }

            const selectedTier = tierSelect?.value;
            console.log(`[DEBUG] Upgrade button clicked, selectedTier=${selectedTier}, time=${new Date().toISOString()}`);

            if (!selectedTier) {
              throw new Error("Tier selection unavailable. Please refresh the page.");
            }

            if (!["starter", "pro", "enterprise"].includes(selectedTier)) {
              throw new Error(`Invalid tier '${selectedTier}'. Choose Developer, Team, or Enterprise.`);
            }

            const user = await fetchUsername();
            if (!user?.username) {
              console.error("[ERROR] No username found, redirecting to /auth");
              window.location.href = "/auth";
              return;
            }

            const username = user.username;
            console.log(`[DEBUG] Initiating tier switch: username=${username}, tier=${selectedTier}, time=${new Date().toISOString()}`);

            const requestBody = JSON.stringify({
              username: username,
              tier: selectedTier
            });

            console.log(`[DEBUG] Sending /create-tier-checkout request with body: ${requestBody}`);

            const response = await fetchWithRetry("/create-tier-checkout", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": token,
                Accept: "application/json",
              },
              body: requestBody,
            }, 3, 2000);

            console.log(`[DEBUG] /create-tier-checkout response status: ${response.status}, ok: ${response.ok}`);

            if (!response.ok) {
              const errorData = await response.json().catch(() => ({}));
              console.error(`[ERROR] /create-tier-checkout failed: status=${response.status}, detail=${errorData.detail || "Unknown error"}`);
              throw new Error(errorData.detail || `Checkout failed (${response.status}). Please try again.`);
            }

            const data = await response.json();
            console.log(`[DEBUG] Stripe checkout session response:`, data);

            if (!data.session_url) {
              console.error("[ERROR] No session_url in response:", data);
              throw new Error("Checkout session created but no redirect URL received. Please try again.");
            }

            console.log(`[DEBUG] Redirecting to Stripe: ${data.session_url}`);
            window.location.href = data.session_url;

          } catch (error) {
            console.error(`[ERROR] Tier switch error: ${error.message}`, error);

            // Show error to user
            if (usageWarning) {
              usageWarning.textContent = `Upgrade error: ${error.message}`;
              usageWarning.classList.add("error");
              usageWarning.style.display = "block";
            }

            // Reset button state
            tierSwitchButton.textContent = originalButtonText;
            tierSwitchButton.disabled = false;
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
            const response = await fetchWithRetry(
              `/enterprise-audit?username=${encodeURIComponent(username)}`,
              {
                method: "POST",
                headers: {
                  "X-CSRFToken": token
                },
                body: formData
              }, 2, 2000
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

            // Security: Escape all on-chain data to prevent XSS
            const proxyType = escapeHtml(proxy.proxy_type || 'Unknown');
            const upgradeRisk = escapeHtml(proxy.upgrade_risk || 'N/A');
            const implementation = escapeHtml(proxy.implementation || '');
            proxyContent.innerHTML = `
              <div class="onchain-status">
                <span class="onchain-status-indicator ${statusClass}"></span>
                <span class="onchain-status-text">${isProxy ? proxyType + ' Proxy' : 'Not a Proxy'}</span>
              </div>
              ${isProxy ? `
                <div class="onchain-detail">
                  <div class="onchain-detail-row">
                    <span class="onchain-detail-label">Upgrade Risk:</span>
                    <span class="onchain-detail-value">${upgradeRisk}</span>
                  </div>
                  ${proxy.implementation ? `
                    <div class="onchain-detail-row">
                      <span class="onchain-detail-label">Implementation:</span>
                    </div>
                    <div class="onchain-address">${implementation}</div>
                  ` : ''}
                </div>
              ` : '<div class="onchain-detail">Contract code is immutable</div>'}
            `;
          }

          // Storage/Ownership
          const storageContent = document.getElementById("onchain-storage-content");
          if (storageContent && onchain.storage) {
            const storage = onchain.storage;
            const centralRisk = escapeHtml(storage.centralization_risk || 'LOW');
            const statusClass = (storage.centralization_risk === 'HIGH' || storage.centralization_risk === 'CRITICAL') ? 'danger' : (storage.centralization_risk === 'MEDIUM' ? 'warning' : 'safe');
            const ownerAddress = escapeHtml(storage.owner || '');

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
                  <div class="onchain-address">${ownerAddress}</div>
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
            const riskLevel = escapeHtml(backdoors.risk_level || 'LOW');
            const statusClass = (backdoors.risk_level === 'CRITICAL' || backdoors.risk_level === 'HIGH') ? 'danger' : (backdoors.risk_level === 'MEDIUM' ? 'warning' : 'safe');

            backdoorsContent.innerHTML = `
              <div class="onchain-status">
                <span class="onchain-status-indicator ${statusClass}"></span>
                <span class="onchain-status-text">${hasBackdoors ? riskLevel + ' Risk Detected' : 'No Backdoors Found'}</span>
              </div>
              ${backdoors.dangerous_functions && backdoors.dangerous_functions.length > 0 ? `
                <ul class="onchain-danger-list">
                  ${backdoors.dangerous_functions.slice(0, 3).map(fn => `
                    <li class="onchain-danger-item">
                      <code>${escapeHtml(fn.name || fn.selector || '')}</code>
                      <span>(${escapeHtml(fn.category || 'unknown')})</span>
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
            const confidence = escapeHtml(honeypot.confidence || 'LOW');
            const statusClass = isHoneypot ? (honeypot.confidence === 'HIGH' ? 'danger' : 'warning') : 'safe';
            const recommendation = escapeHtml(honeypot.recommendation || '');

            honeypotContent.innerHTML = `
              <div class="onchain-status">
                <span class="onchain-status-indicator ${statusClass}"></span>
                <span class="onchain-status-text">${isHoneypot ? '⚠️ Honeypot (' + confidence + ')' : '✅ Not a Honeypot'}</span>
              </div>
              <div class="onchain-detail">
                ${honeypot.recommendation ? '<p style="margin: 0; font-size: 0.75rem;">' + recommendation + '</p>' : ''}
                ${honeypot.indicators && honeypot.indicators.length > 0 ? `
                  <ul class="onchain-danger-list">
                    ${honeypot.indicators.slice(0, 2).map(ind => `
                      <li class="onchain-danger-item">${escapeHtml(ind.description || ind.type || '')}</li>
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
                <span class="onchain-risk-badge ${escapeHtml(level)}">${escapeHtml(overall.level || 'N/A')}</span>
                <span class="onchain-risk-score">${overall.score || 0}/100</span>
              </div>
              <div class="onchain-risk-summary">
                ${escapeHtml(overall.summary || 'On-chain analysis complete.')}
              </div>
            `;
          }

          // Show the section
          onchainSection.style.display = "block";
          log('ONCHAIN', 'On-chain section displayed');
        } else if (onchainSection) {
          onchainSection.style.display = "none";
        }

        // FREE TIER UPGRADE PROMPT - Psychological: Loss aversion, social proof, urgency
        if (report.upgrade_prompt) {
          const upgradePromptEl = document.getElementById("upgrade-prompt");
          if (upgradePromptEl) {
            // Calculate value proposition
            const totalIssues = report.total_issues || report.issues?.length || 0;
            const hiddenCount = totalIssues > 3 ? totalIssues - 3 : 0;

            upgradePromptEl.innerHTML = `
              <div class="upgrade-banner" style="background: linear-gradient(135deg, rgba(231, 76, 60, 0.08), rgba(155, 89, 182, 0.12));
                                                  border: 2px solid var(--accent-purple); border-radius: 16px;
                                                  padding: var(--space-5); margin: var(--space-4) 0;">
                <div style="display: flex; align-items: flex-start; gap: var(--space-4);">
                  <div class="upgrade-icon" style="font-size: 48px; line-height: 1;">⚠️</div>
                  <div class="upgrade-content" style="flex: 1;">
                    <h3 style="color: var(--red); margin: 0 0 var(--space-2) 0; font-size: var(--text-xl);">
                      ${hiddenCount > 0 ? `${hiddenCount} Vulnerabilities Hidden` : 'Limited Analysis'}
                    </h3>
                    <p style="color: var(--text-primary); margin: 0 0 var(--space-3) 0; font-size: var(--text-base);">
                      ${escapeHtml(report.upgrade_prompt || '')}
                    </p>
                    <div style="display: flex; flex-wrap: wrap; gap: var(--space-3); align-items: center;">
                      <a href="#tier-select" class="btn btn-primary"
                         style="background: linear-gradient(135deg, var(--accent-purple), var(--accent-teal));
                                padding: 12px 24px; font-size: var(--text-base); font-weight: 700;
                                border-radius: 8px; text-decoration: none; color: white;"
                         onclick="document.getElementById('tier-select').scrollIntoView({behavior: 'smooth'})">
                        🔓 Unlock Full Report - $99/mo
                      </a>
                      <span style="color: var(--text-tertiary); font-size: var(--text-sm);">
                        <strong style="color: var(--green);">Save 99%</strong> vs traditional audits ($15,000+)
                      </span>
                    </div>
                    <p style="color: var(--text-tertiary); font-size: var(--text-xs); margin-top: var(--space-3);">
                      ✓ See all ${totalIssues} vulnerabilities | ✓ AI-powered fix recommendations |
                      ✓ PDF security report | ✓ MiCA/SEC FIT21 compliance scoring
                    </p>
                  </div>
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
          // Security: Escape all user-controlled data to prevent XSS
          const severity = (issue.severity || "unknown").toLowerCase();
          const severityDisplay = escapeHtml(issue.severity || "Unknown");
          const type = escapeHtml(issue.type || "Unknown Issue");
          const description = escapeHtml(issue.description || "No description available");
          const fix = escapeHtml(issue.fix || "No fix recommendation available");
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
                            Line ${escapeHtml(String(issue.line_number))}${issue.function_name ? ` in <code>${escapeHtml(issue.function_name)}()</code>` : ''}
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
                            <p>${escapeHtml(issue.exploit_scenario)}</p>
                          </div>
                        ` : ''}

                        ${issue.estimated_impact ? `
                          <div class="detail-section impact-estimate">
                            <strong>💰 Estimated Impact:</strong> ${escapeHtml(issue.estimated_impact)}
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
                            ${issue.code_fix.explanation ? `<p class="fix-explanation">${escapeHtml(issue.code_fix.explanation)}</p>` : ''}
                          </div>
                        ` : ''}
                        
                        ${issue.alternatives && issue.alternatives.length > 0 ? `
                          <div class="detail-section alternatives">
                            <strong>🔄 Alternative Fixes:</strong>
                            ${issue.alternatives.map((alt, altIndex) => `
                              <div class="alternative-item">
                                <strong>Option ${altIndex + 1}: ${escapeHtml(alt.approach || '')}</strong>
                                <p><strong>Pros:</strong> ${escapeHtml(alt.pros || '')}</p>
                                <p><strong>Cons:</strong> ${escapeHtml(alt.cons || '')}</p>
                                <p><strong>Gas Impact:</strong> ${escapeHtml(alt.gas_impact || '')}</p>
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
                              ${issue.references.map(ref => {
                                // Security: Validate URL to prevent javascript: XSS
                                const safeUrl = (ref.url && (ref.url.startsWith('https://') || ref.url.startsWith('http://')))
                                  ? escapeHtml(ref.url) : '#';
                                return `<li><a href="${safeUrl}" target="_blank" rel="noopener">${escapeHtml(ref.title || 'Reference')}</a></li>`;
                              }).join('')}
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
      <p>${escapeHtml(issue.description || 'No description available')}</p>
    </section>
  `;

  if (issue.line_number || issue.function_name) {
    modalContent += `
      <section>
        <h3>📍 Location</h3>
        <p><strong>Line:</strong> ${escapeHtml(String(issue.line_number || 'N/A'))}</p>
        ${issue.function_name ? `<p><strong>Function:</strong> <code>${escapeHtml(issue.function_name)}</code></p>` : ''}
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
        <p>${escapeHtml(issue.exploit_scenario)}</p>
      </section>
    `;
  }

  if (issue.estimated_impact) {
    modalContent += `
      <section>
        <h3>💰 Estimated Impact</h3>
        <p>${escapeHtml(issue.estimated_impact)}</p>
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
        ${issue.code_fix.explanation ? `<p style="margin-top: var(--space-4); color: var(--text-secondary);">${escapeHtml(issue.code_fix.explanation)}</p>` : ''}
      </section>
    `;
  }

  if (issue.alternatives && issue.alternatives.length > 0) {
    modalContent += `
      <section>
        <h3>🔄 Alternative Fixes</h3>
        ${issue.alternatives.map((alt, altIndex) => `
          <div class="alternative-fix">
            <h4>Option ${altIndex + 1}: ${escapeHtml(alt.approach || '')}</h4>
            <p><strong>Pros:</strong> ${escapeHtml(alt.pros || '')}</p>
            <p><strong>Cons:</strong> ${escapeHtml(alt.cons || '')}</p>
            <p><strong>Gas Impact:</strong> <code>${escapeHtml(alt.gas_impact || '')}</code></p>
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
          ${issue.references.map(ref => {
            // Security: Validate URL to prevent javascript: XSS
            const safeUrl = (ref.url && (ref.url.startsWith('https://') || ref.url.startsWith('http://')))
              ? escapeHtml(ref.url) : '#';
            return `<li><a href="${safeUrl}" target="_blank" rel="noopener">${escapeHtml(ref.title || 'Reference')}</a></li>`;
          }).join('')}
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
          : report.predictions.map(p => `<li tabindex="0">Scenario: ${escapeHtml(p.scenario || 'N/A')} | Impact: ${escapeHtml(p.impact || 'N/A')}</li>`).join('');
          
        // RECOMMENDATIONS - Categorized by urgency
        if (report.recommendations && typeof report.recommendations === 'object') {
          // New format with immediate/short_term/long_term
          let recHtml = '';
          
          if (report.recommendations.immediate && report.recommendations.immediate.length > 0) {
            recHtml += '<h4 class="rec-category critical">🚨 Immediate (Fix Before Deploy)</h4><ul>';
            recHtml += report.recommendations.immediate.map(r => `<li tabindex="0">${escapeHtml(r)}</li>`).join('');
            recHtml += '</ul>';
          }

          if (report.recommendations.short_term && report.recommendations.short_term.length > 0) {
            recHtml += '<h4 class="rec-category warning">⚠️ Short-Term (Next 7 Days)</h4><ul>';
            recHtml += report.recommendations.short_term.map(r => `<li tabindex="0">${escapeHtml(r)}</li>`).join('');
            recHtml += '</ul>';
          }

          if (report.recommendations.long_term && report.recommendations.long_term.length > 0) {
            recHtml += '<h4 class="rec-category info">💡 Long-Term (Future Improvements)</h4><ul>';
            recHtml += report.recommendations.long_term.map(r => `<li tabindex="0">${escapeHtml(r)}</li>`).join('');
            recHtml += '</ul>';
          }

          recommendationsList.innerHTML = recHtml || '<li>No recommendations available.</li>';
        } else if (Array.isArray(report.recommendations)) {
          // Legacy format - array of strings
          recommendationsList.innerHTML = report.recommendations.length === 0
            ? "<li>No recommendations available.</li>"
            : report.recommendations.map(r => `<li tabindex="0">${escapeHtml(r)}</li>`).join('');
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
                  <span class="status-text">${escapeHtml(parsed.execution_summary || 'Fuzzing Complete')}</span>
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
                  ${parsed.function_tests.map(test => {
                    // Validate icon - only allow safe emoji icons
                    const safeIcons = ['✅', '❌', '⚠️', '🔄', '⏳', '✓', '✗'];
                    const testIcon = safeIcons.includes(test.icon) ? test.icon : (test.passed ? '✅' : '❌');
                    return `
                    <div class="function-test-item ${test.passed ? 'passed' : 'failed'}">
                      <span class="test-icon">${testIcon}</span>
                      <code class="test-name">${escapeHtml(test.function)}</code>
                      <span class="test-status">${escapeHtml(test.status || '')}</span>
                    </div>
                  `}).join('')}
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
            // Validate severity - only allow known severity levels
            const allowedSeverities = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'];
            const rawSeverity = (result.severity || 'HIGH').toUpperCase();
            const severity = allowedSeverities.includes(rawSeverity) ? rawSeverity : 'HIGH';
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
                  <span class="step-content">${escapeHtml(step.trim())}</span>
                </li>
              `)
              .join('');
          } else {
            // Fallback if parsing fails - escape for XSS protection
            remediationRoadmap.innerHTML = `<li tabindex="0">${escapeHtml(roadmapText)}</li>`;
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
              const desc = escapeHtml(issue.description || '');
              const issueType = escapeHtml(issue.type || issue.title || 'Issue');
              const needsExpand = desc.length > 80; // Show expand hint if description is long
              return `
                <div class="mobile-issue-item ${escapeHtml(severity)}" data-issue-index="${index}" onclick="this.classList.toggle('expanded')">
                  <span class="mobile-issue-severity">${severityIcon[severity] || '🔵'}</span>
                  <div class="mobile-issue-content">
                    <div class="mobile-issue-type">${issueType}</div>
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
        LoadingManager.hide(loading);
        if (resultsDiv) {
          resultsDiv.classList.add("show");
          resultsDiv.focus();
          resultsDiv.scrollIntoView({ behavior: "smooth" });
        }
        logMessage(`Audit complete – risk score ${report.risk_score}`);
      };

      // Listen for retrieved audit results (from Access Key retrieval) to populate UI
      // This bridges the retrieveAuditByKey function (global) with handleAuditResponse (local)
      window.addEventListener('retrievedAuditComplete', (event) => {
        debugLog('[AUDIT_RETRIEVE] Received retrievedAuditComplete event:', event.detail);
        handleAuditResponse(event.detail);
      });

      // Flag to prevent double submission
      let isSubmitting = false;

      const handleSubmit = (e) => {
        e.preventDefault();

        // Prevent double submission
        if (isSubmitting) {
          debugLog('[AUDIT] Submission already in progress, ignoring duplicate click');
          return;
        }
        isSubmitting = true;

        // Disable submit button visually
        const submitBtn = auditForm.querySelector('button[type="submit"]');
        if (submitBtn) {
          submitBtn.disabled = true;
          submitBtn.dataset.originalText = submitBtn.textContent;
          submitBtn.textContent = 'Submitting...';
        }

        withCsrfToken(async (token) => {
          // Use LoadingManager for auto-timeout safety
          LoadingManager.show(loading, 10 * 60 * 1000); // 10 min timeout for audits
          if (resultsDiv) resultsDiv.classList.remove("show");
          usageWarning.textContent = "";
          usageWarning.classList.remove("error", "success");

          const file = auditForm.querySelector("#file")?.files[0];
          if (!file) {
            LoadingManager.hide(loading);
            usageWarning.textContent = "Please select a file";
            usageWarning.classList.add("error");
            return;
          }

          // Save audit key for state persistence
          AuditStateManager.setCurrentAudit(null); // Will be set when we get the key

          const username = (await fetchUsername())?.username || "guest";
          const formData = new FormData(auditForm);

          // Get optional API key assignment (Pro/Enterprise feature)
          const apiKeySelect = document.getElementById('api_key_select');
          const selectedApiKeyId = apiKeySelect?.value || '';

          try {
            logMessage("Submitting audit...");

            // Build URL with optional api_key_id parameter
            let auditUrl = `/audit?username=${encodeURIComponent(username)}`;
            if (selectedApiKeyId) {
              auditUrl += `&api_key_id=${encodeURIComponent(selectedApiKeyId)}`;
              logMessage(`Assigning to API key: ${apiKeySelect.options[apiKeySelect.selectedIndex]?.text || selectedApiKeyId}`);
            }

            // Submit to /audit - it will either run immediately or queue
            const response = await fetchWithRetry(auditUrl, {
              method: 'POST',
              headers: { 'X-CSRFToken': token },
              body: formData
            }, 2, 3000); // 2 retries with 3s delay for file uploads
            
            if (!response.ok) {
              const errorData = await response.json().catch(() => ({}));

              // Handle 409 Conflict - One File Per Project Key Policy
              if (response.status === 409 && errorData.detail?.error === 'one_file_per_key') {
                LoadingManager.hide(loading);
                const detail = errorData.detail;

                // Show informative error with action options
                const errorHtml = `
                  <div class="one-file-per-key-error" style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 16px; margin: 16px 0;">
                    <h4 style="color: #856404; margin: 0 0 12px 0;">📁 Project Key Already Assigned</h4>
                    <p style="color: #856404; margin: 0 0 12px 0;">
                      The project key "<strong>${escapeHtml(detail.api_key_label || 'Selected Key')}</strong>" is already assigned to:
                      <strong>${escapeHtml(detail.existing_file || 'another file')}</strong>
                    </p>
                    <p style="color: #856404; margin: 0 0 16px 0;">
                      Each project key can only audit <strong>one file</strong>. You can:
                    </p>
                    <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                      <button onclick="document.getElementById('api_key_select').value=''; this.closest('.one-file-per-key-error').remove();"
                              style="background: #6c757d; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                        🔄 Remove Key Assignment
                      </button>
                      <button onclick="window.showApiKeyModal ? window.showApiKeyModal() : alert('Open Project Keys in Settings'); this.closest('.one-file-per-key-error').remove();"
                              style="background: #28a745; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                        ➕ Create New Project Key
                      </button>
                      <button onclick="window.retrieveAuditByKey && window.retrieveAuditByKey('${escapeHtml(detail.existing_audit_key || '')}'); this.closest('.one-file-per-key-error').remove();"
                              style="background: #007bff; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                        📋 View Existing Audit
                      </button>
                    </div>
                  </div>
                `;

                // Insert error message before the form or in a designated area
                const auditForm = document.getElementById('auditForm');
                const existingError = document.querySelector('.one-file-per-key-error');
                if (existingError) existingError.remove();

                if (auditForm) {
                  auditForm.insertAdjacentHTML('beforebegin', errorHtml);
                } else {
                  usageWarning.innerHTML = errorHtml;
                }

                logMessage(`One file per key: ${detail.api_key_label} already assigned to ${detail.existing_file}`);
                return;
              }

              throw new Error(typeof errorData.detail === 'string' ? errorData.detail : (errorData.detail?.message || 'Audit request failed'));
            }
            
            const data = await response.json();
            
            // Check if audit was queued or ran immediately
            if (data.queued) {
              // QUEUED: Server at capacity, use queue tracker for real-time updates
              logMessage(`Server busy - queued at position ${data.position}`);

              // Show Access Key popup immediately so user can save it
              if (data.audit_key) {
                queueTracker.showAuditKey(data.audit_key);
                logMessage(`Access Key generated: ${data.audit_key.substring(0, 20)}...`);
              }

              queueTracker.jobId = data.job_id;
              
              queueTracker.onUpdate = (status) => {
                logMessage(`Queue status: ${status.status} - ${status.current_phase || 'waiting'}`);
              };
              
              queueTracker.onComplete = async (result) => {
                logMessage("Audit complete!");
                queueTracker.hideQueueUI();
                LoadingManager.hide(loading);

                // Clear form state - audit successful, no need to restore
                FormStateManager.clear();
                AuditStateManager.clearCurrentAudit();
                TierCache.invalidate(); // Force refresh tier data

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
                LoadingManager.hide(loading);
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
              LoadingManager.hide(loading);

              // Clear form state - audit successful, no need to restore
              FormStateManager.clear();
              AuditStateManager.clearCurrentAudit();
              TierCache.invalidate(); // Force refresh tier data

              if (data.tier) window.currentAuditTier = data.tier;

              // Show Access Key popup so user can save it for later retrieval
              if (data.audit_key) {
                queueTracker.showAuditKey(data.audit_key);
                logMessage(`Access Key generated: ${data.audit_key.substring(0, 20)}...`);
              }

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
            LoadingManager.hide(loading);
            usageWarning.textContent = err.message || "Audit error";
            usageWarning.classList.add("error");
          } finally {
            // Reset submission state
            isSubmitting = false;
            const submitBtn = auditForm?.querySelector('button[type="submit"]');
            if (submitBtn) {
              submitBtn.disabled = false;
              submitBtn.textContent = submitBtn.dataset.originalText || 'Audit';
            }
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
          const response = await fetchWithRetry(fetchUrl, {}, 3, 1000);

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

          const response = await fetchWithRetry(fetchUrl, {}, 3, 1000);

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
            
            const response = await fetchWithRetry(`/mint-nft?username=${encodeURIComponent(username)}`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                "X-CSRF-Token": token,
              }
            }, 2, 2000);
            
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
        modalUpgradeButton, modalLogoutButton
      } = els;

      // Certora elements fetched dynamically (only present for enterprise/diamond tiers)
      const modalCertoraSection = document.getElementById("modal-certora-section");
      const certoraJobsTableBody = document.getElementById("certora-jobs-table-body");
      const certoraUploadBtn = document.getElementById("certora-upload-btn");
      const certoraUploadInput = document.getElementById("certora-upload-input");

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
          const tierResponse = await fetchWithRetry(tierUrl, {
            headers: { Accept: "application/json" }
          }, 2, 1000);
          
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
          
          // Member since (from backend)
          if (modalMemberSince) {
            if (user?.member_since) {
              const memberDate = new Date(user.member_since);
              modalMemberSince.textContent = memberDate.toLocaleDateString('en-US', {
                month: 'long',
                year: 'numeric'
              });
            } else {
              modalMemberSince.textContent = "N/A";
            }
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

      // Load and display all project keys
      const loadApiKeys = async () => {
        try {
          const response = await fetchWithRetry("/api/keys", {}, 2, 1000);

          if (!response.ok) throw new Error("Failed to load project keys");

          const data = await response.json();
          const { keys, active_count, max_keys, tier } = data;

          // Update count display
          if (apiKeyCountDisplay) {
            if (max_keys === null) {
              apiKeyCountDisplay.textContent = `Active Project Keys: ${active_count} (Unlimited - Enterprise)`;
            } else {
              apiKeyCountDisplay.textContent = `Active Project Keys: ${active_count}/${max_keys} (Pro Tier)`;
            }
          }

          // Update table
          if (apiKeysTableBody) {
            if (keys.length === 0) {
              apiKeysTableBody.innerHTML = `
                <tr>
                  <td colspan="6" style="padding: var(--space-6); text-align: center; color: var(--text-tertiary);">
                    No project keys yet. Create your first key to organize audits by client or project!
                  </td>
                </tr>
              `;
            } else {
              apiKeysTableBody.innerHTML = keys.map(key => `
                <tr style="border-bottom: 1px solid var(--glass-border);">
                  <td style="padding: var(--space-3); font-weight: 600; color: var(--text-primary);">
                    ${escapeHtml(key.label)}
                  </td>
                  <td style="padding: var(--space-3);">
                    <code style="font-family: 'JetBrains Mono', monospace; font-size: var(--text-sm); color: var(--accent-teal);">
                      ${escapeHtml(key.key_preview || '****')}
                    </code>
                  </td>
                  <td style="padding: var(--space-3); font-size: var(--text-sm); color: var(--text-tertiary);">
                    ${new Date(key.created_at).toLocaleDateString()}
                  </td>
                  <td style="padding: var(--space-3); font-size: var(--text-sm); color: var(--text-tertiary);">
                    ${key.last_used_at ? new Date(key.last_used_at).toLocaleDateString() : 'Never'}
                  </td>
                  <td style="padding: var(--space-3); font-size: var(--text-sm); text-align: center;">
                    <span style="background: rgba(155, 89, 182, 0.2); color: var(--accent-purple); padding: 2px 8px; border-radius: 12px; font-size: 12px;">
                      ${key.audit_count || 0} audits
                    </span>
                  </td>
                  <td style="padding: var(--space-3); text-align: right;">
                    <button class="revoke-key-btn btn btn-secondary btn-sm" data-key-id="${key.id}" data-key-label="${escapeHtml(key.label)}">
                      🗑️ Revoke
                    </button>
                  </td>
                </tr>
              `).join('');

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
                      const response = await fetchWithRetry(`/api/keys/${keyId}`, {
                        method: "DELETE",
                        headers: { "X-CSRFToken": csrfToken }
                      }, 2, 1000);

                      if (!response.ok) throw new Error("Failed to revoke key");

                      alert(`✅ "${keyLabel}" revoked successfully`);
                      await loadApiKeys(); // Reload
                    });
                  } catch (err) {
                    console.error("[ERROR] Failed to revoke:", err);
                    alert("Failed to revoke project key");
                  }
                });
              });
            }
          }

          debugLog("[DEBUG] Project keys loaded successfully");
        } catch (error) {
          console.error("[ERROR] Failed to load project keys:", error);
          if (apiKeysTableBody) {
            apiKeysTableBody.innerHTML = `
              <tr>
                <td colspan="5" style="padding: var(--space-6); text-align: center; color: var(--error);">
                  Failed to load project keys. Please try again.
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
          const response = await fetchWithRetry("/api/certora/jobs", {}, 2, 1000);

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
                // Normalize status to allowed values only
                const safeStatus = ['completed', 'running', 'pending', 'error'].includes(job.status) ? job.status : 'pending';
                const statusStyle = statusStyles[safeStatus];

                // Results display (numbers only, safe)
                let resultsHtml = '—';
                if (safeStatus === 'completed') {
                  const verified = parseInt(job.rules_verified, 10) || 0;
                  const violated = parseInt(job.rules_violated, 10) || 0;
                  resultsHtml = `
                    <span style="color: var(--green);">✓ ${verified}</span> /
                    <span style="color: var(--red);">✗ ${violated}</span>
                  `;
                } else if (safeStatus === 'running' || safeStatus === 'pending') {
                  resultsHtml = '<span style="color: var(--accent-purple);">⏳ Running...</span>';
                }

                // Escape all user-controlled data
                const safeContractName = escapeHtml(job.contract_name || 'Unknown');
                const safeJobId = escapeHtml(job.job_id || '');
                const shortJobId = safeJobId.length > 12
                  ? safeJobId.substring(0, 8) + '...'
                  : safeJobId;

                // Validate URL - only allow https:// URLs to Certora domains
                let safeJobUrl = '#';
                if (job.job_url && typeof job.job_url === 'string') {
                  try {
                    const urlObj = new URL(job.job_url);
                    if (urlObj.protocol === 'https:' &&
                        (urlObj.hostname.endsWith('certora.com') || urlObj.hostname.endsWith('prover.certora.com'))) {
                      safeJobUrl = escapeHtml(job.job_url);
                    }
                  } catch (e) {
                    // Invalid URL, use fallback
                  }
                }

                return `
                  <tr style="border-bottom: 1px solid var(--glass-border);">
                    <td style="padding: var(--space-3); font-weight: 600; color: var(--text-primary); max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                      ${safeContractName}
                    </td>
                    <td style="padding: var(--space-3);">
                      <span style="display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: var(--text-xs); text-transform: uppercase; ${statusStyle}">
                        ${safeStatus}
                      </span>
                    </td>
                    <td style="padding: var(--space-3); text-align: center; font-size: var(--text-sm);">
                      ${resultsHtml}
                    </td>
                    <td style="padding: var(--space-3);">
                      <a href="${safeJobUrl}" target="_blank" rel="noopener noreferrer"
                         style="font-family: 'JetBrains Mono', monospace; font-size: var(--text-sm); color: var(--accent-teal); text-decoration: none;"
                         title="View on Certora Dashboard: ${safeJobId}">
                        ${shortJobId} ↗
                      </a>
                    </td>
                    <td style="padding: var(--space-3); text-align: right;">
                      <button class="copy-job-id-btn btn btn-sm" data-job-id="${safeJobId}" style="margin-right: var(--space-2);">
                        📋
                      </button>
                      ${safeStatus === 'running' || safeStatus === 'pending'
                        ? `<button class="poll-job-btn btn btn-sm" data-job-id="${safeJobId}" style="margin-right: var(--space-2);">🔄</button>`
                        : ''}
                      <button class="delete-job-btn btn btn-secondary btn-sm" data-job-id="${safeJobId}" data-contract="${safeContractName}">
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
                      const response = await fetchWithRetry(`/api/certora/poll/${jobId}`, {
                        method: "POST",
                        headers: { "X-CSRFToken": csrfToken }
                      }, 2, 1000);

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
                      const response = await fetchWithRetry(`/api/certora/job/${jobId}`, {
                        method: "DELETE",
                        headers: { "X-CSRFToken": csrfToken }
                      }, 2, 1000);

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
            const response = await fetchWithRetry("/api/certora/start", {
              method: "POST",
              headers: { "X-CSRFToken": csrfToken },
              body: formData
            }, 2, 2000); // 2 retries with 2s base delay for uploads

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

      // Show Project Key Created Modal - Enterprise-grade copyable key display
      const showApiKeyCreatedModal = (label, apiKey) => {
        // Create modal dynamically
        const existingModal = document.getElementById('api-key-created-modal');
        if (existingModal) existingModal.remove();

        const modalHtml = `
          <div id="api-key-created-modal-backdrop" style="
            position: fixed; inset: 0; background: rgba(0,0,0,0.8);
            z-index: 10001; display: flex; align-items: center; justify-content: center;
          ">
            <div id="api-key-created-modal" style="
              background: var(--glass-bg, #1a1a2e);
              border: 1px solid var(--glass-border, #2d2d44);
              border-radius: var(--radius-xl, 16px);
              padding: var(--space-8, 32px);
              max-width: 600px; width: 90%;
              box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
            ">
              <div style="text-align: center; margin-bottom: var(--space-6, 24px);">
                <div style="font-size: 48px; margin-bottom: var(--space-4, 16px);">📁</div>
                <h2 style="color: var(--text-primary, #fff); margin: 0 0 8px 0; font-size: var(--text-xl, 20px);">
                  Project Key Created Successfully
                </h2>
                <p style="color: var(--text-secondary, #a0a0a0); margin: 0; font-size: var(--text-sm, 14px);">
                  Project: <strong>${escapeHtml(label)}</strong>
                </p>
              </div>

              <div style="
                background: rgba(0,0,0,0.3);
                border: 1px solid var(--glass-border, #2d2d44);
                border-radius: var(--radius-lg, 12px);
                padding: var(--space-4, 16px);
                margin-bottom: var(--space-6, 24px);
              ">
                <div style="display: flex; align-items: center; gap: var(--space-3, 12px);">
                  <code id="new-api-key-display" style="
                    flex: 1;
                    font-family: 'JetBrains Mono', monospace;
                    font-size: var(--text-sm, 14px);
                    color: var(--accent-teal, #00d4aa);
                    word-break: break-all;
                    padding: var(--space-3, 12px);
                    background: rgba(0,212,170,0.1);
                    border-radius: var(--radius-md, 8px);
                  ">${escapeHtml(apiKey)}</code>
                  <button id="copy-new-api-key" class="btn" style="
                    padding: var(--space-3, 12px) var(--space-4, 16px);
                    font-size: var(--text-base, 16px);
                    white-space: nowrap;
                  ">📋 Copy Key</button>
                </div>
              </div>

              <div style="
                background: rgba(241, 196, 15, 0.1);
                border: 1px solid rgba(241, 196, 15, 0.3);
                border-radius: var(--radius-md, 8px);
                padding: var(--space-4, 16px);
                margin-bottom: var(--space-6, 24px);
              ">
                <p style="color: #f1c40f; margin: 0; font-size: var(--text-sm, 14px); display: flex; align-items: flex-start; gap: 8px;">
                  <span style="font-size: 18px;">⚠️</span>
                  <span><strong>Important:</strong> This is the only time you'll see the full key.
                  Make sure to copy and store it securely. You won't be able to view it again.</span>
                </p>
              </div>

              <div style="text-align: center;">
                <button id="close-api-key-created-modal" class="btn" style="
                  padding: var(--space-3, 12px) var(--space-6, 24px);
                ">Done</button>
              </div>
            </div>
          </div>
        `;

        document.body.insertAdjacentHTML('beforeend', modalHtml);
        document.body.style.overflow = 'hidden';

        // Copy button handler
        document.getElementById('copy-new-api-key').addEventListener('click', async () => {
          try {
            await navigator.clipboard.writeText(apiKey);
            const btn = document.getElementById('copy-new-api-key');
            btn.textContent = '✅ Copied!';
            btn.style.background = 'var(--green, #27ae60)';
            setTimeout(() => {
              btn.textContent = '📋 Copy Key';
              btn.style.background = '';
            }, 2000);
          } catch (err) {
            console.error('[ERROR] Failed to copy API key:', err);
            alert('Failed to copy. Please select and copy manually.');
          }
        });

        // Close modal handler
        const closeApiKeyCreatedModal = () => {
          const modal = document.getElementById('api-key-created-modal-backdrop');
          if (modal) {
            modal.remove();
            document.body.style.overflow = 'auto';
          }
        };

        document.getElementById('close-api-key-created-modal').addEventListener('click', closeApiKeyCreatedModal);
        document.getElementById('api-key-created-modal-backdrop').addEventListener('click', (e) => {
          if (e.target.id === 'api-key-created-modal-backdrop') {
            closeApiKeyCreatedModal();
          }
        });

        // ESC key to close
        const escHandler = (e) => {
          if (e.key === 'Escape') {
            closeApiKeyCreatedModal();
            document.removeEventListener('keydown', escHandler);
          }
        };
        document.addEventListener('keydown', escHandler);
      };

      createKeyModalClose?.addEventListener("click", closeCreateKeyModal);
      createKeyCancel?.addEventListener("click", closeCreateKeyModal);
      createKeyModalBackdrop?.addEventListener("click", closeCreateKeyModal);

      // Confirm Create Project Key
      createKeyConfirm?.addEventListener("click", async () => {
        const label = newKeyLabelInput?.value.trim();

        if (!label) {
          alert("Please enter a project/client name for your key");
          return;
        }

        try {
          createKeyConfirm.disabled = true;
          createKeyConfirm.textContent = "Creating...";

          await withCsrfToken(async (csrfToken) => {
            const response = await fetchWithRetry("/api/keys/create", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": csrfToken
              },
              body: JSON.stringify({ label })
            }, 2, 1000);

            if (!response.ok) {
              const error = await response.json();
              throw new Error(error.detail || "Failed to create project key");
            }

            const data = await response.json();

            closeCreateKeyModal();

            // Show the full key in a copyable modal (enterprise-grade UX)
            showApiKeyCreatedModal(data.label, data.api_key);

            // Reload keys table
            await loadApiKeys();

            createKeyConfirm.disabled = false;
            createKeyConfirm.textContent = "Create Key";
          });

        } catch (error) {
          console.error("[ERROR] Failed to create project key:", error);
          alert(`❌ Failed to create project key: ${error.message}`);
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
          const response = await fetchWithRetry("/legal/status", {}, 2, 1000);
          
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
            const response = await fetchWithRetry("/legal/accept", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": csrfToken
              },
              body: JSON.stringify({
                accepted_terms: acceptTermsCheckbox.checked,
                accepted_privacy: acceptPrivacyCheckbox.checked
              })
            }, 2, 1000);
            
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

      // ═══════════════════════════════════════════════════════════════════════
      // CRITICAL FIX: Handle post-payment redirect FIRST, before fetching tier
      // This ensures users see their upgraded tier immediately after Stripe payment
      // ═══════════════════════════════════════════════════════════════════════
      const _initPaymentStart = DebugTracer.enter('init_payment_or_tier');
      try {
        const urlParams = new URLSearchParams(window.location.search);
        const hasPaymentRedirect = urlParams.has("upgrade") || urlParams.has("session_id");

        DebugTracer.snapshot('init_decision', {
          hasPaymentRedirect,
          upgrade: urlParams.get("upgrade"),
          session_id: urlParams.get("session_id")
        });

        if (hasPaymentRedirect) {
          // User is returning from Stripe - handle payment completion FIRST
          debugLog("[PAYMENT] Post-payment redirect detected, handling before tier fetch");
          DebugTracer.snapshot('init_path', { path: 'payment_redirect' });
          await handlePostPaymentRedirect();
          // handlePostPaymentRedirect already calls fetchTierData() and updateAuthStatus()
        } else {
          // Normal page load - fetch tier data normally
          DebugTracer.snapshot('init_path', { path: 'normal_load' });
          await fetchTierData();
          await updateAuthStatus();
        }
        DebugTracer.exit('init_payment_or_tier', _initPaymentStart, { success: true });
      } catch (initErr) {
        DebugTracer.error('init_payment_or_tier', initErr);
        console.error("[INIT] Error during tier/payment initialization:", initErr);
        DebugTracer.exit('init_payment_or_tier', _initPaymentStart, { error: initErr.message });
        // Still continue with rest of initialization
      }

      // Calm, efficient auth+tier refresh every 30 seconds (no spam)
      if (window.authTierInterval) clearInterval(window.authTierInterval);
      window.authTierInterval = setInterval(async () => {
        try {
          await updateAuthStatus();
          await fetchTierData();
        } catch (e) {
          console.error("[REFRESH] Error during periodic refresh:", e);
        }
      }, 30000);

      // DEBUG: Exit the main callback and generate report (only when debugging enabled)
      if (DEBUG_MODE || DebugTracer.enabled) {
        DebugTracer.exit('waitForDOM_callback', _waitForDOMStart, { complete: true });

        // Generate debug report after small delay to ensure all async operations complete
        setTimeout(() => {
          console.log('%c════════════════════════════════════════════════════════════', 'color: #ff00ff');
          console.log('%c🔍 UI INITIALIZATION DEBUG REPORT', 'color: #ff00ff; font-weight: bold; font-size: 14px');
          console.log('%c════════════════════════════════════════════════════════════', 'color: #ff00ff');

          // Check hamburger state
          const hamburgerEl = document.getElementById('hamburger');
          const sidebarEl = document.getElementById('sidebar');
          console.log('%c📱 HAMBURGER STATE:', 'color: #00aaff; font-weight: bold', {
            hamburgerExists: !!hamburgerEl,
            hamburgerInitialized: hamburgerEl?._debugInitialized || false,
            hamburgerClickable: hamburgerEl ? typeof hamburgerEl.onclick === 'function' || hamburgerEl._debugInitialized : false,
            sidebarExists: !!sidebarEl,
            sidebarHasOpenClass: sidebarEl?.classList.contains('open') || false
          });

          // Test hamburger click
          if (hamburgerEl) {
            console.log('%c🧪 Hamburger element found, event listeners attached:', 'color: #00ff00',
              hamburgerEl._debugInitialized ? 'YES' : 'NO');
          } else {
            console.error('%c❌ CRITICAL: Hamburger element NOT FOUND in DOM!', 'color: #ff0000; font-weight: bold');
          }

          DebugTracer.report();
        }, 2000);
      }

    } // Closing brace for waitForDOM callback
  ); // Closing parenthesis for waitForDOM function call

  // FINAL DUPLICATE AUTH CHECK KILLER – NO MORE MULTIPLES EVER
  if (window.authCheckInterval) {
    clearInterval(window.authCheckInterval);
    debugLog("[DEBUG] Cleared duplicate auth check interval");
  }
  window.authCheckInterval = null;
}); // ← THIS CLOSES document.addEventListener("DOMContentLoaded", () => {