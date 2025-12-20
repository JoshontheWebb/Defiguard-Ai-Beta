// ---------------------------------------------------------------------
// GLOBAL DEBUG HELPER – makes console logs easy to spot
// ---------------------------------------------------------------------
const log = (label, ...args) => console.log(`[${label}]`, ...args, `time=${new Date().toISOString()}`);

// ---------------------------------------------------------------------
// AUDIT QUEUE TRACKER - Real-time queue position and status updates
// ---------------------------------------------------------------------
class AuditQueueTracker {
    constructor() {
        this.jobId = null;
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
            
            log('QUEUE', `Audit submitted: job_id=${this.jobId}, position=${data.position}`);
            
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
    
    connectWebSocket() {
        if (!this.jobId) return;
        
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws/job/${this.jobId}`;
        
        try {
            this.ws = new WebSocket(wsUrl);
            
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
            };
            
            // Keep-alive ping every 25 seconds
            this.pingInterval = setInterval(() => {
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send('ping');
                }
            }, 25000);
            
        } catch (e) {
            log('QUEUE_WS', 'Failed to connect WebSocket:', e);
            this.startPolling();
        }
    }
    
    startPolling() {
        if (this.pollInterval) return; // Already polling
        
        this.pollInterval = setInterval(async () => {
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
        }, 3000); // Poll every 3 seconds
    }
    
    stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
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
            'grok': { icon: '🤖', label: 'Claude AI Analysis & Report Generation', progress: 60 },
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
        
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
        
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        
        this.jobId = null;
    }
}

// Global queue tracker instance
const queueTracker = new AuditQueueTracker();

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
            console.log('[DEBUG] All DOM elements found – initializing');
            callback(elements);
        } else if (attempts < maxAttempts) {
            attempts++;
            console.log(`[DEBUG] Waiting for DOM elements, attempt ${attempts}/${maxAttempts}, missing: ${missing.join(', ')}`);
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
      console.log(`[DEBUG] Fresh CSRF token fetched: ${data.csrf_token.substring(0, 10)}...`);
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
      settingsModalButton: "#open-settings-modal",
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

      // WebSocket audit log (single instance)
      const wsUsername = (await fetchUsername())?.username || "guest";
      const ws = new WebSocket(`${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws-audit-log?username=${encodeURIComponent(wsUsername)}`);
      ws.onopen = () => logMessage("Connected to audit log");
      ws.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          if (data.type === "audit_log") logMessage(data.message);
        } catch (_) {}
      };
      ws.onerror = () => logMessage("WebSocket error");
      ws.onclose = () => logMessage("Disconnected from audit log");

      // Hamburger Menu
      if (hamburger && sidebar && mainContent) {
        hamburger.addEventListener("click", () => {
          sidebar.classList.toggle("open");
          hamburger.classList.toggle("open");
          document.body.classList.toggle('sidebar-open');
          mainContent.style.marginLeft = sidebar.classList.contains("open") ? "270px" : "0";
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
          authStatus.innerHTML = `Signed in as <strong>${user.username}</strong> <small>(${displayProvider})</small>`;
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
          console.log(`[DEBUG] Logout initiated, time=${new Date().toISOString()}`);
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
            console.log("[DEBUG] User is logged out, showing logged-out UI");
            
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
            
            console.log("[DEBUG] Logged-out UI state applied");
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
          
          // Build features list for sidebar
          const featuresList = [];
          if (tier === "enterprise") {
            featuresList.push("Unlimited everything", "White-label reports", "Team accounts", "Priority support");
          } else if (tier === "pro") {
            featuresList.push("Unlimited audits", "5MB file uploads", "API access", "Full analysis");
            if (feature_flags.fuzzing) featuresList.push("Fuzzing");
            if (feature_flags.priority_support) featuresList.push("Priority support");
          } else if (tier === "starter") {
            featuresList.push("50 audits/month", "1MB file uploads", "Full analysis", "Fix recommendations");
          } else {
            featuresList.push("3 free audits", "500KB files", "Critical issues only");
          }
          
          sidebarTierFeatures.innerHTML = featuresList
            .map(f => `<div class="feature-item">${f}</div>`)
            .join("");
          
          // Update main tier description
          tierDescription.textContent = `${tierNameCap} Tier: ${
            tier === "enterprise"
              ? "Unlimited file size, full Enterprise audits, fuzzing, priority support"
              : tier === "pro"
              ? "Unlimited audits, 5MB files, API access, fuzzing, priority support"
              : tier === "starter"
              ? `50 audits/month, 1MB file size (${auditCount}/${auditLimit} remaining)`
              : `3 free audits, 500KB files (${auditCount}/${auditLimit} remaining)`
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
          console.log(`[DEBUG] Tier data fetched: tier=${tier}, auditCount=${auditCount}, auditLimit=${auditLimit}, time=${new Date().toISOString()}`);
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
        console.log('[DEBUG] Tier from audit response:', data.tier);
      } else {
        // Fallback: Get tier from already-loaded tier data (from sidebar or initial fetch)
        const sidebarTier = document.getElementById('sidebar-tier-name')?.textContent?.toLowerCase() || 'free';
        window.currentAuditTier = window.currentAuditTier || sidebarTier;
        console.log('[DEBUG] Tier from sidebar fallback:', window.currentAuditTier);
      } // Store tier info
      console.log('[DEBUG] Audit response received:', {
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
          
          const hasProFeatures = issue.line_number || issue.function_name || issue.vulnerable_code;
          
          return `
            <tr class="issue-row ${hasProFeatures ? 'expandable' : ''}" data-issue-id="${index}">
              <td class="severity-cell">
                <span class="severity-badge ${severity}">${severityDisplay}</span>
              </td>
              <td><strong>${type}</strong></td>
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
  
  console.log(`[DEBUG] Modal opened for issue ${index}`);
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
        
        // Store PDF path if available
        if (data.compliance_pdf) {
          window.currentAuditPdfPath = data.compliance_pdf;
          console.log(`[DEBUG] PDF generated: ${data.compliance_pdf}`);
        }
        
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
        console.log("[DEBUG] Audit submit handler attached successfully");
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
        console.log("[DEBUG] Report downloaded");
      });

      // PDF Download Button
      pdfDownloadButton?.addEventListener("click", async () => {
        if (!window.currentAuditPdfPath) {
          usageWarning.textContent = "No PDF available. Please run an audit first.";
          usageWarning.classList.add("error");
          return;
        }
        
        try {
          // The PDF path is a server file path, we need to download it
          const filename = window.currentAuditPdfPath.split('/').pop();
          const response = await fetch(`/static/reports/${filename}`, {
            credentials: "include"
          });
          
          if (!response.ok) {
            throw new Error("PDF not found");
          }
          
          const blob = await response.blob();
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = filename;
          a.click();
          URL.revokeObjectURL(url);
          console.log("[DEBUG] PDF downloaded");
        } catch (error) {
          console.error("[ERROR] PDF download failed:", error);
          usageWarning.textContent = "PDF download failed. The report may have expired.";
          usageWarning.classList.add("error");
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
            console.log(`[DEBUG] NFT minted: ${data.nft_id}`);
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
        settingsModalButton, settingsModal, settingsModalBackdrop, settingsModalClose,
        modalUsername, modalEmail, modalTier, modalMemberSince, modalAuditsUsed,
        modalAuditsRemaining, modalSizeLimit, modalUsageProgress, modalUsageText,
        modalApiSection, apiKeyCountDisplay, apiKeysTableBody, createApiKeyButton,
        createKeyModal, createKeyModalBackdrop, createKeyModalClose,
        newKeyLabelInput, createKeyConfirm, createKeyCancel,
        modalUpgradeButton, modalLogoutButton
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
          
          console.log("[DEBUG] Settings modal populated successfully");
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
          
          console.log("[DEBUG] API keys loaded successfully");
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

      // Open modal
      settingsModalButton?.addEventListener("click", async () => {
        console.log("[DEBUG] Opening settings modal");
        await populateSettingsModal();
        settingsModal?.classList.add("active");
        settingsModalBackdrop?.classList.add("active");
        document.body.style.overflow = "hidden"; // Prevent background scroll
      });

      // Close modal
      const closeSettingsModal = () => {
        console.log("[DEBUG] Closing settings modal");
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
            settingsModalButton?.click();
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
          console.log(`[DEBUG] Logout initiated from settings modal, time=${new Date().toISOString()}`);
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
        console.log(`[DEBUG] Theme loaded: ${savedTheme}`);
      };
      
      // Apply theme
      const applyTheme = (theme) => {
        document.documentElement.setAttribute("data-theme", theme);
        localStorage.setItem("userTheme", theme);
        console.log(`[DEBUG] Theme applied: ${theme}`);
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
    console.log("[DEBUG] Cleared duplicate auth check interval");
  }
  window.authCheckInterval = null;
}); // ← THIS CLOSES document.addEventListener("DOMContentLoaded", () => {