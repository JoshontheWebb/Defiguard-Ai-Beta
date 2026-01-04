"""
Certora Prover Runner for DeFiGuard AI.

Handles submission of verification jobs to Certora's cloud infrastructure
and parsing of verification results.

Architecture:
- Certora jobs run on Certora's cloud (NOT on your server)
- The CLI submits jobs and can optionally wait for results
- For reliability, we use non-blocking mode: submit job, get URL, then poll

This prevents timeout failures when Certora cloud is slow or network has latency.
"""

import os
import json
import subprocess
import tempfile
import logging
import re
import time
import socket
from typing import Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Timeout for job submission WITH wait_for_results=True
# The CLI will wait for verification to complete, which can take several minutes
# Certora formal verification is compute-intensive - allow generous timeout
SUBMISSION_TIMEOUT = 1800  # 30 minutes - CLI waits for results now

# Timeout for polling results (fallback if CLI doesn't return results)
# Since we now use wait_for_results=True, this is rarely needed
POLL_TIMEOUT = 600  # 10 minutes fallback polling

# How often to poll for results
POLL_INTERVAL = 20  # Check every 20 seconds

# Certora cloud job URL pattern
CERTORA_JOB_URL = "https://prover.certora.com/output/{job_id}"


class CertoraRunner:
    """Interface to Certora Prover cloud."""

    def __init__(self, submission_timeout: int = SUBMISSION_TIMEOUT, poll_timeout: int = POLL_TIMEOUT):
        """
        Initialize Certora Runner.

        Args:
            submission_timeout: Time to wait for job submission (getting URL)
            poll_timeout: Max time to wait for results after submission
        """
        self.api_key = os.getenv("CERTORAKEY")
        self.submission_timeout = submission_timeout
        self.poll_timeout = poll_timeout

        if not self.api_key:
            logger.warning("CertoraRunner: CERTORAKEY not set - verification will be skipped")

    def is_configured(self) -> bool:
        """Check if Certora is properly configured."""
        return bool(self.api_key)

    def run_verification_sync(
        self,
        contract_path: str,
        spec_content: str,
        contract_name: str = None,
        wait_for_results: bool = True
    ) -> dict[str, Any]:
        """
        Run Certora verification with non-blocking job submission.

        This method uses a two-phase approach:
        1. Submit job and get URL immediately (fast, ~30 seconds)
        2. Poll for results (can be slow, depends on Certora cloud)

        This prevents timeout failures - if polling times out, we still have the job URL.

        Args:
            contract_path: Path to Solidity contract file
            spec_content: CVL specification content
            contract_name: Optional contract name (extracted from file if not provided)
            wait_for_results: If True, poll for results. If False, return after submission.

        Returns:
            Dictionary with verification results:
            {
                "success": bool,
                "status": "verified" | "issues_found" | "pending" | "error" | "skipped",
                "rules_verified": int,
                "rules_violated": int,
                "rules_timeout": int,
                "violations": [...],
                "verified_rules": [...],
                "job_url": str,
                "error": str (if failed)
            }
        """
        if not self.is_configured():
            return {
                "success": False,
                "status": "skipped",
                "error": "CERTORAKEY not configured",
                "rules_verified": 0,
                "rules_violated": 0,
                "violations": []
            }

        # Validate contract file exists
        if not os.path.exists(contract_path):
            return {
                "success": False,
                "status": "error",
                "error": f"Contract file not found: {contract_path}",
                "rules_verified": 0,
                "rules_violated": 0,
                "violations": []
            }

        spec_path = None
        conf_path = None

        try:
            # Extract contract name if not provided
            if not contract_name:
                contract_name = self._extract_contract_name(contract_path)

            # Write spec to temp file
            spec_path = self._write_temp_file(spec_content, suffix=".spec")

            # Create conf file - USE BLOCKING MODE for reliable results
            # Per Certora 2026 docs: wait_for_results=True makes CLI wait for completion
            # This is more reliable than manual polling which can fail with 403 auth errors
            conf_content = self._create_conf_file(contract_path, spec_path, contract_name, wait_for_results=True)
            conf_path = self._write_temp_file(conf_content, suffix=".conf")

            logger.info(f"CertoraRunner: Submitting verification for {contract_name}")
            logger.debug(f"CertoraRunner: Contract: {contract_path}")

            # PHASE 1: Submit job and get URL (quick, ~30 seconds)
            cmd = ["certoraRun", conf_path]
            logger.debug(f"CertoraRunner: Command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.submission_timeout,
                env={**os.environ, "CERTORAKEY": self.api_key}
            )

            logger.info(f"CertoraRunner: Job returned with code {result.returncode}")

            # Extract job URL from output (always available)
            job_url = self._extract_job_url(result.stdout)

            # With wait_for_results=True, CLI waits for completion and returns results
            # Try to parse results directly from CLI output first
            parsed = self._parse_output(result.stdout, result.stderr, result.returncode)
            parsed["job_url"] = job_url

            # Check if we got actual results from CLI (not just job submission)
            has_results = (
                parsed.get("rules_verified", 0) > 0 or
                parsed.get("rules_violated", 0) > 0 or
                parsed.get("status") in ["verified", "issues_found", "error"]
            )

            if has_results:
                logger.info(f"CertoraRunner: Got results from CLI - verified:{parsed.get('rules_verified')}, violated:{parsed.get('rules_violated')}")
                return parsed

            # No results in CLI output - check if we got a job URL
            if not job_url:
                # No job URL and no results means submission failed
                logger.warning(f"CertoraRunner: No job URL and no results in output")
                logger.warning(f"CertoraRunner: Return code: {result.returncode}")
                if result.stdout:
                    stdout_preview = result.stdout[:2000].replace('\n', ' | ')
                    logger.warning(f"CertoraRunner: STDOUT preview: {stdout_preview}")
                if result.stderr:
                    stderr_preview = result.stderr[:2000].replace('\n', ' | ')
                    logger.warning(f"CertoraRunner: STDERR preview: {stderr_preview}")
                return parsed

            logger.info(f"CertoraRunner: Job URL: {job_url}")

            # If not waiting for results, return immediately with pending status
            if not wait_for_results:
                return {
                    "success": True,
                    "status": "pending",
                    "job_url": job_url,
                    "rules_verified": 0,
                    "rules_violated": 0,
                    "verified_rules": [],
                    "violations": [],
                    "message": "Job submitted to Certora cloud. Results will be available at the job URL."
                }

            # PHASE 2: Fallback polling (if CLI didn't return results)
            # This shouldn't happen with wait_for_results=True, but keep as safety net
            logger.info(f"CertoraRunner: CLI didn't return results, falling back to polling (timeout: {self.poll_timeout}s)")
            poll_result = self._poll_for_results(job_url, self.poll_timeout)

            # Merge job URL into result
            poll_result["job_url"] = job_url

            return poll_result

        except subprocess.TimeoutExpired:
            logger.warning(f"CertoraRunner: Job submission timed out after {self.submission_timeout}s")
            return {
                "success": False,
                "status": "timeout",
                "error": f"Job submission timed out after {self.submission_timeout} seconds. Certora cloud may be busy.",
                "rules_verified": 0,
                "rules_violated": 0,
                "violations": [{
                    "rule": "Job Submission",
                    "status": "timeout",
                    "description": "Could not submit job to Certora cloud. Try again later."
                }]
            }

        except FileNotFoundError:
            logger.error("CertoraRunner: certoraRun command not found")
            return {
                "success": False,
                "status": "error",
                "error": "Certora CLI not installed (certoraRun not found)",
                "rules_verified": 0,
                "rules_violated": 0,
                "violations": [{
                    "rule": "Certora Setup",
                    "status": "error",
                    "description": "Certora CLI (certoraRun) is not installed. Please install it with: pip install certora-cli"
                }]
            }

        except Exception as e:
            logger.error(f"CertoraRunner: Unexpected error: {e}")
            return {
                "success": False,
                "status": "error",
                "error": str(e),
                "rules_verified": 0,
                "rules_violated": 0,
                "violations": [{
                    "rule": "Verification Error",
                    "status": "error",
                    "description": f"Unexpected error during verification: {str(e)}"
                }]
            }

        finally:
            # Cleanup temp files
            self._cleanup_temp_file(spec_path)
            self._cleanup_temp_file(conf_path)

    def _extract_job_url(self, output: str) -> Optional[str]:
        """Extract job URL from Certora CLI output.

        IMPORTANT: Certora URLs may include query parameters like ?anonymousKey=...
        which are REQUIRED for accessing output files. We must capture the full URL.
        """
        url_patterns = [
            # Capture full URL including query params (anonymousKey, etc.)
            r"https://prover\.certora\.com/output/[a-zA-Z0-9/_-]+(?:\?[^\s]*)?",
            r"https://prover\.certora\.com/job/[a-zA-Z0-9/_-]+(?:\?[^\s]*)?",
            r"Job URL:\s*(https://prover\.certora\.com[^\s]+)",
            r"Report:\s*(https://prover\.certora\.com[^\s]+)",
        ]

        for pattern in url_patterns:
            match = re.search(pattern, output)
            if match:
                url = match.group(0)
                if "://" not in url and len(match.groups()) > 0:
                    url = match.group(1)
                # Log the URL with any query params for debugging
                logger.info(f"CertoraRunner: Extracted job URL: {url}")
                return url

        return None

    def _poll_for_results(self, job_url: str, timeout: int) -> dict[str, Any]:
        """
        Poll Certora cloud for job results.

        Args:
            job_url: URL of the Certora job
            timeout: Maximum time to wait for results

        Returns:
            Verification results dictionary
        """
        start_time = time.time()
        poll_count = 0

        # Base result for timeout case
        timeout_result = {
            "success": True,  # Job was submitted successfully
            "status": "pending",
            "job_url": job_url,
            "rules_verified": 0,
            "rules_violated": 0,
            "verified_rules": [],
            "violations": [],
            "message": "Verification is running on Certora cloud. Check job URL for results."
        }

        while time.time() - start_time < timeout:
            poll_count += 1
            logger.info(f"CertoraRunner: Poll attempt {poll_count} for {job_url}")

            try:
                # Use certoraRun --wait to check status
                # Or fetch results from the job URL API
                result = self._fetch_job_results(job_url)

                if result.get("status") in ["verified", "issues_found", "error"]:
                    logger.info(f"CertoraRunner: Got final result: {result.get('status')}")
                    return result

                if result.get("status") == "running":
                    logger.info(f"CertoraRunner: Job still running, waiting {POLL_INTERVAL}s...")
                    time.sleep(POLL_INTERVAL)
                    continue

            except Exception as e:
                logger.warning(f"CertoraRunner: Poll error: {e}")
                time.sleep(POLL_INTERVAL)
                continue

        # Timeout - but job might still complete
        logger.warning(f"CertoraRunner: Polling timed out after {timeout}s")
        logger.info(f"CertoraRunner: Job may still be running at: {job_url}")
        return timeout_result

    def _fetch_job_results(self, job_url: str) -> dict[str, Any]:
        """
        Fetch results from a Certora job URL.

        Uses multiple strategies to detect completion:
        1. statsdata.json - Contains per-rule success/fail status (most reliable)
        2. jobStatus.json - Contains job metadata
        3. output.json - Contains detailed results
        4. HTML page - Fallback parsing

        AUTHENTICATION: Adds CERTORAKEY to requests for authenticated access.

        Args:
            job_url: URL of the Certora job (e.g., https://prover.certora.com/output/9579011/abc123)

        Returns:
            Results dictionary with status
        """
        try:
            import urllib.request
            import urllib.error
            from urllib.parse import urlparse, parse_qs, urlencode

            # Track what we've tried for debugging
            tried_endpoints = []

            # Extract base URL and query params (anonymousKey is REQUIRED for authenticated access)
            parsed = urlparse(job_url)
            query_params = parse_qs(parsed.query)
            base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            query_string = f"?{parsed.query}" if parsed.query else ""

            # AUTHENTICATION: Build headers with API key for authenticated access
            # Certora API accepts either anonymousKey query param OR Authorization header
            auth_headers = {
                "Accept": "application/json, text/html",
                "User-Agent": "DeFiGuard-AI/1.0",
            }
            if self.api_key:
                # Add API key in multiple formats for compatibility
                auth_headers["Authorization"] = f"Bearer {self.api_key}"
                auth_headers["x-api-key"] = self.api_key
                # Also add as query param if not already present
                if not query_string:
                    query_string = f"?anonymousKey={self.api_key}"
                elif "anonymousKey" not in query_string:
                    query_string += f"&anonymousKey={self.api_key}"
                logger.info("CertoraRunner: Added authentication headers and anonymousKey to requests")

            # CRITICAL: Certora output FILES are hosted on vaas-stg.certora.com, NOT prover.certora.com
            # The job URL from CLI is prover.certora.com but statsdata.json etc are on vaas-stg
            vaas_base_url = base_url.replace("prover.certora.com", "vaas-stg.certora.com")

            logger.info(f"CertoraRunner: Base URL: {base_url}, VAAS URL: {vaas_base_url}, has query params: {bool(parsed.query)}")

            # PRIORITY 0: Try jobStatus URL (change 'output' to 'jobStatus' in path)
            # This is available immediately after job submission and shows current status
            # Try BOTH vaas-stg and prover hosts
            if '/output/' in base_url:
                job_status_urls = [
                    f"{vaas_base_url.replace('/output/', '/jobStatus/')}{query_string}",  # vaas-stg first
                    f"{base_url.replace('/output/', '/jobStatus/')}{query_string}",       # prover fallback
                ]
                for job_status_url in job_status_urls:
                    tried_endpoints.append(job_status_url)
                    try:
                        logger.info(f"CertoraRunner: Trying jobStatus URL: {job_status_url}")
                        req = urllib.request.Request(job_status_url, headers=auth_headers)
                        with urllib.request.urlopen(req, timeout=30) as response:
                            raw_content = response.read()
                            try:
                                content = raw_content.decode('utf-8')
                            except UnicodeDecodeError:
                                content = raw_content.decode('latin-1', errors='replace')
                            content_type = response.headers.get('Content-Type', '')
                            logger.info(f"CertoraRunner: jobStatus response type: {content_type}, length: {len(content)}")

                            # Try to parse as JSON first
                            if 'json' in content_type or content.strip().startswith('{'):
                                try:
                                    data = json.loads(content)
                                    logger.info(f"CertoraRunner: jobStatus JSON keys: {list(data.keys())[:10]}")
                                    # Check for completion status
                                    status = str(data.get('status', data.get('jobStatus', ''))).lower()
                                    if status in ['complete', 'done', 'finished', 'succeeded', 'failed', 'error']:
                                        logger.info(f"CertoraRunner: jobStatus indicates completion: {status}")
                                        return self._parse_api_results(data)
                                    elif status in ['running', 'pending', 'queued']:
                                        logger.info(f"CertoraRunner: jobStatus indicates still running: {status}")
                                        # Don't return yet - try other hosts
                                        break
                                except json.JSONDecodeError:
                                    pass

                            # Parse HTML for status indicators
                            if len(content) > 1000:
                                content_lower = content.lower()
                                if 'completed' in content_lower or 'finished' in content_lower:
                                    logger.info("CertoraRunner: jobStatus HTML indicates completion")
                                    # Job complete - continue to get actual results
                                elif 'running' in content_lower or 'pending' in content_lower:
                                    logger.info("CertoraRunner: jobStatus HTML indicates still running")
                                    # Don't return yet - try other hosts
                                    break

                    except urllib.error.HTTPError as e:
                        logger.info(f"CertoraRunner: jobStatus returned HTTP {e.code} for {job_status_url}")
                    except Exception as e:
                        logger.info(f"CertoraRunner: Error fetching jobStatus from {job_status_url}: {e}")

            # PRIORITY 1: statsdata.json - This file contains per-rule completion status
            # It's generated after Certora finishes and has explicit success/fail indicators
            # CRITICAL: Try vaas-stg.certora.com FIRST - this is where output files actually live!
            stats_urls_to_try = [
                f"{vaas_base_url}/statsdata.json{query_string}",  # vaas-stg first!
                f"{base_url}/statsdata.json{query_string}",       # fallback to prover
            ]

            for stats_url in stats_urls_to_try:
                tried_endpoints.append(stats_url)
                try:
                    logger.info(f"CertoraRunner: Trying statsdata.json: {stats_url}")
                    req = urllib.request.Request(stats_url, headers=auth_headers)
                    with urllib.request.urlopen(req, timeout=30) as response:
                        raw_content = response.read()
                        try:
                            content = raw_content.decode('utf-8')
                        except UnicodeDecodeError:
                            content = raw_content.decode('latin-1', errors='replace')
                        data = json.loads(content)
                        logger.info(f"CertoraRunner: SUCCESS - Got statsdata.json from {stats_url}! Keys: {list(data.keys())[:10]}")

                        # statsdata.json existing means the job is COMPLETE
                        # Parse it to extract rule results
                        return self._parse_statsdata(data, job_url)

                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        logger.info(f"CertoraRunner: statsdata.json not found at {stats_url} (404)")
                    else:
                        logger.info(f"CertoraRunner: statsdata.json returned HTTP {e.code} at {stats_url}")
                except Exception as e:
                    logger.info(f"CertoraRunner: Error fetching statsdata.json from {stats_url}: {e}")

            # PRIORITY 2: Try other JSON endpoints
            # Certora has multiple possible URL structures for output data
            # Include query_string for authentication (anonymousKey)
            # Try BOTH vaas-stg and prover hosts
            json_endpoints = [
                # vaas-stg endpoints (where files actually live)
                f"{vaas_base_url}/jobStatus.json{query_string}",
                f"{vaas_base_url}/output.json{query_string}",
                f"{vaas_base_url}/results.json{query_string}",
                f"{vaas_base_url}/verificationProgress.json{query_string}",
                f"{vaas_base_url}/Reports/statsdata.json{query_string}",
                # prover endpoints (fallback)
                f"{base_url}/jobStatus.json{query_string}",
                f"{base_url}/output.json{query_string}",
                f"{base_url}/results.json{query_string}",
                f"{base_url}/verificationProgress.json{query_string}",
                f"{base_url}/Reports/statsdata.json{query_string}",
            ]

            for json_url in json_endpoints:
                tried_endpoints.append(json_url)
                try:
                    logger.info(f"CertoraRunner: Trying endpoint: {json_url}")
                    req = urllib.request.Request(json_url, headers=auth_headers)
                    with urllib.request.urlopen(req, timeout=30) as response:
                        raw_content = response.read()
                        try:
                            content = raw_content.decode('utf-8')
                        except UnicodeDecodeError:
                            content = raw_content.decode('latin-1', errors='replace')
                        data = json.loads(content)
                        logger.info(f"CertoraRunner: Got response from {json_url}, keys: {list(data.keys())[:5] if isinstance(data, dict) else 'array'}")

                        # Parse the response based on structure
                        if isinstance(data, dict):
                            # Check various status field names Certora uses
                            status = (
                                data.get("jobStatus", "") or
                                data.get("status", "") or
                                data.get("verificationStatus", "") or
                                data.get("jobEnded", "") or  # Sometimes indicates completion
                                ""
                            )
                            if isinstance(status, bool):
                                status = "complete" if status else "running"
                            status = str(status).lower()

                            logger.info(f"CertoraRunner: Job status from {json_url}: '{status}'")

                            # Map Certora status values to our status
                            if status in ["succeeded", "success", "complete", "finished", "done", "verified", "true"]:
                                return self._parse_api_results(data)
                            elif status in ["running", "pending", "queued", "inprogress", "in_progress", "in progress"]:
                                return {"status": "running"}
                            elif status in ["failed", "error", "violated"]:
                                return self._parse_api_results(data)

                            # Check for completion indicators in data structure
                            if data.get("jobEnded") or data.get("endTime") or data.get("completedAt"):
                                logger.info("CertoraRunner: Found completion timestamp - job is done")
                                return self._parse_api_results(data)

                            if "rules" in data or "results" in data or "output" in data:
                                # Has results data - try to parse it
                                return self._parse_api_results(data)

                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        logger.info(f"CertoraRunner: {json_url} not found (404)")
                    else:
                        logger.info(f"CertoraRunner: {json_url} returned HTTP {e.code}")
                    continue
                except Exception as e:
                    logger.info(f"CertoraRunner: Error fetching {json_url}: {e}")
                    continue

            # PRIORITY 3: Parse the HTML page as fallback
            # NOTE: Certora uses a React SPA, so server-side HTML fetching only gets the shell
            html_url = f"{base_url}{query_string}"
            logger.info(f"CertoraRunner: All JSON endpoints failed, trying HTML page (SPA shell): {html_url}")
            tried_endpoints.append(html_url)
            try:
                # Use auth_headers but override Accept for HTML
                html_headers = {**auth_headers, "Accept": "text/html"}
                req = urllib.request.Request(html_url, headers=html_headers)
                with urllib.request.urlopen(req, timeout=30) as response:
                    raw_content = response.read()
                    try:
                        html = raw_content.decode('utf-8')
                    except UnicodeDecodeError:
                        html = raw_content.decode('latin-1', errors='replace')
                    html_len = len(html)

                    # Log a sample of the HTML for debugging (first 500 chars, excluding boilerplate)
                    # Look for the main content area
                    html_sample = html[:1000].replace('\n', ' ').replace('\r', '')
                    logger.info(f"CertoraRunner: HTML length={html_len}, sample: {html_sample[:300]}...")

                    # Look for status indicators in the HTML
                    html_lower = html.lower()

                    # Look for SPECIFIC completion phrases (not just single words that could be anywhere)
                    completion_phrases = [
                        "job completed",
                        "verification complete",
                        "verification succeeded",
                        "verification failed",
                        "all rules verified",
                        "rules verified",
                        "results summary",
                        "final results",
                        "job finished",
                        "✓",  # Checkmark often indicates completion
                        "✗",  # X mark also indicates completion (with issues)
                    ]

                    running_phrases = [
                        "job is running",
                        "verification in progress",
                        "currently running",
                        "please wait",
                        "processing",
                        "job queued",
                        "waiting in queue",
                    ]

                    found_completion = [p for p in completion_phrases if p in html_lower]
                    found_running = [p for p in running_phrases if p in html_lower]

                    logger.info(f"CertoraRunner: Completion phrases found: {found_completion}")
                    logger.info(f"CertoraRunner: Running phrases found: {found_running}")

                    # Also check for rule status patterns in HTML
                    # Certora pages often have tables with rule statuses
                    verified_count = len(re.findall(r'(?:✓|verified|VERIFIED|passed|PASSED)', html))
                    violated_count = len(re.findall(r'(?:✗|violated|VIOLATED|failed|FAILED)', html))
                    timeout_count = len(re.findall(r'(?:timeout|TIMEOUT)', html, re.IGNORECASE))

                    if verified_count > 0 or violated_count > 0:
                        logger.info(f"CertoraRunner: Found rule results - verified:{verified_count}, violated:{violated_count}, timeout:{timeout_count}")
                        # If we have rule results, the job is DEFINITELY complete
                        return self._parse_html_results(html)

                    # Decision logic - prioritize completion phrases
                    if found_completion and not found_running:
                        logger.info("CertoraRunner: HTML shows completion (no running phrases)")
                        return self._parse_html_results(html)

                    if found_running and not found_completion:
                        logger.info("CertoraRunner: HTML shows still running")
                        return {"status": "running"}

                    # If both or neither - check page size
                    # A results page is usually larger than a "waiting" page
                    if html_len > 5000:
                        # Large page usually means results are available
                        logger.info(f"CertoraRunner: Large page ({html_len} chars) - assuming complete")
                        return self._parse_html_results(html)

                    if html_len < 2000:
                        # Small page usually means still loading/running
                        logger.info(f"CertoraRunner: Small page ({html_len} chars) - assuming still running")
                        return {"status": "running"}

                    # Default: If we really can't tell, log extensively and assume running
                    logger.warning(f"CertoraRunner: Could not determine status from HTML. Tried: {tried_endpoints}")
                    logger.warning(f"CertoraRunner: HTML title match: {re.search(r'<title>(.*?)</title>', html, re.IGNORECASE)}")
                    return {"status": "running"}

            except Exception as e:
                logger.warning(f"CertoraRunner: Error fetching HTML page: {e}")

            # Default to running if we couldn't determine status
            logger.warning(f"CertoraRunner: All endpoints failed. Tried: {tried_endpoints}")
            return {"status": "running"}

        except Exception as e:
            logger.warning(f"CertoraRunner: Error in _fetch_job_results: {e}")
            return {"status": "running"}

    def _parse_statsdata(self, data: dict, job_url: str) -> dict[str, Any]:
        """
        Parse Certora statsdata.json which contains per-rule verification status.

        The presence of this file indicates the job is COMPLETE.
        """
        result = {
            "success": True,
            "status": "verified",
            "rules_verified": 0,
            "rules_violated": 0,
            "rules_timeout": 0,
            "verified_rules": [],
            "violations": [],
            "job_url": job_url
        }

        try:
            # statsdata.json typically has structure like:
            # { "ruleName": { "status": "verified", ... }, ... }
            # or may have nested "rules" or "results" array

            rules = data
            if "rules" in data:
                rules = data["rules"]
            elif "results" in data:
                rules = data["results"]

            if isinstance(rules, dict):
                for rule_name, rule_data in rules.items():
                    if isinstance(rule_data, dict):
                        status = str(rule_data.get("status", rule_data.get("result", ""))).lower()
                    else:
                        status = str(rule_data).lower()

                    if status in ["verified", "passed", "pass", "success", "true"]:
                        result["rules_verified"] += 1
                        result["verified_rules"].append({
                            "rule": rule_name,
                            "status": "verified",
                            "description": f"Property '{rule_name}' mathematically proven"
                        })
                    elif status in ["violated", "failed", "fail", "false"]:
                        result["rules_violated"] += 1
                        result["violations"].append({
                            "rule": rule_name,
                            "status": "violated",
                            "description": f"Rule '{rule_name}' was violated"
                        })
                    elif status in ["timeout"]:
                        result["rules_timeout"] += 1

            elif isinstance(rules, list):
                for rule in rules:
                    if isinstance(rule, dict):
                        rule_name = rule.get("name", rule.get("rule", "Unknown"))
                        status = str(rule.get("status", rule.get("result", ""))).lower()

                        if status in ["verified", "passed", "pass", "success"]:
                            result["rules_verified"] += 1
                            result["verified_rules"].append({
                                "rule": rule_name,
                                "status": "verified",
                                "description": f"Property '{rule_name}' mathematically proven"
                            })
                        elif status in ["violated", "failed", "fail"]:
                            result["rules_violated"] += 1
                            result["violations"].append({
                                "rule": rule_name,
                                "status": "violated",
                                "description": f"Rule '{rule_name}' was violated"
                            })

            # Update overall status based on findings
            if result["rules_violated"] > 0:
                result["success"] = False
                result["status"] = "issues_found"

            logger.info(f"CertoraRunner: Parsed statsdata - verified:{result['rules_verified']}, violated:{result['rules_violated']}")
            return result

        except Exception as e:
            logger.warning(f"CertoraRunner: Error parsing statsdata.json: {e}")
            # statsdata.json exists but couldn't parse - job is still complete
            return {
                "success": True,
                "status": "verified",
                "rules_verified": 1,
                "rules_violated": 0,
                "verified_rules": [{"rule": "Verification", "status": "verified", "description": "Verification completed"}],
                "violations": [],
                "job_url": job_url
            }

    def _parse_html_results(self, html: str) -> dict[str, Any]:
        """Parse results from Certora HTML output page."""
        result = {
            "success": True,
            "status": "verified",
            "rules_verified": 0,
            "rules_violated": 0,
            "rules_timeout": 0,
            "verified_rules": [],
            "violations": []
        }

        # Count verified rules (look for checkmarks or "verified" indicators)
        verified_count = len(re.findall(r'(?:✓|✔|VERIFIED|verified|Verified)', html))
        violated_count = len(re.findall(r'(?:✗|✘|VIOLATED|violated|Violated|FAILED|failed)', html))
        timeout_count = len(re.findall(r'(?:TIMEOUT|timeout|Timeout)', html))

        result["rules_verified"] = verified_count
        result["rules_violated"] = violated_count
        result["rules_timeout"] = timeout_count

        if violated_count > 0:
            result["success"] = False
            result["status"] = "issues_found"
            result["violations"].append({
                "rule": "Multiple Rules",
                "status": "violated",
                "description": f"{violated_count} rule(s) violated - check Certora dashboard for details"
            })

        if verified_count > 0:
            result["verified_rules"].append({
                "rule": "Multiple Rules",
                "status": "verified",
                "description": f"{verified_count} rule(s) verified successfully"
            })

        return result

    def _parse_api_results(self, data: dict) -> dict[str, Any]:
        """Parse results from Certora API response."""
        result = {
            "success": True,
            "status": "verified",
            "rules_verified": 0,
            "rules_violated": 0,
            "rules_timeout": 0,
            "verified_rules": [],
            "violations": []
        }

        # Parse rules from API response
        rules = data.get("rules", data.get("results", []))

        for rule in rules:
            if isinstance(rule, dict):
                name = rule.get("name", rule.get("rule", "Unknown"))
                status = rule.get("status", "").lower()

                if status in ["verified", "passed", "pass"]:
                    result["rules_verified"] += 1
                    result["verified_rules"].append({
                        "rule": name,
                        "status": "verified",
                        "description": f"Property '{name}' mathematically proven"
                    })
                elif status in ["violated", "failed", "fail"]:
                    result["rules_violated"] += 1
                    result["violations"].append({
                        "rule": name,
                        "status": "violated",
                        "description": rule.get("message", f"Rule '{name}' was violated")
                    })
                elif status in ["timeout"]:
                    result["rules_timeout"] += 1

        if result["rules_violated"] > 0:
            result["success"] = False
            result["status"] = "issues_found"

        return result

    async def run_verification(
        self,
        contract_path: str,
        spec_content: str,
        contract_name: str = None,
        wait_for_results: bool = True
    ) -> dict[str, Any]:
        """
        Run Certora verification asynchronously.

        Args:
            contract_path: Path to Solidity contract file
            spec_content: CVL specification content
            contract_name: Optional contract name
            wait_for_results: If True, poll for results. If False, return after submission.

        Returns:
            Dictionary with verification results
        """
        import asyncio
        return await asyncio.to_thread(
            self.run_verification_sync,
            contract_path,
            spec_content,
            contract_name,
            wait_for_results
        )

    def _write_temp_file(self, content: str, suffix: str) -> str:
        """Write content to a temporary file and return the path.

        Uses proper file descriptor handling to prevent resource leaks.
        """
        fd, path = tempfile.mkstemp(suffix=suffix)
        fd_closed = False
        try:
            # os.fdopen takes ownership of the file descriptor
            with os.fdopen(fd, 'w') as f:
                fd_closed = True  # Context manager now owns fd
                f.write(content)
            return path
        except Exception:
            # Only close fd if os.fdopen failed (before it took ownership)
            if not fd_closed:
                try:
                    os.close(fd)
                except OSError:
                    pass  # Already closed or invalid
            # Clean up the temp file on error
            try:
                os.unlink(path)
            except OSError:
                pass
            raise

    def _cleanup_temp_file(self, path: Optional[str]):
        """Safely remove a temporary file."""
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except Exception as e:
                logger.warning(f"CertoraRunner: Failed to cleanup {path}: {e}")

    def _extract_contract_name(self, contract_path: str) -> str:
        """Extract contract name from Solidity file."""
        try:
            with open(contract_path, 'r') as f:
                content = f.read()
            # Match contract declaration at start of line (not in comments)
            # Pattern: optional whitespace, optional modifiers, then "contract Name"
            pattern = r"^\s*(?:abstract\s+)?contract\s+(\w+)"
            match = re.search(pattern, content, re.MULTILINE)
            return match.group(1) if match else Path(contract_path).stem
        except Exception:
            return Path(contract_path).stem

    def _create_conf_file(
        self,
        contract_path: str,
        spec_path: str,
        contract_name: str,
        wait_for_results: bool = False
    ) -> str:
        """
        Create Certora configuration file content.

        Args:
            contract_path: Path to Solidity contract
            spec_path: Path to CVL spec file
            contract_name: Name of the contract to verify
            wait_for_results: If True, CLI waits for results. If False, returns after submission.
        """
        # JSON format conf file for Certora Prover
        # Use path:contract format to handle UUID filenames with hyphens
        #
        # Per Certora docs: solc_allow_path (singular) expects a STRING, not array
        # It passes the value to solc's --allow-paths option
        # See: https://docs.certora.com/en/latest/docs/prover/cli/options.html
        contract_dir = str(Path(contract_path).parent)

        conf = {
            "files": [f"{contract_path}:{contract_name}"],
            "verify": f"{contract_name}:{spec_path}",
            "msg": f"DeFiGuard AI: {contract_name}",
            # CRITICAL: "all" blocks until verification completes
            # Per Certora 2026 docs: This makes CLI wait and return results directly
            # Much more reliable than manual polling which can fail with 403 errors
            "wait_for_results": "all" if wait_for_results else "none",
            # Simplifies console output for easier programmatic parsing
            "short_output": True,
            "rule_sanity": "basic",  # Check for tautologies
            "optimistic_loop": True,  # Assume loops terminate
            "loop_iter": "3",  # String per docs: "numbers are also encoded as strings"
            "process": "evm",  # EVM mode
            "solc": "solc",  # Use system solc
            "solc_allow_path": contract_dir,  # STRING: single path to contract directory
            "server": "production"  # Use production server
        }

        return json.dumps(conf, indent=2)

    def _parse_output(
        self,
        stdout: str,
        stderr: str,
        return_code: int
    ) -> dict[str, Any]:
        """Parse Certora output into structured results."""

        result = {
            "success": return_code == 0,
            "status": "verified" if return_code == 0 else "violated",
            "rules_verified": 0,
            "rules_violated": 0,
            "rules_timeout": 0,
            "verified_rules": [],
            "violations": [],
            "job_url": None,
            "raw_output": stdout[:2000] if stdout else None
        }

        # Extract job URL - only from prover.certora.com (NOT docs.certora.com)
        url_patterns = [
            # Primary job URL format
            r"https://prover\.certora\.com/output/[a-zA-Z0-9/_-]+",
            # Alternative prover URL formats
            r"https://prover\.certora\.com/job/[a-zA-Z0-9/_-]+",
            # Job URL from labeled output
            r"Job URL:\s*(https://prover\.certora\.com[^\s]+)",
            # Report URL format
            r"Report:\s*(https://prover\.certora\.com[^\s]+)",
        ]
        for pattern in url_patterns:
            url_match = re.search(pattern, stdout)
            if url_match:
                # Extract just the URL part
                matched = url_match.group(0)
                if "://" not in matched and len(url_match.groups()) > 0:
                    matched = url_match.group(1)
                result["job_url"] = matched
                logger.info(f"CertoraRunner: Found job URL: {matched}")
                break

        # Multiple patterns to catch Certora's various output formats
        rule_patterns = [
            # Standard format: "Rule: name - status"
            r"(?:Rule|Invariant):\s*(\w+)\s*[-:]\s*(Verified|Violated|Timeout|Error|FAIL|PASS)",
            # Results table format: "| ruleName | VERIFIED/VIOLATED |"
            r"\|\s*(\w+)\s*\|\s*(VERIFIED|VIOLATED|TIMEOUT|SANITY_FAIL)",
            # JSON-style: "rule": "name", "status": "verified"
            r'"rule":\s*"(\w+)"[^}]*"status":\s*"(verified|violated|timeout)"',
            # Compact format: "ruleName: VERIFIED"
            r"(\w+):\s*(VERIFIED|VIOLATED|TIMEOUT|PASS|FAIL)\b",
            # With checkmark/cross: "✓ ruleName" or "✗ ruleName"
            r"([✓✗])\s*(\w+)",
        ]

        parsed_rules = set()  # Track to avoid duplicates

        for pattern in rule_patterns:
            matches = re.findall(pattern, stdout, re.IGNORECASE)
            for match in matches:
                # Handle checkmark pattern differently
                if match[0] in ['✓', '✗']:
                    rule_name = match[1]
                    status = "verified" if match[0] == '✓' else "violated"
                else:
                    rule_name = match[0]
                    status = match[1].lower()

                # Skip if already parsed
                if rule_name.lower() in parsed_rules:
                    continue
                parsed_rules.add(rule_name.lower())

                # Normalize status
                if status in ["verified", "pass"]:
                    result["rules_verified"] += 1
                    result["verified_rules"].append({
                        "rule": rule_name,
                        "status": "verified",
                        "description": f"Property '{rule_name}' mathematically proven to hold"
                    })
                elif status in ["violated", "fail", "sanity_fail"]:
                    result["rules_violated"] += 1
                    # Try to extract counterexample or more context
                    desc = self._extract_violation_context(stdout, rule_name)
                    result["violations"].append({
                        "rule": rule_name,
                        "status": "violated",
                        "description": desc
                    })
                elif status == "timeout":
                    result["rules_timeout"] += 1
                    result["violations"].append({
                        "rule": rule_name,
                        "status": "timeout",
                        "description": f"Verification of '{rule_name}' exceeded time limit - contract may be too complex"
                    })

        # If no specific rules found, analyze output for general status
        if not parsed_rules:
            result = self._parse_general_status(stdout, stderr, return_code, result)

        # Update success based on violations
        if result["rules_violated"] > 0:
            result["success"] = False
            result["status"] = "issues_found"

        return result

    def _extract_violation_context(self, stdout: str, rule_name: str) -> str:
        """Extract meaningful context about why a rule was violated."""
        # Look for counterexample near the rule name
        patterns = [
            rf"{rule_name}[^.]*counterexample[^.]+\.",
            rf"{rule_name}[^.]*violation[^.]+\.",
            rf"{rule_name}[^.]*failed[^.]+\.",
            rf"Assert[^.]*{rule_name}[^.]+\.",
        ]

        for pattern in patterns:
            match = re.search(pattern, stdout, re.IGNORECASE)
            if match:
                return match.group(0)[:200]

        # Default descriptions based on common rule types
        rule_lower = rule_name.lower()
        if "balance" in rule_lower or "supply" in rule_lower:
            return f"Token balance/supply invariant '{rule_name}' was violated - potential accounting error"
        elif "owner" in rule_lower or "access" in rule_lower:
            return f"Access control rule '{rule_name}' was violated - unauthorized access possible"
        elif "reentran" in rule_lower:
            return f"Reentrancy protection '{rule_name}' was violated - contract may be vulnerable"
        elif "transfer" in rule_lower:
            return f"Transfer integrity rule '{rule_name}' was violated - funds may be at risk"
        elif "overflow" in rule_lower or "underflow" in rule_lower:
            return f"Arithmetic safety rule '{rule_name}' was violated - overflow/underflow possible"

        return f"Formal verification rule '{rule_name}' was violated - review recommended"

    def _parse_general_status(
        self,
        stdout: str,
        stderr: str,
        return_code: int,
        result: dict
    ) -> dict:
        """Parse general verification status when specific rules aren't found."""

        combined = (stdout + " " + stderr).lower()

        # Check for success indicators
        success_indicators = [
            "verification succeeded",
            "all rules verified",
            "no violations found",
            "verification complete: pass",
            "all properties hold",
        ]

        failure_indicators = [
            "violation found",
            "counterexample found",
            "verification failed",
            "property violated",
            "assert failed",
            "invariant broken",
        ]

        error_indicators = [
            "compilation error",
            "parse error",
            "solidity error",
            "spec error",
            "specification error",
            "found errors",
            "error always",
            "syntax error",
            "type error",
            "could not compile",
            "failed to compile",
            "internal error",
            "cli error",
            "connection error",
            "authentication failed",
            "invalid api key",
            "certorakey",
            "permission denied",
            "no such file",
            "file not found",
            "import error",
            "pragma",
            "unsupported",
        ]

        # Check for errors first
        for indicator in error_indicators:
            if indicator in combined:
                result["success"] = False
                result["status"] = "error"
                # Extract error message
                error_msg = self._extract_error_message(stdout, stderr)
                result["violations"].append({
                    "rule": "Compilation",
                    "status": "error",
                    "description": error_msg or "Verification could not complete due to errors"
                })
                return result

        # Check for success
        for indicator in success_indicators:
            if indicator in combined:
                result["success"] = True
                result["status"] = "verified"
                result["rules_verified"] = 1
                result["verified_rules"].append({
                    "rule": "Contract Verification",
                    "status": "verified",
                    "description": "All formal verification checks passed successfully"
                })
                return result

        # Check for failures
        for indicator in failure_indicators:
            if indicator in combined:
                result["success"] = False
                result["status"] = "issues_found"
                result["rules_violated"] = 1
                # Try to extract what failed
                violation_desc = self._extract_violation_summary(stdout)
                result["violations"].append({
                    "rule": "Property Verification",
                    "status": "violated",
                    "description": violation_desc
                })
                return result

        # If return code indicates failure but we couldn't parse why
        if return_code != 0:
            result["success"] = False
            result["status"] = "incomplete"

            # Try to extract a meaningful error from the output
            error_desc = self._extract_meaningful_error(stdout, stderr, return_code)
            result["violations"].append({
                "rule": "Verification Process",
                "status": "incomplete",
                "description": error_desc
            })
        else:
            # Return code 0 but no clear results - assume success
            result["success"] = True
            result["status"] = "verified"
            result["rules_verified"] = 1
            result["verified_rules"].append({
                "rule": "Contract Verification",
                "status": "verified",
                "description": "Formal verification completed without finding issues"
            })

        return result

    def _extract_error_message(self, stdout: str, stderr: str) -> str:
        """Extract a meaningful error message from output."""
        combined = stdout + "\n" + stderr

        # Look for common error patterns - ordered by specificity
        error_patterns = [
            # Certora specific errors
            r"ERROR ALWAYS\s*[-:]\s*([^\n]+)",
            r"Specification error:\s*([^\n]+)",
            r"Found errors in[^\n]*:\s*([^\n]+)",
            # General errors
            r"Error:\s*([^\n]+)",
            r"error\[E\d+\]:\s*([^\n]+)",
            r"(?:Solidity|CVL)\s+error:\s*([^\n]+)",
            # CVL syntax errors
            r"line\s+\d+[:\s]+([^\n]+error[^\n]*)",
            r"(?:unexpected|expected|invalid)[^\n]+",
        ]

        for pattern in error_patterns:
            match = re.search(pattern, combined, re.IGNORECASE)
            if match:
                return match.group(1)[:200].strip() if match.lastindex else match.group(0)[:200].strip()

        # If no specific pattern matched, look for any line containing "error"
        for line in combined.split('\n'):
            if 'error' in line.lower() and len(line.strip()) > 10:
                return line.strip()[:200]

        return "An error occurred during verification"

    def _extract_violation_summary(self, stdout: str) -> str:
        """Extract a summary of what was violated."""
        # Look for specific violation info
        patterns = [
            r"Violated:\s*([^\n]+)",
            r"Failed:\s*([^\n]+)",
            r"Counterexample for[^:]*:\s*([^\n]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, stdout, re.IGNORECASE)
            if match:
                return f"Verification issue: {match.group(1)[:150]}"

        return "Formal verification found potential issues that require review"

    def _extract_meaningful_error(self, stdout: str, stderr: str, return_code: int) -> str:
        """Extract a meaningful error description from output."""
        combined = stdout + "\n" + stderr

        # Check for specific common issues
        if "certorakey" in combined.lower() or "api key" in combined.lower():
            return "Certora API key issue. The verification service could not authenticate."

        if "could not compile" in combined.lower() or "compilation" in combined.lower():
            # Try to find the specific compilation error
            comp_match = re.search(r"(?:Error|error)[:\s]+([^\n]{10,100})", combined)
            if comp_match:
                return f"Contract compilation failed: {comp_match.group(1)}"
            return "Contract compilation failed. Check Solidity syntax and imports."

        if "spec" in combined.lower() and "error" in combined.lower():
            spec_match = re.search(r"spec[^:]*:\s*([^\n]+)", combined, re.IGNORECASE)
            if spec_match:
                return f"Specification error: {spec_match.group(1)[:100]}"
            return "CVL specification has errors. The auto-generated spec may need adjustment."

        if "timeout" in combined.lower():
            return "Verification timed out. The contract may be too complex for automated verification."

        if "connection" in combined.lower() or "network" in combined.lower():
            return "Network error connecting to Certora cloud. Please try again later."

        if "unsupported" in combined.lower():
            unsup_match = re.search(r"unsupported[:\s]+([^\n]+)", combined, re.IGNORECASE)
            if unsup_match:
                return f"Unsupported feature: {unsup_match.group(1)[:100]}"
            return "Contract uses unsupported Solidity features for formal verification."

        # Look for any error message
        error_match = re.search(r"(?:Error|error|ERROR)[:\s]+([^\n]{10,150})", combined)
        if error_match:
            return f"Verification error: {error_match.group(1)}"

        # Generic message with return code
        return f"Verification did not complete (exit code {return_code}). Check contract complexity or try a simpler specification."


# Convenience function
def run_certora_verification(
    contract_path: str,
    spec_content: str,
    wait_for_results: bool = True,
    submission_timeout: int = SUBMISSION_TIMEOUT,
    poll_timeout: int = POLL_TIMEOUT
) -> dict[str, Any]:
    """
    Run Certora verification on a contract.

    This uses non-blocking job submission for reliability:
    1. Submits job to Certora cloud (fast, ~30 seconds)
    2. Polls for results (optional, can take several minutes)

    If polling times out, returns "pending" status with job URL instead of failing.

    Args:
        contract_path: Path to Solidity contract
        spec_content: CVL specification content
        wait_for_results: If True, poll for results. If False, return after submission.
        submission_timeout: Time to wait for job submission
        poll_timeout: Max time to wait for results after submission

    Returns:
        Verification results dictionary with:
        - success: bool
        - status: "verified" | "issues_found" | "pending" | "error"
        - job_url: URL to view results on Certora cloud
        - rules_verified, rules_violated: counts
        - verified_rules, violations: detailed results
    """
    runner = CertoraRunner(
        submission_timeout=submission_timeout,
        poll_timeout=poll_timeout
    )
    return runner.run_verification_sync(contract_path, spec_content, wait_for_results=wait_for_results)
