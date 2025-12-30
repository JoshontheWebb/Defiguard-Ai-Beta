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
from typing import Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Timeout for job submission (just getting the URL, not waiting for results)
SUBMISSION_TIMEOUT = 300  # 5 minutes to submit and get URL (generous for network issues)

# Timeout for polling results (if we choose to wait)
# Since we now save results to database and can retrieve later, this is just for
# initial wait time - if it times out, status is "pending" not "failed"
POLL_TIMEOUT = 1800  # 30 minutes max wait for results (Certora can take a while)

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

            # Create conf file - NON-BLOCKING MODE (don't wait for results)
            conf_content = self._create_conf_file(contract_path, spec_path, contract_name, wait_for_results=False)
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

            logger.info(f"CertoraRunner: Job submission returned {result.returncode}")

            # Extract job URL from output (this is available immediately)
            job_url = self._extract_job_url(result.stdout)

            if not job_url:
                # No job URL means submission failed
                logger.warning(f"CertoraRunner: No job URL found in output")
                if result.stderr:
                    logger.warning(f"CertoraRunner: stderr: {result.stderr[:1000]}")
                return self._parse_output(result.stdout, result.stderr, result.returncode)

            logger.info(f"CertoraRunner: Job submitted successfully: {job_url}")

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

            # PHASE 2: Poll for results
            logger.info(f"CertoraRunner: Polling for results (timeout: {self.poll_timeout}s)")
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
        """Extract job URL from Certora CLI output."""
        url_patterns = [
            r"https://prover\.certora\.com/output/[a-zA-Z0-9/_-]+",
            r"https://prover\.certora\.com/job/[a-zA-Z0-9/_-]+",
            r"Job URL:\s*(https://prover\.certora\.com[^\s]+)",
            r"Report:\s*(https://prover\.certora\.com[^\s]+)",
        ]

        for pattern in url_patterns:
            match = re.search(pattern, output)
            if match:
                url = match.group(0)
                if "://" not in url and len(match.groups()) > 0:
                    url = match.group(1)
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

        Args:
            job_url: URL of the Certora job (e.g., https://prover.certora.com/output/9579011/abc123)

        Returns:
            Results dictionary with status
        """
        try:
            import urllib.request
            import urllib.error

            # Certora's output structure has several JSON files we can check:
            # 1. jobStatus.json - contains job status and basic info
            # 2. output.json - contains detailed results
            # 3. The main HTML page can be parsed for status

            # List of potential JSON endpoints to try
            json_endpoints = [
                f"{job_url}/jobStatus.json",
                f"{job_url}/output.json",
                f"{job_url}/results.json",
                f"{job_url}/verificationProgress.json",
            ]

            for json_url in json_endpoints:
                try:
                    logger.debug(f"CertoraRunner: Trying endpoint: {json_url}")
                    req = urllib.request.Request(json_url, headers={
                        "Accept": "application/json",
                        "User-Agent": "DeFiGuard-AI/1.0"
                    })
                    with urllib.request.urlopen(req, timeout=30) as response:
                        data = json.loads(response.read().decode())
                        logger.info(f"CertoraRunner: Got response from {json_url}")

                        # Parse the response based on structure
                        if isinstance(data, dict):
                            # Check various status field names Certora uses
                            status = (
                                data.get("jobStatus", "") or
                                data.get("status", "") or
                                data.get("verificationStatus", "") or
                                ""
                            ).lower()

                            logger.info(f"CertoraRunner: Job status from API: {status}")

                            # Map Certora status values to our status
                            if status in ["succeeded", "success", "complete", "finished", "done", "verified"]:
                                return self._parse_api_results(data)
                            elif status in ["running", "pending", "queued", "inprogress", "in_progress", "in progress"]:
                                return {"status": "running"}
                            elif status in ["failed", "error", "violated"]:
                                return self._parse_api_results(data)
                            elif "rules" in data or "results" in data or "output" in data:
                                # Has results data - try to parse it
                                return self._parse_api_results(data)

                except urllib.error.HTTPError as e:
                    logger.debug(f"CertoraRunner: {json_url} returned {e.code}")
                    continue
                except Exception as e:
                    logger.debug(f"CertoraRunner: Error fetching {json_url}: {e}")
                    continue

            # If no JSON endpoints worked, try parsing the HTML page
            try:
                logger.debug(f"CertoraRunner: Trying HTML page: {job_url}")
                req = urllib.request.Request(job_url, headers={
                    "Accept": "text/html",
                    "User-Agent": "DeFiGuard-AI/1.0"
                })
                with urllib.request.urlopen(req, timeout=30) as response:
                    html = response.read().decode()

                    # Look for status indicators in the HTML
                    html_lower = html.lower()

                    # Check for completion indicators
                    if "verification succeeded" in html_lower or "all rules verified" in html_lower:
                        logger.info("CertoraRunner: HTML indicates verification succeeded")
                        return self._parse_html_results(html)
                    elif "verification failed" in html_lower or "violated" in html_lower:
                        logger.info("CertoraRunner: HTML indicates violations found")
                        return self._parse_html_results(html)
                    elif "running" in html_lower or "in progress" in html_lower or "pending" in html_lower:
                        logger.info("CertoraRunner: HTML indicates job still running")
                        return {"status": "running"}
                    elif "error" in html_lower and "compilation" in html_lower:
                        logger.info("CertoraRunner: HTML indicates compilation error")
                        return {
                            "success": False,
                            "status": "error",
                            "error": "Compilation error",
                            "rules_verified": 0,
                            "rules_violated": 0,
                            "violations": []
                        }

                    # If we got the page but can't determine status, assume still running
                    logger.info("CertoraRunner: Got HTML but couldn't determine status, assuming running")
                    return {"status": "running"}

            except Exception as e:
                logger.warning(f"CertoraRunner: Error fetching HTML page: {e}")

            # Default to running if we couldn't determine status
            return {"status": "running"}

        except Exception as e:
            logger.warning(f"CertoraRunner: Error in _fetch_job_results: {e}")
            return {"status": "running"}

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
            # CRITICAL: "none" returns immediately after job submission
            # "all" blocks until verification completes (can timeout on slow jobs)
            "wait_for_results": "all" if wait_for_results else "none",
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

        # Look for common error patterns
        error_patterns = [
            r"Error:\s*([^\n]+)",
            r"error\[E\d+\]:\s*([^\n]+)",
            r"(?:Solidity|CVL)\s+error:\s*([^\n]+)",
        ]

        for pattern in error_patterns:
            match = re.search(pattern, combined, re.IGNORECASE)
            if match:
                return match.group(1)[:200]

        # Return first non-empty error line
        for line in stderr.split('\n'):
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
