"""
Certora Prover Runner for DeFiGuard AI.

Handles submission of verification jobs to Certora's cloud infrastructure
and parsing of verification results.
"""

import os
import json
import subprocess
import tempfile
import logging
import re
from typing import Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Default timeout for verification (10 minutes)
DEFAULT_TIMEOUT = 600

# Certora cloud job URL pattern
CERTORA_JOB_URL = "https://prover.certora.com/output/{job_id}"


class CertoraRunner:
    """Interface to Certora Prover cloud."""

    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        """
        Initialize Certora Runner.

        Args:
            timeout: Maximum time to wait for verification (seconds)
        """
        self.api_key = os.getenv("CERTORAKEY")
        self.timeout = timeout

        if not self.api_key:
            logger.warning("CertoraRunner: CERTORAKEY not set - verification will be skipped")

    def is_configured(self) -> bool:
        """Check if Certora is properly configured."""
        return bool(self.api_key)

    def run_verification_sync(
        self,
        contract_path: str,
        spec_content: str,
        contract_name: str = None
    ) -> dict[str, Any]:
        """
        Run Certora verification synchronously.

        Args:
            contract_path: Path to Solidity contract file
            spec_content: CVL specification content
            contract_name: Optional contract name (extracted from file if not provided)

        Returns:
            Dictionary with verification results:
            {
                "success": bool,
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

            # Create conf file
            conf_content = self._create_conf_file(contract_path, spec_path, contract_name)
            conf_path = self._write_temp_file(conf_content, suffix=".conf")

            logger.info(f"CertoraRunner: Starting verification for {contract_name}")
            logger.debug(f"CertoraRunner: Contract: {contract_path}")
            logger.debug(f"CertoraRunner: Spec: {spec_path}")

            # Run certoraRun with config file as positional argument
            # Certora CLI expects: certoraRun <config.conf>
            result = subprocess.run(
                ["certoraRun", conf_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "CERTORAKEY": self.api_key}
            )

            logger.info(f"CertoraRunner: certoraRun returned {result.returncode}")
            logger.info(f"CertoraRunner: stdout length: {len(result.stdout)}")

            # Log output for debugging verification issues - always log for visibility
            if result.returncode != 0:
                logger.warning(f"CertoraRunner: Non-zero exit ({result.returncode})")
                logger.warning(f"CertoraRunner: stdout: {result.stdout[:2000]}")
                if result.stderr:
                    logger.warning(f"CertoraRunner: stderr: {result.stderr[:1000]}")

            # Parse output
            return self._parse_output(result.stdout, result.stderr, result.returncode)

        except subprocess.TimeoutExpired:
            logger.warning(f"CertoraRunner: Verification timed out after {self.timeout}s")
            return {
                "success": False,
                "status": "timeout",
                "error": f"Verification timed out after {self.timeout} seconds",
                "rules_verified": 0,
                "rules_violated": 0,
                "violations": []
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

    async def run_verification(
        self,
        contract_path: str,
        spec_content: str,
        contract_name: str = None
    ) -> dict[str, Any]:
        """
        Run Certora verification asynchronously.

        Args:
            contract_path: Path to Solidity contract file
            spec_content: CVL specification content
            contract_name: Optional contract name

        Returns:
            Dictionary with verification results
        """
        import asyncio
        return await asyncio.to_thread(
            self.run_verification_sync,
            contract_path,
            spec_content,
            contract_name
        )

    def _write_temp_file(self, content: str, suffix: str) -> str:
        """Write content to a temporary file and return the path."""
        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(content)
            return path
        except Exception:
            os.close(fd)
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
            pattern = r"contract\s+(\w+)"
            match = re.search(pattern, content)
            return match.group(1) if match else Path(contract_path).stem
        except Exception:
            return Path(contract_path).stem

    def _create_conf_file(
        self,
        contract_path: str,
        spec_path: str,
        contract_name: str
    ) -> str:
        """Create Certora configuration file content."""

        # JSON format conf file for Certora Prover
        # Use path:contract format to handle UUID filenames with hyphens
        # Get the directory containing the contract for solc allow_path
        contract_dir = str(Path(contract_path).parent)

        conf = {
            "files": [f"{contract_path}:{contract_name}"],
            "verify": f"{contract_name}:{spec_path}",
            "msg": f"DeFiGuard AI: {contract_name}",
            "wait_for_results": "all",  # Wait for verification to complete
            "rule_sanity": "basic",  # Check for tautologies
            "optimistic_loop": True,  # Assume loops terminate
            "loop_iter": 3,  # Unroll loops 3 times
            "process": "evm",  # EVM mode
            "solc": "solc",  # Use system solc
            "solc_allow_path": [contract_dir, "/opt/render/project/data", "/tmp"],  # Allow temp dirs
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
    timeout: int = DEFAULT_TIMEOUT
) -> dict[str, Any]:
    """
    Run Certora verification on a contract.

    Args:
        contract_path: Path to Solidity contract
        spec_content: CVL specification content
        timeout: Maximum verification time in seconds

    Returns:
        Verification results dictionary
    """
    runner = CertoraRunner(timeout=timeout)
    return runner.run_verification_sync(contract_path, spec_content)
