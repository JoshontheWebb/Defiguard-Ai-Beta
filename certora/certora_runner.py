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

            # Run certoraRun
            result = subprocess.run(
                [
                    "certoraRun",
                    conf_path,
                    "--wait",  # Wait for results
                    "--msg", f"DeFiGuard AI verification: {contract_name}"
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "CERTORAKEY": self.api_key}
            )

            logger.info(f"CertoraRunner: certoraRun returned {result.returncode}")
            logger.debug(f"CertoraRunner: stdout length: {len(result.stdout)}")

            if result.stderr:
                logger.warning(f"CertoraRunner: stderr: {result.stderr[:500]}")

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
                "violations": []
            }

        except Exception as e:
            logger.error(f"CertoraRunner: Unexpected error: {e}")
            return {
                "success": False,
                "status": "error",
                "error": str(e),
                "rules_verified": 0,
                "rules_violated": 0,
                "violations": []
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

        # JSON5 format conf file
        conf = {
            "files": [contract_path],
            "verify": f"{contract_name}:{spec_path}",
            "msg": f"DeFiGuard AI: {contract_name}",
            "rule_sanity": "basic",  # Check for tautologies
            "optimistic_loop": True,  # Assume loops terminate
            "loop_iter": 3,  # Unroll loops 3 times
            "process": "emv",  # EVM mode
            "solc": "solc",  # Use system solc
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

        # Extract job URL
        url_pattern = r"https://prover\.certora\.com/output/[a-zA-Z0-9/]+"
        url_match = re.search(url_pattern, stdout)
        if url_match:
            result["job_url"] = url_match.group(0)

        # Parse rule results from output
        # Pattern: "Rule: <name> - <status>"
        rule_pattern = r"(?:Rule|Invariant):\s*(\w+)\s*[-:]\s*(Verified|Violated|Timeout|Error)"
        rule_matches = re.findall(rule_pattern, stdout, re.IGNORECASE)

        for rule_name, status in rule_matches:
            status_lower = status.lower()
            if status_lower == "verified":
                result["rules_verified"] += 1
                result["verified_rules"].append({
                    "rule": rule_name,
                    "status": "verified"
                })
            elif status_lower == "violated":
                result["rules_violated"] += 1
                result["violations"].append({
                    "rule": rule_name,
                    "status": "violated",
                    "description": f"Rule {rule_name} was violated"
                })
            elif status_lower == "timeout":
                result["rules_timeout"] += 1
                result["violations"].append({
                    "rule": rule_name,
                    "status": "timeout",
                    "description": f"Rule {rule_name} timed out"
                })

        # If we couldn't parse specific rules, check for general success/failure
        if not rule_matches:
            if "verification succeeded" in stdout.lower():
                result["success"] = True
                result["status"] = "verified"
                result["rules_verified"] = 1
                result["verified_rules"].append({
                    "rule": "all",
                    "status": "verified"
                })
            elif "violation" in stdout.lower() or "counterexample" in stdout.lower():
                result["success"] = False
                result["status"] = "violated"
                result["rules_violated"] = 1
                result["violations"].append({
                    "rule": "unknown",
                    "status": "violated",
                    "description": "Verification found violations"
                })

        # Check for errors in stderr
        if stderr and "error" in stderr.lower():
            if not result["violations"]:
                result["success"] = False
                result["status"] = "error"
                result["error"] = stderr[:500]

        return result


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
