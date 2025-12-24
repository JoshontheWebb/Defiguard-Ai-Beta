#!/usr/bin/env python3
"""
Load test script for DeFiGuard AI concurrent audit handling.
Tests how the system handles multiple simultaneous audit requests.

Usage:
    python load_test.py --url https://your-app.onrender.com --users 5
    python load_test.py --url http://localhost:8000 --users 10
"""

import asyncio
import aiohttp
import argparse
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
import statistics


@dataclass
class AuditResult:
    user_id: int
    success: bool
    queued: bool
    queue_position: Optional[int]
    response_time_ms: float
    error: Optional[str] = None
    job_id: Optional[str] = None


# Simple test contract
TEST_CONTRACT = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TestContract {
    uint public value;
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    function setValue(uint _value) public {
        value = _value;
    }

    function withdraw() public {
        // Intentional vulnerability for testing
        payable(msg.sender).transfer(address(this).balance);
    }
}
"""


async def submit_audit(session: aiohttp.ClientSession, base_url: str, user_id: int) -> AuditResult:
    """Submit a single audit request and track the result."""
    start_time = time.time()

    try:
        # Create form data with the contract file
        data = aiohttp.FormData()
        data.add_field(
            'file',
            TEST_CONTRACT.encode('utf-8'),
            filename=f'test_user_{user_id}.sol',
            content_type='text/plain'
        )

        async with session.post(
            f"{base_url}/audit",
            data=data,
            timeout=aiohttp.ClientTimeout(total=300)  # 5 min timeout
        ) as response:
            elapsed_ms = (time.time() - start_time) * 1000
            result_json = await response.json()

            if response.status == 200:
                # Check if queued or processed immediately
                queued = result_json.get('queued', False)
                return AuditResult(
                    user_id=user_id,
                    success=True,
                    queued=queued,
                    queue_position=result_json.get('position'),
                    response_time_ms=elapsed_ms,
                    job_id=result_json.get('job_id')
                )
            else:
                return AuditResult(
                    user_id=user_id,
                    success=False,
                    queued=False,
                    queue_position=None,
                    response_time_ms=elapsed_ms,
                    error=f"HTTP {response.status}: {result_json.get('detail', 'Unknown error')}"
                )

    except asyncio.TimeoutError:
        elapsed_ms = (time.time() - start_time) * 1000
        return AuditResult(
            user_id=user_id,
            success=False,
            queued=False,
            queue_position=None,
            response_time_ms=elapsed_ms,
            error="Request timed out (>5 min)"
        )
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        return AuditResult(
            user_id=user_id,
            success=False,
            queued=False,
            queue_position=None,
            response_time_ms=elapsed_ms,
            error=str(e)
        )


async def run_load_test(base_url: str, num_users: int, stagger_ms: int = 100):
    """Run concurrent audit requests and collect results."""
    print(f"\n{'='*60}")
    print(f"LOAD TEST: {num_users} concurrent users")
    print(f"Target: {base_url}")
    print(f"Stagger: {stagger_ms}ms between requests")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    results: list[AuditResult] = []

    async with aiohttp.ClientSession() as session:
        # Create tasks with slight stagger to simulate realistic traffic
        tasks = []
        for i in range(num_users):
            task = asyncio.create_task(submit_audit(session, base_url, i + 1))
            tasks.append(task)
            if stagger_ms > 0 and i < num_users - 1:
                await asyncio.sleep(stagger_ms / 1000)

        # Wait for all to complete
        print(f"All {num_users} requests submitted, waiting for responses...\n")
        results = await asyncio.gather(*tasks)

    return results


def print_results(results: list[AuditResult]):
    """Print a summary of the load test results."""
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}\n")

    # Individual results
    print("Individual Results:")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x.user_id):
        status = "OK" if r.success else "FAIL"
        queue_info = f"(queued #{r.queue_position})" if r.queued else "(immediate)"
        error_info = f" - {r.error}" if r.error else ""
        print(f"  User {r.user_id:2d}: {status:4s} {queue_info:20s} {r.response_time_ms:8.0f}ms{error_info}")

    # Statistics
    print(f"\n{'='*60}")
    print("STATISTICS")
    print(f"{'='*60}\n")

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    queued = [r for r in results if r.queued]
    immediate = [r for r in successful if not r.queued]

    print(f"  Total requests:     {len(results)}")
    print(f"  Successful:         {len(successful)} ({100*len(successful)/len(results):.1f}%)")
    print(f"  Failed:             {len(failed)} ({100*len(failed)/len(results):.1f}%)")
    print(f"  Processed immediately: {len(immediate)}")
    print(f"  Queued:             {len(queued)}")

    if queued:
        positions = [r.queue_position for r in queued if r.queue_position]
        if positions:
            print(f"  Max queue position: {max(positions)}")

    response_times = [r.response_time_ms for r in results]
    if response_times:
        print(f"\n  Response times:")
        print(f"    Min:    {min(response_times):,.0f}ms")
        print(f"    Max:    {max(response_times):,.0f}ms")
        print(f"    Mean:   {statistics.mean(response_times):,.0f}ms")
        print(f"    Median: {statistics.median(response_times):,.0f}ms")

    # Errors breakdown
    if failed:
        print(f"\n  Errors:")
        error_counts = {}
        for r in failed:
            error_counts[r.error] = error_counts.get(r.error, 0) + 1
        for error, count in error_counts.items():
            print(f"    {count}x: {error}")

    print(f"\n{'='*60}")

    # Verdict
    if len(successful) == len(results):
        print("VERDICT: ALL REQUESTS HANDLED SUCCESSFULLY")
    elif len(successful) >= len(results) * 0.9:
        print("VERDICT: MOSTLY SUCCESSFUL (>90%)")
    elif len(successful) >= len(results) * 0.5:
        print("VERDICT: PARTIAL SUCCESS (50-90%)")
    else:
        print("VERDICT: SIGNIFICANT FAILURES (<50% success)")
    print(f"{'='*60}\n")


async def main():
    parser = argparse.ArgumentParser(description='Load test DeFiGuard AI audit endpoint')
    parser.add_argument('--url', type=str, default='http://localhost:8000',
                       help='Base URL of the DeFiGuard API')
    parser.add_argument('--users', type=int, default=5,
                       help='Number of concurrent users to simulate')
    parser.add_argument('--stagger', type=int, default=100,
                       help='Milliseconds between request starts (0 for true simultaneous)')

    args = parser.parse_args()

    results = await run_load_test(args.url, args.users, args.stagger)
    print_results(results)


if __name__ == "__main__":
    asyncio.run(main())
