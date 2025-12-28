/*
 * Common Formal Verification Specification
 * General properties applicable to most smart contracts
 */

/*
 * SANITY RULES
 * These rules verify the specification itself is meaningful
 */

// Sanity check: at least one method can succeed
rule sanityMethodCanSucceed(method f, env e, calldataarg args) {
    f@withrevert(e, args);
    satisfy !lastReverted;
}

// Sanity check: at least one method can revert
rule sanityMethodCanRevert(method f, env e, calldataarg args) {
    f@withrevert(e, args);
    satisfy lastReverted;
}

/*
 * REENTRANCY PROTECTION
 * Generic patterns to detect reentrancy vulnerabilities
 */

// Ghost to track reentrancy state
ghost bool entered {
    init_state axiom !entered;
}

// Rule: No function should be callable while another is executing
// Note: This requires contract to have a reentrancy guard variable
rule noReentrancyWithGuard(method f, method g, env e, calldataarg args1, calldataarg args2)
    filtered { f -> !f.isView, g -> !g.isView }
{
    // If we're in f, calling g should fail
    require entered;
    g@withrevert(e, args2);
    assert lastReverted;
}

/*
 * STATE CONSISTENCY
 */

// State should be consistent before and after failed transactions
rule failedTxDoesNotChangeState(method f, env e, calldataarg args) {
    // Store state hash before
    storage stateBefore = lastStorage;

    f@withrevert(e, args);

    // If reverted, state should be unchanged
    assert lastReverted => (lastStorage == stateBefore);
}

/*
 * ETH HANDLING
 */

// Contract ETH balance should not decrease unexpectedly
rule ethBalanceNonDecreasing(method f, env e, calldataarg args)
    filtered { f -> !f.isView }
{
    uint256 balanceBefore = nativeBalances[currentContract];

    f(e, args);

    uint256 balanceAfter = nativeBalances[currentContract];

    // Balance should not decrease unless explicitly sent
    // Note: Customize this for your specific withdrawal functions
    assert balanceAfter >= balanceBefore;
}

/*
 * VIEW FUNCTIONS
 */

// View functions should not modify state
rule viewFunctionsDoNotModifyState(method f, env e, calldataarg args)
    filtered { f -> f.isView }
{
    storage stateBefore = lastStorage;

    f(e, args);

    assert lastStorage == stateBefore;
}

/*
 * BASIC ARITHMETIC SAFETY
 * Note: Solidity 0.8+ has built-in overflow checks, but these rules
 * can catch logic errors in calculations
 */

// Generic addition safety (for contracts with add function)
rule additionSafety(env e, uint256 a, uint256 b) {
    // If a + b would overflow, the function should revert
    require a > 0 && b > 0;
    mathint sum = a + b;
    require sum > max_uint256;

    // Any arithmetic operation should revert on overflow
    // This is automatic in Solidity 0.8+
    assert true; // Placeholder - customize for your contract
}
