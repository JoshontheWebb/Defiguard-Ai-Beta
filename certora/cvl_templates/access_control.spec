/*
 * Access Control Formal Verification Specification
 * Properties for contracts with ownership/role-based access
 */

methods {
    function owner() external returns (address) envfree;
    function transferOwnership(address) external;
    function renounceOwnership() external;
}

/*
 * INVARIANTS
 */

// Owner is never zero address (unless renounced)
invariant ownerIsNonZero()
    owner() != 0;

/*
 * RULES
 */

// Only owner can transfer ownership
rule onlyOwnerCanTransferOwnership(env e, address newOwner) {
    address ownerBefore = owner();
    require e.msg.sender != ownerBefore;

    transferOwnership@withrevert(e, newOwner);

    assert lastReverted;
}

// Ownership transfer sets new owner correctly
rule transferOwnershipSetsNewOwner(env e, address newOwner) {
    require e.msg.sender == owner();
    require newOwner != 0;

    transferOwnership(e, newOwner);

    assert owner() == newOwner;
}

// Cannot transfer to zero address
rule cannotTransferToZeroAddress(env e) {
    require e.msg.sender == owner();

    transferOwnership@withrevert(e, 0);

    assert lastReverted;
}

// Only owner can renounce
rule onlyOwnerCanRenounce(env e) {
    require e.msg.sender != owner();

    renounceOwnership@withrevert(e);

    assert lastReverted;
}

// After renounce, owner is zero
rule renounceSetOwnerToZero(env e) {
    require e.msg.sender == owner();

    renounceOwnership(e);

    assert owner() == 0;
}

/*
 * Generic rule: restricted functions should revert for non-owner
 * Usage: Filter this rule for your specific restricted functions
 */
rule restrictedFunctionRevertsForNonOwner(method f, env e)
    filtered { f -> f.selector == sig:transferOwnership(address).selector
                 || f.selector == sig:renounceOwnership().selector }
{
    require e.msg.sender != owner();

    f@withrevert(e);

    assert lastReverted;
}
