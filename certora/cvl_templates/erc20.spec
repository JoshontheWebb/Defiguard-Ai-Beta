/*
 * ERC20 Token Formal Verification Specification
 * Standard properties for ERC20 token contracts
 */

methods {
    function balanceOf(address) external returns (uint256) envfree;
    function totalSupply() external returns (uint256) envfree;
    function allowance(address, address) external returns (uint256) envfree;
    function transfer(address, uint256) external returns (bool);
    function transferFrom(address, address, uint256) external returns (bool);
    function approve(address, uint256) external returns (bool);
}

/*
 * Ghost variable to track sum of all balances
 */
ghost mathint sumOfBalances {
    init_state axiom sumOfBalances == 0;
}

hook Sstore balanceOf[KEY address account] uint256 newValue (uint256 oldValue) {
    sumOfBalances = sumOfBalances + newValue - oldValue;
}

/*
 * INVARIANTS
 */

// Total supply equals sum of all balances
invariant totalSupplyIsSumOfBalances()
    to_mathint(totalSupply()) == sumOfBalances;

// No single balance exceeds total supply
invariant balanceDoesNotExceedTotalSupply(address account)
    balanceOf(account) <= totalSupply();

/*
 * RULES
 */

// Transfer preserves total supply
rule transferPreservesTotalSupply(env e, address to, uint256 amount) {
    uint256 supplyBefore = totalSupply();

    transfer(e, to, amount);

    uint256 supplyAfter = totalSupply();
    assert supplyAfter == supplyBefore;
}

// Transfer moves exact amount
rule transferMovesExactAmount(env e, address to, uint256 amount) {
    address sender = e.msg.sender;
    require sender != to; // Different addresses

    uint256 senderBalanceBefore = balanceOf(sender);
    uint256 toBalanceBefore = balanceOf(to);

    bool success = transfer(e, to, amount);

    uint256 senderBalanceAfter = balanceOf(sender);
    uint256 toBalanceAfter = balanceOf(to);

    assert success => (
        senderBalanceAfter == senderBalanceBefore - amount &&
        toBalanceAfter == toBalanceBefore + amount
    );
}

// Cannot transfer more than balance
rule cannotTransferMoreThanBalance(env e, address to, uint256 amount) {
    uint256 balance = balanceOf(e.msg.sender);
    require amount > balance;

    transfer@withrevert(e, to, amount);

    assert lastReverted;
}

// Approve sets correct allowance
rule approveSetAllowance(env e, address spender, uint256 amount) {
    approve(e, spender, amount);

    assert allowance(e.msg.sender, spender) == amount;
}

// TransferFrom respects allowance
rule transferFromRespectsAllowance(env e, address from, address to, uint256 amount) {
    uint256 allowanceBefore = allowance(from, e.msg.sender);
    require amount > allowanceBefore;

    transferFrom@withrevert(e, from, to, amount);

    assert lastReverted;
}
