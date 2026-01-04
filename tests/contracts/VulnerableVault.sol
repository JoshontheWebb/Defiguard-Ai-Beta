// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title VulnerableVault
 * @notice A deliberately vulnerable contract for testing DeFiGuard AI security tools
 * @dev Contains multiple vulnerability patterns for comprehensive testing:
 *      - Reentrancy (multiple variants)
 *      - Access control issues
 *      - Oracle manipulation
 *      - Flash loan vulnerabilities
 *      - Integer issues
 *      - Unchecked calls
 *      - Centralization risks
 *      - Front-running vulnerabilities
 *      - Delegate call dangers
 *      - Tx.origin authentication
 */

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
}

interface IOracle {
    function getPrice(address token) external view returns (uint256);
}

interface IFlashLoanReceiver {
    function executeOperation(uint256 amount, uint256 fee, bytes calldata data) external;
}

contract VulnerableVault {
    // ═══════════════════════════════════════════════════════════════════════════
    // STATE VARIABLES
    // ═══════════════════════════════════════════════════════════════════════════

    // Structs - Tests CVL struct handling
    struct UserInfo {
        uint256 depositAmount;
        uint256 rewardDebt;
        uint256 lastDepositTime;
        string nickname;  // String in struct - CVL should handle
    }

    struct Proposal {
        uint256 id;
        address proposer;
        bytes data;  // Bytes in struct
        uint256 votes;
        bool executed;
    }

    // Enums - Tests CVL enum handling
    enum VaultState { Active, Paused, Emergency, Deprecated }
    enum UserTier { Bronze, Silver, Gold, Platinum }

    // Mappings
    mapping(address => uint256) public balances;
    mapping(address => uint256) public shares;
    mapping(address => UserInfo) public userInfo;
    mapping(address => mapping(address => uint256)) public allowances;
    mapping(uint256 => Proposal) public proposals;
    mapping(address => bool) public isAdmin;
    mapping(address => UserTier) public userTiers;

    // State variables
    uint256 public totalSupply;
    uint256 public totalShares;
    uint256 public totalDeposits;
    uint256 public feePercent = 100; // 1% = 100 basis points
    uint256 public proposalCount;
    uint256 public minDeposit = 0.01 ether;
    uint256 public withdrawalDelay = 1 days;

    address public owner;
    address public pendingOwner;
    address public feeRecipient;
    address public oracle;
    address public token;

    VaultState public vaultState;

    bool public locked;
    bool private _notEntered = true;

    // Constants
    uint256 public constant MAX_FEE = 1000; // 10%
    uint256 public constant PRECISION = 1e18;

    // ═══════════════════════════════════════════════════════════════════════════
    // EVENTS
    // ═══════════════════════════════════════════════════════════════════════════

    event Deposit(address indexed user, uint256 amount, uint256 shares);
    event Withdraw(address indexed user, uint256 amount, uint256 shares);
    event FlashLoan(address indexed receiver, uint256 amount, uint256 fee);
    event PriceUpdated(address indexed token, uint256 price);
    event OwnershipTransferred(address indexed oldOwner, address indexed newOwner);

    // ═══════════════════════════════════════════════════════════════════════════
    // MODIFIERS
    // ═══════════════════════════════════════════════════════════════════════════

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier nonReentrant() {
        require(_notEntered, "Reentrant call");
        _notEntered = false;
        _;
        _notEntered = true;
    }

    modifier whenNotPaused() {
        require(vaultState == VaultState.Active, "Vault paused");
        _;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════════════

    constructor(address _token, address _oracle) {
        owner = msg.sender;
        token = _token;
        oracle = _oracle;
        feeRecipient = msg.sender;
        vaultState = VaultState.Active;
        isAdmin[msg.sender] = true;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // VULNERABILITY 1: REENTRANCY (Classic)
    // External call before state update
    // ═══════════════════════════════════════════════════════════════════════════

    function withdrawUnsafe(uint256 amount) external whenNotPaused {
        require(balances[msg.sender] >= amount, "Insufficient balance");

        // VULNERABLE: External call before state update
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");

        // State update after external call - REENTRANCY!
        balances[msg.sender] -= amount;
        totalDeposits -= amount;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // VULNERABILITY 2: REENTRANCY (Cross-function)
    // State shared between functions
    // ═══════════════════════════════════════════════════════════════════════════

    function depositETH() external payable whenNotPaused {
        uint256 sharesToMint = _calculateShares(msg.value);

        // State updates
        balances[msg.sender] += msg.value;
        shares[msg.sender] += sharesToMint;
        totalDeposits += msg.value;
        totalShares += sharesToMint;

        emit Deposit(msg.sender, msg.value, sharesToMint);
    }

    function withdrawShares(uint256 shareAmount) external whenNotPaused {
        require(shares[msg.sender] >= shareAmount, "Insufficient shares");

        uint256 ethAmount = _calculateETHForShares(shareAmount);

        // VULNERABLE: Cross-function reentrancy possible
        // An attacker can call depositETH during this call
        (bool success, ) = msg.sender.call{value: ethAmount}("");
        require(success, "Transfer failed");

        shares[msg.sender] -= shareAmount;
        totalShares -= shareAmount;
        balances[msg.sender] -= ethAmount;
        totalDeposits -= ethAmount;

        emit Withdraw(msg.sender, ethAmount, shareAmount);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // VULNERABILITY 3: TX.ORIGIN AUTHENTICATION
    // Using tx.origin instead of msg.sender
    // ═══════════════════════════════════════════════════════════════════════════

    function transferOwnershipUnsafe(address newOwner) external {
        // VULNERABLE: tx.origin can be manipulated via phishing
        require(tx.origin == owner, "Not owner");
        owner = newOwner;
        emit OwnershipTransferred(msg.sender, newOwner);
    }

    function emergencyWithdrawByOrigin() external {
        // VULNERABLE: tx.origin authentication
        require(tx.origin == owner, "Not authorized");
        uint256 balance = balances[tx.origin];
        balances[tx.origin] = 0;
        payable(tx.origin).transfer(balance);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // VULNERABILITY 4: UNCHECKED EXTERNAL CALLS
    // Return values not checked
    // ═══════════════════════════════════════════════════════════════════════════

    function transferTokensUnsafe(address to, uint256 amount) external onlyOwner {
        // VULNERABLE: Return value not checked
        IERC20(token).transfer(to, amount);
    }

    function batchTransferUnsafe(address[] calldata recipients, uint256[] calldata amounts) external onlyOwner {
        for (uint256 i = 0; i < recipients.length; i++) {
            // VULNERABLE: Unchecked low-level call
            recipients[i].call{value: amounts[i]}("");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // VULNERABILITY 5: ORACLE MANIPULATION / PRICE DEPENDENCY
    // Using spot price without TWAP
    // ═══════════════════════════════════════════════════════════════════════════

    function swapWithOraclePrice(address tokenIn, uint256 amountIn) external whenNotPaused {
        // VULNERABLE: Spot price can be manipulated in same block
        uint256 price = IOracle(oracle).getPrice(tokenIn);
        uint256 amountOut = (amountIn * price) / PRECISION;

        require(IERC20(tokenIn).transferFrom(msg.sender, address(this), amountIn), "Transfer in failed");

        // User could manipulate price via flash loan before this call
        (bool success, ) = msg.sender.call{value: amountOut}("");
        require(success, "Transfer out failed");
    }

    function liquidate(address user) external {
        // VULNERABLE: Price manipulation can trigger unfair liquidation
        uint256 collateralValue = _getCollateralValue(user);
        uint256 debtValue = balances[user];

        // No minimum health factor check, easily manipulatable
        require(collateralValue < debtValue, "User is healthy");

        // Liquidator gets all collateral
        uint256 collateral = shares[user];
        shares[user] = 0;
        shares[msg.sender] += collateral;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // VULNERABILITY 6: FLASH LOAN WITHOUT PROPER CHECKS
    // Missing repayment verification
    // ═══════════════════════════════════════════════════════════════════════════

    function flashLoan(uint256 amount, bytes calldata data) external whenNotPaused {
        uint256 balanceBefore = address(this).balance;
        require(balanceBefore >= amount, "Insufficient liquidity");

        uint256 fee = (amount * feePercent) / 10000;

        // Send ETH to receiver
        (bool sent, ) = msg.sender.call{value: amount}("");
        require(sent, "Flash loan transfer failed");

        // Execute callback
        IFlashLoanReceiver(msg.sender).executeOperation(amount, fee, data);

        // VULNERABLE: Balance check can be manipulated
        // Attacker could deposit during callback to satisfy check
        require(address(this).balance >= balanceBefore + fee, "Flash loan not repaid");

        emit FlashLoan(msg.sender, amount, fee);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // VULNERABILITY 7: DELEGATE CALL TO ARBITRARY ADDRESS
    // Allows code execution in vault context
    // ═══════════════════════════════════════════════════════════════════════════

    function executeTransaction(address target, bytes calldata data) external onlyOwner {
        // VULNERABLE: Owner can delegatecall to any contract
        // This could be used to change storage maliciously
        (bool success, ) = target.delegatecall(data);
        require(success, "Delegatecall failed");
    }

    function upgradeImplementation(address newImpl) external {
        // VULNERABLE: No access control on upgrade
        (bool success, ) = newImpl.delegatecall(
            abi.encodeWithSignature("initialize()")
        );
        require(success, "Upgrade failed");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // VULNERABILITY 8: INTEGER ISSUES
    // Division before multiplication, unchecked math
    // ═══════════════════════════════════════════════════════════════════════════

    function calculateReward(uint256 amount, uint256 rate, uint256 time) public pure returns (uint256) {
        // VULNERABLE: Division before multiplication causes precision loss
        return amount / PRECISION * rate * time;
    }

    function calculateFee(uint256 amount) public view returns (uint256) {
        // VULNERABLE: Can overflow on large amounts (pre-0.8.0 pattern, but still bad practice)
        return amount * feePercent / 10000;
    }

    function unsafeIncrement(uint256 value) external pure returns (uint256) {
        // Note: Solidity 0.8+ has built-in overflow checks
        // But unchecked blocks bypass them
        unchecked {
            // VULNERABLE: Unchecked arithmetic
            return value + 1;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // VULNERABILITY 9: MISSING ACCESS CONTROL
    // Critical functions without proper restrictions
    // ═══════════════════════════════════════════════════════════════════════════

    function setFeeRecipient(address _feeRecipient) external {
        // VULNERABLE: No access control - anyone can change fee recipient
        feeRecipient = _feeRecipient;
    }

    function setOracle(address _oracle) external {
        // VULNERABLE: No access control on critical oracle update
        oracle = _oracle;
    }

    function pause() external {
        // VULNERABLE: Anyone can pause the vault
        vaultState = VaultState.Paused;
    }

    function addAdmin(address admin) external {
        // VULNERABLE: No check on who can add admins
        isAdmin[admin] = true;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // VULNERABILITY 10: FRONT-RUNNING VULNERABLE
    // Predictable operations that can be front-run
    // ═══════════════════════════════════════════════════════════════════════════

    function claimReward(bytes32 secretHash, bytes32 secret) external {
        // VULNERABLE: Secret revealed in mempool, can be front-run
        require(keccak256(abi.encodePacked(secret)) == secretHash, "Invalid secret");

        uint256 reward = _calculatePendingReward(msg.sender);
        balances[msg.sender] += reward;
    }

    function submitLimitOrder(uint256 price, uint256 amount, bool isBuy) external {
        // VULNERABLE: Order visible in mempool, can be sandwiched
        if (isBuy) {
            require(msg.value >= price * amount, "Insufficient ETH");
            shares[msg.sender] += amount;
        } else {
            require(shares[msg.sender] >= amount, "Insufficient shares");
            shares[msg.sender] -= amount;
            payable(msg.sender).transfer(price * amount);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // VULNERABILITY 11: FIRST DEPOSITOR / INFLATION ATTACK
    // Share calculation vulnerable to manipulation
    // ═══════════════════════════════════════════════════════════════════════════

    function deposit(uint256 amount) external whenNotPaused returns (uint256 sharesToMint) {
        require(amount >= minDeposit, "Below minimum deposit");

        require(IERC20(token).transferFrom(msg.sender, address(this), amount), "Transfer failed");

        // VULNERABLE: First depositor can manipulate share price
        if (totalShares == 0) {
            sharesToMint = amount;
        } else {
            // VULNERABLE: Can be manipulated by donating to vault before others deposit
            sharesToMint = (amount * totalShares) / totalDeposits;
        }

        // VULNERABLE: No minimum shares check - could mint 0 shares
        shares[msg.sender] += sharesToMint;
        totalShares += sharesToMint;
        totalDeposits += amount;

        emit Deposit(msg.sender, amount, sharesToMint);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // VULNERABILITY 12: UNPROTECTED SELFDESTRUCT
    // Can destroy contract and steal funds
    // ═══════════════════════════════════════════════════════════════════════════

    function destroy() external {
        // VULNERABLE: No access control on selfdestruct
        selfdestruct(payable(msg.sender));
    }

    function destroyWithRecipient(address payable recipient) external onlyOwner {
        // Still dangerous even with onlyOwner - funds sent to arbitrary address
        selfdestruct(recipient);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // VULNERABILITY 13: CENTRALIZATION RISKS
    // Single admin with too much power
    // ═══════════════════════════════════════════════════════════════════════════

    function emergencyWithdrawAll() external onlyOwner {
        // VULNERABLE: Owner can drain all funds
        payable(owner).transfer(address(this).balance);
    }

    function setFee(uint256 _feePercent) external onlyOwner {
        // VULNERABLE: No max fee limit enforced
        feePercent = _feePercent;
    }

    function blacklistUser(address user) external onlyOwner {
        // VULNERABLE: Can lock user funds indefinitely
        balances[user] = 0;
        shares[user] = 0;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // VULNERABILITY 14: SIGNATURE REPLAY
    // Missing nonce and chain ID validation
    // ═══════════════════════════════════════════════════════════════════════════

    function withdrawWithSignature(
        uint256 amount,
        uint256 deadline,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external {
        require(block.timestamp <= deadline, "Signature expired");

        // VULNERABLE: No nonce - same signature can be replayed
        // VULNERABLE: No chain ID - replay across chains
        bytes32 hash = keccak256(abi.encodePacked(msg.sender, amount, deadline));
        address signer = ecrecover(hash, v, r, s);

        require(signer == owner, "Invalid signature");
        require(balances[msg.sender] >= amount, "Insufficient balance");

        balances[msg.sender] -= amount;
        payable(msg.sender).transfer(amount);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SAFE FUNCTIONS (For comparison and CVL verification)
    // ═══════════════════════════════════════════════════════════════════════════

    function withdrawSafe(uint256 amount) external nonReentrant whenNotPaused {
        require(balances[msg.sender] >= amount, "Insufficient balance");

        // Checks-Effects-Interactions pattern
        balances[msg.sender] -= amount;
        totalDeposits -= amount;

        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");

        emit Withdraw(msg.sender, amount, 0);
    }

    function transferOwnershipSafe(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid address");
        pendingOwner = newOwner;
    }

    function acceptOwnership() external {
        require(msg.sender == pendingOwner, "Not pending owner");
        address oldOwner = owner;
        owner = pendingOwner;
        pendingOwner = address(0);
        emit OwnershipTransferred(oldOwner, owner);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // VIEW FUNCTIONS (For CVL verification)
    // ═══════════════════════════════════════════════════════════════════════════

    function getBalance(address user) external view returns (uint256) {
        return balances[user];
    }

    function getShares(address user) external view returns (uint256) {
        return shares[user];
    }

    function getVaultState() external view returns (VaultState) {
        return vaultState;
    }

    function getUserTier(address user) external view returns (UserTier) {
        return userTiers[user];
    }

    function getTotalValue() external view returns (uint256) {
        return address(this).balance + totalDeposits;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // INTERNAL FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    function _calculateShares(uint256 amount) internal view returns (uint256) {
        if (totalShares == 0) {
            return amount;
        }
        return (amount * totalShares) / totalDeposits;
    }

    function _calculateETHForShares(uint256 shareAmount) internal view returns (uint256) {
        if (totalShares == 0) {
            return 0;
        }
        return (shareAmount * totalDeposits) / totalShares;
    }

    function _getCollateralValue(address user) internal view returns (uint256) {
        return shares[user] * IOracle(oracle).getPrice(token) / PRECISION;
    }

    function _calculatePendingReward(address user) internal view returns (uint256) {
        UserInfo storage info = userInfo[user];
        return info.depositAmount * feePercent / 10000;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // RECEIVE FUNCTION
    // ═══════════════════════════════════════════════════════════════════════════

    receive() external payable {
        balances[msg.sender] += msg.value;
        totalDeposits += msg.value;
    }

    fallback() external payable {
        // VULNERABLE: Accepts any call with ETH
        balances[msg.sender] += msg.value;
    }
}
