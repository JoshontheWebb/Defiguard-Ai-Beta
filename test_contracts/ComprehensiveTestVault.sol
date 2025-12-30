// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title ComprehensiveTestVault
 * @notice A test contract that includes ALL patterns detected by DeFiGuard's CVL generator
 * @dev This contract is designed to test formal verification coverage
 *
 * PATTERNS INCLUDED:
 * ✓ ERC20/Token (totalSupply, balances)
 * ✓ Transfer (transfer function)
 * ✓ TransferFrom/Allowance
 * ✓ Minting
 * ✓ Burning
 * ✓ Ownable (owner, transferOwnership, renounceOwnership)
 * ✓ Pausable (pause, unpause, paused)
 * ✓ Vault (deposit, withdraw)
 * ✓ Staking (stake, unstake)
 * ✓ Oracle (getPrice, setPrice)
 * ✓ FlashLoan
 * ✓ ReentrancyGuard (nonReentrant modifier)
 */
contract ComprehensiveTestVault {
    // ═══════════════════════════════════════════════════════════════════
    // STATE VARIABLES
    // ═══════════════════════════════════════════════════════════════════

    // ERC20 Token State
    string public name = "Test Vault Token";
    string public symbol = "TVT";
    uint8 public decimals = 18;
    uint256 public totalSupply;

    mapping(address => uint256) public balances;
    mapping(address => mapping(address => uint256)) public allowance;

    // Ownership State
    address public owner;

    // Pausable State
    bool public paused;

    // Staking State
    mapping(address => uint256) public stakedBalance;
    uint256 public totalStaked;
    mapping(address => uint256) public rewards;
    uint256 public rewardRate = 100; // basis points per period

    // Oracle State
    uint256 private _price = 1e18; // 1 USD default
    address public oracle;

    // Vault State
    uint256 public totalAssets;

    // Reentrancy Guard
    uint256 private _status;
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;

    // Flash Loan State
    uint256 public flashLoanFee = 9; // 0.09%

    // ═══════════════════════════════════════════════════════════════════
    // EVENTS
    // ═══════════════════════════════════════════════════════════════════

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event Deposit(address indexed user, uint256 assets, uint256 shares);
    event Withdraw(address indexed user, uint256 shares, uint256 assets);
    event Staked(address indexed user, uint256 amount);
    event Unstaked(address indexed user, uint256 amount);
    event RewardsClaimed(address indexed user, uint256 amount);
    event Paused(address account);
    event Unpaused(address account);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event PriceUpdated(uint256 oldPrice, uint256 newPrice);
    event FlashLoan(address indexed receiver, uint256 amount, uint256 fee);

    // ═══════════════════════════════════════════════════════════════════
    // MODIFIERS
    // ═══════════════════════════════════════════════════════════════════

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier whenNotPaused() {
        require(!paused, "Paused");
        _;
    }

    modifier whenPaused() {
        require(paused, "Not paused");
        _;
    }

    modifier nonReentrant() {
        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");
        _status = _ENTERED;
        _;
        _status = _NOT_ENTERED;
    }

    // ═══════════════════════════════════════════════════════════════════
    // CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════

    constructor() {
        owner = msg.sender;
        oracle = msg.sender;
        _status = _NOT_ENTERED;

        // Initial mint to deployer
        uint256 initialSupply = 1000000 * 10**decimals;
        balances[msg.sender] = initialSupply;
        totalSupply = initialSupply;
        emit Transfer(address(0), msg.sender, initialSupply);
    }

    // ═══════════════════════════════════════════════════════════════════
    // ERC20 FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════

    function balanceOf(address account) public view returns (uint256) {
        return balances[account];
    }

    function transfer(address to, uint256 amount) public whenNotPaused returns (bool) {
        require(to != address(0), "Transfer to zero address");
        require(balances[msg.sender] >= amount, "Insufficient balance");

        balances[msg.sender] -= amount;
        balances[to] += amount;

        emit Transfer(msg.sender, to, amount);
        return true;
    }

    function approve(address spender, uint256 amount) public returns (bool) {
        require(spender != address(0), "Approve to zero address");

        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) public whenNotPaused returns (bool) {
        require(from != address(0), "Transfer from zero address");
        require(to != address(0), "Transfer to zero address");
        require(balances[from] >= amount, "Insufficient balance");
        require(allowance[from][msg.sender] >= amount, "Insufficient allowance");

        balances[from] -= amount;
        balances[to] += amount;
        allowance[from][msg.sender] -= amount;

        emit Transfer(from, to, amount);
        return true;
    }

    // ═══════════════════════════════════════════════════════════════════
    // MINTING & BURNING
    // ═══════════════════════════════════════════════════════════════════

    function mint(address to, uint256 amount) public onlyOwner {
        require(to != address(0), "Mint to zero address");
        require(amount > 0, "Mint amount must be positive");

        totalSupply += amount;
        balances[to] += amount;

        emit Transfer(address(0), to, amount);
    }

    function burn(uint256 amount) public {
        require(amount > 0, "Burn amount must be positive");
        require(balances[msg.sender] >= amount, "Burn exceeds balance");

        balances[msg.sender] -= amount;
        totalSupply -= amount;

        emit Transfer(msg.sender, address(0), amount);
    }

    function burnFrom(address from, uint256 amount) public {
        require(amount > 0, "Burn amount must be positive");
        require(balances[from] >= amount, "Burn exceeds balance");
        require(allowance[from][msg.sender] >= amount, "Burn exceeds allowance");

        balances[from] -= amount;
        totalSupply -= amount;
        allowance[from][msg.sender] -= amount;

        emit Transfer(from, address(0), amount);
    }

    // ═══════════════════════════════════════════════════════════════════
    // OWNERSHIP (Ownable Pattern)
    // ═══════════════════════════════════════════════════════════════════

    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "New owner is zero address");

        address oldOwner = owner;
        owner = newOwner;

        emit OwnershipTransferred(oldOwner, newOwner);
    }

    function renounceOwnership() public onlyOwner {
        address oldOwner = owner;
        owner = address(0);

        emit OwnershipTransferred(oldOwner, address(0));
    }

    // ═══════════════════════════════════════════════════════════════════
    // PAUSABLE
    // ═══════════════════════════════════════════════════════════════════

    function pause() public onlyOwner whenNotPaused {
        paused = true;
        emit Paused(msg.sender);
    }

    function unpause() public onlyOwner whenPaused {
        paused = false;
        emit Unpaused(msg.sender);
    }

    // ═══════════════════════════════════════════════════════════════════
    // VAULT FUNCTIONS (Deposit/Withdraw)
    // ═══════════════════════════════════════════════════════════════════

    function deposit(uint256 assets) public whenNotPaused nonReentrant returns (uint256 shares) {
        require(assets > 0, "Deposit amount must be positive");
        require(balances[msg.sender] >= assets, "Insufficient balance for deposit");

        // Calculate shares based on current exchange rate
        if (totalSupply == 0 || totalAssets == 0) {
            shares = assets; // 1:1 for first deposit
        } else {
            shares = (assets * totalSupply) / totalAssets;
        }

        require(shares > 0, "Zero shares minted");

        // Transfer assets to vault
        balances[msg.sender] -= assets;
        totalAssets += assets;

        // Mint shares
        balances[msg.sender] += shares;
        totalSupply += shares;

        emit Deposit(msg.sender, assets, shares);
        return shares;
    }

    function withdraw(uint256 shares) public whenNotPaused nonReentrant returns (uint256 assets) {
        require(shares > 0, "Withdraw amount must be positive");
        require(balances[msg.sender] >= shares, "Insufficient shares");

        // Calculate assets based on current exchange rate
        assets = (shares * totalAssets) / totalSupply;
        require(assets > 0, "Zero assets withdrawn");

        // Burn shares
        balances[msg.sender] -= shares;
        totalSupply -= shares;

        // Transfer assets from vault
        totalAssets -= assets;
        balances[msg.sender] += assets;

        emit Withdraw(msg.sender, shares, assets);
        return assets;
    }

    function convertToShares(uint256 assets) public view returns (uint256) {
        if (totalSupply == 0 || totalAssets == 0) {
            return assets;
        }
        return (assets * totalSupply) / totalAssets;
    }

    function convertToAssets(uint256 shares) public view returns (uint256) {
        if (totalSupply == 0) {
            return shares;
        }
        return (shares * totalAssets) / totalSupply;
    }

    // ═══════════════════════════════════════════════════════════════════
    // STAKING FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════

    function stake(uint256 amount) public whenNotPaused nonReentrant {
        require(amount > 0, "Stake amount must be positive");
        require(balances[msg.sender] >= amount, "Insufficient balance for staking");

        // Claim any pending rewards first
        _claimRewards(msg.sender);

        // Transfer tokens to staking
        balances[msg.sender] -= amount;
        stakedBalance[msg.sender] += amount;
        totalStaked += amount;

        emit Staked(msg.sender, amount);
    }

    function unstake(uint256 amount) public whenNotPaused nonReentrant {
        require(amount > 0, "Unstake amount must be positive");
        require(stakedBalance[msg.sender] >= amount, "Insufficient staked balance");

        // Claim any pending rewards first
        _claimRewards(msg.sender);

        // Transfer tokens from staking
        stakedBalance[msg.sender] -= amount;
        totalStaked -= amount;
        balances[msg.sender] += amount;

        emit Unstaked(msg.sender, amount);
    }

    function claimRewards() public whenNotPaused nonReentrant {
        _claimRewards(msg.sender);
    }

    function _claimRewards(address user) internal {
        uint256 reward = earned(user);
        if (reward > 0) {
            rewards[user] = 0;
            balances[user] += reward;
            totalSupply += reward;
            emit RewardsClaimed(user, reward);
        }
    }

    function earned(address user) public view returns (uint256) {
        return rewards[user] + (stakedBalance[user] * rewardRate / 10000);
    }

    // ═══════════════════════════════════════════════════════════════════
    // ORACLE FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════

    function getPrice() public view returns (uint256) {
        return _price;
    }

    function latestAnswer() public view returns (uint256) {
        return _price;
    }

    function setPrice(uint256 newPrice) public {
        require(msg.sender == oracle, "Only oracle can set price");
        require(newPrice > 0, "Price must be positive");

        uint256 oldPrice = _price;
        _price = newPrice;

        emit PriceUpdated(oldPrice, newPrice);
    }

    function updatePrice(uint256 newPrice) public {
        setPrice(newPrice);
    }

    function setOracle(address newOracle) public onlyOwner {
        require(newOracle != address(0), "Oracle cannot be zero address");
        oracle = newOracle;
    }

    // ═══════════════════════════════════════════════════════════════════
    // FLASH LOAN FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════

    function flashLoan(address receiver, uint256 amount) public nonReentrant {
        require(amount > 0, "Flash loan amount must be positive");
        require(totalAssets >= amount, "Insufficient liquidity");

        uint256 balanceBefore = totalAssets;
        uint256 fee = (amount * flashLoanFee) / 10000;

        // Transfer assets to receiver
        totalAssets -= amount;
        balances[receiver] += amount;

        // Call receiver callback (simplified - in real impl would use interface)
        // IFlashLoanReceiver(receiver).onFlashLoan(msg.sender, amount, fee, data);

        // Verify repayment
        require(balances[receiver] >= amount + fee, "Flash loan not repaid");

        // Take back principal + fee
        balances[receiver] -= (amount + fee);
        totalAssets += (amount + fee);

        require(totalAssets >= balanceBefore, "Flash loan: insufficient repayment");

        emit FlashLoan(receiver, amount, fee);
    }

    function setFlashLoanFee(uint256 newFee) public onlyOwner {
        require(newFee <= 1000, "Fee too high"); // Max 10%
        flashLoanFee = newFee;
    }

    // ═══════════════════════════════════════════════════════════════════
    // EMERGENCY FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════

    function emergencyWithdraw() public onlyOwner {
        uint256 amount = totalAssets;
        totalAssets = 0;
        balances[owner] += amount;
    }

    // ═══════════════════════════════════════════════════════════════════
    // VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════

    function getStakedBalance(address user) public view returns (uint256) {
        return stakedBalance[user];
    }

    function getTotalStaked() public view returns (uint256) {
        return totalStaked;
    }

    function getRewardRate() public view returns (uint256) {
        return rewardRate;
    }
}
