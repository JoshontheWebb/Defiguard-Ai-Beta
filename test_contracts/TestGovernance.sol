// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title TestGovernance
 * @notice Simple governance contract for formal verification testing
 * @dev Tests governance pattern detection in CVL generator
 *
 * PATTERNS INCLUDED:
 * ✓ Governance (propose, vote/castVote, execute)
 * ✓ Voting power (getVotes)
 * ✓ ERC20 voting token
 * ✓ Ownable
 */
contract TestGovernance {
    // ═══════════════════════════════════════════════════════════════════
    // STATE VARIABLES
    // ═══════════════════════════════════════════════════════════════════

    // Token state (for voting power)
    string public name = "Governance Token";
    string public symbol = "GOV";
    uint256 public totalSupply;
    mapping(address => uint256) public balances;

    // Governance state
    struct Proposal {
        uint256 id;
        address proposer;
        string description;
        uint256 forVotes;
        uint256 againstVotes;
        uint256 startBlock;
        uint256 endBlock;
        bool executed;
        bool canceled;
        mapping(address => bool) hasVoted;
    }

    mapping(uint256 => Proposal) public proposals;
    uint256 public proposalCount;
    uint256 public votingDelay = 1; // blocks
    uint256 public votingPeriod = 100; // blocks
    uint256 public proposalThreshold = 1000 * 10**18; // 1000 tokens to propose
    uint256 public quorumVotes = 10000 * 10**18; // 10000 tokens for quorum

    // Ownership
    address public owner;

    // ═══════════════════════════════════════════════════════════════════
    // EVENTS
    // ═══════════════════════════════════════════════════════════════════

    event Transfer(address indexed from, address indexed to, uint256 value);
    event ProposalCreated(uint256 indexed proposalId, address proposer, string description);
    event VoteCast(address indexed voter, uint256 indexed proposalId, uint8 support, uint256 votes);
    event ProposalExecuted(uint256 indexed proposalId);
    event ProposalCanceled(uint256 indexed proposalId);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    // ═══════════════════════════════════════════════════════════════════
    // MODIFIERS
    // ═══════════════════════════════════════════════════════════════════

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    // ═══════════════════════════════════════════════════════════════════
    // CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════

    constructor() {
        owner = msg.sender;

        // Initial supply to deployer
        uint256 initialSupply = 1000000 * 10**18;
        balances[msg.sender] = initialSupply;
        totalSupply = initialSupply;
        emit Transfer(address(0), msg.sender, initialSupply);
    }

    // ═══════════════════════════════════════════════════════════════════
    // TOKEN FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════

    function balanceOf(address account) public view returns (uint256) {
        return balances[account];
    }

    function transfer(address to, uint256 amount) public returns (bool) {
        require(to != address(0), "Transfer to zero address");
        require(balances[msg.sender] >= amount, "Insufficient balance");

        balances[msg.sender] -= amount;
        balances[to] += amount;

        emit Transfer(msg.sender, to, amount);
        return true;
    }

    function getVotes(address account) public view returns (uint256) {
        return balances[account];
    }

    function getPastVotes(address account, uint256) public view returns (uint256) {
        // Simplified: just return current balance
        return balances[account];
    }

    // ═══════════════════════════════════════════════════════════════════
    // GOVERNANCE FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════

    function propose(string memory description) public returns (uint256) {
        require(getVotes(msg.sender) >= proposalThreshold, "Below proposal threshold");

        uint256 proposalId = proposalCount++;

        Proposal storage proposal = proposals[proposalId];
        proposal.id = proposalId;
        proposal.proposer = msg.sender;
        proposal.description = description;
        proposal.startBlock = block.number + votingDelay;
        proposal.endBlock = proposal.startBlock + votingPeriod;

        emit ProposalCreated(proposalId, msg.sender, description);
        return proposalId;
    }

    function castVote(uint256 proposalId, uint8 support) public returns (uint256) {
        return _castVote(proposalId, msg.sender, support);
    }

    function vote(uint256 proposalId, bool support) public returns (uint256) {
        return _castVote(proposalId, msg.sender, support ? 1 : 0);
    }

    function _castVote(uint256 proposalId, address voter, uint8 support) internal returns (uint256) {
        Proposal storage proposal = proposals[proposalId];

        require(block.number >= proposal.startBlock, "Voting not started");
        require(block.number <= proposal.endBlock, "Voting ended");
        require(!proposal.hasVoted[voter], "Already voted");
        require(!proposal.canceled, "Proposal canceled");

        uint256 votes = getVotes(voter);
        require(votes > 0, "No voting power");

        proposal.hasVoted[voter] = true;

        if (support == 1) {
            proposal.forVotes += votes;
        } else {
            proposal.againstVotes += votes;
        }

        emit VoteCast(voter, proposalId, support, votes);
        return votes;
    }

    function execute(uint256 proposalId) public {
        Proposal storage proposal = proposals[proposalId];

        require(block.number > proposal.endBlock, "Voting not ended");
        require(!proposal.executed, "Already executed");
        require(!proposal.canceled, "Proposal canceled");
        require(proposal.forVotes >= quorumVotes, "Quorum not reached");
        require(proposal.forVotes > proposal.againstVotes, "Proposal defeated");

        proposal.executed = true;

        // Execute proposal logic here (simplified)
        emit ProposalExecuted(proposalId);
    }

    function cancel(uint256 proposalId) public {
        Proposal storage proposal = proposals[proposalId];

        require(
            msg.sender == proposal.proposer || msg.sender == owner,
            "Only proposer or owner can cancel"
        );
        require(!proposal.executed, "Already executed");

        proposal.canceled = true;
        emit ProposalCanceled(proposalId);
    }

    // ═══════════════════════════════════════════════════════════════════
    // GOVERNANCE VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════

    function proposalVotes(uint256 proposalId) public view returns (uint256) {
        return proposals[proposalId].forVotes;
    }

    function proposalSnapshot(uint256 proposalId) public view returns (uint256) {
        return proposals[proposalId].startBlock;
    }

    function proposalEta(uint256 proposalId) public view returns (uint256) {
        return proposals[proposalId].endBlock;
    }

    function state(uint256 proposalId) public view returns (string memory) {
        Proposal storage proposal = proposals[proposalId];

        if (proposal.canceled) return "Canceled";
        if (proposal.executed) return "Executed";
        if (block.number < proposal.startBlock) return "Pending";
        if (block.number <= proposal.endBlock) return "Active";
        if (proposal.forVotes < quorumVotes) return "Defeated";
        if (proposal.forVotes <= proposal.againstVotes) return "Defeated";
        return "Succeeded";
    }

    function hasVoted(uint256 proposalId, address account) public view returns (bool) {
        return proposals[proposalId].hasVoted[account];
    }

    // ═══════════════════════════════════════════════════════════════════
    // OWNERSHIP
    // ═══════════════════════════════════════════════════════════════════

    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "New owner is zero address");
        address oldOwner = owner;
        owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }

    // ═══════════════════════════════════════════════════════════════════
    // ADMIN FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════

    function setVotingDelay(uint256 newDelay) public onlyOwner {
        votingDelay = newDelay;
    }

    function setVotingPeriod(uint256 newPeriod) public onlyOwner {
        require(newPeriod > 0, "Voting period must be positive");
        votingPeriod = newPeriod;
    }

    function setProposalThreshold(uint256 newThreshold) public onlyOwner {
        proposalThreshold = newThreshold;
    }

    function setQuorumVotes(uint256 newQuorum) public onlyOwner {
        quorumVotes = newQuorum;
    }
}
