# pyright: reportMissingImports=false
# pyright: reportUnknownMemberType=false
# pyright: reportGeneralTypeIssues=false
import logging
import os
from dotenv import load_dotenv
import sys

print("=" * 80)
print("🔍 PYTHON STARTUP DEBUG:")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {os.path.abspath(__file__)}")
print(f".env file exists at CWD: {os.path.exists('.env')}")
print(f".env file exists at script dir: {os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))}")

# Load from script directory (not CWD)
script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, '.env')
print(f"Loading .env from: {dotenv_path}")
load_dotenv(dotenv_path=dotenv_path)

grok_test = os.getenv("GROK_API_KEY")
print(f"GROK_API_KEY loaded: {'YES ✅' if grok_test else 'NO ❌'}")
if grok_test:
    print(f"First 20 chars: {grok_test[:20]}...")
print("=" * 80)

# NOW initialize clients AFTER .env is loaded
print("=" * 80)
print("🔍 CALLING INITIALIZE_CLIENT EARLY:")
print("=" * 80)

# Import what we need for initialize_client
from openai import OpenAI
import anthropic
from web3 import Web3

# Initialize clients RIGHT NOW
claude_client = None
grok_client = None
w3 = None

# Primary: Claude
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
if anthropic_key and anthropic_key.strip():
    claude_client = anthropic.Anthropic(api_key=anthropic_key.strip())
    print("[CLAUDE] Primary client initialized successfully ✅")
else:
    print("[CLAUDE] ANTHROPIC_API_KEY missing - primary client is None ❌")

# Fallback: Grok
grok_key = os.getenv("GROK_API_KEY")
if grok_key and grok_key.strip():
    grok_client = OpenAI(
        api_key=grok_key.strip(),
        base_url="https://api.x.ai/v1"
    )
    print("[GROK] Fallback client initialized successfully ✅")
else:
    print("[GROK] GROK_API_KEY missing - fallback client is None ❌")

# Web3
infura_url = f"https://mainnet.infura.io/v3/{os.getenv('INFURA_PROJECT_ID')}"
w3 = Web3(Web3.HTTPProvider(infura_url))
print(f"[WEB3] Connected: {w3.is_connected()}")

print("=" * 80)

import platform
import json
import time
import uuid
from datetime import datetime, timedelta, timezone
import secrets
from tempfile import NamedTemporaryFile
from typing import Optional, Callable, Awaitable, Any, cast
from fastapi import FastAPI, File, UploadFile, Request, Query, HTTPException, Depends, Response, Header, WebSocket, Body
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response as StarletteResponse
from starlette.websockets import WebSocketState
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware
from urllib.parse import quote_plus, urlencode
from fastapi.middleware.cors import CORSMiddleware
from web3 import Web3
import stripe
import re  # For username sanitization

# === EARLY LOGGER SETUP (fixes NameError) ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("DeFiGuard")

# Import stripe.error directly for exception handling
try:
    from stripe import error as stripe_error
except ImportError:
    # Fallback stub so references like stripe_error.InvalidRequestError won't raise AttributeError
    class _StripeErrorStub(Exception):
        pass
    class _StripeModuleStub:
        InvalidRequestError = _StripeErrorStub
        StripeError = _StripeErrorStub
        SignatureVerificationError = _StripeErrorStub
    stripe_error = _StripeModuleStub()

from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text, ForeignKey, Float
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.orm import sessionmaker, Session, Mapped, mapped_column
from slither.slither import Slither
from slither.exceptions import SlitherError
from openai import OpenAI  # Sync
from tenacity import retry, stop_after_attempt, wait_fixed
import uvicorn
from pydantic import BaseModel, Field, field_validator
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import requests
import subprocess
import asyncio
import heapq
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Flowable
from reportlab.lib.styles import getSampleStyleSheet
from jinja2 import Environment, FileSystemLoader
from compliance_checker import get_compliance_analysis, ComplianceChecker

# ============================================================================
# PHASE 1: IN-MEMORY AUDIT QUEUE SYSTEM
# ============================================================================
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class AuditStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AuditJob:
    """Represents a single audit job in the queue."""
    job_id: str
    username: str
    filename: str
    file_content: bytes
    tier: str
    contract_address: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: AuditStatus = AuditStatus.QUEUED
    position: int = 0
    result: Optional[dict] = None
    error: Optional[str] = None
    current_phase: Optional[str] = None  # For progress tracking
    progress_percent: int = 0

class AuditQueue:
    """
    Priority queue for managing concurrent audit requests.
    Higher tiers (Enterprise > Pro > Starter > Free) get processed first.
    Phase 1 implementation - handles ~50 audits/hour with 2GB RAM.
    """
    
    # Priority values (lower number = higher priority = processed first)
    TIER_PRIORITY = {
        "enterprise": 0,
        "diamond": 0,      # Legacy - same as enterprise
        "pro": 1,
        "starter": 2,
        "beginner": 2,     # Legacy - same as starter
        "free": 3
    }
    
    def __init__(self, max_concurrent: int = 1):
        # Priority queue: list of (priority, counter, job) tuples
        # Counter ensures FIFO order within same priority tier
        self.queue: list[tuple[int, int, AuditJob]] = []
        self.jobs: dict[str, AuditJob] = {}
        self.processing: set[str] = set()
        self.max_concurrent = max_concurrent
        self.lock = asyncio.Lock()
        self._processor_task: Optional[asyncio.Task] = None
        self._counter = 0  # Monotonic counter for tie-breaking
        logger.info(f"[QUEUE] AuditQueue initialized with max_concurrent={max_concurrent} (PRIORITY ENABLED)")
    
    async def submit(self, username: str, file_content: bytes, 
                     filename: str, tier: str, contract_address: Optional[str] = None) -> AuditJob:
        """Submit a new audit job to the priority queue."""
        async with self.lock:
            self._counter += 1
            counter = self._counter
            
            # Calculate position based on priority
            priority = self.TIER_PRIORITY.get(tier, 3)  # Default to free tier priority
            
            job = AuditJob(
                job_id=str(uuid.uuid4()),
                username=username,
                file_content=file_content,
                filename=filename,
                tier=tier,
                contract_address=contract_address,
                position=0  # Will be calculated dynamically
            )
            
            self.jobs[job.job_id] = job
            heapq.heappush(self.queue, (priority, counter, job))
            
            # Calculate actual position (how many jobs ahead with higher/equal priority)
            position = self._calculate_position(job.job_id, priority, counter)
            job.position = position
        
        logger.info(f"[QUEUE] Job {job.job_id[:8]}... submitted for {username} (tier={tier}, priority={priority}), position {position}, queue size: {len(self.queue)}")
        return job
    
    def _calculate_position(self, job_id: str, job_priority: int, job_counter: int) -> int:
        """Calculate queue position based on priority ordering."""
        position = 1  # Start at 1 (next to be processed)
        
        for priority, counter, queued_job in self.queue:
            if queued_job.job_id == job_id:
                continue
            if queued_job.status != AuditStatus.QUEUED:
                continue
            # Job is ahead if: lower priority number, OR same priority but earlier submission
            if priority < job_priority or (priority == job_priority and counter < job_counter):
                position += 1
        
        # Add currently processing jobs
        position += len(self.processing)
        
        return position
    
    async def get_position(self, job_id: str) -> int:
        """Get current queue position for a job."""
        job = self.jobs.get(job_id)
        if not job:
            return -1
        if job.status == AuditStatus.PROCESSING:
            return 0
        if job.status in (AuditStatus.COMPLETED, AuditStatus.FAILED):
            return -1
        
        # Find job in queue and calculate position
        async with self.lock:
            for priority, counter, queued_job in self.queue:
                if queued_job.job_id == job_id:
                    return self._calculate_position(job_id, priority, counter)
        
        return -1
    
    async def get_status(self, job_id: str) -> dict:
        """Get full status of a job."""
        job = self.jobs.get(job_id)
        if not job:
            return {"error": "Job not found", "status": "not_found"}
        
        position = await self.get_position(job_id)
        
        # Estimate wait time based on position (avg 130s per audit)
        estimated_wait = max(0, (position - 1) * 130) if position > 0 else 0
        
        # Count users ahead by tier for transparency
        users_ahead = {"enterprise": 0, "pro": 0, "starter": 0, "free": 0}
        async with self.lock:
            for _, _, queued_job in self.queue:
                if queued_job.job_id == job_id:
                    continue
                if queued_job.status == AuditStatus.QUEUED:
                    tier_key = queued_job.tier
                    if tier_key in ["diamond"]:
                        tier_key = "enterprise"
                    elif tier_key in ["beginner"]:
                        tier_key = "starter"
                    if tier_key in users_ahead:
                        users_ahead[tier_key] += 1
        
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "position": position,
            "queue_length": len(self.queue),
            "processing_count": len(self.processing),
            "created_at": job.created_at.isoformat(),
            "current_phase": job.current_phase,
            "progress_percent": job.progress_percent,
            "estimated_wait_seconds": estimated_wait,
            "tier": job.tier,
            "users_ahead": users_ahead,
            "result": job.result if job.status == AuditStatus.COMPLETED else None,
            "error": job.error if job.status == AuditStatus.FAILED else None
        }
    
    async def update_phase(self, job_id: str, phase: str, progress: int):
        """Update the current processing phase for UI feedback."""
        if job_id in self.jobs:
            self.jobs[job_id].current_phase = phase
            self.jobs[job_id].progress_percent = progress
            logger.debug(f"[QUEUE] Job {job_id[:8]}... phase: {phase} ({progress}%)")
    
    async def process_next(self) -> Optional[AuditJob]:
        """Get highest priority job from queue if capacity available."""
        if len(self.processing) >= self.max_concurrent:
            return None
        
        async with self.lock:
            # Find and remove the highest priority QUEUED job
            while self.queue:
                try:
                    priority, counter, job = heapq.heappop(self.queue)
                    
                    # Skip if job was already processed/cancelled
                    if job.status != AuditStatus.QUEUED:
                        continue
                    
                    job.status = AuditStatus.PROCESSING
                    self.processing.add(job.job_id)
                    logger.info(f"[QUEUE] Job {job.job_id[:8]}... started processing (tier={job.tier}, priority={priority})")
                    return job
                    
                except IndexError:
                    return None
        
        return None
    
    async def complete(self, job_id: str, result: dict):
        """Mark job as completed with results."""
        if job_id in self.jobs:
            self.jobs[job_id].status = AuditStatus.COMPLETED
            self.jobs[job_id].result = result
            self.jobs[job_id].progress_percent = 100
            self.jobs[job_id].current_phase = "complete"
            self.processing.discard(job_id)
            logger.info(f"[QUEUE] Job {job_id[:8]}... completed successfully")
    
    async def fail(self, job_id: str, error: str):
        """Mark job as failed with error."""
        if job_id in self.jobs:
            self.jobs[job_id].status = AuditStatus.FAILED
            self.jobs[job_id].error = error
            self.jobs[job_id].current_phase = "failed"
            self.processing.discard(job_id)
            logger.error(f"[QUEUE] Job {job_id[:8]}... failed: {error}")
    
    def get_stats(self) -> dict:
        """Get queue statistics for monitoring."""
        # Count by tier
        by_tier = {"enterprise": 0, "pro": 0, "starter": 0, "free": 0}
        for _, _, job in self.queue:
            if job.status == AuditStatus.QUEUED:
                tier_key = job.tier
                if tier_key in ["diamond"]:
                    tier_key = "enterprise"
                elif tier_key in ["beginner"]:
                    tier_key = "starter"
                if tier_key in by_tier:
                    by_tier[tier_key] += 1
        
        return {
            "queued": sum(1 for j in self.jobs.values() if j.status == AuditStatus.QUEUED),
            "queued_by_tier": by_tier,
            "processing": len(self.processing),
            "completed": sum(1 for j in self.jobs.values() if j.status == AuditStatus.COMPLETED),
            "failed": sum(1 for j in self.jobs.values() if j.status == AuditStatus.FAILED),
            "total": len(self.jobs),
            "max_concurrent": self.max_concurrent
        }
    
    def cleanup_old_jobs(self, max_age_hours: int = 1):
        """Remove completed/failed jobs older than max_age_hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        to_remove = [
            jid for jid, job in self.jobs.items()
            if job.status in (AuditStatus.COMPLETED, AuditStatus.FAILED)
            and job.created_at < cutoff
        ]
        for jid in to_remove:
            del self.jobs[jid]
        if to_remove:
            logger.info(f"[QUEUE] Cleaned up {len(to_remove)} old jobs")

# Initialize global audit queue with admission control
# For 2GB RAM: Allow 2 concurrent audits before queuing
audit_queue = AuditQueue(max_concurrent=1)

# Track currently executing audits (for admission control)
active_audit_count = 0
active_audit_lock = asyncio.Lock()
MAX_CONCURRENT_AUDITS = 1  # Conservative limit for 2GB RAM with parallel tools

# WebSocket connections for job status updates
active_job_websockets: dict[str, list[WebSocket]] = {}

try:
    from celery import Celery  # type: ignore[reportMissingTypeStubs]
except Exception:
    class Celery:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.conf = {}
        def task(self, *args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
                return fn
            return decorator
        def send_task(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Celery is not installed in this environment")

import requests  # For cloud fallback
import asyncio
from jose import jwt, JWTError

# === JINJA2 TEMPLATE SETUP (required for /ui and /auth) ===
templates_dir = "templates"
jinja_env = Environment(loader=FileSystemLoader(templates_dir))

# === FASTAPI APP INSTANCE (MUST COME BEFORE ANY @app ROUTES) ===
app = FastAPI()

# Trust proxy headers from Render (ensures HTTPS URLs are generated correctly)
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts=["*"])

# CRITICAL: Session middleware for Auth0
_secret_key = os.getenv("APP_SECRET_KEY")
if not _secret_key:
    _secret_key = secrets.token_urlsafe(32)
    logger.warning("APP_SECRET_KEY not set; using generated temporary secret (not for production)")

app.add_middleware(
    SessionMiddleware,
    secret_key=_secret_key,
    session_cookie="session",
    max_age=14 * 24 * 60 * 60,  # 2 weeks
    same_site="lax",
    https_only=False  # Set to True in production
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "https://defiguard-ai-fresh-private-test.onrender.com",
        "https://defiguard-ai-fresh-private.onrender.com",
        "https://defiguard-ai-beta.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global clients — already initialized at top of file
# client and w3 are set above, no need to reset them here
oauth = OAuth()
active_audit_websockets: dict[str, WebSocket] = {}

# === AUTH0 REGISTRATION ===
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")

if AUTH0_DOMAIN and AUTH0_CLIENT_ID and AUTH0_CLIENT_SECRET:
    oauth.register(
        name='auth0',
        client_id=AUTH0_CLIENT_ID,
        client_secret=AUTH0_CLIENT_SECRET,
        server_metadata_url=f"https://{AUTH0_DOMAIN}/.well-known/openid-configuration",
        audience=os.getenv("AUTH0_AUDIENCE"),
        client_kwargs={
            "scope": "openid email profile offline_access"
        }
    )
    logger.info("Auth0 OAuth client registered successfully")
else:
    logger.warning("Auth0 env vars missing – running in legacy/local mode")

@app.get("/debug-files")
async def debug_files():
    import os
    files = os.listdir("templates") if os.path.exists("templates") else "NO templates folder"
    cwd = os.getcwd()
    return {
        "current_working_directory": cwd,
        "templates_folder_exists": os.path.exists("templates"),
        "files_in_templates": files,
        "absolute_path_attempted": os.path.abspath("templates/auth.html")
    }

@app.get("/login")
async def login(request: Request, screen_hint: Optional[str] = None):
    if not AUTH0_DOMAIN:
        return HTMLResponse("Auth0 not configured – <a href='/ui'>continue locally</a>")
    redirect_uri = request.url_for("callback")
    response = await oauth.auth0.authorize_redirect(request, redirect_uri)
    if screen_hint:
        from urllib.parse import urlparse, parse_qs, urlencode
        location = response.headers["Location"]
        parsed = urlparse(location)
        params = parse_qs(parsed.query)
        params["screen_hint"] = [screen_hint]
        new_query = urlencode(params, doseq=True)
        new_location = parsed._replace(query=new_query).geturl()
        response.headers["Location"] = new_location
    logger.info(f"Redirecting to Auth0 for login, screen_hint={screen_hint}")
    return response

@app.get("/logout")
async def logout(request: Request):
    # Clear ALL session data
    request.session.clear()
    
    # Build Auth0 logout URL
    base_url = str(request.base_url).rstrip('/')
    logout_url = (
        f"https://{AUTH0_DOMAIN}/v2/logout?"
        + urlencode(
            {
                "returnTo": base_url,
                "client_id": AUTH0_CLIENT_ID,
            },
            quote_via=quote_plus,
        )
    )
    
    # Create redirect response and DELETE the session cookie
    response = RedirectResponse(url=logout_url, status_code=307)
    response.delete_cookie("session")  # ← THIS IS THE CRITICAL FIX
    return response

# === DATABASE SETUP AND DEPENDENCIES (MUST BE BEFORE ANY ROUTES) ===
import os

# Use local path for development, Render path for production
import os
from pathlib import Path

# Use local path for development, Render path for production
if os.getenv("RENDER"):
    DATABASE_URL = "sqlite:////tmp/users.db"
else:
    # Create instance folder if it doesn't exist
    Path("./instance").mkdir(exist_ok=True)
    DATABASE_URL = "sqlite:///./instance/users.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String, unique=True, index=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    password_hash: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    tier: Mapped[str] = mapped_column(String, default="free")
    has_diamond: Mapped[bool] = mapped_column(Boolean, default=False)
    last_reset: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    api_key: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    auth0_sub: Mapped[Optional[str]] = mapped_column(String, unique=True, nullable=True)
    stripe_customer_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    stripe_subscription_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    stripe_subscription_item_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    audit_history: Mapped[str] = mapped_column(Text, default="[]")
    
    # API Keys relationship (new multi-key system)
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")

    # Legal acceptance tracking
    terms_accepted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    terms_version: Mapped[Optional[str]] = mapped_column(String, default=None, nullable=True)
    privacy_accepted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    privacy_version: Mapped[Optional[str]] = mapped_column(String, default=None, nullable=True)

    # ============================================================================
    # GAMIFICATION FIELDS (Phase 0 - Silent tracking for future features)
    # ============================================================================
    # These fields track user activity and achievements without showing UI yet
    # When we launch gamification in Week 2, users will see their full history
    
    # Audit tracking
    total_audits: Mapped[int] = mapped_column(Integer, default=0)
    total_issues_found: Mapped[int] = mapped_column(Integer, default=0)
    total_critical_issues: Mapped[int] = mapped_column(Integer, default=0)
    total_high_issues: Mapped[int] = mapped_column(Integer, default=0)
    
    # Achievement tracking
    best_security_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_streak_days: Mapped[int] = mapped_column(Integer, default=0)
    last_audit_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Leveling system
    xp_points: Mapped[int] = mapped_column(Integer, default=0)
    level: Mapped[int] = mapped_column(Integer, default=1)
    
    # NFT minting (for future)
    wallet_address: Mapped[Optional[str]] = mapped_column(String, nullable=True, unique=True)
   
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship

class APIKey(Base):
    """Multi-key API key management for Pro/Enterprise users"""
    __tablename__ = "api_keys"
    __table_args__ = {'extend_existing': True}
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey('users.id'), nullable=False)
    key: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    label: Mapped[str] = mapped_column(String, default="API Key", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Relationship
    user = relationship("User", back_populates="api_keys")
class PendingAudit(Base):
    __tablename__ = "pending_audits"
    __table_args__ = {'extend_existing': True}
    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String, index=True)
    temp_path: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="pending")
    results: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

Base.metadata.create_all(bind=engine, checkfirst=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_authenticated_user(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    # 1. Try Auth0 session first
    auth0_user = request.session.get("user")
    if auth0_user and auth0_user.get("sub"):
        auth0_sub = auth0_user["sub"]
        # JWKS decode for provider
        try:
            token = request.session.get("id_token")
            if token:
                jwks_url = f"https://{os.getenv('AUTH0_DOMAIN')}/.well-known/jwks.json"
                jwks = requests.get(jwks_url).json()
                unverified_header = jwt.get_unverified_header(token)
                rsa_key = next((k for k in jwks['keys'] if k['kid'] == unverified_header['kid']), None)
                if rsa_key:
                    payload = jwt.decode(
                        token, rsa_key,
                        algorithms=['RS256'],
                        audience=os.getenv('AUTH0_CLIENT_ID'),
                        issuer=f"https://{os.getenv('AUTH0_DOMAIN')}/"
                    )
                    provider = payload.get('identities', [{}])[0].get('provider', 'unknown')
                    auth0_user['provider'] = provider
                else:
                    auth0_user['provider'] = 'unknown'
        except (JWTError, Exception) as e:
            logger.warning(f"JWT decode failed in get_authenticated_user: {e}")
            auth0_user['provider'] = 'unknown'
        
        user = db.query(User).filter(User.auth0_sub == auth0_sub).first()
        if user:
            return user
        
        # Auto-create user on first Auth0 login
        new_user = User(
            username=auth0_user.get("nickname") or auth0_user.get("email", "auth0_user").split("@")[0],
            email=auth0_user.get("email", ""),
            auth0_sub=auth0_sub,
            tier="free",
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user
    
    # 2. Fallback to legacy session user_id
    user_id = request.session.get("user_id")
    if user_id:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            return user
    
    return None

# Ensure logging directory exists (Render-specific)
LOG_DIR = "/opt/render/project/data"
os.makedirs(LOG_DIR, exist_ok=True)

# Initialize logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "debug.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Load environment variables at startup
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 80)
    print("🔍 LIFESPAN STARTUP:")
    print("About to call initialize_client()...")
    print("=" * 80)
    
    required_env_vars = [
        "GROK_API_KEY",
        "INFURA_PROJECT_ID",
        "STRIPE_API_KEY",
        "STRIPE_WEBHOOK_SECRET",
        "AUTH0_DOMAIN",
        "AUTH0_CLIENT_ID",
        "AUTH0_CLIENT_SECRET",
        "APP_SECRET_KEY",
        "REDIS_URL"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing critical environment variables: {', '.join(missing_vars)}")
        raise RuntimeError(f"Missing critical environment variables: {', '.join(missing_vars)}")
    
    # Log Stripe price IDs
    for var in ["STRIPE_PRICE_PRO", "STRIPE_PRICE_STARTER", "STRIPE_PRICE_ENTERPRISE", 
                "STRIPE_PRICE_BEGINNER", "STRIPE_PRICE_DIAMOND", "STRIPE_METERED_PRICE_DIAMOND"]:
        logger.info(f"Environment variable {var}: {'set' if os.getenv(var) else 'NOT set'}")
    
    # Log routes
    logger.info(f"Registered routes: {[getattr(r, 'path', str(r)) for r in app.routes]}")
    if "/create-tier-checkout" in [getattr(r, 'path', None) for r in app.routes]:
        logger.info("Confirmed: /create-tier-checkout is registered")
    else:
        logger.error("Error: /create-tier-checkout is NOT registered")
    
    # Initialize clients
    global client, w3
    client, w3 = initialize_client()
    
    # Start background queue processor
    processor_task = asyncio.create_task(process_audit_queue())
    logger.info("[QUEUE] Background audit queue processor started")
    
    # Start periodic cleanup task
    cleanup_task = asyncio.create_task(periodic_queue_cleanup())
    logger.info("[QUEUE] Periodic cleanup task started")
    
    yield  # App running
    
    # Cleanup on shutdown
    processor_task.cancel()
    cleanup_task.cancel()
    logger.info("[QUEUE] Background tasks cancelled")

from fastapi import Request

async def require_login(request: Request):
    if "user_id" not in request.session:
        raise HTTPException(
            status_code=401,
            detail="Login required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return request.session["user_id"]

@app.get("/callback")
async def callback(request: Request):
    try:
        token = cast(dict[str, Any], await oauth.auth0.authorize_access_token(request))
        userinfo = token.get("userinfo")
        
        if not userinfo:
            logger.warning("Invalid Auth0 userinfo – no data")
            return RedirectResponse(url="/ui")
        
        request.session["userinfo"] = userinfo
        request.session["id_token"] = token.get("id_token")
        
        db = next(get_db())
        email = userinfo.get("email", "")
        sub = userinfo.get("sub")
        
        # Extract username
        username = (
            userinfo.get("preferred_username")
            or userinfo.get("name")
            or userinfo.get("nickname")
            or userinfo.get("login")
            or userinfo.get("username")
            or (email.split("@")[0] if email else "user")
        ).strip()[:50]
        
        # Sanitize username
        username = re.sub(r"[^a-zA-Z0-9_.-]", "_", username)
        
        # Find or create user
        user = db.query(User).filter(User.email == email).first()
        if not user:
            user = User(
                username=username,
                email=email,
                auth0_sub=sub,
                tier="free",
                audit_history="[]",
                last_reset=datetime.now(timezone.utc),
                has_diamond=False
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info(f"New user created: {username}")
        else:
            if not user.auth0_sub:
                user.auth0_sub = sub
                db.commit()
        
        # Set session
        request.session["user_id"] = user.id
        request.session["username"] = user.username
        request.session["csrf_token"] = secrets.token_urlsafe(32)
        
        # Provider from userinfo
        provider = 'unknown'
        if 'userinfo' in locals():
            provider = userinfo.get('identities', [{}])[0].get('provider', 'unknown')
        request.session["user"] = {"sub": sub, "email": email, "provider": provider}
        
        response = RedirectResponse(url="/ui")
        response.set_cookie("username", str(user.username), httponly=False, secure=True, samesite="lax", max_age=2592000)
        logger.info(f"Auth0 login successful: {user.username}")
        return response
        
    except Exception as e:
        logger.error(f"Auth0 callback error: {e}")
        request.session["auth_error"] = str(e)
        return RedirectResponse(url="/ui")

@app.get("/me")
async def me_endpoint(request: Request, db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        user = await get_authenticated_user(request, db)
        if not user:
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        provider = getattr(user, 'provider', 'unknown')
        if provider == 'unknown' and 'userinfo' in request.session:
            ui = request.session['userinfo']
            provider = ui.get('identities', [{}])[0].get('provider', 'unknown') if ui.get('identities') else 'unknown'
        
        return {
            "sub": user.auth0_sub,
            "email": user.email,
            "username": user.username,
            "provider": user.auth0_sub.split("|")[0] if user.auth0_sub and "|" in user.auth0_sub else "auth0",
            "logged_in": True,
            "tier": user.tier
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/me error: {e}")
        raise HTTPException(status_code=500, detail="Internal error")

@app.post("/api/regenerate-key")
async def regenerate_api_key(
    request: Request,
    db: Session = Depends(get_db)
) -> dict[str, Any]:
    """
    Regenerate API key for Pro/Enterprise users
    """
    try:
        # Verify CSRF token
        await verify_csrf_token(request)
        
        # Get authenticated user
        user = await get_authenticated_user(request, db)
        if not user:
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        # Check if user has Pro or Enterprise tier (only they should have API keys)
        if user.tier not in ["pro", "enterprise"]:
            raise HTTPException(
                status_code=403, 
                detail="API key regeneration is only available for Pro and Enterprise tiers"
            )
        
        # Generate new API key
        new_api_key = str(uuid.uuid4())
        
        # Update user's API key in database
        user.api_key = new_api_key
        db.commit()
        db.refresh(user)
        
        logger.info(f"[API_KEY_REGEN] User {user.username} regenerated API key")
        
        return {
            "success": True,
            "api_key": new_api_key,
            "message": "API key regenerated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API_KEY_REGEN] Error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to regenerate API key")

# ============================================================================
# MULTI-KEY API MANAGEMENT (Pro/Enterprise)
# ============================================================================

@app.get("/api/keys")
async def list_api_keys(
    request: Request,
    db: Session = Depends(get_db)
) -> dict[str, Any]:
    """
    List all API keys for the authenticated user.
    Returns truncated keys for security.
    """
    try:
        user = await get_authenticated_user(request, db)
        if not user:
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        if user.tier not in ["pro", "enterprise"]:
            raise HTTPException(status_code=403, detail="API keys require Pro or Enterprise tier")
        
        # Get all active keys for user
        keys = db.query(APIKey).filter(
            APIKey.user_id == user.id,
            APIKey.is_active == True
        ).order_by(APIKey.created_at.desc()).all()
        
        # Determine max keys based on tier
        max_keys = None if user.tier == "enterprise" else 5  # Pro = 5, Enterprise = unlimited
        
        keys_data = [{
            "id": key.id,
            "key": key.key,  # Full key (frontend will truncate for display)
            "label": key.label,
            "created_at": key.created_at.isoformat(),
            "last_used_at": key.last_used_at.isoformat() if key.last_used_at else None,
            "is_active": key.is_active
        } for key in keys]
        
        return {
            "keys": keys_data,
            "active_count": len(keys),
            "max_keys": max_keys,
            "tier": user.tier
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API_KEYS_LIST] Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to load API keys")


@app.post("/api/keys/create")
async def create_api_key(
    request: Request,
    body: dict = Body(...),
    db: Session = Depends(get_db)
) -> dict[str, Any]:
    """
    Create a new API key with a custom label.
    Pro tier: max 5 keys, Enterprise: unlimited.
    """
    try:
        await verify_csrf_token(request)
        
        user = await get_authenticated_user(request, db)
        if not user:
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        if user.tier not in ["pro", "enterprise"]:
            raise HTTPException(status_code=403, detail="API keys require Pro or Enterprise tier")
        
        label = body.get("label", "").strip()
        if not label:
            raise HTTPException(status_code=400, detail="Label is required")
        
        # Check tier limits
        active_keys_count = db.query(APIKey).filter(
            APIKey.user_id == user.id,
            APIKey.is_active == True
        ).count()
        
        if user.tier == "pro" and active_keys_count >= 5:
            raise HTTPException(
                status_code=403,
                detail="Pro tier is limited to 5 active API keys. Upgrade to Enterprise for unlimited keys."
            )
        
        # Generate new API key
        new_key = str(uuid.uuid4())
        
        # Create APIKey record
        api_key_obj = APIKey(
            user_id=user.id,
            key=new_key,
            label=label,
            created_at=datetime.now(),
            is_active=True
        )
        
        db.add(api_key_obj)
        db.commit()
        db.refresh(api_key_obj)
        
        logger.info(f"[API_KEY_CREATE] User {user.username} created key: {label}")
        
        return {
            "success": True,
            "key_id": api_key_obj.id,
            "api_key": new_key,  # Show full key ONCE
            "label": label,
            "created_at": api_key_obj.created_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API_KEY_CREATE] Error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create API key")


@app.delete("/api/keys/{key_id}")
async def revoke_api_key(
    key_id: int,
    request: Request,
    db: Session = Depends(get_db)
) -> dict[str, Any]:
    """
    Revoke (delete) a specific API key.
    Only the key owner can revoke it.
    """
    try:
        await verify_csrf_token(request)
        
        user = await get_authenticated_user(request, db)
        if not user:
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        # Find the key
        api_key = db.query(APIKey).filter(
            APIKey.id == key_id,
            APIKey.user_id == user.id
        ).first()
        
        if not api_key:
            raise HTTPException(status_code=404, detail="API key not found")
        
        # Delete the key (immediate revocation)
        label = api_key.label
        db.delete(api_key)
        db.commit()
        
        logger.info(f"[API_KEY_REVOKE] User {user.username} revoked key: {label}")
        
        return {
            "success": True,
            "key_id": key_id,
            "deleted": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API_KEY_REVOKE] Error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to revoke API key")


@app.patch("/api/keys/{key_id}")
async def update_api_key_label(
    key_id: int,
    request: Request,
    body: dict = Body(...),
    db: Session = Depends(get_db)
) -> dict[str, Any]:
    """
    Update API key label (for future use).
    Only the key owner can update it.
    """
    try:
        await verify_csrf_token(request)
        
        user = await get_authenticated_user(request, db)
        if not user:
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        new_label = body.get("label", "").strip()
        if not new_label:
            raise HTTPException(status_code=400, detail="Label is required")
        
        # Find the key
        api_key = db.query(APIKey).filter(
            APIKey.id == key_id,
            APIKey.user_id == user.id
        ).first()
        
        if not api_key:
            raise HTTPException(status_code=404, detail="API key not found")
        
        # Update label
        api_key.label = new_label
        db.commit()
        db.refresh(api_key)
        
        logger.info(f"[API_KEY_UPDATE] User {user.username} updated key {key_id} label to: {new_label}")
        
        return {
            "success": True,
            "key_id": key_id,
            "label": new_label
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API_KEY_UPDATE] Error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update API key")
# ============================================================================
# LEGAL DOCUMENT ACCEPTANCE TRACKING
# ============================================================================

CURRENT_TERMS_VERSION = "1.0"
CURRENT_PRIVACY_VERSION = "1.0"

@app.get("/legal/status")
async def check_legal_acceptance(
    request: Request,
    db: Session = Depends(get_db)
) -> dict[str, Any]:
    """
    Check if user has accepted current versions of legal documents.
    Returns acceptance status and document versions.
    """
    try:
        user = await get_authenticated_user(request, db)
        if not user:
            # Guest users don't need acceptance
            return {
                "needs_acceptance": False,
                "is_guest": True
            }
        
        # Check if user has accepted current versions
        needs_terms = (
            not user.terms_accepted_at or
            user.terms_version != CURRENT_TERMS_VERSION
        )
        
        needs_privacy = (
            not user.privacy_accepted_at or
            user.privacy_version != CURRENT_PRIVACY_VERSION
        )
        
        return {
            "needs_acceptance": needs_terms or needs_privacy,
            "is_guest": False,
            "terms_accepted": not needs_terms,
            "privacy_accepted": not needs_privacy,
            "current_terms_version": CURRENT_TERMS_VERSION,
            "current_privacy_version": CURRENT_PRIVACY_VERSION,
            "user_terms_version": user.terms_version,
            "user_privacy_version": user.privacy_version
        }
        
    except Exception as e:
        logger.error(f"[LEGAL_STATUS] Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to check legal status")


@app.post("/legal/accept")
async def accept_legal_documents(
    request: Request,
    body: dict = Body(...),
    db: Session = Depends(get_db)
) -> dict[str, Any]:
    """
    Record user's acceptance of Terms of Service and Privacy Policy.
    Both must be accepted to proceed.
    """
    try:
        await verify_csrf_token(request)
        
        user = await get_authenticated_user(request, db)
        if not user:
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        accepted_terms = body.get("accepted_terms", False)
        accepted_privacy = body.get("accepted_privacy", False)
        
        if not accepted_terms or not accepted_privacy:
            raise HTTPException(
                status_code=400,
                detail="Both Terms of Service and Privacy Policy must be accepted"
            )
        
        # Record acceptance
        now = datetime.now()
        user.terms_accepted_at = now
        user.terms_version = CURRENT_TERMS_VERSION
        user.privacy_accepted_at = now
        user.privacy_version = CURRENT_PRIVACY_VERSION
        
        db.commit()
        
        logger.info(f"[LEGAL_ACCEPT] User {user.username} accepted ToS v{CURRENT_TERMS_VERSION} and Privacy v{CURRENT_PRIVACY_VERSION}")
        
        return {
            "success": True,
            "accepted_at": now.isoformat(),
            "terms_version": CURRENT_TERMS_VERSION,
            "privacy_version": CURRENT_PRIVACY_VERSION
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[LEGAL_ACCEPT] Error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to record acceptance")
@app.get("/usage")
def get_usage(db: Session = Depends(get_db), current_user: User = Depends(get_authenticated_user)) -> dict[str, Any]:
    raw_history = getattr(current_user, "audit_history", None)
    if raw_history is None:
        history_list: list[Any] = []
    else:
        history_str = raw_history if isinstance(raw_history, str) else str(raw_history)
        try:
            parsed_obj: Any = json.loads(history_str) if history_str.strip() else []
            if isinstance(parsed_obj, list):
                parsed: list[Any] = parsed_obj
            else:
                parsed = []
            history_list: list[Any] = parsed
        except Exception:
            logger.warning("Failed to parse audit_history for user %s, returning empty history", getattr(current_user, "username", "unknown"))
            history_list: list[Any] = []
    
    return {"audits": len(history_list), "tier": current_user.tier}

# HTTP middleware for global request logging
@app.middleware("http")
async def log_requests(request: Request, call_next: Callable[[Request], Awaitable[Response]]):
    client_host = request.client.host if request.client else "unknown"
    logger.debug(f"Incoming request: {request.method} {request.url}, headers={request.headers}, client={client_host}")
    response = await call_next(request)
    logger.debug(f"Response: status={response.status_code}, headers={response.headers}")
    return response

# Root endpoint to redirect to /ui
@app.get("/")
async def root(request: Request):
    # Check if user is logged in
    user = request.session.get("user")
    
    if user:
        # Logged in → go to UI
        logger.info(f"Root endpoint accessed by {user.get('nickname', 'unknown')}, redirecting to /ui")
        return RedirectResponse(url="/ui", status_code=307)
    else:
        # Logged out → redirect to login
        logger.info("Root endpoint accessed (not authenticated), redirecting to /login")
        return RedirectResponse(url="/login", status_code=307)

@app.get("/csrf-token", response_model=dict)
async def csrf_token(request: Request):
    return {"csrf_token": await get_csrf_token(request)}

# Manual CSRF Protection
async def get_csrf_token(request: Request) -> str:
    try:
        if "csrf_token" not in request.session:
            request.session["csrf_token"] = secrets.token_urlsafe(32)
            request.session["csrf_last_refresh"] = datetime.now(timezone.utc).isoformat()
            logger.debug(f"Initialized new CSRF token for session: {request.session['csrf_token']}")
            return request.session["csrf_token"]
        
        token = request.session["csrf_token"]
        last_refresh_str = request.session.get("csrf_last_refresh")
        now = datetime.now(timezone.utc)
        
        if not last_refresh_str:
            token = secrets.token_urlsafe(32)
            request.session["csrf_token"] = token
            request.session["csrf_last_refresh"] = now.isoformat()
            logger.debug("CSRF token missing last_refresh – refreshed")
            return token
        
        last_refresh = datetime.fromisoformat(last_refresh_str)
        if not last_refresh.tzinfo:
            last_refresh = last_refresh.replace(tzinfo=timezone.utc)
        
        if (now - last_refresh) > timedelta(minutes=15):
            token = secrets.token_urlsafe(32)
            request.session["csrf_token"] = token
            request.session["csrf_last_refresh"] = now.isoformat()
            logger.debug(f"Refreshed CSRF token (age: {(now - last_refresh).seconds}s)")
        else:
            logger.debug(f"Reusing CSRF token (age: {(now - last_refresh).seconds}s)")
        
        return token
    except Exception as e:
        logger.error(f"Failed to get CSRF token: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate CSRF token")

async def verify_csrf_token(request: Request):
    if request.method in ["GET", "HEAD", "OPTIONS"]:
        return
    
    token = request.headers.get("X-CSRFToken")
    expected = request.session.get("csrf_token")
    
    if not expected:
        request.session["csrf_token"] = secrets.token_urlsafe(32)
        expected = request.session["csrf_token"]
    
    if not token or token != expected:
        logger.error(f"CSRF failed: header={token} session={expected}")
        raise HTTPException(status_code=403, detail="CSRF token invalid")
    
    logger.debug("CSRF valid")

# Sync client init (STABLE)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def initialize_client() -> tuple[Optional[OpenAI], Web3]:
    logger.info("Initializing clients...")
    global client, w3
    
    # DEBUG: Check if env var is available
    grok_key = os.getenv("GROK_API_KEY")
    print("=" * 80)
    print("🔍 INITIALIZE_CLIENT DEBUG:")
    print(f"GROK_API_KEY from os.getenv: {'YES ✅' if grok_key else 'NO ❌'}")
    if grok_key:
        print(f"First 20 chars: {grok_key[:20]}...")
    print("=" * 80)
    
    # Grok client – safe with fallback
    if grok_key and grok_key.strip():
        client = OpenAI(
            api_key=grok_key.strip(),
            base_url="https://api.x.ai/v1"
        )
        logger.info("[GROK] Client initialized successfully")
    else:
        client = None
        logger.warning("[GROK] GROK_API_KEY missing or empty – Grok analysis will be skipped")
    
    # Web3 (unchanged)
    infura_url = f"https://mainnet.infura.io/v3/{os.getenv('INFURA_PROJECT_ID')}"
    w3 = Web3(Web3.HTTPProvider(infura_url))
    if not w3.is_connected():
        logger.error("Failed to connect to Infura")
        raise ConnectionError("Infura connection failed")
    
    logger.info("Clients initialized")
    return client, w3

def extract_json_from_response(raw_response: str) -> str:
    """
    Robustly extract JSON object from AI response.
    Handles markdown code blocks, extra text, and malformed responses.
    """
    import re
    
    if not raw_response or not raw_response.strip():
        logger.warning("[JSON_EXTRACT] Empty response received")
        return "{}"
    
    text = raw_response.strip()
    
    # Step 1: Remove markdown code blocks (```json ... ``` or ``` ... ```)
    code_block_patterns = [
        r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
        r'```\s*([\s\S]*?)\s*```',       # ``` ... ```
    ]
    
    for pattern in code_block_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            if extracted.startswith('{') and extracted.endswith('}'):
                logger.info("[JSON_EXTRACT] Extracted JSON from markdown code block")
                return extracted
    
    # Step 2: Find JSON object with proper brace matching
    # This handles nested objects correctly
    start_idx = text.find('{')
    if start_idx == -1:
        logger.warning("[JSON_EXTRACT] No JSON object found in response")
        return "{}"
    
    brace_count = 0
    end_idx = -1
    in_string = False
    escape_next = False
    
    for i in range(start_idx, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"':
            in_string = not in_string
            continue
        
        if in_string:
            continue
        
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i
                break
    
    if end_idx == -1:
        # Fallback: use last } (original behavior)
        end_idx = text.rfind('}')
        logger.warning("[JSON_EXTRACT] Brace matching failed, using fallback")
    
    if end_idx > start_idx:
        result = text[start_idx:end_idx + 1]
        logger.info(f"[JSON_EXTRACT] Extracted {len(result)} chars of JSON")
        return result
    
    logger.error("[JSON_EXTRACT] Failed to extract valid JSON")
    return "{}"


def normalize_audit_response(audit_json: dict, tier: str) -> dict:
    """
    Normalize and validate AI audit response to ensure all required fields exist.
    This restores the field population that worked with Grok's strict JSON schema.
    """
    
    # ===== REQUIRED CORE FIELDS =====
    # Risk score - ensure it's a number or valid string
    risk_score = audit_json.get("risk_score")
    if risk_score is None:
        audit_json["risk_score"] = 50  # Default moderate risk
    elif isinstance(risk_score, str):
        # Try to extract number from string like "75/100" or "High (85)"
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', str(risk_score))
        if numbers:
            audit_json["risk_score"] = float(numbers[0])
        else:
            audit_json["risk_score"] = 50
    
    # Executive summary
    if not audit_json.get("executive_summary"):
        audit_json["executive_summary"] = "AI analysis completed. Review the identified issues below for detailed findings."
    
    # ===== ISSUES ARRAY =====
    issues = audit_json.get("issues", [])
    if not isinstance(issues, list):
        issues = []
    
    normalized_issues = []
    for idx, issue in enumerate(issues):
        if not isinstance(issue, dict):
            continue
        
        # Normalize each issue to match expected schema
        normalized_issue = {
            "id": issue.get("id") or f"ISSUE-{idx + 1:03d}",
            "type": issue.get("type") or issue.get("title") or issue.get("name") or "Security Issue",
            "severity": _normalize_severity(issue.get("severity", "Medium")),
            "description": issue.get("description") or issue.get("details") or "No description provided",
            "fix": issue.get("fix") or issue.get("recommendation") or issue.get("remediation") or "Manual review recommended",
        }
        
        # Pro tier fields
        if tier in ["pro", "enterprise", "diamond"]:
            normalized_issue["line_number"] = issue.get("line_number") or issue.get("line") or issue.get("lineNumber")
            normalized_issue["function_name"] = issue.get("function_name") or issue.get("function") or issue.get("functionName")
            normalized_issue["vulnerable_code"] = issue.get("vulnerable_code") or issue.get("code") or issue.get("vulnerableCode")
            normalized_issue["exploit_scenario"] = issue.get("exploit_scenario") or issue.get("exploit") or issue.get("exploitScenario")
            normalized_issue["estimated_impact"] = issue.get("estimated_impact") or issue.get("impact") or issue.get("estimatedImpact")
            
            # Code fix object
            code_fix = issue.get("code_fix") or issue.get("codeFix") or issue.get("fix_code")
            if isinstance(code_fix, dict):
                normalized_issue["code_fix"] = {
                    "before": code_fix.get("before") or code_fix.get("original") or "",
                    "after": code_fix.get("after") or code_fix.get("fixed") or "",
                    "explanation": code_fix.get("explanation") or code_fix.get("reason") or ""
                }
            
            # Alternatives array
            alts = issue.get("alternatives") or issue.get("alternative_fixes") or []
            if isinstance(alts, list):
                normalized_issue["alternatives"] = [
                    {
                        "approach": alt.get("approach") or alt.get("method") or "",
                        "pros": alt.get("pros") or alt.get("advantages") or "",
                        "cons": alt.get("cons") or alt.get("disadvantages") or "",
                        "gas_impact": alt.get("gas_impact") or alt.get("gasImpact") or "Unknown"
                    }
                    for alt in alts if isinstance(alt, dict)
                ]
        
        # Enterprise tier fields
        if tier in ["enterprise", "diamond"]:
            normalized_issue["proof_of_concept"] = issue.get("proof_of_concept") or issue.get("poc") or issue.get("proofOfConcept")
            
            refs = issue.get("references") or []
            if isinstance(refs, list):
                normalized_issue["references"] = [
                    {
                        "title": ref.get("title") or ref.get("name") or "Reference",
                        "url": ref.get("url") or ref.get("link") or "#"
                    }
                    for ref in refs if isinstance(ref, dict)
                ]
        
        normalized_issues.append(normalized_issue)
    
    audit_json["issues"] = normalized_issues
    
    # ===== SEVERITY COUNTS =====
    audit_json["critical_count"] = sum(1 for i in normalized_issues if i.get("severity", "").lower() == "critical")
    audit_json["high_count"] = sum(1 for i in normalized_issues if i.get("severity", "").lower() == "high")
    audit_json["medium_count"] = sum(1 for i in normalized_issues if i.get("severity", "").lower() == "medium")
    audit_json["low_count"] = sum(1 for i in normalized_issues if i.get("severity", "").lower() == "low")
    
    # ===== PREDICTIONS =====
    predictions = audit_json.get("predictions", [])
    if not isinstance(predictions, list):
        predictions = []
    
    normalized_predictions = []
    for pred in predictions:
        if isinstance(pred, dict):
            normalized_predictions.append({
                "scenario": pred.get("scenario") or pred.get("description") or "Potential vulnerability scenario",
                "impact": pred.get("impact") or pred.get("consequence") or "Unknown impact"
            })
        elif isinstance(pred, str):
            normalized_predictions.append({"scenario": pred, "impact": "See description"})
    
    # Default predictions if empty
    if not normalized_predictions and normalized_issues:
        if audit_json["critical_count"] > 0:
            normalized_predictions.append({
                "scenario": "Critical vulnerabilities may be exploited within hours of deployment",
                "impact": "Potential total loss of contract funds"
            })
        if audit_json["high_count"] > 0:
            normalized_predictions.append({
                "scenario": "High-severity issues could be targeted by sophisticated attackers",
                "impact": "Significant financial or operational damage"
            })
    
    audit_json["predictions"] = normalized_predictions
    
    # ===== RECOMMENDATIONS =====
    recommendations = audit_json.get("recommendations", {})
    
    if isinstance(recommendations, list):
        # Convert flat list to categorized object
        audit_json["recommendations"] = {
            "immediate": recommendations[:2] if len(recommendations) > 0 else ["Review critical issues immediately"],
            "short_term": recommendations[2:4] if len(recommendations) > 2 else ["Schedule security review"],
            "long_term": recommendations[4:] if len(recommendations) > 4 else ["Consider formal verification"]
        }
    elif isinstance(recommendations, dict):
        audit_json["recommendations"] = {
            "immediate": recommendations.get("immediate") or recommendations.get("critical") or ["Review critical issues"],
            "short_term": recommendations.get("short_term") or recommendations.get("shortTerm") or ["Schedule testing"],
            "long_term": recommendations.get("long_term") or recommendations.get("longTerm") or ["Plan security roadmap"]
        }
    else:
        audit_json["recommendations"] = {
            "immediate": ["Address critical vulnerabilities before deployment"],
            "short_term": ["Implement comprehensive testing"],
            "long_term": ["Establish ongoing security monitoring"]
        }
    
    # ===== REMEDIATION ROADMAP (Enterprise) =====
    if tier in ["enterprise", "diamond"]:
        if not audit_json.get("remediation_roadmap"):
            critical_count = audit_json["critical_count"]
            high_count = audit_json["high_count"]
            audit_json["remediation_roadmap"] = f"""Day 1-2: Fix {critical_count} critical issues immediately
Day 3-5: Address {high_count} high-severity vulnerabilities
Week 2: Implement medium/low priority fixes
Week 3: Conduct comprehensive re-audit
Week 4: Deploy to testnet for final validation"""
    
    logger.info(f"[NORMALIZE] Normalized response: {len(normalized_issues)} issues, {len(normalized_predictions)} predictions")
    return audit_json


def _normalize_severity(severity: str) -> str:
    """Normalize severity string to expected values."""
    if not severity:
        return "Medium"
    
    severity_lower = str(severity).lower().strip()
    
    severity_map = {
        "critical": "Critical",
        "crit": "Critical",
        "severe": "Critical",
        "high": "High",
        "major": "High",
        "medium": "Medium",
        "moderate": "Medium",
        "med": "Medium",
        "low": "Low",
        "minor": "Low",
        "info": "Low",
        "informational": "Low"
    }
    
    return severity_map.get(severity_lower, "Medium")

async def broadcast_audit_log(username: str, message: str):

    """Send audit log message to user's active WebSocket."""
    ws = active_audit_websockets.get(username)
    if ws and ws.application_state == WebSocketState.CONNECTED:
        try:
            await ws.send_json({
                "type": "audit_log",
                "message": f"{message}"
            })
            logger.debug(f"Sent audit log to {username}: {message}")
        except Exception as e:
            logger.error(f"Failed to send audit log to {username}: {str(e)}")
            active_audit_websockets.pop(username, None)

# Stripe setup
STRIPE_API_KEY = os.getenv("STRIPE_API_KEY")
stripe.api_key = STRIPE_API_KEY
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

# ============================================================================
# NEW PRICING STRUCTURE (December 7, 2025)
# ============================================================================
# Env fallback: Parse prices.txt if env missing
import csv

# NEW TIER PRICES (Primary)
stripe_price_starter = os.getenv("STRIPE_PRICE_STARTER")
if not stripe_price_starter:
    stripe_price_starter = "price_1SbnJ0EqXlKjClpj2CDRYguO"  # Default
STRIPE_PRICE_STARTER = stripe_price_starter

stripe_price_pro = os.getenv("STRIPE_PRICE_PRO")
if not stripe_price_pro:
    stripe_price_pro = "price_1SbnM2EqXlKjClpjt9DeOFQw"  # Default  
STRIPE_PRICE_PRO = stripe_price_pro

stripe_price_enterprise = os.getenv("STRIPE_PRICE_ENTERPRISE")
if not stripe_price_enterprise:
    stripe_price_enterprise = "price_1SbnNTEqXlKjClpjDYVndZ98"  # Default
STRIPE_PRICE_ENTERPRISE = stripe_price_enterprise

# LEGACY PRICES (For backward compatibility)
stripe_price_beginner = os.getenv("STRIPE_PRICE_BEGINNER")
if not stripe_price_beginner:
    try:
        with open('prices.txt') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row[2] == "DeFiGuard Beginner Tier":
                    stripe_price_beginner = row[0]
    except FileNotFoundError:
        stripe_price_beginner = "price_1SFoJGEqXlKjClpjj2RZ10bf"
STRIPE_PRICE_BEGINNER = stripe_price_beginner

stripe_price_diamond = os.getenv("STRIPE_PRICE_DIAMOND")
if not stripe_price_diamond:
    try:
        with open('prices.txt') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row[2] == "DeFiGuard Diamond Tier":
                    stripe_price_diamond = row[0]
    except FileNotFoundError:
        stripe_price_diamond = "price_1SFoVMEqXlKjClpjTyRtHJcD"
STRIPE_PRICE_DIAMOND = stripe_price_diamond

stripe_metered_price_diamond = os.getenv("STRIPE_METERED_PRICE_DIAMOND")
if not stripe_metered_price_diamond:
    try:
        with open('prices.txt') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row[2] == "DeFiGuard Metered Diamond":
                    stripe_metered_price_diamond = row[0]
    except FileNotFoundError:
        stripe_metered_price_diamond = "price_1SFpPTEqXlKjClpjeGFNYSgF"
STRIPE_METERED_PRICE_DIAMOND = stripe_metered_price_diamond

# ============================================================================
# TIER LIMITS AND MAPPING
# ============================================================================
FREE_LIMIT = 3
STARTER_LIMIT = 50  # Changed from BEGINNER_LIMIT = 10
PRO_LIMIT = 9999
ENTERPRISE_LIMIT = 9999  # New tier

# Support both old and new tier names for backward compatibility
level_map = {
    "free": 0,
    "starter": 1,
    "beginner": 1,  # Legacy support
    "pro": 2,
    "enterprise": 3,
    "diamond": 3,  # Legacy support
}

class AuditResponse(BaseModel):
    report: dict[str, Any]
    risk_score: str = Field(..., description="Risk score as a string (e.g., '8.5')")
    overage_cost: Optional[float] = None
    
    @field_validator("risk_score", mode='before')
    def coerce_risk_score(cls, v: Any) -> str:
        if isinstance(v, (int, float)):
            return str(v)
        return v

AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "risk_score": {"type": "number"},
        "executive_summary": {"type": "string"},
        "critical_count": {"type": "integer", "default": 0},
        "high_count": {"type": "integer", "default": 0},
        "medium_count": {"type": "integer", "default": 0},
        "low_count": {"type": "integer", "default": 0},
        "upgrade_prompt": {"type": ["string", "null"]},  # For free tier upsell
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string"},
                    "severity": {"type": "string", "enum": ["Critical", "High", "Medium", "Low"]},
                    "description": {"type": ["string", "null"]},
                    "fix": {"type": "string"},
                    # Pro+ tier fields
                    "line_number": {"type": ["integer", "null"]},
                    "function_name": {"type": ["string", "null"]},
                    "vulnerable_code": {"type": ["string", "null"]},
                    "exploit_scenario": {"type": ["string", "null"]},
                    "estimated_impact": {"type": ["string", "null"]},
                    "code_fix": {
                        "type": ["object", "null"],
                        "properties": {
                            "before": {"type": "string"},
                            "after": {"type": "string"},
                            "explanation": {"type": "string"}
                        }
                    },
                    "alternatives": {
                        "type": ["array", "null"],
                        "items": {
                            "type": "object",
                            "properties": {
                                "approach": {"type": "string"},
                                "pros": {"type": "string"},
                                "cons": {"type": "string"},
                                "gas_impact": {"type": "string"}
                            }
                        }
                    },
                    # Enterprise tier fields
                    "proof_of_concept": {"type": ["string", "null"]},
                    "references": {
                        "type": ["array", "null"],
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "url": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["type", "severity", "fix"]
            }
        },
        "predictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scenario": {"type": "string"},
                    "impact": {"type": "string"}
                },
                "required": ["scenario", "impact"]
            }
        },
        "recommendations": {
            "type": "object",
            "properties": {
                "immediate": {"type": "array", "items": {"type": "string"}},
                "short_term": {"type": "array", "items": {"type": "string"}},
                "long_term": {"type": "array", "items": {"type": "string"}}
            }
        },
        "code_quality_metrics": {
            "type": ["object", "null"],
            "properties": {
                "lines_of_code": {"type": "integer"},
                "functions_count": {"type": "integer"},
                "complexity_score": {"type": "number"}
            }
        },
        "remediation_roadmap": {"type": ["string", "null"]},
        "fuzzing_results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "vulnerability": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["vulnerability", "description"]
            }
        }
    },
    "required": ["risk_score", "issues", "executive_summary"]
}

# Define PROMPT_TEMPLATE for audit endpoint
PROMPT_TEMPLATE = """
You are an elite smart contract security auditor with expertise in:
- Solidity security (10+ years experience)
- DeFi protocol design and attack vectors
- Regulatory compliance (EU MiCA, SEC FIT21, AML/KYC)
- Formal verification and cryptographic security

Your analysis MUST be comprehensive, precise, and actionable.

═══════════════════════════════════════════════════════════════════
TIER: {tier}
CONTRACT TYPE: {contract_type}
STATIC ANALYSIS CONTEXT: {context}
FUZZING RESULTS: {fuzzing_results}
COMPLIANCE PRE-SCAN: {compliance_scan}
═══════════════════════════════════════════════════════════════════

CODE TO AUDIT:
{code}

PROTOCOL DETAILS: {details}

═══════════════════════════════════════════════════════════════════
CRITICAL AUDIT METHODOLOGY (FOLLOW STRICTLY):
═══════════════════════════════════════════════════════════════════

PHASE 1: STATIC VULNERABILITY ANALYSIS
1. Reentrancy attacks (CEI pattern violations)
2. Integer overflow/underflow (even with Solidity 0.8+)
3. Access control bypasses (role-based security)
4. Unchecked external calls (return value validation)
5. Gas optimization issues (DoS via gas limits)
6. Delegate call injection (proxy vulnerabilities)
7. Flash loan attack vectors (price manipulation)
8. Oracle manipulation (TWAP, multi-source validation)
9. Front-running vulnerabilities (MEV protection)
10. Timestamp dependence (miner manipulation)

PHASE 2: DEFI-SPECIFIC ATTACK VECTORS
11. Liquidity pool attacks (sandwich attacks, IL)
12. Governance attacks (voting manipulation, flash loans)
13. Cross-chain bridge vulnerabilities
14. MEV extraction opportunities
15. Composability risks (protocol integration failures)

PHASE 3: REGULATORY COMPLIANCE ANALYSIS
═══════════════════════════════════════════════════════════════════
EU MiCA COMPLIANCE (Markets in Crypto-Assets Regulation):

Article 50 - Custody & Safeguarding:
- Are withdrawal functions protected by access control?
- Is multi-signature custody implemented for high-value assets?
- Does the contract have emergency pause functionality?
- Are user funds segregated from operational funds?

Article 30 - Market Abuse Prevention:
- Are there slippage limits to prevent price manipulation?
- Does the contract protect against front-running attacks?
- Are there circuit breakers for abnormal price movements?

Article 68 - Whitepaper Requirements:
- Does code match documented tokenomics?
- Are all risks adequately disclosed in comments?
- Is technical architecture clearly documented?

Article 120 - AML/KYC Requirements (if applicable):
- Does contract handle user identification requirements?
- Are transaction monitoring hooks present?
- Can suspicious activity be flagged/paused?

═══════════════════════════════════════════════════════════════════
US SEC COMPLIANCE (FIT21 & Howey Test):

Howey Test Analysis (Is this a security?):
1. Investment of money? (Does user contribute assets?)
2. Common enterprise? (Pooled assets/shared outcome?)
3. Expectation of profits? (Yield farming, staking rewards?)
4. Efforts of others? (Centralized team control?)

If 3+ factors present → Likely a SECURITY → Recommend:
- Increase decentralization (reduce admin powers)
- Consider Reg D (accredited investors) or Reg A+ exemptions
- Evaluate SEC registration requirements

FIT21 Decentralization Threshold:
- Is governance sufficiently decentralized?
- Are admin keys timelocked or multi-sig?
- Can the protocol function without the founding team?

═══════════════════════════════════════════════════════════════════

PHASE 4: CODE QUALITY & GAS OPTIMIZATION
- Contract complexity (cyclomatic complexity score)
- Gas efficiency (expensive operations, storage optimization)
- Code maintainability (comments, naming conventions)
- Upgrade safety (proxy patterns, storage collisions)

═══════════════════════════════════════════════════════════════════
ANALYSIS REQUIREMENTS BY TIER:
═══════════════════════════════════════════════════════════════════

FREE TIER:
- Risk score (0-100) with justification
- Top 3 CRITICAL or HIGH severity issues ONLY
- For each issue: type, severity, 2-3 sentence description
- NO fix recommendations (upgrade required)
- Calculate exact counts: critical_count, high_count, medium_count, low_count
- Set upgrade_prompt: "⚠️ [X] critical and [Y] high-severity issues detected. [Z] total issues found. Upgrade to Starter ($29/mo) to get fix recommendations and see all issues."
- Executive summary under 100 words focusing on most critical risk

STARTER TIER ($29/mo):
- Full executive summary (2-3 paragraphs with regulatory context)
- ALL issues with: type, severity, detailed description (4-5 sentences), basic fix
- Fix recommendations must be SPECIFIC and ACTIONABLE:
  ✅ GOOD: "Use OpenZeppelin's ReentrancyGuard by importing '@openzeppelin/contracts/security/ReentrancyGuard.sol' and adding 'nonReentrant' modifier to withdraw()"
  ❌ BAD: "Add reentrancy protection"
- Predictions: 3-5 realistic attack scenarios with quantified impact
- Recommendations in THREE categories:
  * immediate: Actions before deployment (fix critical bugs)
  * short_term: Actions for next 7-30 days (audits, testing)
  * long_term: Strategic improvements (formal verification, monitoring)
- Basic MiCA/SEC compliance analysis (high-level only)
- NO line numbers, code snippets, or PoC exploits (Pro+ features)

PRO TIER ($149/mo):
- Everything in Starter PLUS:
- MANDATORY FOR EVERY ISSUE - ALL fields required:
  * line_number: INTEGER - Exact line number (1-indexed) where vulnerability exists
  * function_name: STRING - Full function signature (e.g., "withdraw(uint256 amount)")
  * vulnerable_code: STRING - Extract 3-5 lines of actual Solidity code showing the vulnerability
  * exploit_scenario: STRING - Detailed attack walkthrough (MINIMUM 50 words) with step-by-step exploitation
  * estimated_impact: STRING - Quantified financial impact (e.g., "$500K at risk", "100% of TVL vulnerable")
  * code_fix: OBJECT with three MANDATORY fields:
    - before: STRING - Vulnerable code (5-10 lines of valid Solidity)
    - after: STRING - Fixed code (5-10 lines of valid Solidity)
    - explanation: STRING - Why this fixes it (minimum 30 words)
  * alternatives: ARRAY of 2-3 alternative fix approaches, each with:
    - approach: STRING - Alternative method
    - pros: STRING - Advantages (minimum 20 words)
    - cons: STRING - Disadvantages (minimum 20 words)
    - gas_impact: STRING - Quantified gas cost (e.g., "+15%", "-20%", "negligible")

- COMPREHENSIVE REGULATORY COMPLIANCE:
  * MiCA compliance score (0-100) with specific article violations
  * SEC Howey Test analysis with decentralization recommendations
  * Specific regulatory risks with mitigation strategies

- CODE QUALITY METRICS (REQUIRED):
  * lines_of_code: INTEGER - Count ALL non-empty, non-comment lines in the ENTIRE file
  * functions_count: INTEGER - Count ALL function declarations in the ENTIRE file
  * complexity_score: FLOAT (0.0-10.0) - Cyclomatic complexity across ENTIRE file
  * CRITICAL: Do NOT limit metrics to a single contract - analyze the FULL file

- DETAILED REMEDIATION ROADMAP:
  * Day-by-day fix schedule with priorities
  * Testing strategy (unit tests, integration tests, fuzz tests)
  * External audit recommendations

ENTERPRISE TIER ($499/mo):
- Everything in Pro PLUS:
- PROOF OF CONCEPT (PoC) for EVERY Critical/High issue:
  * proof_of_concept: STRING - Complete, runnable exploit code (Solidity or JavaScript)
  * Must be syntactically valid with comments explaining each step
  * Minimum 15 lines of code demonstrating the exploit

- REFERENCES (for every Critical/High issue):
  * references: ARRAY of 2-3 objects with:
    - title: STRING - Reference title (e.g., "Similar reentrancy on Protocol X")
    - url: STRING - Full URL to rekt.news, past audit, or security writeup

- ADVANCED ANALYSIS:
  * Multi-AI consensus validation note
  * Formal verification results (if Certora available)
  * Attack surface analysis with trust boundaries
  * Post-deployment monitoring recommendations with specific metrics
  * Custom alerting rules for runtime protection

═══════════════════════════════════════════════════════════════════
CRITICAL OUTPUT REQUIREMENTS:
═══════════════════════════════════════════════════════════════════

1. ALWAYS calculate and include severity counts:
   - critical_count: INTEGER
   - high_count: INTEGER  
   - medium_count: INTEGER
   - low_count: INTEGER

2. For FREE tier with >3 issues, set upgrade_prompt with EXACT COUNTS

3. Use ONLY these severity levels: "Critical", "High", "Medium", "Low"

4. Line numbers must be ACTUAL parsed line numbers from the code (1-indexed)

5. Exploit scenarios must be REALISTIC and SPECIFIC to THIS code

6. Code fixes must be VALID SOLIDITY that compiles

7. Gas impacts must be QUANTIFIED with percentages or "negligible"

8. Risk score (0-100) calculation:
   - Critical issues: +30 each
   - High issues: +15 each
   - Medium issues: +5 each
   - Low issues: +1 each
   - Cap at 100

9. REGULATORY COMPLIANCE is MANDATORY - always include:
   - MiCA compliance analysis
   - SEC Howey Test evaluation
   - Specific article violations with remediation

10. All code_fix examples must show BEFORE and AFTER with clear explanation

Return analysis in the EXACT JSON schema provided. Do not skip any required fields.
"""

## Section 2
import os.path

DATA_DIR = "/opt/render/project/data"
USAGE_STATE_FILE = os.path.join(DATA_DIR, "usage_state.json")
USAGE_COUNT_FILE = os.path.join(DATA_DIR, "usage_count.txt")

class UsageTracker:
    def __init__(self):
        self.count = 0
        self.last_reset = datetime.now()
        os.makedirs(DATA_DIR, exist_ok=True)
        
        if os.path.exists(USAGE_STATE_FILE):
            try:
                with open(USAGE_STATE_FILE, "r") as f:
                    state = json.load(f)
                self.last_tier = state.get("last_tier", "free")
                self.last_change_time = datetime.fromisoformat(state.get("last_change_time", datetime.now().isoformat()))
            except Exception as e:
                logger.error(f"Failed to load usage state: {str(e)}")
                self.last_tier = "free"
                self.last_change_time = datetime.now()
        else:
            self.last_tier = "free"
            self.last_change_time = datetime.now()
            self._save_state()
        
        if os.path.exists(USAGE_COUNT_FILE):
            try:
                with open(USAGE_COUNT_FILE, "r") as f:
                    legacy_count = int(f.read().strip() or 0)
                if legacy_count > self.count:
                    self.count = legacy_count
                    self._save_state()
            except Exception as e:
                logger.error(f"Failed to load usage count: {str(e)}")
        
        # NEW: Updated size limits for new tier structure
        self.size_limits = {
            "free": 500 * 1024,              # 500KB (reduced from 1MB)
            "starter": 1024 * 1024,          # 1MB
            "beginner": 1024 * 1024,         # Legacy support
            "pro": 5 * 1024 * 1024,          # 5MB (increased)
            "enterprise": float("inf"),      # Unlimited
            "diamond": float("inf")          # Legacy support
        }
        
        # NEW: Enhanced feature flags with detailed free tier restrictions
        self.feature_flags = {
            "free": {
                "show_all_issues": False,           # Only show top 3 issues
                "issue_limit": 3,                   # Max issues displayed
                "fix_recommendations": False,       # No fix suggestions
                "predictions": False,
                "onchain": False,
                "pdf_export": False,
                "fuzzing": False,
                "priority_support": False,
                "watermark": True,                  # Add "Upgrade to see more"
                "slither": True,                    # Basic static analysis only
                "mythril": False,
                "echidna": False,
                "certora": False,
                "api_access": False,
                "reports": False,
                "nft_rewards": False,
                # Pro+ features
                "line_by_line_analysis": False,
                "code_snippets": False,
                "exploit_scenarios": False,
                "fix_alternatives": False,
                "impact_estimates": False,
                "code_diffs": False,
                # Enterprise features
                "poc_generation": False,
                "multi_ai_consensus": False,
                "continuous_monitoring": False,
                "interactive_remediation": False,
            },
            "starter": {
                "show_all_issues": True,
                "issue_limit": None,
                "fix_recommendations": "basic",
                "predictions": True,
                "onchain": False,
                "pdf_export": "plain",
                "fuzzing": False,
                "priority_support": False,
                "watermark": False,
                "slither": True,
                "mythril": False,
                "echidna": False,
                "certora": False,
                "api_access": False,
                "reports": True,
                "nft_rewards": False,
                # Pro+ features (disabled for starter)
                "line_by_line_analysis": False,
                "code_snippets": False,
                "exploit_scenarios": False,
                "fix_alternatives": False,
                "impact_estimates": False,
                "code_diffs": False,
                # Enterprise features
                "poc_generation": False,
                "multi_ai_consensus": False,
                "continuous_monitoring": False,
                "interactive_remediation": False,
            },
            "pro": {
                "show_all_issues": True,
                "issue_limit": None,
                "fix_recommendations": "advanced",
                "predictions": True,
                "onchain": True,
                "pdf_export": "branded",
                "fuzzing": True,
                "priority_support": True,
                "watermark": False,
                "slither": True,
                "mythril": True,
                "echidna": True,
                "certora": False,
                "api_access": True,
                "reports": True,
                "nft_rewards": False,
                # Pro+ features (ENABLED)
                "line_by_line_analysis": True,
                "code_snippets": True,
                "exploit_scenarios": True,
                "fix_alternatives": True,
                "impact_estimates": True,
                "code_diffs": True,
                # Enterprise features (disabled for pro)
                "poc_generation": False,
                "multi_ai_consensus": False,
                "continuous_monitoring": False,
                "interactive_remediation": False,
            },
            "enterprise": {
                "show_all_issues": True,
                "issue_limit": None,
                "fix_recommendations": "advanced",
                "predictions": True,
                "onchain": True,
                "pdf_export": "whitelabel",
                "fuzzing": True,
                "priority_support": True,
                "watermark": False,
                "slither": True,
                "mythril": True,
                "echidna": True,
                "certora": True,
                "api_access": True,
                "reports": True,
                "team_accounts": True,
                "runtime_monitoring": True,
                "white_label": True,
                "nft_rewards": True,
                # Pro+ features (all enabled)
                "line_by_line_analysis": True,
                "code_snippets": True,
                "exploit_scenarios": True,
                "fix_alternatives": True,
                "impact_estimates": True,
                "code_diffs": True,
                # Enterprise features (all enabled)
                "poc_generation": True,
                "multi_ai_consensus": True,
                "continuous_monitoring": True,
                "interactive_remediation": True,
            },
            # Legacy support
            "beginner": {
                "show_all_issues": True,
                "issue_limit": None,
                "fix_recommendations": "basic",
                "predictions": True,
                "onchain": False,
                "pdf_export": "plain",
                "fuzzing": False,
                "priority_support": False,
                "watermark": False,
                "slither": True,
                "mythril": False,
                "echidna": False,
                "certora": False,
                "api_access": False,
                "reports": True,
                "nft_rewards": False,
            },
            "diamond": {
                "diamond": True,
                "predictions": True,
                "onchain": True,
                "reports": True,
                "fuzzing": True,
                "priority_support": True,
                "nft_rewards": True
            }
        }
    
    def calculate_diamond_overage(self, file_size: int) -> int:
        """Calculate progressive overage for Diamond tier files >1MB."""
        if file_size <= 1024 * 1024:
            return 0
        
        overage_mb = (file_size - 1024 * 1024) / (1024 * 1024)
        total_cost = 0
        
        if overage_mb <= 10:
            total_cost = overage_mb * 0.50
        else:
            total_cost += 10 * 0.50
            remaining_mb = overage_mb - 10
            if remaining_mb <= 40:
                total_cost += remaining_mb * 1.00
            else:
                total_cost += 40 * 1.00
                remaining_mb -= 40
                if remaining_mb <= 2:
                    total_cost += remaining_mb * 2.00
                else:
                    total_cost += 2 * 2.00
                    total_cost += (remaining_mb - 2) * 5.00
        
        return round(total_cost * 100)
    
    def increment(self, file_size: int, username: Optional[str] = None, db: Optional[Session] = None, commit: bool = True):
        if username:
            if db is None:
                raise HTTPException(status_code=500, detail="Database session is not available")
            
            user = db.query(User).filter(User.username == username).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            current_time = datetime.now()
            
            if not user.last_reset:
                user.last_reset = current_time
                if commit:
                    db.commit()
                lr = getattr(user, "last_reset", None)
                lr_str = lr.isoformat() if isinstance(lr, datetime) else "None"
                logger.debug(f"Initialized last_reset for {username} to {lr_str}")
            
            elapsed_days = (current_time - user.last_reset).days
            
            if user.tier == "free" and elapsed_days >= 30:
                self.count = 0
                user.last_reset = current_time
                logger.info(f"Reset usage for {username} on free tier after 30 days")
            elif user.tier in ["beginner", "starter", "pro"] and elapsed_days >= 30:
                user.tier = "free"
                user.has_diamond = False
                self.count = 0
                user.last_reset = current_time
                if commit:
                    db.commit()
                logger.info(f"Downgraded {username} to free tier due to non-payment")
            
            tier_name = getattr(user, "tier", "free") or "free"
            has_diamond_flag = bool(getattr(user, "has_diamond", False))
            
            raw_limit = cast(float, self.size_limits.get(tier_name, 1024 * 1024))
            try:
                size_limit_for_tier: float = float(raw_limit)
            except (TypeError, ValueError):
                size_limit_for_tier = float(1024 * 1024)
            
            if file_size > size_limit_for_tier and not has_diamond_flag:
                overage_cost = self.calculate_diamond_overage(file_size) / 100
                raise HTTPException(
                    status_code=400,
                    detail=f"File size exceeds tier limit. Upgrade to Pro ($149/mo) or Enterprise ($499/mo) for larger files."
                )
            
            self.count += 1
            user.last_reset = current_time
            if commit:
                db.commit()
            
            logger.info(f"UsageTracker incremented to: {self.count} for {username}, current tier: {user.tier}, has_diamond: {user.has_diamond}")
            return self.count
        else:
            current_tier = os.getenv("TIER", "free")
            current_time = datetime.now()
            
            if current_tier != self.last_tier:
                old_level = level_map.get(self.last_tier, 0)
                new_level = level_map.get(current_tier, 0)
                days_since_change = (current_time - self.last_change_time).days
                
                if new_level > old_level:
                    logger.info(f"Upgrade detected from {self.last_tier} to {current_tier}, resetting count")
                    self.count = 0
                elif new_level < old_level:
                    if days_since_change > 30:
                        logger.info(f"Downgrade from {self.last_tier} to {current_tier} after 30+ days, resetting count")
                        self.count = 0
                    else:
                        logger.info(f"Downgrade from {self.last_tier} to {current_tier} within 30 days, keeping count")
                
                self.last_tier = current_tier
                self.last_change_time = current_time
                self._save_state()
            
            if file_size > self.size_limits[current_tier]:
                overage_cost = self.calculate_diamond_overage(file_size) / 100
                raise HTTPException(
                    status_code=400,
                    detail=f"File size exceeds tier limit. Upgrade to Pro or Enterprise for larger files."
                )
            
            self.count += 1
            self._save_state()
            
            limits = {"free": FREE_LIMIT, "starter": STARTER_LIMIT, "beginner": STARTER_LIMIT, "pro": PRO_LIMIT, "enterprise": ENTERPRISE_LIMIT, "diamond": ENTERPRISE_LIMIT}
            if self.count > limits.get(current_tier, FREE_LIMIT):
                raise HTTPException(status_code=403, detail=f"Usage limit exceeded for {current_tier} tier. Limit is {limits.get(current_tier, FREE_LIMIT)}. Upgrade tier.")
            
            logger.info(f"UsageTracker incremented to: {self.count}, current tier: {current_tier}")
            return self.count
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _save_state(self):
        state: dict[str, Any] = {
            "count": int(self.count),
            "last_tier": str(self.last_tier),
            "last_change_time": self.last_change_time.isoformat()
        }
        try:
            with open(USAGE_STATE_FILE, "w") as f:
                json.dump(state, f)
            with open(USAGE_COUNT_FILE, "w") as f:
                f.write(str(self.count))
        except PermissionError as e:
            logger.error(f"Failed to save usage state: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to save usage state due to permissions")
    
    def reset_usage(self, username: Optional[str] = None, db: Optional[Session] = None):
        try:
            if username:
                if db is None:
                    logger.error("Reset usage called with username but no DB session provided")
                    raise HTTPException(status_code=500, detail="Database session is required when specifying a username")
                
                user = db.query(User).filter(User.username == username).first()
                if not user:
                    logger.error(f"Reset usage failed: User {username} not found")
                    raise HTTPException(status_code=404, detail=f"User {username} not found")
                
                self.count = 0
                user.last_reset = datetime.now()
                db.commit()
                logger.info(f"Reset usage for {username}")
            else:
                self.count = 0
                self.last_change_time = datetime.now()
                self._save_state()
                logger.info("Reset usage for anonymous session")
            
            return self.count
        except Exception as e:
            logger.error(f"Reset usage error for {username or 'anonymous'}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to reset usage: {str(e)}")
    
    def set_tier(self, tier: str, has_diamond: bool = False, username: Optional[str] = None, db: Optional[Session] = None):
        # Support both old and new tier names
        tier_mapping = {
            "beginner": "starter",
            "diamond": "enterprise"
        }
        normalized_tier = tier_mapping.get(tier, tier)
        
        if normalized_tier not in level_map and tier not in level_map:
            raise HTTPException(status_code=400, detail=f"Invalid tier: {tier}. Use 'free', 'starter', 'pro', or 'enterprise'")
        
        if username and db:
            user = db.query(User).filter(User.username == username).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            # Handle legacy diamond tier
            if tier == "diamond" and user.tier != "pro":
                raise HTTPException(status_code=400, detail="Diamond add-on requires Pro tier")
            
            user.tier = normalized_tier
            user.has_diamond = has_diamond if normalized_tier == "pro" else False
            user.last_reset = datetime.now()
            
            if normalized_tier == "pro" and not user.api_key:
                user.api_key = cast(Optional[str], secrets.token_urlsafe(32))
            
            if tier == "diamond":
                user.tier = "pro"
                user.has_diamond = True
                user.last_reset = datetime.now() + timedelta(days=30)
            
            db.commit()
            logger.info(f"Set tier for {username} to {normalized_tier}, has_diamond: {user.has_diamond}")
        else:
            self.last_tier = normalized_tier
            self.last_change_time = datetime.now()
            os.environ["TIER"] = normalized_tier
            self._save_state()
            logger.info(f"Tier switched to: {normalized_tier}")
        
        return f"Switched to {normalized_tier} tier" + (f" with Diamond add-on" if has_diamond else "")
    
    def mock_purchase(self, tier: str, has_diamond: bool = False, username: str = "", db: Optional[Session] = None):
        if tier in level_map and level_map[tier] > level_map.get(self.last_tier, 0):
            result = self.set_tier(tier, has_diamond, username, db)
            self.count = 0
            return f"Purchase successful. {result}"
        return f"Purchase failed. Cannot downgrade from {self.last_tier} to {tier} or invalid tier."

usage_tracker = UsageTracker()
usage_tracker.set_tier("free")

## Section 3

def parse_echidna_output(raw_output: str) -> dict[str, Any]:
    """
    Parse raw Echidna output into structured, frontend-friendly JSON.
    """
    import re
    
    result = {
        "status": "complete",
        "all_passed": True,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_total": 0,
        "function_tests": [],
        "coverage": {
            "instructions": 0,
            "contracts": 0,
            "corpus_size": 0,
            "codehashes": 0
        },
        "fuzzing_iterations": 0,
        "test_limit": 0,
        "gas_per_second": 0,
        "compile_time": None,
        "slither_time": None,
        "execution_summary": "",
        "raw_log": raw_output[-500:] if len(raw_output) > 500 else raw_output
    }
    
    if not raw_output:
        result["status"] = "no_output"
        return result
    
    # Parse function test results
    test_pattern = r'^([a-zA-Z_][a-zA-Z0-9_]*(?:\([^)]*\))?)\s*:\s*(passing|failed|reverted)$'
    for line in raw_output.split('\n'):
        line = line.strip()
        match = re.match(test_pattern, line)
        if match:
            func_name, status = match.groups()
            is_passed = status == "passing"
            result["function_tests"].append({
                "function": func_name,
                "status": status,
                "passed": is_passed,
                "icon": "✅" if is_passed else "❌"
            })
            result["tests_total"] += 1
            if is_passed:
                result["tests_passed"] += 1
            else:
                result["tests_failed"] += 1
                result["all_passed"] = False
    
    # Parse status line
    status_match = re.search(r'\[status\]\s*tests:\s*(\d+)/(\d+),\s*fuzzing:\s*(\d+)/(\d+).*?cov:\s*(\d+).*?corpus:\s*(\d+).*?gas/s:\s*(\d+)', raw_output)
    if status_match:
        result["tests_failed"] = int(status_match.group(1))
        result["tests_total"] = int(status_match.group(2)) if result["tests_total"] == 0 else result["tests_total"]
        result["fuzzing_iterations"] = int(status_match.group(3))
        result["test_limit"] = int(status_match.group(4))
        result["coverage"]["instructions"] = int(status_match.group(5))
        result["coverage"]["corpus_size"] = int(status_match.group(6))
        result["gas_per_second"] = int(status_match.group(7))
    
    # Parse coverage details
    instr_match = re.search(r'Unique instructions:\s*(\d+)', raw_output)
    if instr_match and result["coverage"]["instructions"] == 0:
        result["coverage"]["instructions"] = int(instr_match.group(1))
    
    codehash_match = re.search(r'Unique codehashes:\s*(\d+)', raw_output)
    if codehash_match:
        result["coverage"]["codehashes"] = int(codehash_match.group(1))
    
    corpus_match = re.search(r'Corpus size:\s*(\d+)', raw_output)
    if corpus_match and result["coverage"]["corpus_size"] == 0:
        result["coverage"]["corpus_size"] = int(corpus_match.group(1))
    
    compile_match = re.search(r'Compiling.*?Done!\s*\(([0-9.]+)s\)', raw_output)
    if compile_match:
        result["compile_time"] = float(compile_match.group(1))
    
    slither_match = re.search(r'Running slither.*?Done!\s*\(([0-9.]+)s\)', raw_output)
    if slither_match:
        result["slither_time"] = float(slither_match.group(1))
    
    contract_match = re.search(r'Analyzing contract:\s*[^:]+:(\w+)', raw_output)
    if contract_match:
        result["contract_name"] = contract_match.group(1)
    
    contracts_match = re.search(r'(\d+)\s+contracts', raw_output)
    if contracts_match:
        result["coverage"]["contracts"] = int(contracts_match.group(1))
    
    # Build execution summary
    if result["tests_total"] > 0:
        if result["all_passed"]:
            result["execution_summary"] = f"All {result['tests_total']} tests passed"
            result["status"] = "success"
        else:
            result["execution_summary"] = f"{result['tests_failed']} of {result['tests_total']} tests failed"
            result["status"] = "issues_found"
    else:
        result["execution_summary"] = "Fuzzing completed"
        result["status"] = "complete"
    
    return result
def run_echidna(temp_path: str) -> list[dict[str, str]]:
    """Run Echidna fuzzing via Docker image binary with memory limits."""
    logger.info(f"[ECHIDNA] Starting fuzzing for {temp_path}")
    
    # Skip on Windows dev environment
    env = os.getenv('ENV', 'dev')
    if env == 'dev' and platform.system() == 'Windows':
        logger.info("[ECHIDNA] Skipped in Windows dev env")
        return [{"vulnerability": "Echidna unavailable", "description": "Skipped in dev"}]
    
    echidna_cmd = "echidna"
    logger.info(f"[ECHIDNA] Using Echidna from Docker image PATH")
    
    try:
        # CRITICAL: Limit test count to prevent OOM
        cmd = [
            echidna_cmd, 
            temp_path, 
            "--test-mode", "assertion",
            "--test-limit", "1000",      # Limit iterations (default is 50000!)
            "--seq-len", "50",           # Shorter sequences
            "--shrink-limit", "100"      # Limit shrinking attempts
        ]
        
        logger.info(f"[ECHIDNA] Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # Hard timeout
        )
        
        logger.info(f"[ECHIDNA] Return code: {result.returncode}")
        logger.info(f"[ECHIDNA] STDOUT length: {len(result.stdout)} chars")
        logger.info(f"[ECHIDNA] STDERR length: {len(result.stderr)} chars")
        
        if result.stdout:
            logger.info(f"[ECHIDNA] STDOUT first 500 chars: {result.stdout[:500]}")
        if result.stderr:
            logger.warning(f"[ECHIDNA] STDERR first 500 chars: {result.stderr[:500]}")
        
        if result.returncode == 0 or result.stdout:
            logger.info(f"[ECHIDNA] Completed successfully")
            
            # Parse the output into structured data
            parsed = parse_echidna_output(result.stdout)
            
            logger.info(f"[ECHIDNA] Parsed: {parsed['tests_passed']}/{parsed['tests_total']} tests passed, {parsed['fuzzing_iterations']} iterations")
            
            return [{
                "vulnerability": "Fuzzing complete",
                "description": parsed["execution_summary"],
                "parsed": parsed
            }]
        else:
            logger.warning(f"[ECHIDNA] Failed with code {result.returncode}")
            return [{"vulnerability": "Echidna completed", "description": result.stderr[:500] or "No output"}]
    
    except subprocess.TimeoutExpired:
        logger.warning("[ECHIDNA] Timed out after 60 seconds")
        return [{"vulnerability": "Echidna timeout", "description": "Fuzzing exceeded 60s limit - partial results may be available"}]
    
    except FileNotFoundError:
        logger.warning("[ECHIDNA] Binary not found")
        return [{"vulnerability": "Echidna unavailable", "description": "Echidna not installed"}]
    
    except Exception as e:
        logger.error(f"[ECHIDNA] Error: {str(e)}")
        return [{"vulnerability": "Echidna error", "description": str(e)}]

def run_mythril(temp_path: str) -> list[dict[str, str]]:
    """Run mythril analysis with Docker environment support."""
    if not os.path.exists(temp_path):
        return []
   
    try:
        logger.info(f"[MYTHRIL] Starting analysis for {temp_path}")
        
        # Mythril command with execution limits for faster analysis
        result = subprocess.run(
            [
                "myth", "analyze", temp_path, 
                "-o", "json",
                "--execution-timeout", "60",      # Limit symbolic execution
                "--max-depth", "22",              # Limit search depth
                "--solver-timeout", "10000"       # 10s solver timeout
            ],
            capture_output=True,
            text=True,
            timeout=90  # Reduced from 120s
        )
        
        logger.info(f"[MYTHRIL] Return code: {result.returncode}")
        logger.info(f"[MYTHRIL] STDOUT length: {len(result.stdout)} chars")
        if result.stderr:
            logger.warning(f"[MYTHRIL] STDERR: {result.stderr[:300]}")
       
        if result.stdout:
            try:
                output = json.loads(result.stdout)
                if output.get("success") and output.get("issues"):
                    issues = output.get("issues", [])
                    formatted = [{
                        "vulnerability": issue.get("title", "Unknown"),
                        "description": issue.get("description", "No description")
                    } for issue in issues]
                    logger.info(f"[MYTHRIL] Found {len(formatted)} issues")
                    return formatted if formatted else [{"vulnerability": "No issues", "description": "Mythril found no vulnerabilities"}]
                else:
                    logger.info("[MYTHRIL] Analysis complete, no issues found")
                    return [{"vulnerability": "No issues", "description": "Mythril found no vulnerabilities"}]
            except json.JSONDecodeError:
                # Text output fallback
                logger.info(f"[MYTHRIL] Non-JSON output: {result.stdout[:200]}")
                return [{"vulnerability": "Analysis complete", "description": result.stdout[:500]}]
        else:
            logger.warning(f"[MYTHRIL] No output, stderr: {result.stderr[:300]}")
            return [{"vulnerability": "Mythril completed", "description": result.stderr[:500] or "No output"}]
   
    except subprocess.TimeoutExpired:
        logger.warning("[MYTHRIL] Timed out after 90 seconds")
        return [{"vulnerability": "Mythril timeout", "description": "Analysis exceeded 90s limit - contract may be complex"}]
    except FileNotFoundError:
        logger.warning("[MYTHRIL] Binary not found")
        return [{"vulnerability": "Mythril unavailable", "description": "Mythril not installed"}]
    except Exception as e:
        logger.error(f"[MYTHRIL] Error: {str(e)}")
        return [{"vulnerability": "Mythril failed", "description": str(e)}]

def filter_issues_for_free_tier(report: dict[str, Any], tier: str) -> dict[str, Any]:
    """
    Filter audit report for free tier to show only top 3 issues.
    Adds upgrade messaging and watermark.
    """
    if tier != "free":
        return report  # No filtering for paid tiers
    
    issues = report.get("issues", [])
    
    if len(issues) <= 3:
        return report
    
    # Sort by severity: CRITICAL > HIGH > MEDIUM > LOW
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "UNKNOWN": 4}
    sorted_issues = sorted(
        issues,
        key=lambda x: severity_order.get(x.get("severity", "UNKNOWN").upper(), 4)
    )
    
    # Show only top 3
    top_3 = sorted_issues[:3]
    hidden_count = len(issues) - 3
    
    # Add upgrade message
    filtered_report = report.copy()
    filtered_report["issues"] = top_3
    filtered_report["upgrade_message"] = (
        f"🔒 {hidden_count} more issue{'s' if hidden_count > 1 else ''} hidden. "
        f"Upgrade to Starter ($29/mo) to see all vulnerabilities."
    )
    filtered_report["watermark"] = "FREE TIER - Upgrade for full analysis"
    
    # Remove sensitive details
    filtered_report["recommendations"] = [
        "Upgrade to see detailed fix recommendations"
    ]
    
    return filtered_report

def generate_compliance_pdf(report: dict[str, Any], username: str, file_size: int) -> str | None:
    """Generate MiCA/FIT21 compliance PDF report."""
    try:
        pdf_path = os.path.join(DATA_DIR, f"compliance_report_{username}_{int(time.time())}.pdf")
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story: list[Flowable] = []
        
        story.append(Paragraph(f"<b>DeFiGuard AI Compliance Report</b>", styles['Title']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"<b>User:</b> {username}", styles['Normal']))
        story.append(Paragraph(f"<b>File Size:</b> {file_size / 1024 / 1024:.2f} MB", styles['Normal']))
        story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>MiCA/SEC FIT21 Compliance Summary:</b>", styles['Heading2']))
        story.append(Paragraph("• Custody: High-severity reentrancy risks must be mitigated.", styles['Normal']))
        story.append(Paragraph("• Transparency: All findings disclosed in audit report.", styles['Normal']))
        story.append(Paragraph("• Risk Score: Below 30/100 recommended for production.", styles['Normal']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"<b>Risk Score:</b> {report['risk_score']}/100", styles['Normal']))
        story.append(Paragraph(f"<b>Issues Found:</b> {len(report['issues'])}", styles['Normal']))
        
        doc.build(story)
        logger.info(f"Compliance PDF generated: {pdf_path}")
        return pdf_path
    except Exception as e:
        logger.error(f"PDF generation failed: {str(e)}")
        return None

# Celery/Redis for scale
celery = Celery(__name__, broker=os.getenv("REDIS_URL"), backend=os.getenv("REDIS_URL"))

# FIXED: Added stub for x_semantic_search to prevent NameError
from typing import Optional, List
from datetime import datetime

def x_semantic_search(
    query: str,
    limit: int = 10,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    exclude_usernames: Optional[List[str]] = None,
    usernames: Optional[List[str]] = None,
    min_score_threshold: float = 0.18
):
    logger.warning(f"Stub x_semantic_search called with query: {query}")
    return {"posts": [{"text": "Dummy regulatory update: No real data fetched."}]}

# /regs for regulatory
@app.get("/regs")
async def regs():
    search_result = x_semantic_search(query="MiCA SEC FIT21 updates 2025", limit=5)
    msg = search_result['posts'][0]['text'] if search_result['posts'] else "No updates found"
    return {"message": msg}

# /ws-alerts for scale
@app.websocket("/ws-alerts")
async def ws_alerts(websocket: WebSocket):
    await websocket.accept()
    while True:
        await asyncio.sleep(300)  # 5min
        search_result = x_semantic_search(query="latest DeFi exploits", limit=5)
        await websocket.send_json(search_result)

# /overage for pre-calc
@app.post("/overage")
async def overage(file_size: int = Body(...)):
    cost = usage_tracker.calculate_diamond_overage(file_size) / 100
    return {"cost": cost}

# /push for mobile
@app.post("/push")
async def push(msg: str = Body(...)):
    logger.info(f"Push sent: {msg}")
    return {"message": "Push sent"}

# API key verifier
from sqlalchemy.orm import Session

async def verify_api_key(
    api_key: str = Header(None),
    db: Session = Depends(get_db)
) -> User:
    """
    Verify API key from either new APIKey table or legacy User.api_key field.
    Supports both Pro and Enterprise tiers.
    Updates last_used_at timestamp on successful auth.
    """
    if not api_key:
        raise HTTPException(status_code=403, detail="API key required")
    
    # Try new APIKey table first
    api_key_obj = db.query(APIKey).filter(
        APIKey.key == api_key,
        APIKey.is_active == True
    ).first()
    
    if api_key_obj:
        user = db.query(User).filter(User.id == api_key_obj.user_id).first()
        if user and user.tier in ["pro", "enterprise"]:
            # Update last used timestamp
            api_key_obj.last_used_at = datetime.now()
            db.commit()
            logger.info(f"API auth successful: user={user.username}, key_label={api_key_obj.label}")
            return user
    
    # Fallback to legacy User.api_key field
    user = db.query(User).filter(
        User.api_key == api_key,
        User.tier.in_(["pro", "enterprise"])
    ).first()
    
    if user:
        logger.warning(f"User {user.username} using legacy API key - should migrate to new system")
        return user
    
    raise HTTPException(status_code=403, detail="Invalid API key or insufficient tier")

## Section 4.1: Prompt and Debug Endpoints

@app.get("/debug")
async def debug_log():
    logger.debug("Debug endpoint called")
    logger.info("Test INFO log")
    logger.warning("Test WARNING log")
    logger.error("Test ERROR log")
    return {"message": "Debug logs written to debug.log and console"}

from fastapi.responses import FileResponse

@app.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    logger.info(f"Serving static file: /static/{file_path}")
    file_full_path = os.path.join("static", file_path)
    if not os.path.isfile(file_full_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_full_path)

## Section 4.2: UI and Auth Endpoints

@app.get("/ui")
async def read_ui(
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_authenticated_user),
    session_id: str = Query(None),
    tier: str = Query(None),
    has_diamond: bool = Query(False),
    temp_id: str = Query(None),
    username: str = Query(None),
    upgrade: str = Query(None),
    message: str = Query(None)
):
    try:
        if current_user is None:
            # Guest mode
            template = jinja_env.get_template("index.html")
            html_content = template.render(
                session=request.session,
                userinfo={},
                username="Guest",
                upgrade=upgrade,
                message=message or "Please sign in for full access"
            )
            return HTMLResponse(content=html_content)
        
        # Logged-in user flow
        if "csrf_token" not in request.session:
            request.session["csrf_token"] = secrets.token_urlsafe(32)
        
        session_username = request.session.get("username")
        logger.debug(f"UI request, session_id={session_id}, tier={tier}, has_diamond={has_diamond}, temp_id={temp_id}, username={username}, session_username={session_username}, upgrade={upgrade}, message={message}, session: {request.session}")
        
        if session_id:
            effective_username = username or session_username
            if not effective_username:
                logger.error("No username provided for post-payment redirect; redirecting to login")
                return RedirectResponse(url="/auth?redirect_reason=no_username")
            
            if temp_id:
                logger.info(f"Processing post-payment redirect for Diamond audit, username={effective_username}, session_id={session_id}, temp_id={temp_id}")
                return RedirectResponse(url=f"/complete-diamond-audit?session_id={session_id}&temp_id={temp_id}&username={effective_username}")
            
            if tier:
                logger.info(f"Processing post-payment redirect for tier upgrade, username={effective_username}, session_id={session_id}, tier={tier}, has_diamond={has_diamond}")
                return RedirectResponse(url=f"/complete-tier-checkout?session_id={session_id}&tier={tier}&has_diamond={has_diamond}&username={effective_username}")
        
        template = jinja_env.get_template("index.html")
        userinfo = request.session.get("userinfo", {})
        init_script = ""
        init_js_cookie = request.cookies.get("init_js")
        if init_js_cookie:
            init_script = f"<script>{init_js_cookie}</script>"
        
        username = getattr(current_user, "username", None) or userinfo.get("email") or userinfo.get("name") or "Guest"
        html_content = template.render(
            session=request.session,
            userinfo=userinfo,
            username=username,
            upgrade=upgrade,
            message=message
        )
        
        if init_script:
            html_content = html_content.replace("</body>", f"{init_script}</body>")
        
        response = HTMLResponse(content=html_content)
        response.delete_cookie("init_js")
        logger.info(f"Rendered UI with init script for {username or 'Guest'}")
        return response
    
    except FileNotFoundError:
        logger.error(f"UI file not found: {os.path.abspath('templates/index.html')}")
        return HTMLResponse(content="<h1>UI file not found. Check templates/index.html.</h1>")
    except Exception as e:
        logger.exception(f"Unexpected error in /ui: {e}")
        return HTMLResponse(content=f"<h1>Internal error: {str(e)}</h1>", status_code=500)

@app.get("/queue-monitor", response_class=HTMLResponse)
async def queue_monitor(request: Request):
    """
    Queue Monitor Dashboard - Shows real-time audit queue status by tier.
    Publicly accessible for transparency about system load.
    """
    try:
        template = jinja_env.get_template("queue-monitor.html")
        html_content = template.render()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        logger.error(f"Queue monitor template not found: {os.path.abspath('templates/queue-monitor.html')}")
        return HTMLResponse(content="<h1>Queue monitor not found. Check templates/queue-monitor.html.</h1>")
    except Exception as e:
        logger.exception(f"Unexpected error in /queue-monitor: {e}")
        return HTMLResponse(content=f"<h1>Internal error: {str(e)}</h1>", status_code=500)

@app.get("/auth", response_class=HTMLResponse)
async def read_auth(request: Request):
    try:
        logger.debug(f"Auth page accessed, session: {request.session}")
        with open("templates/auth.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        logger.info(f"Loading auth from: {os.path.abspath('templates/auth.html')}")
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        logger.error(f"Auth file not found: {os.path.abspath('templates/auth.html')}")
        return HTMLResponse(content="<h1>Auth file not found. Check templates folder.</h1>")

## Section 4.3: User and Tier Management Endpoints

from fastapi import Body
from pydantic import BaseModel
import urllib.parse

class TierUpgradeRequest(BaseModel):
    username: Optional[str] = None
    tier: str
    has_diamond: bool = False

@app.get("/tier")
async def get_tier(request: Request, username: str = Query(None), db: Session = Depends(get_db)) -> dict[str, Any]:
    session_username = request.session.get("username")
    logger.debug(f"Tier request: Query username={username}, Session username={session_username}, session: {request.session}")
    
    effective_username = username or session_username
    if not effective_username:
        logger.debug("No username provided for /tier; returning free tier defaults")
        return {
            "tier": "free",
            "size_limit": "500KB",
            "feature_flags": usage_tracker.feature_flags["free"],
            "api_key": None,
            "audit_count": usage_tracker.count,
            "audit_limit": FREE_LIMIT,
            "has_diamond": False,
            "username": None
        }
    
    user = db.query(User).filter(User.username == effective_username).first()
    if not user:
        logger.error(f"Tier fetch failed: User {effective_username} not found")
        raise HTTPException(status_code=404, detail="User not found")
    
    user_tier = user.tier
    
    # Map size limits based on tier
    size_limit_map = {
        "free": "500KB",
        "starter": "1MB",
        "beginner": "1MB",  # Legacy
        "pro": "5MB",
        "enterprise": "Unlimited",
        "diamond": "Unlimited"  # Legacy
    }
    size_limit = size_limit_map.get(user_tier, "500KB")
    
    feature_flags = usage_tracker.feature_flags.get(user_tier, usage_tracker.feature_flags["free"])
    api_key = user.api_key if user.tier in ["pro", "enterprise"] else None
    audit_count = usage_tracker.count
    
    # Map audit limits
    audit_limit_map = {
        "free": FREE_LIMIT,
        "starter": STARTER_LIMIT,
        "beginner": STARTER_LIMIT,  # Legacy
        "pro": PRO_LIMIT,
        "enterprise": ENTERPRISE_LIMIT,
        "diamond": ENTERPRISE_LIMIT  # Legacy
    }
    audit_limit = audit_limit_map.get(user.tier, FREE_LIMIT)
    
    if audit_limit == float("inf"):
        audit_limit = 9999
    
    has_diamond = user.has_diamond
    
    logger.debug(f"Retrieved tier for {effective_username}: {user_tier}, audit count: {audit_count}, has_diamond: {has_diamond}")
    logger.debug("Flushing log file after tier retrieval")
    for handler in logging.getLogger().handlers:
        handler.flush()
    
    return {
        "tier": user_tier,
        "size_limit": size_limit,
        "feature_flags": feature_flags,
        "api_key": api_key,
        "audit_count": audit_count,
        "audit_limit": audit_limit,
        "has_diamond": has_diamond,
        "username": user.username
    }

@app.post("/set-tier/{username}/{tier}")
async def set_tier(username: str, tier: str, request: Request, has_diamond: bool = Query(False), db: Session = Depends(get_db)):
    await verify_csrf_token(request)
    logger.debug(f"Set-tier request for {username}, tier: {tier}, has_diamond: {has_diamond}, session: {request.session}")
    
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Map old tier names to new ones
    tier_mapping = {
        "beginner": "starter",
        "diamond": "enterprise"
    }
    normalized_tier = tier_mapping.get(tier, tier)
    
    if normalized_tier not in level_map and tier not in level_map:
        raise HTTPException(status_code=400, detail=f"Invalid tier: {tier}. Use 'free', 'starter', 'pro', or 'enterprise'")
    
    if tier == "diamond" and user.tier != "pro":
        raise HTTPException(status_code=400, detail="Diamond add-on requires Pro tier")
    
    if not STRIPE_API_KEY:
        logger.error(f"Stripe checkout creation failed for {username} to {tier}: STRIPE_API_KEY not set")
        raise HTTPException(status_code=503, detail="Payment processing unavailable: Please set STRIPE_API_KEY in environment variables.")
    
    lock_file = os.path.join(DATA_DIR, "set_tier.lock")
    
    try:
        with open(lock_file, "w") as f:
            f.write(str(os.getpid()))
        
        # Price mapping with both old and new tier names
        price_map = {
            "starter": STRIPE_PRICE_STARTER,
            "beginner": STRIPE_PRICE_BEGINNER,  # Legacy
            "pro": STRIPE_PRICE_PRO,
            "enterprise": STRIPE_PRICE_ENTERPRISE,
            "diamond": STRIPE_PRICE_DIAMOND  # Legacy
        }
        
        price_id = price_map.get(tier)
        if not price_id:
            logger.error(f"Invalid price_id for tier {tier}: price_id={price_id}")
            raise HTTPException(status_code=400, detail="Cannot downgrade or select invalid tier")
        
        # Check all required price IDs are set
        required_prices = {
            "STRIPE_PRICE_STARTER": STRIPE_PRICE_STARTER,
            "STRIPE_PRICE_PRO": STRIPE_PRICE_PRO,
            "STRIPE_PRICE_ENTERPRISE": STRIPE_PRICE_ENTERPRISE
        }
        missing_prices = [name for name, value in required_prices.items() if not value]
        
        if missing_prices:
            logger.error(f"Stripe checkout creation failed for {username} to {tier}: Missing Stripe price IDs: {', '.join(missing_prices)}")
            raise HTTPException(status_code=503, detail=f"Payment processing unavailable: Missing Stripe price IDs: {', '.join(missing_prices)}")
        
        line_items: list[dict[str, Any]] = []
        
        if tier in ["beginner", "starter", "pro"]:
            line_items.append({"price": price_id, "quantity": 1})
            if has_diamond and tier == "pro" and not user.has_diamond:
                line_items.append({"price": STRIPE_PRICE_DIAMOND, "quantity": 1})
        elif tier in ["diamond", "enterprise"] and user.tier not in ["pro", "diamond", "enterprise"]:
            line_items.append({"price": STRIPE_PRICE_PRO, "quantity": 1})
            line_items.append({"price": STRIPE_PRICE_DIAMOND if tier == "diamond" else STRIPE_PRICE_ENTERPRISE, "quantity": 1})
        elif tier in ["diamond", "enterprise"] and user.tier == "pro":
            line_items.append({"price": STRIPE_PRICE_DIAMOND if tier == "diamond" else STRIPE_PRICE_ENTERPRISE, "quantity": 1})
        
        logger.debug(f"Creating Stripe checkout session for {username} to {tier}, line_items={line_items}")
        
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=line_items,
            mode="subscription",
            success_url=f"https://defiguard-ai-fresh-private.onrender.com/complete-tier-checkout?session_id={{CHECKOUT_SESSION_ID}}&tier={urllib.parse.quote(tier)}&has_diamond={urllib.parse.quote(str(has_diamond).lower())}&username={urllib.parse.quote(username)}",
            cancel_url=f"https://defiguard-ai-fresh-private.onrender.com/ui?username={urllib.parse.quote(username)}",
            metadata={"username": username, "tier": tier, "has_diamond": str(has_diamond).lower()}
        )
        
        logger.info(f"Redirecting {username} to Stripe checkout for {tier} tier, has_diamond: {has_diamond}, session: {request.session}")
        logger.debug(f"Success URL: {session.url}, params: tier={tier}, has_diamond={has_diamond}, username={username}")
        
        if session.subscription:
            sub_id = session.subscription if isinstance(session.subscription, str) else getattr(session.subscription, "id", None)
            user.stripe_subscription_id = sub_id
            
            if sub_id:
                try:
                    sub = stripe.Subscription.retrieve(sub_id)
                    for item in sub.get("items", {}).get("data", []):
                        item_id: Optional[str] = None
                        price_obj: Optional[Any] = None
                        
                        if isinstance(item, dict):
                            price_obj = cast(Optional[Any], item.get("price"))
                            item_id = cast(Optional[str], item.get("id"))
                        else:
                            price_obj = getattr(item, "price", None)
                            item_id = getattr(item, "id", None)
                        
                        price_id: Optional[str] = None
                        if isinstance(price_obj, dict):
                            pid = cast(Optional[str], price_obj.get("id"))
                            price_id = pid if isinstance(pid, str) else None
                        elif price_obj is not None:
                            if hasattr(price_obj, "id"):
                                price_id = getattr(price_obj, "id", None)
                            else:
                                try:
                                    pid = cast(Optional[str], price_obj.get("id"))
                                    price_id = pid if isinstance(pid, str) else None
                                except Exception:
                                    try:
                                        pid2 = cast(Optional[str], getattr(price_obj, "id", None))
                                        price_id = pid2 if isinstance(pid2, str) else None
                                    except Exception:
                                        price_id = None
                        else:
                            price_id = None
                        
                        if price_id == STRIPE_METERED_PRICE_DIAMOND:
                            user.stripe_subscription_item_id = item_id
                except Exception:
                    logger.debug("Failed to retrieve subscription items for subscription id: %s", sub_id)
        
        db.commit()
        return {"session_url": session.url}
    
    except stripe.error.InvalidRequestError as e:
        logger.error(
            "Stripe InvalidRequestError for %s to %s: %s, error_code=%s, param=%s",
            str(username),
            str(tier),
            str(e),
            str(getattr(cast(Any, e), "code", None)) if hasattr(e, "code") else "None",
            str(getattr(e, "param", None)) if hasattr(e, "param") else "None"
        )
        raise HTTPException(status_code=400, detail=f"Invalid Stripe request: {getattr(e, 'user_message', None) or str(e)}")
    except stripe_error.StripeError as e:
        logger.error(f"Stripe error for {username} to {tier}: {str(e)}, error_code={getattr(e, 'code', None)}, param={getattr(e, 'param', None)}")
        raise HTTPException(status_code=503, detail=f"Failed to create checkout session: {getattr(e, 'user_message', None) or 'Payment processing error. Please try again or contact support.'}")
    finally:
        if os.path.exists(lock_file):
            os.unlink(lock_file)

@app.post("/create-tier-checkout")
async def create_tier_checkout(
    request: Request,
    tier_request: TierUpgradeRequest = Body(...),
    db: Session = Depends(get_db)
):
    await verify_csrf_token(request)
    
    session_username = request.session.get("username")
    logger.debug(f"/create-tier-checkout called – body: {tier_request}, session_username: {session_username}")
    
    effective_username = tier_request.username or session_username
    if not effective_username:
        raise HTTPException(status_code=401, detail="Login required")
    
    user = db.query(User).filter(User.username == effective_username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    tier = tier_request.tier
    has_diamond = tier_request.has_diamond
    
    # Map old tier names to new
    tier_mapping = {
        "beginner": "starter",
        "diamond": "enterprise"
    }
    normalized_tier = tier_mapping.get(tier, tier)
    
    if normalized_tier not in level_map and tier not in level_map:
        raise HTTPException(status_code=400, detail=f"Invalid tier: {tier}")
    
    if tier == "diamond" and user.tier != "pro":
        raise HTTPException(status_code=400, detail="Diamond add-on requires Pro tier")
    
    if not STRIPE_API_KEY:
        raise HTTPException(status_code=503, detail="Payment processing unavailable – contact admin")
    
    price_map = {
        "starter": STRIPE_PRICE_STARTER,
        "beginner": STRIPE_PRICE_BEGINNER,
        "pro": STRIPE_PRICE_PRO,
        "enterprise": STRIPE_PRICE_ENTERPRISE,
        "diamond": STRIPE_PRICE_DIAMOND
    }
    
    price_id = price_map.get(tier)
    if not price_id:
        raise HTTPException(status_code=400, detail="Invalid tier for checkout")
    
    line_items = [{"price": price_id, "quantity": 1}]
    if has_diamond and tier == "pro":
        line_items.append({"price": STRIPE_PRICE_DIAMOND, "quantity": 1})
    
    base_url = f"{request.base_url}".rstrip("/")
    success_url = f"{base_url}/complete-tier-checkout?session_id={{CHECKOUT_SESSION_ID}}&tier={tier}&has_diamond={has_diamond}&username={effective_username}"
    cancel_url = f"{base_url}/ui?upgrade=cancel"
    
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=line_items,
            mode="subscription",
            customer_email=user.email,
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={
                "username": effective_username,
                "tier": tier,
                "has_diamond": str(has_diamond),
                "user_id": str(user.id)
            }
        )
        logger.info(f"Stripe session created for {effective_username} → {tier} (diamond: {has_diamond})")
        return {"session_url": session.url}
    except Exception as e:
        logger.error(f"Stripe session creation failed for {effective_username}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create checkout session")

@app.get("/complete-tier-checkout")
async def complete_tier_checkout(
    request: Request,
    db: Session = Depends(get_db),
    session_id: str = Query(...),
    tier: str = Query(...),
    has_diamond: bool = Query(False),
    username: str = Query(...),
):
    logger.debug(f"Complete-tier-checkout request: session_id={session_id}, tier={tier}, has_diamond={has_diamond}, username={username}")
    
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        logger.info(f"Retrieved Stripe session: payment_status={session.payment_status}")
        
        if session.payment_status != "paid":
            logger.error(f"Payment not completed for {username}, status={session.payment_status}")
            return RedirectResponse(url="/ui?upgrade=failed&message=Payment%20failed")
        
        # Find and update the user
        user = db.query(User).filter(User.username == username).first()
        if not user:
            logger.error(f"User {username} not found after Stripe payment")
            return RedirectResponse(url="/auth?redirect_reason=user_not_found")
        
        # Map old tier names to new
        tier_mapping = {
            "beginner": "starter",
            "diamond": "enterprise"
        }
        normalized_tier = tier_mapping.get(tier, tier)
        
        # Apply tier upgrade
        user.tier = normalized_tier
        user.has_diamond = has_diamond if normalized_tier == "pro" else False
        
        if normalized_tier in ["pro", "enterprise"] and not user.api_key:
            user.api_key = cast(Optional[str], secrets.token_urlsafe(32))
        
        if tier == "diamond":
            user.tier = "pro"
            user.has_diamond = True
            user.last_reset = datetime.now() + timedelta(days=30)
        
        # Save Stripe IDs
        if session.subscription:
            user.stripe_subscription_id = session.subscription if isinstance(session.subscription, str) else getattr(session.subscription, "id", None)
            user.stripe_customer_id = session.customer
            
            subscription_id = session.subscription if isinstance(session.subscription, str) else getattr(session.subscription, "id", None)
            if subscription_id:
                try:
                    for item in stripe.Subscription.retrieve(subscription_id).get("items", {}).get("data", []):
                        if item.price.id == STRIPE_METERED_PRICE_DIAMOND:
                            user.stripe_subscription_item_id = item.id
                except Exception as e:
                    logger.error(f"Failed to retrieve subscription items: {e}")
        
        db.commit()
        
        usage_tracker.set_tier(normalized_tier, has_diamond, username, db)
        usage_tracker.reset_usage(username, db)
        
        # Re-establish session
        if request is not None:
            request.session["user_id"] = user.id
            request.session["username"] = user.username
            logger.info(f"Session re-established for {username} after successful Stripe payment")
        
        return RedirectResponse(url="/ui?upgrade=success&message=Tier%20upgrade%20completed")
    
    except Exception as e:
        logger.error(f"Complete-tier-checkout error: {str(e)}")
        return RedirectResponse(url=f"/ui?upgrade=error&message={str(e)}")

## Section 4.4: Webhook Endpoint

@app.post("/webhook")
async def webhook(request: Request, db: Session = Depends(get_db)):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    logger.debug(f"Webhook received, payload: {payload[:100]}, sig_header: {sig_header}, session: {request.session}")
    
    if not STRIPE_API_KEY or not STRIPE_WEBHOOK_SECRET:
        logger.error("Stripe webhook processing failed: STRIPE_API_KEY or STRIPE_WEBHOOK_SECRET not set")
        return Response(status_code=503, content="Webhook processing unavailable: Please set STRIPE_API_KEY and STRIPE_WEBHOOK_SECRET in environment variables.")
    
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
        logger.info(f"Webhook event received: type={event['type']}, id={event['id']}")
    except ValueError as e:
        logger.error(f"Stripe webhook error: Invalid payload - {str(e)}, payload={payload[:200]}")
        return Response(status_code=400, content=f"Invalid payload: {str(e)}")
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Stripe webhook error: Invalid signature - {str(e)}, sig_header={sig_header}")
        return Response(status_code=400, content=f"Invalid signature: {str(e)}")
    except Exception as e:
        logger.error(f"Stripe webhook unexpected error: {str(e)}, payload={payload[:200]}")
        return Response(status_code=500, content=f"Webhook processing failed: {str(e)}")
    
    try:
        if event["type"] == "checkout.session.completed":
            session = event["data"]["object"]
            username = session["metadata"].get("username")
            pending_id = session["metadata"].get("pending_id")
            audit_type = session["metadata"].get("audit_type")
            
            if not username:
                logger.warning(f"Webhook: Missing username in metadata, event_id={event['id']}")
                return Response(status_code=200)
            
            user = db.query(User).filter(User.username == username).first()
            if not user:
                logger.error(f"Webhook: User {username} not found")
                return Response(status_code=200)
            
            # 1. DIAMOND OVERAGE AUDIT
            if pending_id and audit_type == "diamond_overage":
                pending_audit = db.query(PendingAudit).filter(PendingAudit.id == pending_id).first()
                if pending_audit and pending_audit.status == "pending":
                    pending_audit.status = "processing"
                    db.commit()
                    asyncio.create_task(process_pending_audit(db, pending_id))
                    logger.info(f"Webhook: Started background Diamond audit for {username}, pending_id={pending_id}")
                else:
                    logger.warning(f"Webhook: Pending audit {pending_id} not found or already processed")
                return Response(status_code=200)
            
            # 2. TIER UPGRADE
            tier = session["metadata"].get("tier")
            has_diamond = session["metadata"].get("has_diamond") == "true"
            
            if user and tier:
                # Map old tier names to new
                tier_mapping = {
                    "beginner": "starter",
                    "diamond": "enterprise"
                }
                normalized_tier = tier_mapping.get(tier, tier)
                
                user.stripe_customer_id = session.customer
                user.stripe_subscription_id = session.subscription
                
                if session.subscription:
                    try:
                        for item in stripe.Subscription.retrieve(session.subscription).get("items", {}).get("data", []):
                            if item.price.id == STRIPE_METERED_PRICE_DIAMOND:
                                user.stripe_subscription_item_id = item.id
                    except Exception as e:
                        logger.error(f"Failed to retrieve subscription items: {e}")
                
                current_tier = user.tier
                
                if normalized_tier == "pro" and current_tier in ["free", "beginner", "starter"]:
                    user.tier = "pro"
                    if not user.api_key:
                        user.api_key = cast(Optional[str], secrets.token_urlsafe(32))
                elif tier == "diamond" and current_tier == "pro":
                    user.has_diamond = True
                    user.last_reset = datetime.now() + timedelta(days=30)
                elif normalized_tier == "enterprise":
                    user.tier = "enterprise"
                    if not user.api_key:
                        user.api_key = cast(Optional[str], secrets.token_urlsafe(32))
                
                usage_tracker.set_tier(normalized_tier, has_diamond, username, db)
                usage_tracker.reset_usage(username, db)
                db.commit()
                
                logger.info(f"Webhook: Tier upgrade completed for {username} to {normalized_tier}")
                
                # Auto-resume pending audit
                pending_audit = db.query(PendingAudit).filter(PendingAudit.username == username, PendingAudit.status == "pending").first()
                if pending_audit:
                    pending_audit.status = "processing"
                    db.commit()
                    asyncio.create_task(process_pending_audit(db, pending_audit.id))
            else:
                logger.debug(f"Webhook event ignored: unhandled type {event['type']}, event_id={event['id']}")
        
        elif event["type"] in ("invoice.payment_failed", "customer.subscription.deleted"):
            session = event["data"]["object"]
            user_id = session.metadata.get("user_id")
            
            if user_id:
                user = db.query(User).filter(User.id == int(user_id)).first()
                if user:
                    user.tier = "free"
                    user.has_diamond = False
                    user.stripe_subscription_id = None
                    user.stripe_subscription_item_id = None
                    db.commit()
                    logger.info(f"User {user_id} reverted to free tier ({event['type']})")
            else:
                logger.warning(f"Webhook: No user_id in metadata for {event['type']}, cannot revert tier")
        
        return Response(status_code=200)
    
    except Exception as e:
        logger.error(f"Webhook processing error for event {event['id']}: {str(e)}")
        return Response(status_code=500, content=f"Webhook processing failed: {str(e)}")

# WebSocket for real-time audit logging
@app.websocket("/ws-audit-log")
async def websocket_audit_log(websocket: WebSocket, username: str = Query(None)):
    await websocket.accept()
    effective_username = username or "guest"
    active_audit_websockets[effective_username] = websocket
    logger.debug(f"WebSocket audit log connected for {effective_username}")
    
    try:
        while True:
            await asyncio.sleep(1)  # Keep alive
    except Exception as e:
        logger.debug(f"WebSocket audit log disconnected for {effective_username}: {str(e)}")
    finally:
        active_audit_websockets.pop(effective_username, None)

async def process_pending_audit(db: Session, pending_id: str):
    pending_audit = None
    try:
        pending_audit = db.query(PendingAudit).filter(PendingAudit.id == pending_id).first()
        if not pending_audit or pending_audit.status != "processing":
            logger.warning(f"Pending audit not found or invalid status for id {pending_id}")
            return
        
        temp_path = pending_audit.temp_path
        if not os.path.exists(temp_path):
            logger.error(f"Temp file not found for pending audit {pending_id}")
            pending_audit.status = "complete"
            pending_audit.results = json.dumps({"error": "Temp file not found"})
            db.commit()
            return
        
        file_size = os.path.getsize(temp_path)
        with open(temp_path, "rb") as f:
            file = UploadFile(filename="temp.sol", file=f, size=file_size)
            raw_result: dict[str, Any] | None = await audit_contract(file, "", pending_audit.username, db, None)
        
        os.unlink(temp_path)
        
        try:
            if raw_result is None:
                result_obj = {"report": None, "risk_score": None, "overage_cost": None}
            else:
                dict_method = getattr(cast(Any, raw_result), "dict", None)
                if callable(dict_method):
                    try:
                        result_obj = cast(dict[str, Any], dict_method())
                    except Exception as e:
                        logger.debug(f".dict() method call failed for pending audit {pending_id}: {e}")
                        result_obj: dict[str, Any] = {"report": None, "risk_score": None, "overage_cost": None}
                else:
                    try:
                        temp = dict(raw_result)
                        result_obj = temp
                    except Exception:
                        try:
                            result_obj = json.loads(json.dumps(raw_result, default=lambda o: getattr(o, "__dict__", str(o))))
                        except Exception as inner_e:
                            logger.debug(f"Normalization fallbacks failed for pending audit {pending_id}: {inner_e}")
                            result_obj = {"report": None, "risk_score": None, "overage_cost": None}
                
                try:
                    if hasattr(result_obj, "__dict__"):
                        result_obj = vars(result_obj)
                    else:
                        result_obj = {"report": None, "risk_score": None, "overage_cost": None}
                except Exception:
                    result_obj = {"report": None, "risk_score": None, "overage_cost": None}
        except Exception as e:
            logger.error(f"Failed to normalize audit result for pending audit {pending_id}: {e}")
            result_obj: dict[str, Any] = {"report": None, "risk_score": None, "overage_cost": None, "error": str(e)}
        
        pending_audit.results = json.dumps(result_obj)
        pending_audit.status = "complete"
        db.commit()
        logger.info(f"Background pending audit completed for id {pending_id}")
    
    except Exception as e:
        logger.error(f"Background pending audit failed for id {pending_id}: {str(e)}")
        if pending_audit is not None:
            pending_audit.status = "complete"
            pending_audit.results = json.dumps({"error": str(e)})
            db.commit()

## Section 4.5: Audit Endpoints

from io import BytesIO

@app.post("/upload-temp")
async def upload_temp(
    request: Request,
    file: UploadFile = File(...),
    username: str = Query(None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if request is None:
        raise HTTPException(status_code=400, detail="Request object is required")
    
    await verify_csrf_token(request)
    
    session_username = request.session.get("username")
    logger.debug(f"Upload-temp request: Query username={username}, Session username={session_username}, session: {request.session}")
    
    effective_username = username or session_username
    if not effective_username:
        logger.error("No username provided for /upload-temp; redirecting to login")
        raise HTTPException(status_code=401, detail="Please login to continue")
    
    user = db.query(User).filter(User.username == effective_username).first()
    if not user or not user.has_diamond:
        raise HTTPException(status_code=403, detail="Temporary file upload requires Diamond add-on")
    
    temp_id = str(uuid.uuid4())
    temp_dir = os.path.join(DATA_DIR, "temp_files")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{temp_id}.sol")
    
    try:
        code_bytes = await file.read()
        file_size = len(code_bytes)
        
        with open(temp_path, "wb") as f:
            f.write(code_bytes)
    except PermissionError as e:
        logger.error(f"Failed to write temp file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save temporary file due to permissions")
    except Exception as e:
        logger.error(f"Upload temp file failed for {effective_username}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload temporary file: {str(e)}")
    
    logger.info(f"Temporary file uploaded for {effective_username}: {temp_id}, size: {file_size / 1024 / 1024:.2f}MB")
    return {"temp_id": temp_id, "file_size": file_size}

@app.post("/diamond-audit")
async def diamond_audit(
    request: Request,
    file: UploadFile = File(...),
    username: str = Query(None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    await verify_csrf_token(request)
    
    session_username = request.session.get("username") if request is not None else None
    logger.debug(f"Diamond-audit request: Query username={username}, Session username={session_username}, session: {getattr(request, 'session', None)}")
    
    effective_username = username or session_username
    if not effective_username:
        logger.error("No username provided for /diamond-audit; redirecting to login")
        raise HTTPException(status_code=401, detail="Please login to continue")
    
    user = db.query(User).filter(User.username == effective_username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        code_bytes = await file.read()
        file_size = len(code_bytes)
        
        if file_size == 0:
            logger.error(f"Empty file uploaded for {effective_username}")
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        if file_size > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 50MB limit")
        
        overage_cost = usage_tracker.calculate_diamond_overage(file_size)
        logger.info(f"Preparing Diamond audit for {effective_username} with overage ${overage_cost / 100:.2f} for file size {file_size / 1024 / 1024:.2f}MB")
        
        if user.has_diamond:
            # Process audit directly
            new_file = UploadFile(filename=file.filename, file=BytesIO(code_bytes), size=file_size)
            result = cast(dict[str, Any], await audit_contract(new_file, "", effective_username, db, request) or {})
            
            # Report overage post-audit
            overage_mb = (file_size - 1024 * 1024) / (1024 * 1024)
            if overage_mb > 0 and user.stripe_subscription_id and user.stripe_subscription_item_id:
                try:
                    stripe.SubscriptionItem.create_usage_record(
                        user.stripe_subscription_item_id,
                        quantity=int(overage_mb),
                        timestamp=int(time.time()),
                        action="increment"
                    )
                    logger.info(f"Reported {overage_mb:.2f}MB overage for {effective_username} to Stripe post-audit")
                except Exception as e:
                    logger.error(f"Failed to report overage for {effective_username}: {str(e)}")
            
            return result
        else:
            # Persist to PendingAudit
            pending_id = str(uuid.uuid4())
            temp_dir = os.path.join(DATA_DIR, "temp_files")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"{pending_id}.sol")
            
            with open(temp_path, "wb") as f:
                f.write(code_bytes)
            
            pending_audit = PendingAudit(id=pending_id, username=effective_username, temp_path=temp_path)
            db.add(pending_audit)
            db.commit()
            
            if user.tier == "pro":
                line_items = [{"price": STRIPE_PRICE_DIAMOND, "quantity": 1}]
            else:
                line_items = [{"price": STRIPE_PRICE_PRO, "quantity": 1}, {"price": STRIPE_PRICE_DIAMOND, "quantity": 1}]
            
            if not STRIPE_API_KEY:
                logger.error(f"Stripe checkout creation failed for {effective_username} Diamond add-on: STRIPE_API_KEY not set")
                os.unlink(temp_path)
                db.delete(pending_audit)
                db.commit()
                raise HTTPException(status_code=503, detail="Payment processing unavailable: Please set STRIPE_API_KEY in environment variables.")
            
            if request is not None:
                base_url = f"{request.url.scheme}://{request.url.netloc}"
            else:
                base_url = "https://defiguard-ai-fresh-private.onrender.com"
            
            success_url = f"{base_url}/ui?upgrade=success&audit=complete&pending_id={urllib.parse.quote(pending_id)}"
            cancel_url = f"{base_url}/ui"
            
            session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=line_items,
                mode="subscription",
                success_url=success_url,
                cancel_url=cancel_url,
                metadata={"pending_id": pending_id, "username": effective_username, "audit_type": "diamond_overage"}
            )
            
            logger.info(f"Redirecting {effective_username} to Stripe checkout for Diamond add-on")
            return {"session_url": session.url}
    
    except Exception as e:
        logger.error(f"Diamond audit error for {effective_username}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/pending-status/{pending_id}")
async def pending_status(pending_id: str, db: Session = Depends(get_db)):
    pending_audit = db.query(PendingAudit).filter(PendingAudit.id == pending_id).first()
    if not pending_audit:
        raise HTTPException(status_code=404, detail="Pending audit not found")
    
    if pending_audit.status == "complete" and pending_audit.results:
        return json.loads(pending_audit.results)
    
    return {"status": pending_audit.status}

@app.get("/complete-diamond-audit")
async def complete_diamond_audit(
    request: Request,
    db: Session = Depends(get_db),
    session_id: str = Query(...),
    temp_id: str = Query(...),
    username: str = Query(None),
):
    session_username = request.session.get("username") if request is not None else None
    logger.debug(f"Complete-diamond-audit request: Query username={username}, Session username={session_username}, session_id={session_id}, temp_id={temp_id}, session: {getattr(request, 'session', None)}")
    
    effective_username = username or session_username
    if not effective_username:
        logger.error("No username provided for /complete-diamond-audit; redirecting to login")
        return RedirectResponse(url="/auth?redirect_reason=no_username")
    
    user = db.query(User).filter(User.username == effective_username).first()
    if not user:
        logger.error(f"User {effective_username} not found for /complete-diamond-audit")
        return RedirectResponse(url="/auth?redirect_reason=user_not_found")
    
    if not STRIPE_API_KEY:
        logger.error(f"Complete diamond audit failed for {effective_username}: STRIPE_API_KEY not set")
        return RedirectResponse(url="/ui?upgrade=error&message=Payment%20processing%20unavailable")
    
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        if session.payment_status == "paid":
            temp_path = os.path.join(DATA_DIR, "temp_files", f"{temp_id}.sol")
            if not os.path.exists(temp_path):
                raise HTTPException(status_code=404, detail="Temporary file not found")
            
            file_size = os.path.getsize(temp_path)
            with open(temp_path, "rb") as f:
                file = UploadFile(filename="temp.sol", file=f, size=file_size)
                _result: dict[str, Any] | None = await audit_contract(file, None, effective_username, db, request)
            
            os.unlink(temp_path)
            logger.info(f"Diamond audit completed for {effective_username} after payment, session: {request.session}")
            return RedirectResponse(url="/ui?upgrade=success")
        else:
            logger.error(f"Payment not completed for {effective_username}, session_id={session_id}, payment_status={session.payment_status}")
            return RedirectResponse(url="/ui?upgrade=failed")
    except Exception as e:
        logger.error(f"Complete diamond audit failed for {effective_username}: {str(e)}")
        return RedirectResponse(url=f"/ui?upgrade=error&message={str(e)}")

@app.get("/api/audit")
async def api_audit(username: str, api_key: str, db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user or user.api_key != api_key or user.tier not in ["pro", "enterprise"]:
            raise HTTPException(status_code=403, detail="API access requires Pro/Enterprise tier and valid API key")
        
        logger.info(f"API audit endpoint accessed by {username}")
        return {"message": "API audit endpoint (Pro/Enterprise tier)"}
    except Exception as e:
        logger.error(f"API audit error for {username}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/mint-nft")
async def mint_nft(request: Request, username: str = Query(...), db: Session = Depends(get_db)):
    await verify_csrf_token(request)
    
    user = db.query(User).filter(User.username == username).first()
    if not user or user.tier != "enterprise":
        raise HTTPException(status_code=403, detail="NFT mint requires Enterprise tier")
    
    token_id = secrets.token_hex(8)
    logger.info(f"Minted NFT for {username}: token_id={token_id}")
    return {"token_id": token_id}

@app.get("/oauth-google")
async def oauth_google():
    return RedirectResponse(url="https://accounts.google.com/o/oauth2/auth?client_id=YOUR_CLIENT_ID&redirect_uri=http://localhost:8000/callback&scope=openid email profile&response_type=code")

@app.post("/refer")
async def refer(request: Request, link: str = Query(...), db: Session = Depends(get_db)):
    await verify_csrf_token(request)
    logger.info(f"Referral tracked for link: {link}")
    return {"message": "Referral tracked"}

@app.get("/upgrade")
async def upgrade_page():
    try:
        logger.debug("Upgrade page accessed")
        return {"message": "Upgrade at /ui for Starter ($29/mo), Pro ($149/mo), or Enterprise ($499/mo)."}
    except Exception as e:
        logger.error(f"Upgrade page error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/facets/{contract_address}", dependencies=[Depends(verify_api_key)])
async def get_facets(contract_address: str, request: Request, username: str = Query(None), db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        logger.debug(f"Received /facets request for {contract_address} by {username or 'anonymous'}, session: {request.session}")
        
        if w3 is None or not w3.is_address(contract_address):
            logger.error(f"Invalid Ethereum address or Web3 not initialized: {contract_address}")
            raise HTTPException(status_code=400, detail="Invalid Ethereum address or Web3 not initialized")
        
        session_username = request.session.get("username")
        effective_username = username or session_username
        user = db.query(User).filter(User.username == effective_username).first() if effective_username else None
        
        current_tier = user.tier if user else os.getenv("TIER", "free")
        has_diamond = user.has_diamond if user else False
        
        if current_tier not in ["pro", "enterprise", "diamond"] and not has_diamond:
            logger.warning(f"Facet preview denied for {effective_username or 'anonymous'} (tier: {current_tier}, has_diamond: {has_diamond})")
            raise HTTPException(status_code=403, detail="Facet preview requires Pro/Enterprise tier. Upgrade at /ui.")
        
        if not os.getenv("INFURA_PROJECT_ID"):
            logger.error(f"Facet fetch failed for {effective_username}: INFURA_PROJECT_ID not set")
            raise HTTPException(status_code=503, detail="On-chain analysis unavailable: Please set INFURA_PROJECT_ID in environment variables.")
        
        diamond_abi: list[dict[str, Any]] = [
            {
                "inputs": [{"internalType": "bytes4", "name": "_functionSelector", "type": "bytes4"}],
                "name": "facetAddress",
                "outputs": [{"internalType": "address", "name": "", "type": "address"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "facets",
                "outputs": [
                    {
                        "components": [
                            {"internalType": "address", "name": "facetAddress", "type": "address"},
                            {"internalType": "bytes4[]", "name": "functionSelectors", "type": "bytes4[]"}
                        ],
                        "internalType": "struct IDiamondLoupe.Facet[]",
                        "name": "",
                        "type": "tuple[]"
                    }
                ],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        
        checksum_address = Web3.to_checksum_address(contract_address)
        contract: Any = w3.eth.contract(address=checksum_address, abi=diamond_abi)
        facets = contract.functions.facets().call()
        
        facet_data: list[dict[str, Any]] = []
        for facet in facets:
            raw_selectors = facet[1] if isinstance(facet[1], (list, tuple)) else list(facet[1])
            selector_hex_list: list[str] = []
            selector_short_list: list[str] = []
            
            for selector in raw_selectors:
                try:
                    if isinstance(selector, (bytes, bytearray)):
                        hexstr = selector.hex()
                    elif isinstance(selector, str):
                        hexstr = selector
                    else:
                        hexstr = str(selector)
                except Exception:
                    hexstr = str(selector)
                
                short = hexstr[:10]
                selector_hex_list.append(hexstr)
                selector_short_list.append(short)
            
            # Preview rules
            if current_tier == "pro" and not has_diamond:
                selector_hex_out = selector_hex_list[:2]
                selector_short_out = selector_short_list[:2]
            else:
                selector_hex_out = selector_hex_list
                selector_short_out = selector_short_list
            
            facet_data.append({
                "facetAddress": facet[0],
                "functionSelectors": selector_hex_out,
                "functions": selector_short_out
            })
        
        logger.info(f"Retrieved {len(facet_data)} facets for {contract_address}")
        return {"facets": facet_data, "is_preview": current_tier == "pro" and not has_diamond}
    
    except Exception as e:
        logger.error(f"Facet endpoint error for {username or 'anonymous'}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

## Section 4.6 Main Audit Endpoint

def run_certora(temp_path: str) -> list[dict[str, str]]:
    """Minimal typed stub for Certora invocation"""
    logger.warning(f"Stub run_certora called for {temp_path}")
    return [{"rule": "Sample rule", "status": "Passed (dummy)"}]

def analyze_slither(temp_path: str) -> list[dict[str, Any]]:
    """Run Slither via Docker image (solc pre-installed)."""
    if not os.path.exists(temp_path):
        logger.error(f"Slither failed: file not found at {temp_path}")
        return []
    
    try:
        # Docker environment: solc is pre-installed in Echidna image
        logger.info("Starting Slither analysis (Docker environment)")
        sl = Slither(temp_path, solc="solc")
        detection_results = sl.run_detectors()
        findings = [det for dets in detection_results for det in dets if det]
        formatted = [{"name": det.get("check"), "details": det.get("description")} for det in findings]
        logger.info(f"Slither found {len(formatted)} issues")
        return formatted
    except SlitherError as e:
        logger.error(f"Slither failed: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Slither crashed: {str(e)}")
        return []

def summarize_context(context: str) -> str:
    if len(context) > 5000:
        return context[:5000] + " ... summarized top findings"
    return context

# ============================================================================
# QUEUE PROCESSING SYSTEM
# ============================================================================

async def process_audit_queue():
    """Background task that continuously processes the audit queue."""
    logger.info("[QUEUE_PROCESSOR] Starting background queue processor")
    
    while True:
        try:
            job = await audit_queue.process_next()
            if job:
                try:
                    # Route through MAIN audit_contract for 100% feature parity
                    from io import BytesIO
                    
                    file_obj = UploadFile(
                        filename=job.filename,
                        file=BytesIO(job.file_content),
                        size=len(job.file_content)
                    )
                    
                    # Create fresh DB session for this job
                    queue_db = SessionLocal()
                    
                    try:
                        # Update phase for WebSocket subscribers
                        await audit_queue.update_phase(job.job_id, "starting", 5)
                        await notify_job_subscribers(job.job_id, {
                            "status": "processing",
                            "phase": "starting",
                            "progress": 5
                        })
                        
                        # Call MAIN audit function with _from_queue=True
                        result = await audit_contract(
                            file=file_obj,
                            contract_address=job.contract_address,
                            username=job.username,
                            db=queue_db,
                            request=None,
                            _from_queue=True
                        )
                        
                        await audit_queue.complete(job.job_id, result)
                        await notify_job_subscribers(job.job_id, {
                            "status": "completed",
                            "result": result
                        })
                        
                    finally:
                        queue_db.close()
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.exception(f"[QUEUE_PROCESSOR] Audit failed for job {job.job_id[:8]}...")
                    await audit_queue.fail(job.job_id, error_msg)
                    
                    await notify_job_subscribers(job.job_id, {
                        "status": "failed",
                        "error": error_msg
                    })
            else:
                # No jobs available, wait before checking again
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            logger.info("[QUEUE_PROCESSOR] Queue processor cancelled")
            break
        except Exception as e:
            logger.error(f"[QUEUE_PROCESSOR] Unexpected error: {e}")
            await asyncio.sleep(5)


async def periodic_queue_cleanup():
    """Periodically clean up old completed/failed jobs."""
    while True:
        try:
            await asyncio.sleep(3600)  # Every hour
            audit_queue.cleanup_old_jobs(max_age_hours=1)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[QUEUE_CLEANUP] Error: {e}")


async def notify_job_subscribers(job_id: str, data: dict):
    """Send update to all WebSocket subscribers for a job."""
    if job_id in active_job_websockets:
        dead_connections = []
        for ws in active_job_websockets[job_id]:
            try:
                if ws.application_state == WebSocketState.CONNECTED:
                    await ws.send_json(data)
            except Exception:
                dead_connections.append(ws)
        
        # Clean up dead connections
        for ws in dead_connections:
            active_job_websockets[job_id].remove(ws)
        
        if not active_job_websockets[job_id]:
            del active_job_websockets[job_id]


async def execute_queued_audit(job: AuditJob) -> dict:
    """
    Execute a queued audit job. This is the core audit logic extracted
    for queue-based processing.
    """
    from io import BytesIO
    
    username = job.username
    file_content = job.file_content
    filename = job.filename
    tier = job.tier
    contract_address = job.contract_address
    
    # Create temp file
    temp_id = str(uuid.uuid4())
    temp_dir = os.path.join(DATA_DIR, "temp_files")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{temp_id}.sol")
    
    try:
        with open(temp_path, "wb") as f:
            f.write(file_content)
        
        file_size = len(file_content)
        code_str = file_content.decode("utf-8")

        # Phase 1: Static Analysis
        await audit_queue.update_phase(job.job_id, "slither", 10)
        await notify_job_subscribers(job.job_id, {"status": "processing", "phase": "slither", "progress": 10})
        await broadcast_audit_log(username, "Running Slither analysis")

        slither_task = asyncio.create_task(asyncio.to_thread(analyze_slither, temp_path))

        # Phase 2: Mythril (parallel with Slither completion)
        await audit_queue.update_phase(job.job_id, "mythril", 25)
        await notify_job_subscribers(job.job_id, {"status": "processing", "phase": "mythril", "progress": 25})
        await broadcast_audit_log(username, "Running Mythril analysis")

        mythril_task = asyncio.create_task(asyncio.to_thread(run_mythril, temp_path))

        
        # Phase 3: Echidna (if enabled)
        tier_for_flags = tier
        tier_flags_map = {"beginner": "starter", "diamond": "enterprise"}
        tier_for_flags = tier_flags_map.get(tier_for_flags, tier_for_flags)
        fuzzing_enabled = usage_tracker.feature_flags.get(tier_for_flags, {}).get("fuzzing", False)
        
        echidna_task = None
        if fuzzing_enabled:
            await audit_queue.update_phase(job.job_id, "echidna", 40)
            await notify_job_subscribers(job.job_id, {"status": "processing", "phase": "echidna", "progress": 40})
            await broadcast_audit_log(username, "Running Echidna fuzzing")
            echidna_task = asyncio.create_task(asyncio.to_thread(run_echidna, temp_path))

        
        # Wait for all analysis tasks
        if echidna_task:
            results = await asyncio.gather(slither_task, mythril_task, echidna_task, return_exceptions=True)
            slither_findings, mythril_results, fuzzing_results = results
        else:
            results = await asyncio.gather(slither_task, mythril_task, return_exceptions=True)
            slither_findings, mythril_results = results
            fuzzing_results = []
        
        # Handle exceptions and log results
        if isinstance(slither_findings, Exception):
            logger.error(f"Slither failed: {slither_findings}")
            slither_findings = []
        await broadcast_audit_log(username, f"Slither found {len(slither_findings)} issues")

        if isinstance(mythril_results, Exception):
            logger.error(f"Mythril failed: {mythril_results}")
            mythril_results = []
        await broadcast_audit_log(username, f"Mythril found {len(mythril_results)} issues")

        if isinstance(fuzzing_results, Exception):
            logger.error(f"Echidna failed: {fuzzing_results}")
            fuzzing_results = []
        if fuzzing_enabled:
            await broadcast_audit_log(username, f"Echidna completed with {len(fuzzing_results)} results")

        # Build context

        context = json.dumps([f if isinstance(f, dict) else getattr(f, "__dict__", str(f)) for f in slither_findings]).replace('"', '\"') if slither_findings else "No static issues found"
        context = summarize_context(context)
        
        # Pre-calculate metrics
        lines_of_code = len([line for line in code_str.split('\n') if line.strip() and not line.strip().startswith('//')])
        functions_count = code_str.count('function ') + code_str.count('constructor(')
        complexity_keywords = ['if', 'for', 'while', 'case', 'catch', '&&', '||', '?']
        complexity_score = sum(code_str.count(keyword) for keyword in complexity_keywords) / max(functions_count, 1)
        complexity_score = min(round(complexity_score, 1), 10.0)
        
        # Phase 4: AI Analysis
        await audit_queue.update_phase(job.job_id, "grok", 60)
        await notify_job_subscribers(job.job_id, {"status": "processing", "phase": "grok", "progress": 60})
        
        # Compliance pre-scan
        compliance_scan = {}
        try:
            compliance_scan = get_compliance_analysis(code_str, contract_type="defi")
        except Exception as e:
            logger.warning(f"[COMPLIANCE] Pre-scan failed: {e}")
            compliance_scan = {"error": str(e)}
        
        details = "Uploaded Solidity code for analysis."
        if contract_address:
            details += f" Contract address: {contract_address}"
        
        report: dict = {}
        
        try:
            if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("GROK_API_KEY"):
                raise Exception("No AI API keys configured (ANTHROPIC_API_KEY or GROK_API_KEY)")
            
            prompt = PROMPT_TEMPLATE.format(
                context=context,
                fuzzing_results=json.dumps(fuzzing_results),
                code=code_str,
                details=details,
                tier=tier_for_flags,
                contract_type="defi",
                compliance_scan=json.dumps(compliance_scan, indent=2)
            )
            
            # Try Claude first, fallback to Grok
            raw_response = ""
            used_model = "unknown"
            
            if claude_client:
                try:
                    logger.info("[AI] Attempting Claude (primary)...")
                    
                    # Enhanced system prompt for strict JSON output
                    claude_system_prompt = """You are an expert smart contract security auditor. You MUST respond with ONLY a valid JSON object - no markdown, no code blocks, no backticks, no explanatory text before or after.

CRITICAL OUTPUT REQUIREMENTS:
1. Start your response with { and end with }
2. Do NOT wrap in ```json or any markdown
3. Do NOT add any text before or after the JSON
4. All string values must be properly escaped
5. Use double quotes for all keys and string values
6. Ensure all arrays and objects are properly closed

The JSON MUST include these exact keys:
- "risk_score": number (0-100)
- "executive_summary": string
- "critical_count": integer
- "high_count": integer  
- "medium_count": integer
- "low_count": integer
- "issues": array of issue objects
- "predictions": array of prediction objects
- "recommendations": object with "immediate", "short_term", "long_term" arrays

Each issue object MUST have: "id", "type", "severity" (Critical/High/Medium/Low), "description", "fix"
For Pro tier, also include: "line_number", "function_name", "vulnerable_code", "exploit_scenario", "estimated_impact", "code_fix", "alternatives"
For Enterprise tier, also include: "proof_of_concept", "references"
"""
                    
                    response = claude_client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=16384,
                        system=claude_system_prompt,
                        messages=[{
                            "role": "user",
                            "content": prompt
                        }]
                    )
                    raw_response = response.content[0].text or ""
                    used_model = "claude-sonnet-4"
                    logger.info("[AI] Claude response received ✅")
                except Exception as claude_err:
                    logger.warning(f"[AI] Claude failed: {claude_err}, trying Grok fallback...")
            
            if not raw_response and grok_client:
                try:
                    logger.info("[AI] Attempting Grok (fallback)...")
                    response = grok_client.chat.completions.create(
                        model="grok-4-1-fast-reasoning",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        stream=False,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "defi_audit_report",
                                "strict": True,
                                "schema": AUDIT_SCHEMA
                            }
                        }
                    )
                    raw_response = response.choices[0].message.content or ""
                    used_model = "grok-4-fast"
                    logger.info("[AI] Grok fallback response received ✅")
                except Exception as grok_err:
                    logger.error(f"[AI] Grok fallback also failed: {grok_err}")
                    raise Exception("Both Claude and Grok API calls failed")
            
            if not raw_response:
                raise Exception("No AI client available - check API keys")
            
            logger.info(f"[AI] Used model: {used_model}")
            logger.debug(f"[AI] Raw response length: {len(raw_response)} chars")
            logger.debug(f"[AI] Raw response first 500 chars: {raw_response[:500]}")
            
            clean_response = extract_json_from_response(raw_response)
            
            try:
                audit_json = json.loads(clean_response)
            except json.JSONDecodeError as json_err:
                logger.error(f"[AI] JSON parse error: {json_err}")
                logger.error(f"[AI] Cleaned response: {clean_response[:1000]}")
                # Create minimal valid response
                audit_json = {
                    "risk_score": 50,
                    "executive_summary": "AI analysis completed but response parsing failed. Manual review recommended.",
                    "issues": [],
                    "predictions": [],
                    "recommendations": {"immediate": ["Manual security review required"], "short_term": [], "long_term": []}
                }
            
            # CRITICAL: Normalize the response to match expected schema (restores Grok-like behavior)
            audit_json = normalize_audit_response(audit_json, tier_for_flags)
            logger.info(f"[AI] Normalized response: {audit_json.get('critical_count', 0)} critical, {audit_json.get('high_count', 0)} high, {audit_json.get('medium_count', 0)} medium, {audit_json.get('low_count', 0)} low issues")
            
            # Override metrics
            audit_json["code_quality_metrics"] = {
                "lines_of_code": lines_of_code,
                "functions_count": functions_count,
                "complexity_score": complexity_score
            }
            
            # Add fuzzing results for enterprise
            if tier_for_flags in ["enterprise", "diamond"]:
                audit_json["fuzzing_results"] = fuzzing_results
            
            report = audit_json
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            fallback_issues = mythril_results + [
                {"type": d.get("name", "Slither finding"), "severity": "Medium", "description": str(d.get("details", "N/A")), "fix": "Manual review required"}
                for d in (slither_findings or [])
            ]
            
            report = {
                "risk_score": "Unknown (AI analysis failed)",
                "critical_count": sum(1 for i in fallback_issues if i.get("severity", "").lower() == "critical"),
                "high_count": sum(1 for i in fallback_issues if i.get("severity", "").lower() == "high"),
                "medium_count": sum(1 for i in fallback_issues if i.get("severity", "").lower() == "medium"),
                "low_count": sum(1 for i in fallback_issues if i.get("severity", "").lower() == "low"),
                "issues": fallback_issues,
                "predictions": [],
                "recommendations": ["AI analysis unavailable – review static analysis results above"],
                "error": str(e)
            }
        
        # Phase 5: Finalization
        await audit_queue.update_phase(job.job_id, "finalizing", 90)
        await notify_job_subscribers(job.job_id, {"status": "processing", "phase": "finalizing", "progress": 90})
        
        # Apply free tier filtering
        if tier == "free":
            report = filter_issues_for_free_tier(report, tier)
        
        return {
            "report": report,
            "risk_score": str(report.get("risk_score", "N/A")),
            "overage_cost": None,
            "tier": tier
        }
        
    finally:
        # Cleanup temp file
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception as e:
            logger.error(f"Failed to delete temp file: {e}")


# ============================================================================
# QUEUE API ENDPOINTS
# ============================================================================

@app.post("/audit/submit")
async def submit_audit_to_queue(
    request: Request,
    file: UploadFile = File(...),
    contract_address: str = Query(None),
    username: str = Query(None),
    db: Session = Depends(get_db)
) -> dict:
    """
    Submit an audit to the queue. Returns immediately with job_id.
    Client should then poll /audit/status/{job_id} or connect to WebSocket.
    """
    await verify_csrf_token(request)
    
    session_username = request.session.get("username")
    userinfo = request.session.get("userinfo")
    session_email = userinfo.get("email") if userinfo else None
    
    effective_username = username or session_username or session_email or "guest"
    
    # Get user tier
    user = None
    tier = "free"
    if effective_username != "guest":
        user = db.query(User).filter(
            (User.username == effective_username) |
            (User.email == effective_username)
        ).first()
        if user:
            tier = user.tier
    
    # Read file
    code_bytes = await file.read()
    file_size = len(code_bytes)
    
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    
    # Check file size limits
    size_limit = usage_tracker.size_limits.get(tier, 500 * 1024)
    if file_size > size_limit and tier not in ["enterprise", "diamond"]:
        raise HTTPException(
            status_code=400,
            detail=f"File size ({file_size / 1024:.1f}KB) exceeds {tier} tier limit ({size_limit / 1024:.1f}KB). Upgrade to continue."
        )
    
    # Submit to queue
    job = await audit_queue.submit(
        username=effective_username,
        file_content=code_bytes,
        filename=file.filename or "contract.sol",
        tier=tier,
        contract_address=contract_address
    )
    
    # Get initial status
    status = await audit_queue.get_status(job.job_id)
    
    logger.info(f"[QUEUE_SUBMIT] Job {job.job_id[:8]}... submitted by {effective_username}, position: {status['position']}")
    
    return {
        "job_id": job.job_id,
        "status": "queued",
        "position": status["position"],
        "queue_length": status["queue_length"],
        "estimated_wait_seconds": status["estimated_wait_seconds"],
        "message": f"Audit queued successfully. Position: {status['position']}"
    }


@app.get("/audit/status/{job_id}")
async def get_audit_status(job_id: str) -> dict:
    """Get the current status of a queued audit job."""
    status = await audit_queue.get_status(job_id)
    
    if status.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")
    
    return status


@app.get("/audit/queue-stats")
async def get_queue_stats() -> dict:
    """Get current queue statistics (for monitoring/debugging)."""
    return audit_queue.get_stats()


@app.websocket("/ws/job/{job_id}")
async def websocket_job_status(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time job status updates.
    Client connects here after submitting an audit to receive live updates.
    """
    await websocket.accept()
    
    # Register this WebSocket for the job
    if job_id not in active_job_websockets:
        active_job_websockets[job_id] = []
    active_job_websockets[job_id].append(websocket)
    
    logger.debug(f"[WS_JOB] Client connected for job {job_id[:8]}...")
    
    try:
        # Send initial status
        status = await audit_queue.get_status(job_id)
        await websocket.send_json(status)
        
        # Keep connection alive until job completes or client disconnects
        while True:
            try:
                # Wait for client messages (ping/pong or close)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                
                if data == "ping":
                    await websocket.send_text("pong")
                elif data == "status":
                    status = await audit_queue.get_status(job_id)
                    await websocket.send_json(status)
                    
            except asyncio.TimeoutError:
                # Send periodic status update
                status = await audit_queue.get_status(job_id)
                await websocket.send_json(status)
                
                # If job is done, close connection
                if status.get("status") in ["completed", "failed", "not_found"]:
                    break
                    
    except Exception as e:
        logger.debug(f"[WS_JOB] Connection closed for job {job_id[:8]}...: {e}")
    finally:
        # Unregister WebSocket
        if job_id in active_job_websockets:
            try:
                active_job_websockets[job_id].remove(websocket)
                if not active_job_websockets[job_id]:
                    del active_job_websockets[job_id]
            except ValueError:
                pass

@app.post("/audit", response_model=None)
async def audit_contract(
    file: UploadFile = File(...),
    contract_address: str = Query(None),
    username: str = Query(None),
    db: Session = Depends(get_db),
    request: Request = None,
    _from_queue: bool = False
):
    # Skip CSRF for queue-originated requests (no request object)
    if not _from_queue:
        await verify_csrf_token(request)
    
    # Handle request=None for queue-originated requests
    if request is not None:
        session_username = request.session.get("username")
        userinfo = request.session.get("userinfo")
        session_email = userinfo.get("email") if userinfo else None
    else:
        session_username = None
        userinfo = None
        session_email = None
    
    logger.debug(f"Audit request: Query username={username}, Session username={session_username}, Session email={session_email}, from_queue={_from_queue}")
    
    # Resolve effective username
    effective_username = username or session_username or session_email or "guest"
    user = None
    
    if effective_username != "guest":
        user = db.query(User).filter(
            (User.username == effective_username) |
            (User.email == effective_username)
        ).first()
    
    if not user:
        logger.info(f"Proceeding as guest for audit: {effective_username}")
        effective_username = "guest"
        current_tier = "free"
        has_diamond = False
        user = None
    else:
        current_tier = user.tier
        has_diamond = bool(user.has_diamond)
    
    # Initialize locals
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    code_bytes: bytes = b""
    file_size = 0
    temp_path: Optional[str] = None
    context = ""
    fuzzing_results: list[dict[str, Any]] = []
    mythril_results: list[dict[str, Any]] = []
    audit_start_time = datetime.now()
    pdf_path: Optional[str] = None
    
    report: dict[str, Any] = {
        "risk_score": 0.0,
        "issues": [],
        "predictions": [],
        "recommendations": [],
        "remediation_roadmap": None,
        "fuzzing_results": [],
        "mythril_results": [],
        "compliance_pdf": None,
        "error": None,
    }
    
    overage_cost: Optional[float] = None
    
    # Read file
    try:
        if getattr(file, "size", None) and file.size > 100 * 1024 * 1024:
            logger.error(f"File size {file.size / 1024 / 1024:.2f}MB exceeds 100MB limit for {effective_username}")
            raise HTTPException(status_code=400, detail="File exceeds 100MB limit")
        
        code_bytes = await file.read()
        file_size = len(code_bytes)
        
        if file_size == 0:
            logger.error(f"Empty file uploaded for {effective_username}")
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        logger.info(f"File read successfully: {file_size} bytes for user {effective_username}")
        
        # Always create temp file
        temp_id = str(uuid.uuid4())
        temp_dir = os.path.join(DATA_DIR, "temp_files")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"{temp_id}.sol")
        
        try:
            with open(temp_path, "wb") as f:
                f.write(code_bytes)
            logger.debug(f"Temporary file saved: {temp_path} for {effective_username}")
        except PermissionError as e:
            logger.error(f"Failed to write temp file: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to save temporary file due to permissions")
        
        # Diamond large file redirect
        if file_size > usage_tracker.size_limits.get(current_tier, 500 * 1024) and not has_diamond:
            overage_cost = usage_tracker.calculate_diamond_overage(file_size) / 100
            
            if not STRIPE_API_KEY:
                os.unlink(temp_path)
                raise HTTPException(status_code=503, detail="Payment processing unavailable: Please set STRIPE_API_KEY in environment variables.")
            
            line_items = [{"price": STRIPE_PRICE_DIAMOND, "quantity": 1}] if current_tier == "pro" else [{"price": STRIPE_PRICE_PRO, "quantity": 1}, {"price": STRIPE_PRICE_DIAMOND, "quantity": 1}]
            
            base_url = f"{request.url.scheme}://{request.url.netloc}" if request is not None else "https://defiguard-ai-fresh-private.onrender.com"
            success_url = f"{base_url}/complete-diamond-audit?session_id={{CHECKOUT_SESSION_ID}}&temp_id={temp_id}&username={urllib.parse.quote(effective_username)}"
            cancel_url = f"{base_url}/ui"
            
            session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=line_items,
                mode="subscription",
                success_url=success_url,
                cancel_url=cancel_url,
                metadata={"temp_id": temp_id, "username": effective_username, "audit_type": "diamond_overage"}
            )
            
            logger.info(f"Redirecting {effective_username} to Stripe for Diamond add-on due to file size")
            return {"session_url": session.url}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File read failed for {effective_username}: {e}")
        raise HTTPException(status_code=400, detail=f"File read failed: {str(e)}")
    
    # Tier & usage checks
    try:
        current_tier = getattr(user, "tier", os.getenv("TIER", "free"))
        has_diamond_flag = bool(getattr(user, "has_diamond", False))
        
        current_count = usage_tracker.increment(file_size, effective_username, db, commit=False)
        logger.info(f"Audit request {current_count} processed for contract {contract_address or 'uploaded'} with tier {current_tier} for user {effective_username}")
    
    except HTTPException as e:
        # Handle size/usage limit redirects
        if file_size > usage_tracker.size_limits.get(current_tier, 500 * 1024) and not has_diamond_flag:
            logger.info(f"File size exceeds limit for {effective_username}; redirecting to upgrade")
            
            if not temp_path:
                temp_id = str(uuid.uuid4())
                temp_dir = os.path.join(DATA_DIR, "temp_files")
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, f"{temp_id}.sol")
                try:
                    with open(temp_path, "wb") as f:
                        f.write(code_bytes)
                except Exception as ex:
                    logger.error(f"Failed to write temp file for upgrade redirect: {ex}")
                    raise HTTPException(status_code=500, detail="Failed to save temporary file due to permissions")
            
            if not STRIPE_API_KEY:
                logger.error("Stripe checkout creation failed: STRIPE_API_KEY not set")
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
                raise HTTPException(status_code=503, detail="Payment processing unavailable: Please set STRIPE_API_KEY in environment variables.")
            
            # Create appropriate checkout based on tier
            if current_tier in ["free", "starter", "beginner"]:
                # Upgrade to Pro
                price_id = STRIPE_PRICE_PRO
            else:
                # Upgrade to Enterprise
                price_id = STRIPE_PRICE_ENTERPRISE
            
            try:
                session = stripe.checkout.Session.create(
                    payment_method_types=["card"],
                    line_items=[{"price": price_id, "quantity": 1}],
                    mode="subscription",
                    success_url=f"https://defiguard-ai-fresh-private.onrender.com/complete-diamond-audit?session_id={{CHECKOUT_SESSION_ID}}&temp_id={urllib.parse.quote(os.path.basename(temp_path).split('.')[0])}&username={urllib.parse.quote(effective_username)}",
                    cancel_url="https://defiguard-ai-fresh-private.onrender.com/ui",
                    metadata={"temp_id": os.path.basename(temp_path).split('.')[0], "username": effective_username}
                )
                logger.info(f"Redirecting {effective_username} to Stripe for tier upgrade due to file size")
                return {"session_url": session.url}
            except Exception as exc:
                logger.error(f"Stripe session creation failed: {exc}")
                raise HTTPException(status_code=503, detail="Failed to create checkout session")
        
        # Usage limit exceeded
        if e.status_code == 403 or (isinstance(e, HTTPException) and "Usage limit exceeded" in getattr(e, "detail", "")):
            logger.info(f"Usage limit exceeded for {effective_username}; redirecting to upgrade")
            
            if not STRIPE_API_KEY or not STRIPE_PRICE_STARTER:
                logger.error("Stripe checkout creation failed: missing STRIPE config for Starter upgrade")
                raise HTTPException(status_code=503, detail="Payment processing unavailable: Missing Stripe config")
            
            try:
                session = stripe.checkout.Session.create(
                    payment_method_types=["card"],
                    line_items=[{"price": STRIPE_PRICE_STARTER, "quantity": 1}],
                    mode="subscription",
                    success_url=f"https://defiguard-ai-fresh-private.onrender.com/ui?session_id={{CHECKOUT_SESSION_ID}}&tier=starter",
                    cancel_url="https://defiguard-ai-fresh-private.onrender.com/ui",
                    metadata={"username": effective_username, "tier": "starter"}
                )
                return {"session_url": session.url}
            except Exception as exc:
                logger.error(f"Stripe Starter checkout failed: {exc}")
                raise HTTPException(status_code=503, detail="Failed to create checkout session")
        
        raise
    
    # ADMISSION CONTROL: Check if we have capacity to run immediately
    # Skip when called from queue (process_one_queued_job already acquired the slot)
    global active_audit_count
    
    if not _from_queue:
        async with active_audit_lock:
            if active_audit_count >= MAX_CONCURRENT_AUDITS:
                # At capacity - queue the audit
                logger.info(f"[ADMISSION] At capacity ({active_audit_count}/{MAX_CONCURRENT_AUDITS}), queuing audit for {effective_username}")
                
                job = await audit_queue.submit(
                    username=effective_username,
                    file_content=code_bytes,
                    filename=file.filename or "contract.sol",
                    tier=current_tier,
                    contract_address=contract_address
                )
                
                status = await audit_queue.get_status(job.job_id)
                
                return {
                    "queued": True,
                    "job_id": job.job_id,
                    "status": "queued",
                    "position": status["position"],
                    "queue_length": status["queue_length"],
                    "estimated_wait_seconds": status["estimated_wait_seconds"],
                    "message": f"Server busy. Audit queued at position {status['position']}."
                }
            
            # Have capacity - increment counter and proceed
            active_audit_count += 1
            logger.info(f"[ADMISSION] Capacity available ({active_audit_count}/{MAX_CONCURRENT_AUDITS}), running immediately for {effective_username}")
    else:
        logger.info(f"[ADMISSION] From queue - slot already acquired for {effective_username}")
    
    # Main audit processing
    try:
        await broadcast_audit_log(effective_username, "Audit started")
        logger.info(f"Starting audit process for {effective_username}")
        
        # Decode and validate
        try:
            code_str = code_bytes.decode("utf-8")
            logger.debug(f"File decoded successfully for {effective_username}")
            await broadcast_audit_log(effective_username, "File decoded")
        except UnicodeDecodeError as decode_err:
            logger.error(f"File decoding failed for {effective_username}: {decode_err}")
            await broadcast_audit_log(effective_username, "File decoding failed")
            report["error"] = f"Audit failed: File decoding failed: {str(decode_err)}"
            return {"report": report, "risk_score": "N/A", "overage_cost": None}
        
        if not code_str.strip():
            logger.error(f"Empty file uploaded for {effective_username}")
            await broadcast_audit_log(effective_username, "Empty file uploaded")
            report["error"] = "Audit failed: Empty file uploaded"
            
            if code_str.count('"') % 2 != 0 or code_str.count("'") % 2 != 0:
                context = "Invalid Solidity code: unbalanced quotes"
                logger.warning(f"Unbalanced quotes detected in code for {effective_username}")
                await broadcast_audit_log(effective_username, "Unbalanced quotes detected")
            
            return {"report": report, "risk_score": "N/A", "overage_cost": None}
        
        # Parallel static analysis
        # Determine if fuzzing is enabled BEFORE starting tasks
        tier_for_flags = "enterprise" if current_tier == "enterprise" else ("diamond" if getattr(user, "has_diamond", False) else current_tier)
        tier_flags_map = {"beginner": "starter", "diamond": "enterprise"}
        tier_for_flags = tier_flags_map.get(tier_for_flags, tier_for_flags)
        fuzzing_enabled = usage_tracker.feature_flags.get(tier_for_flags, {}).get("fuzzing", False)
        
        # Helper to get job_id if this came from queue (for WebSocket updates)
        job_id = None
        if _from_queue:
            # Find job by username (most recent processing job)
            for jid, job in audit_queue.jobs.items():
                if job.username == effective_username and job.status == AuditStatus.PROCESSING:
                    job_id = jid
                    break
        
        # Run analysis tools SEQUENTIALLY to reduce RAM pressure
        # Each tool uses 300MB-1GB, sequential keeps peak RAM at ~1GB vs ~2.5GB parallel
        # Yield points (asyncio.sleep(0)) allow event loop to handle other requests
        
        # Phase 1: Slither
        await broadcast_audit_log(effective_username, "Running Slither analysis")
        if job_id:
            await audit_queue.update_phase(job_id, "slither", 10)
            await notify_job_subscribers(job_id, {"status": "processing", "phase": "slither", "progress": 10})
        await asyncio.sleep(0)  # Yield to event loop
        try:
            slither_findings = await asyncio.to_thread(analyze_slither, temp_path)
        except Exception as e:
            logger.error(f"Slither failed: {e}")
            slither_findings = []
        await asyncio.sleep(0)  # Yield to event loop
        await broadcast_audit_log(effective_username, f"Slither found {len(slither_findings)} issues")

        # Phase 2: Mythril
        await broadcast_audit_log(effective_username, "Running Mythril analysis")
        if job_id:
            await audit_queue.update_phase(job_id, "mythril", 30)
            await notify_job_subscribers(job_id, {"status": "processing", "phase": "mythril", "progress": 30})
        await asyncio.sleep(0)  # Yield to event loop
        try:
            mythril_results = await asyncio.to_thread(run_mythril, temp_path)
        except Exception as e:
            logger.error(f"Mythril failed: {e}")
            mythril_results = []
        await asyncio.sleep(0)  # Yield to event loop
        await broadcast_audit_log(effective_username, f"Mythril found {len(mythril_results)} issues")

        fuzzing_results = []
        if fuzzing_enabled:
            # Phase 3: Echidna
            await broadcast_audit_log(effective_username, "Running Echidna fuzzing")
            if job_id:
                await audit_queue.update_phase(job_id, "echidna", 50)
                await notify_job_subscribers(job_id, {"status": "processing", "phase": "echidna", "progress": 50})
            await asyncio.sleep(0)  # Yield to event loop
            try:
                fuzzing_results = await asyncio.to_thread(run_echidna, temp_path)
            except Exception as e:
                logger.error(f"Echidna failed: {e}")
                fuzzing_results = []
            await asyncio.sleep(0)  # Yield to event loop
            await broadcast_audit_log(effective_username, f"Echidna completed with {len(fuzzing_results)} results")

        # Results already handled above in sequential execution
        
        context = json.dumps([f if isinstance(f, dict) else getattr(f, "__dict__", str(f)) for f in slither_findings]).replace('"', '\"') if slither_findings else "No static issues found"
        context = summarize_context(context)
        
        # On-chain analysis
        details = "Uploaded Solidity code for analysis."
        if contract_address:
            if not usage_tracker.feature_flags.get(tier_for_flags, {}).get("onchain", False):
                logger.warning(f"On-chain analysis denied for {effective_username} (tier: {current_tier})")
                await broadcast_audit_log(effective_username, "On-chain analysis denied (tier restriction)")
                raise HTTPException(status_code=403, detail="On-chain analysis requires Starter tier or higher.")
            
            if not os.getenv("INFURA_PROJECT_ID"):
                logger.error("INFURA_PROJECT_ID not set")
                await broadcast_audit_log(effective_username, "On-chain analysis failed: INFURA_PROJECT_ID not set")
                raise HTTPException(status_code=503, detail="On-chain analysis unavailable: Please set INFURA_PROJECT_ID")
            
            if not w3.is_address(contract_address):
                logger.error(f"Invalid Ethereum address: {contract_address}")
                await broadcast_audit_log(effective_username, "Invalid Ethereum address")
                raise HTTPException(status_code=400, detail="Invalid Ethereum address.")
            
            try:
                onchain_code = w3.eth.get_code(contract_address)
                details += f" On-chain code fetched for {contract_address} (bytecode length: {len(onchain_code)})"
                await broadcast_audit_log(effective_username, "On-chain code fetched")
            except Exception as e:
                logger.error(f"On-chain code fetch failed: {e}")
                await broadcast_audit_log(effective_username, "No deployed code found")
        
        # Pre-calculate code metrics (don't rely on AI)
        lines_of_code = len([line for line in code_str.split('\n') if line.strip() and not line.strip().startswith('//')])
        functions_count = code_str.count('function ') + code_str.count('constructor(')
        
        # Calculate cyclomatic complexity (simple heuristic)
        complexity_keywords = ['if', 'for', 'while', 'case', 'catch', '&&', '||', '?']
        complexity_score = sum(code_str.count(keyword) for keyword in complexity_keywords) / max(functions_count, 1)
        complexity_score = min(round(complexity_score, 1), 10.0)
        
        logger.info(f"[METRICS] Calculated: {lines_of_code} LOC, {functions_count} functions, complexity {complexity_score}")
        
        # Grok API call
        # Phase 4: AI Analysis
        await asyncio.sleep(0)  # Yield to event loop before heavy API call
        await broadcast_audit_log(effective_username, "Sending to Claude AI")
        if job_id:
            await audit_queue.update_phase(job_id, "ai_analysis", 70)
            await notify_job_subscribers(job_id, {"status": "processing", "phase": "ai_analysis", "progress": 70})
                # Run compliance pre-scan
        compliance_scan = {}
        try:
            compliance_scan = get_compliance_analysis(code_str, contract_type="defi")
            logger.info(f"[COMPLIANCE] Pre-scan completed: {len(compliance_scan.get('attack_vectors', []))} attack vectors detected")
        except Exception as e:
            logger.warning(f"[COMPLIANCE] Pre-scan failed: {e}")
            compliance_scan = {"error": str(e)}

        try:
            if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("GROK_API_KEY"):
                raise Exception("No AI API keys configured (ANTHROPIC_API_KEY or GROK_API_KEY)")
            
            prompt = PROMPT_TEMPLATE.format(
                context=context,
                fuzzing_results=json.dumps(fuzzing_results),
                code=code_str,
                details=details,
                tier=tier_for_flags,
                contract_type="defi",
                compliance_scan=json.dumps(compliance_scan, indent=2)
            )
            
            # Try Claude first, fallback to Grok
            raw_response = ""
            used_model = "unknown"
            
            if claude_client:
                try:
                    logger.info("[AI] Attempting Claude (primary)...")
                    
                    # Enhanced system prompt for strict JSON output
                    claude_system_prompt = """You are an expert smart contract security auditor. You MUST respond with ONLY a valid JSON object - no markdown, no code blocks, no backticks, no explanatory text before or after.

CRITICAL OUTPUT REQUIREMENTS:
1. Start your response with { and end with }
2. Do NOT wrap in ```json or any markdown
3. Do NOT add any text before or after the JSON
4. All string values must be properly escaped
5. Use double quotes for all keys and string values
6. Ensure all arrays and objects are properly closed

The JSON MUST include these exact keys:
- "risk_score": number (0-100)
- "executive_summary": string
- "critical_count": integer
- "high_count": integer  
- "medium_count": integer
- "low_count": integer
- "issues": array of issue objects
- "predictions": array of prediction objects
- "recommendations": object with "immediate", "short_term", "long_term" arrays

Each issue object MUST have: "id", "type", "severity" (Critical/High/Medium/Low), "description", "fix"
For Pro tier, also include: "line_number", "function_name", "vulnerable_code", "exploit_scenario", "estimated_impact", "code_fix", "alternatives"
For Enterprise tier, also include: "proof_of_concept", "references"
"""
                    
                    response = claude_client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=16384,
                        system=claude_system_prompt,
                        messages=[{
                            "role": "user",
                            "content": prompt
                        }]
                    )
                    raw_response = response.content[0].text or ""
                    used_model = "claude-sonnet-4"
                    logger.info("[AI] Claude response received ✅")
                except Exception as claude_err:
                    logger.warning(f"[AI] Claude failed: {claude_err}, trying Grok fallback...")
            
            if not raw_response and grok_client:
                try:
                    logger.info("[AI] Attempting Grok (fallback)...")
                    response = grok_client.chat.completions.create(
                        model="grok-4-1-fast-reasoning",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        stream=False,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "defi_audit_report",
                                "strict": True,
                                "schema": AUDIT_SCHEMA
                            }
                        }
                    )
                    raw_response = response.choices[0].message.content or ""
                    used_model = "grok-4-fast"
                    logger.info("[AI] Grok fallback response received ✅")
                except Exception as grok_err:
                    logger.error(f"[AI] Grok fallback also failed: {grok_err}")
                    raise Exception("Both Claude and Grok API calls failed")
            
            if not raw_response:
                raise Exception("No AI client available - check API keys")
            
            logger.info(f"[AI] Used model: {used_model}")
            # === CRITICAL DEBUG LOGGING ===
            logger.info(f"[AI_DEBUG] ===== RAW RESPONSE START =====")
            logger.info(f"[AI_DEBUG] Model used: {used_model}")
            logger.info(f"[AI_DEBUG] Length: {len(raw_response)} characters")
            logger.info(f"[AI_DEBUG] First 1000 chars: {raw_response[:1000]}")
            logger.info(f"[AI_DEBUG] Last 500 chars: {raw_response[-500:]}")
            logger.info(f"[AI_DEBUG] ===== RAW RESPONSE END =====")

            try:
                audit_json = json.loads(raw_response)
                logger.info(f"[AI_DEBUG] Parsed JSON successfully")
                logger.info(f"[AI_DEBUG] Keys in response: {list(audit_json.keys())}")
                logger.info(f"[AI_DEBUG] Issues count: {len(audit_json.get('issues', []))}")
                logger.info(f"[AI_DEBUG] Issues array: {json.dumps(audit_json.get('issues', []), indent=2)}")
            except json.JSONDecodeError as e:
                logger.error(f"[AI_DEBUG] JSON parse failed: {e}")
                logger.error(f"[AI_DEBUG] Full response: {raw_response}")
            # Debug logging
            logger.info(f"[AI] Response length: {len(raw_response)} chars")
            logger.debug(f"[AI] First 500 chars: {raw_response[:500]}")
            
            clean_response = extract_json_from_response(raw_response)
            
            try:
                audit_json = json.loads(clean_response)
            except json.JSONDecodeError as json_err:
                logger.error(f"[AI] JSON parse error: {json_err}")
                logger.error(f"[AI] Cleaned response: {clean_response[:1000]}")
                # Create minimal valid response
                audit_json = {
                    "risk_score": 50,
                    "executive_summary": "AI analysis completed but response parsing failed. Manual review recommended.",
                    "issues": [],
                    "predictions": [],
                    "recommendations": {"immediate": ["Manual security review required"], "short_term": [], "long_term": []}
                }
            
            # CRITICAL: Normalize the response to match expected schema (restores Grok-like behavior)
            audit_json = normalize_audit_response(audit_json, tier_for_flags)
            logger.info(f"[AI] Normalized response: {audit_json.get('critical_count', 0)} critical, {audit_json.get('high_count', 0)} high, {audit_json.get('medium_count', 0)} medium, {audit_json.get('low_count', 0)} low issues")
            
            # Calculate severity counts if AI didn't return them
            if "critical_count" not in audit_json or audit_json.get("critical_count") is None:
                issues = audit_json.get("issues", [])
                critical_count = sum(1 for i in issues if i.get("severity", "").lower() == "critical")
                high_count = sum(1 for i in issues if i.get("severity", "").lower() == "high")
                medium_count = sum(1 for i in issues if i.get("severity", "").lower() == "medium")
                low_count = sum(1 for i in issues if i.get("severity", "").lower() == "low")
                
                audit_json["critical_count"] = critical_count
                audit_json["high_count"] = high_count
                audit_json["medium_count"] = medium_count
                audit_json["low_count"] = low_count
                
                logger.info(f"[AI] Calculated severity counts: C={critical_count}, H={high_count}, M={medium_count}, L={low_count}")
            
            # Override code_quality_metrics with our accurate calculation
            if "code_quality_metrics" in audit_json:
                audit_json["code_quality_metrics"]["lines_of_code"] = lines_of_code
                audit_json["code_quality_metrics"]["functions_count"] = functions_count
                audit_json["code_quality_metrics"]["complexity_score"] = complexity_score
            else:
                audit_json["code_quality_metrics"] = {
                    "lines_of_code": lines_of_code,
                    "functions_count": functions_count,
                    "complexity_score": complexity_score
                }
            logger.info(f"[METRICS] Overrode AI metrics with accurate counts: {lines_of_code} LOC")
            
            # Verify Pro+ fields are present
            
            # Add Enterprise/Diamond extras
            if tier_for_flags in ["enterprise", "diamond"] or getattr(user, "has_diamond", False):
                audit_json["fuzzing_results"] = fuzzing_results
                try:
                    certora_result = await asyncio.to_thread(run_certora, temp_path)
                    audit_json["formal_verification"] = certora_result
                except Exception as e:
                    audit_json["formal_verification"] = f"Certora failed: {e}"
            
            report = audit_json
        
        except Exception as e:
            logger.error(f"AI analysis failed for {effective_username}: {e}")
            logger.exception("FULL AI ERROR TRACEBACK:")
            await broadcast_audit_log(effective_username, f"AI analysis failed: {str(e)}")
            
            # Fallback: still show Slither/Mythril results with calculated counts
            fallback_issues = mythril_results + [
                {"type": d.get("name", "Slither finding"), "severity": "Medium", "description": str(d.get("details", "N/A")), "fix": "Manual review required"}
                for d in (slither_findings or [])
            ]
            
            # Calculate severity counts for fallback
            critical_count = sum(1 for i in fallback_issues if i.get("severity", "").lower() == "critical")
            high_count = sum(1 for i in fallback_issues if i.get("severity", "").lower() == "high")
            medium_count = sum(1 for i in fallback_issues if i.get("severity", "").lower() == "medium")
            low_count = sum(1 for i in fallback_issues if i.get("severity", "").lower() == "low")
            
            report.update({
                "risk_score": "Unknown (AI analysis failed)",
                "critical_count": critical_count,
                "high_count": high_count,
                "medium_count": medium_count,
                "low_count": low_count,
                "issues": fallback_issues,
                "predictions": [],
                "recommendations": ["AI analysis unavailable – review static analysis results above"],
                "error": str(e)
            })
        
        # Phase 5: Finalization
        await broadcast_audit_log(effective_username, "Audit complete")
        if job_id:
            await audit_queue.update_phase(job_id, "finalizing", 95)
            await notify_job_subscribers(job_id, {"status": "processing", "phase": "finalizing", "progress": 95})
        
        # Finalization: PDF, overage reporting, history, DB commit, cleanup
        if usage_tracker.feature_flags.get(tier_for_flags, {}).get("reports", False):
            pdf_path = generate_compliance_pdf(report, effective_username, file_size)
        
        try:
            overage_mb = max(0, (file_size - 1024 * 1024) / (1024 * 1024))
            overage_cost = usage_tracker.calculate_diamond_overage(file_size) / 100 if overage_mb > 0 else None
            
            if overage_mb > 0 and user and getattr(user, "stripe_subscription_item_id", None):
                stripe.SubscriptionItem.create_usage_record(
                    user.stripe_subscription_item_id,
                    quantity=int(overage_mb),
                    timestamp=int(time.time()),
                    action="increment",
                )
                logger.info(f"Reported {overage_mb:.2f}MB overage for {effective_username} to Stripe")
        except Exception as e:
            logger.error(f"Failed to report overage: {e}")
        
        try:
            if user:
                history = json.loads(user.audit_history or "[]")
                history.append({"contract": contract_address or "uploaded", "timestamp": datetime.now().isoformat(), "risk_score": report.get("risk_score", "N/A")})
                user.audit_history = json.dumps(history)
                db.commit()
        except Exception as e:
            logger.error(f"Failed to update audit history: {e}")
        
        # Cleanup temp file
        try:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                logger.debug(f"Temp file deleted: {temp_path}")
        except Exception as e:
            logger.error(f"Failed to delete temp file: {e}")
        
        # Persist usage increment
        try:
            usage_tracker.increment(file_size, effective_username, db, commit=True)
        except Exception as e:
            logger.error(f"Failed to increment usage: {e}")
        
        try:
            if user:
                user.last_reset = datetime.now()
                db.commit()
        except Exception as e:
            logger.error(f"Failed to update last_reset: {e}")
        
        if (datetime.now() - audit_start_time).total_seconds() > 6 * 3600:
            logger.warning(f"Audit timeout for {effective_username} — resetting usage")
            usage_tracker.reset_usage(effective_username, db)
        
        # Apply free tier filtering
        if current_tier == "free":
            report = filter_issues_for_free_tier(report, current_tier)
            
         # ============================================================================
        # GAMIFICATION: Update user stats (Phase 0 - Silent tracking)
        # ============================================================================
        if user:  # Only track for authenticated users (not guests)
            try:
                logger.info(f"[GAMIFICATION] Updating stats for user: {user.username}")
                
                # 1. Increment total audits
                user.total_audits += 1
                logger.debug(f"[GAMIFICATION] Total audits now: {user.total_audits}")
                
                # 2. Count issues by severity
                issues = report.get("issues", [])
                issues_added_this_audit = 0
                critical_added = 0
                high_added = 0
                
                for issue in issues:
                    severity = issue.get("severity", "").lower()
                    if severity == "critical":
                        user.total_critical_issues += 1
                        critical_added += 1
                    elif severity == "high":
                        user.total_high_issues += 1
                        high_added += 1
                    
                    user.total_issues_found += 1
                    issues_added_this_audit += 1
                
                logger.debug(f"[GAMIFICATION] Issues added: {issues_added_this_audit} (Critical: {critical_added}, High: {high_added})")
                
                # 3. Update best security score
                try:
                    current_score = float(report.get("risk_score", 0))
                    if not user.best_security_score or current_score > user.best_security_score:
                        old_best = user.best_security_score or 0
                        user.best_security_score = current_score
                        logger.info(f"[GAMIFICATION] New best score: {current_score} (was {old_best})")
                except (ValueError, TypeError) as e:
                    logger.warning(f"[GAMIFICATION] Could not parse risk score: {e}")
                
                # 4. Update streak tracking
                from datetime import date
                today = datetime.now(timezone.utc).date()
                
                if user.last_audit_date:
                    last_date = user.last_audit_date.date()
                    days_diff = (today - last_date).days
                    
                    if days_diff == 1:
                        # Consecutive day - increment streak
                        user.current_streak_days += 1
                        logger.info(f"[GAMIFICATION] Streak continued: {user.current_streak_days} days 🔥")
                    elif days_diff > 1:
                        # Streak broken - reset to 1
                        old_streak = user.current_streak_days
                        user.current_streak_days = 1
                        logger.info(f"[GAMIFICATION] Streak broken (was {old_streak} days), reset to 1")
                    else:
                        # Same day - no change
                        logger.debug(f"[GAMIFICATION] Same day audit, streak unchanged: {user.current_streak_days}")
                else:
                    # First audit ever
                    user.current_streak_days = 1
                    logger.info(f"[GAMIFICATION] First audit! Streak started: 1 day")
                
                user.last_audit_date = datetime.now(timezone.utc)
                
                # 5. Award XP and calculate level
                # XP formula: Base 10 per audit + bonus for finding issues
                xp_awarded = 10  # Base XP
                
                if critical_added > 0:
                    xp_awarded += critical_added * 5  # +5 XP per critical issue
                if high_added > 0:
                    xp_awarded += high_added * 3  # +3 XP per high issue
                if issues_added_this_audit > 10:
                    xp_awarded += 10  # Bonus for comprehensive scan
                
                old_xp = user.xp_points
                user.xp_points += xp_awarded
                logger.info(f"[GAMIFICATION] XP awarded: +{xp_awarded} (total: {user.xp_points})")
                
                # Calculate level (exponential curve)
                old_level = user.level
                if user.xp_points < 100:
                    user.level = 1
                elif user.xp_points < 300:
                    user.level = 2
                elif user.xp_points < 600:
                    user.level = 3
                elif user.xp_points < 1000:
                    user.level = 4
                elif user.xp_points < 1500:
                    user.level = 5
                elif user.xp_points < 2200:
                    user.level = 6
                elif user.xp_points < 3000:
                    user.level = 7
                elif user.xp_points < 4000:
                    user.level = 8
                elif user.xp_points < 5500:
                    user.level = 9
                elif user.xp_points < 7500:
                    user.level = 10
                elif user.xp_points < 10000:
                    user.level = 11
                elif user.xp_points < 15000:
                    user.level = 12
                elif user.xp_points < 20000:
                    user.level = 13
                elif user.xp_points < 30000:
                    user.level = 14
                else:
                    user.level = 15  # Max level
                
                if user.level > old_level:
                    logger.info(f"[GAMIFICATION] 🎉 LEVEL UP! {old_level} → {user.level}")
                
                # 6. Commit all stat updates
                db.commit()
                
                # 7. Log summary
                logger.info(
                    f"[GAMIFICATION] Stats updated for {user.username}: "
                    f"Audits={user.total_audits}, "
                    f"Issues={user.total_issues_found} (Critical={user.total_critical_issues}, High={user.total_high_issues}), "
                    f"Level={user.level}, "
                    f"XP={user.xp_points}, "
                    f"Streak={user.current_streak_days} days, "
                    f"Best Score={user.best_security_score}"
                )
                
            except Exception as e:
                logger.error(f"[GAMIFICATION] Error updating user stats: {e}")
                logger.exception(e)
                # Don't fail the audit if stats tracking fails
                # Just log the error and continue
                try:
                    db.rollback()
                except:
                    pass
        else:
            logger.debug("[GAMIFICATION] Guest user - stats not tracked")
        
        response = {"report": report, "risk_score": str(report.get("risk_score", "N/A")), "overage_cost": overage_cost}
        if pdf_path:
            response["compliance_pdf"] = pdf_path
        
        return response
    
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Unexpected error during audit for {effective_username}: {exc}")
        await broadcast_audit_log(effective_username, "Audit failed: unexpected error")
        report["error"] = f"Audit failed: unexpected error: {exc}"
        return {"report": report, "risk_score": "N/A", "overage_cost": None}
    
    finally:
        # ADMISSION CONTROL: Release slot and process next queued job
        # Skip when called from queue (process_one_queued_job handles slot release)
        if not _from_queue:
            async with active_audit_lock:
                active_audit_count = max(0, active_audit_count - 1)
                logger.info(f"[ADMISSION] Slot released ({active_audit_count}/{MAX_CONCURRENT_AUDITS})")
            
            # Try to process next queued job
            asyncio.create_task(process_one_queued_job())


async def process_one_queued_job():
    """Process one job from queue if capacity available."""
    global active_audit_count
    
    async with active_audit_lock:
        if active_audit_count >= MAX_CONCURRENT_AUDITS:
            return  # Still at capacity
        
        # Check if there's a queued job (priority queue handles ordering)
        async with audit_queue.lock:
            if not audit_queue.queue:
                return  # Queue empty
            
            # Find highest priority QUEUED job
            job = None
            temp_items = []
            while audit_queue.queue:
                priority, counter, candidate = heapq.heappop(audit_queue.queue)
                if candidate.status == AuditStatus.QUEUED:
                    job = candidate
                    # Don't put this one back - we're processing it
                    break
                else:
                    temp_items.append((priority, counter, candidate))
            
            # Put back any items we popped but didn't use
            for item in temp_items:
                heapq.heappush(audit_queue.queue, item)
            
            if not job:
                return  # No queued jobs found
        
        active_audit_count += 1
        logger.info(f"[QUEUE_DRAIN] Processing queued job {job.job_id[:8]}... (tier={job.tier}, {active_audit_count}/{MAX_CONCURRENT_AUDITS})")
    
    try:
        job.status = AuditStatus.PROCESSING
        audit_queue.processing.add(job.job_id)
        
        # Route through full audit_contract for 100% feature parity
        from io import BytesIO
        file_obj = UploadFile(
            filename=job.filename,
            file=BytesIO(job.file_content),
            size=len(job.file_content)
        )
        queue_db = SessionLocal()
        
        try:
            result = await audit_contract(
                file=file_obj,
                contract_address=job.contract_address,
                username=job.username,
                db=queue_db,
                request=None,
                _from_queue=True
            )
        finally:
            queue_db.close()
        
        await audit_queue.complete(job.job_id, result)
        await notify_job_subscribers(job.job_id, {"status": "completed", "result": result})
        
    except Exception as e:
        logger.exception(f"[QUEUE_DRAIN] Job {job.job_id[:8]}... failed: {e}")
        await audit_queue.fail(job.job_id, str(e))
        await notify_job_subscribers(job.job_id, {"status": "failed", "error": str(e)})
    
    finally:
        async with active_audit_lock:
            active_audit_count = max(0, active_audit_count - 1)
        
        # Recursively drain queue
        asyncio.create_task(process_one_queued_job())

# ============================================================================
# GAMIFICATION DEBUG ENDPOINT (Phase 0)
# ============================================================================
@app.get("/debug/my-stats")
async def debug_my_stats(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Debug endpoint to check your gamification stats.
    Only shows data for the authenticated user.
    """
    try:
        user = await get_authenticated_user(request, db)
        if not user:
            raise HTTPException(status_code=401, detail="Not authenticated - please log in")
        
        # Calculate XP needed for next level
        current_xp = user.xp_points
        next_level_xp = 100  # Default
        
        level_thresholds = [100, 300, 600, 1000, 1500, 2200, 3000, 4000, 5500, 7500, 10000, 15000, 20000, 30000]
        for threshold in level_thresholds:
            if current_xp < threshold:
                next_level_xp = threshold
                break
        
        xp_to_next_level = next_level_xp - current_xp
        
        return {
            "username": user.username,
            "email": user.email,
            "tier": user.tier,
            "member_since": user.last_reset.isoformat() if user.last_reset else "Unknown",
            
            # Gamification stats
            "gamification": {
            "total_audits": user.total_audits,
                "total_issues_found": user.total_issues_found,
                "total_critical_issues": user.total_critical_issues,
                "total_high_issues": user.total_high_issues,
                "best_security_score": user.best_security_score,
                "current_streak_days": user.current_streak_days,
                "last_audit_date": user.last_audit_date.isoformat() if user.last_audit_date else None,
                
                # Level & XP
                "level": user.level,
                "xp_points": user.xp_points,
                "xp_to_next_level": xp_to_next_level,
                "next_level": user.level + 1 if user.level < 15 else 15,
                
                # NFT
                "wallet_address": user.wallet_address
            },
            
            # Achievements unlocked (will be calculated in Week 2)
            "achievements_preview": {
                "rookie_auditor": user.total_audits >= 1,
                "bug_hunter": user.total_critical_issues >= 1,
                "security_student": user.total_audits >= 3,
                "diligent_developer": user.total_audits >= 10,
                "audit_machine": user.total_audits >= 50,
                "perfect_code": user.best_security_score and user.best_security_score >= 10.0
            },
            
            "message": "Stats are being tracked silently. Gamification UI coming in Week 2! 🎮"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[DEBUG_STATS] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")

# ============================================================================
# NEW: Admin endpoint for tier migration
# ============================================================================
@app.post("/admin/migrate-tiers")
async def migrate_tiers(request: Request, db: Session = Depends(get_db), admin_key: str = Query(None)):
    """
    Database migration endpoint to rename old tiers to new names.
    ADMIN ONLY - requires admin_key from environment.
    """
    if admin_key != os.getenv("ADMIN_KEY"):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Migrate beginner → starter
        beginner_count = db.query(User).filter(User.tier == "beginner").update({"tier": "starter"})
        
        # Migrate diamond → enterprise (only if has_diamond=True and tier="diamond")
        diamond_count = db.query(User).filter(
            User.tier == "diamond",
            User.has_diamond == True
        ).update({"tier": "enterprise"})
        
        db.commit()
        
        return {
            "success": True,
            "migrated": {
                "beginner_to_starter": beginner_count,
                "diamond_to_enterprise": diamond_count
            }
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Migration failed: {str(e)}")
@app.get("/debug/echidna-env")
async def debug_echidna_env():
    """Check Echidna installation and environment"""
    import subprocess
    
    checks = {
        "timestamp": datetime.now().isoformat(),
        "env": os.getenv("ENV", "unknown"),
        "path": os.environ.get("PATH", "NOT SET"),
    }
    
    # Check binary locations
    paths_to_check = [
        "/opt/render/project/.local/bin/echidna",
        "/usr/local/bin/echidna",
        "/usr/bin/echidna"
    ]
    
    checks["paths"] = {}
    for path in paths_to_check:
        checks["paths"][path] = {
            "exists": os.path.exists(path),
            "is_file": os.path.isfile(path) if os.path.exists(path) else False,
            "is_executable": os.access(path, os.X_OK) if os.path.exists(path) else False,
            "size": os.path.getsize(path) if os.path.exists(path) else 0
        }
    
    # Try 'which echidna'
    try:
        which_result = subprocess.run(["which", "echidna"], capture_output=True, text=True, timeout=5)
        checks["which_echidna"] = {
            "returncode": which_result.returncode,
            "stdout": which_result.stdout.strip(),
            "stderr": which_result.stderr.strip()
        }
    except Exception as e:
        checks["which_echidna"] = {"error": str(e)}
    
    # List /opt/render/project/.local/bin/
    try:
        bin_dir = "/opt/render/project/.local/bin"
        if os.path.exists(bin_dir):
            checks["bin_directory"] = {
                "exists": True,
                "contents": os.listdir(bin_dir)
            }
        else:
            checks["bin_directory"] = {"exists": False}
    except Exception as e:
        checks["bin_directory"] = {"error": str(e)}
    
    # Try running echidna --version
    for path in ["/opt/render/project/.local/bin/echidna", "echidna"]:
        try:
            version_result = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=5)
            checks[f"version_check_{path}"] = {
                "returncode": version_result.returncode,
                "stdout": version_result.stdout.strip(),
                "stderr": version_result.stderr.strip()
            }
            break
        except FileNotFoundError:
            checks[f"version_check_{path}"] = {"error": "FileNotFoundError"}
        except Exception as e:
            checks[f"version_check_{path}"] = {"error": str(e)}
    
    return checks

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
