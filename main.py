# pyright: reportMissingImports=false
# pyright: reportUnknownMemberType=false
# pyright: reportGeneralTypeIssues=false
import logging
import os
from dotenv import load_dotenv
import sys

# Load from script directory (not CWD)
script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path=dotenv_path)

# NOW initialize clients AFTER .env is loaded

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

# Fallback: Grok
grok_key = os.getenv("GROK_API_KEY")
if grok_key and grok_key.strip():
    grok_client = OpenAI(
        api_key=grok_key.strip(),
        base_url="https://api.x.ai/v1"
    )

# Web3
infura_url = f"https://mainnet.infura.io/v3/{os.getenv('INFURA_PROJECT_ID')}"
w3 = Web3(Web3.HTTPProvider(infura_url))

# On-chain Analyzer (lazy initialization - imports after all modules loaded)
onchain_analyzer = None
_onchain_init_error = None  # Store error for later logging

def get_onchain_analyzer():
    """Get or create on-chain analyzer instance."""
    global onchain_analyzer, _onchain_init_error
    if onchain_analyzer is None and _onchain_init_error is None:
        try:
            from onchain_analyzer import OnChainAnalyzer as OCA
            onchain_analyzer = OCA(rpc_url=infura_url)
        except Exception as e:
            _onchain_init_error = str(e)
            # Will be logged once logger is available
    return onchain_analyzer

import platform
import json
import time
import uuid
from datetime import datetime, timedelta, timezone
import secrets
from tempfile import NamedTemporaryFile
from typing import Optional, Callable, Awaitable, Any, cast
from fastapi import FastAPI, File, UploadFile, Request, Query, HTTPException, Depends, Response, Header, WebSocket, Body
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response as StarletteResponse
from starlette.websockets import WebSocketState
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware
from urllib.parse import quote_plus, urlencode, urlparse, parse_qs
from fastapi.middleware.cors import CORSMiddleware
# Note: Web3 already imported at line 19 for early client initialization
import stripe
import re  # For username sanitization
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# On-chain analysis module
from onchain_analyzer import OnChainAnalyzer

# === EARLY LOGGER SETUP (fixes NameError) ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("DeFiGuard")

# Unique worker ID for Redis pub/sub message deduplication (prevents duplicate audit log messages)
WORKER_ID = str(uuid.uuid4())

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

from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text, ForeignKey, Float, LargeBinary
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.orm import sessionmaker, Session, Mapped, mapped_column
from slither.slither import Slither
from slither.exceptions import SlitherError
# Note: OpenAI already imported at line 17 for early client initialization
from tenacity import retry, stop_after_attempt, wait_fixed
import uvicorn
from pydantic import BaseModel, Field, field_validator
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import requests
import httpx  # Async HTTP client for non-blocking requests
import subprocess
import asyncio
import hmac
import base64
import heapq
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Flowable, Table, TableStyle, PageBreak, ListFlowable, ListItem, Preformatted, Image, BaseDocTemplate, Frame, PageTemplate, NextPageTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas as pdf_canvas
from jinja2 import Environment, FileSystemLoader
from compliance_checker import get_compliance_analysis, ComplianceChecker

# ============================================================================
# RATE LIMITING - Prevents abuse and DoS attacks
# ============================================================================
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from collections import defaultdict
import time as time_module

class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""

    def __init__(self):
        self.requests: dict[str, list[float]] = defaultdict(list)
        self.lock = asyncio.Lock()

    async def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """Check if request is allowed under rate limit."""
        async with self.lock:
            now = time_module.time()
            window_start = now - window_seconds

            # Clean old requests outside window
            self.requests[key] = [t for t in self.requests[key] if t > window_start]

            if len(self.requests[key]) >= max_requests:
                return False

            self.requests[key].append(now)
            return True

    async def get_retry_after(self, key: str, window_seconds: int) -> int:
        """Get seconds until next request is allowed."""
        async with self.lock:
            if not self.requests[key]:
                return 0
            oldest = min(self.requests[key])
            return max(0, int(window_seconds - (time_module.time() - oldest)))

# Global rate limiter instance
rate_limiter = RateLimiter()

# Rate limit configuration
RATE_LIMITS = {
    "audit_submit": {"max_requests": 10, "window_seconds": 60},  # 10 audits per minute
    "audit_submit_guest": {"max_requests": 3, "window_seconds": 60},  # 3 for guests
    "api_call": {"max_requests": 100, "window_seconds": 60},  # 100 API calls per minute
    "push": {"max_requests": 5, "window_seconds": 60},  # 5 push notifications per minute
    "overage": {"max_requests": 10, "window_seconds": 60},  # 10 overage checks per minute
    # Security: Rate limits for sensitive endpoints
    "auth": {"max_requests": 10, "window_seconds": 60},  # 10 auth attempts per minute
    "api_key_create": {"max_requests": 5, "window_seconds": 60},  # 5 key creations per minute
    "api_key_modify": {"max_requests": 10, "window_seconds": 60},  # 10 key modifications per minute
    "wallet_connect": {"max_requests": 5, "window_seconds": 60},  # 5 wallet connects per minute
    "email_resend": {"max_requests": 3, "window_seconds": 300},  # 3 email resends per 5 minutes
}

# Queue limits
MAX_QUEUE_SIZE = 1000  # Maximum jobs in queue
MAX_JOBS_PER_USER = 10  # Maximum concurrent jobs per user

# ============================================================================
# WEBSOCKET TOKEN AUTHENTICATION
# ============================================================================

WS_TOKEN_EXPIRY_SECONDS = 3600  # 1 hour

def generate_ws_token(username: str, secret_key: str) -> str:
    """Generate a signed token for WebSocket authentication."""
    timestamp = int(time_module.time())
    message = f"{username}:{timestamp}"
    signature = hmac.new(
        secret_key.encode(),
        message.encode(),
        'sha256'
    ).hexdigest()
    token = base64.urlsafe_b64encode(f"{message}:{signature}".encode()).decode()
    return token

def verify_ws_token(token: str, secret_key: str) -> Optional[str]:
    """Verify a WebSocket token and return the username if valid."""
    try:
        decoded = base64.urlsafe_b64decode(token.encode()).decode()
        parts = decoded.split(':')
        if len(parts) != 3:
            return None
        username, timestamp_str, provided_signature = parts
        timestamp = int(timestamp_str)

        # Check expiry
        if time_module.time() - timestamp > WS_TOKEN_EXPIRY_SECONDS:
            return None

        # Verify signature
        message = f"{username}:{timestamp_str}"
        expected_signature = hmac.new(
            secret_key.encode(),
            message.encode(),
            'sha256'
        ).hexdigest()

        if not hmac.compare_digest(provided_signature, expected_signature):
            return None

        return username
    except Exception:
        return None

# ============================================================================
# PHASE 1: IN-MEMORY AUDIT QUEUE SYSTEM
# ============================================================================

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
    api_key_id: Optional[int] = None  # Pro/Enterprise: Track which API key this audit belongs to
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
                     filename: str, tier: str, contract_address: Optional[str] = None,
                     api_key_id: Optional[int] = None) -> AuditJob:
        """Submit a new audit job to the priority queue with limits enforcement."""
        async with self.lock:
            # Security: Enforce queue depth limit to prevent memory exhaustion
            if len(self.queue) >= MAX_QUEUE_SIZE:
                raise HTTPException(
                    status_code=503,
                    detail=f"Queue is full ({MAX_QUEUE_SIZE} jobs). Please try again later."
                )

            # Security: Enforce per-user job limit
            user_jobs = sum(1 for j in self.jobs.values()
                           if j.username == username and j.status in [AuditStatus.QUEUED, AuditStatus.PROCESSING])
            if user_jobs >= MAX_JOBS_PER_USER:
                raise HTTPException(
                    status_code=429,
                    detail=f"Too many pending audits ({MAX_JOBS_PER_USER} max). Please wait for current audits to complete."
                )

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
                api_key_id=api_key_id,  # Track API key assignment for Pro/Enterprise
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
# For 2GB RAM: Allow 3 concurrent audits before queuing
audit_queue = AuditQueue(max_concurrent=3)

# Track currently executing audits (for admission control)
active_audit_count = 0
active_audit_lock = asyncio.Lock()
MAX_CONCURRENT_AUDITS = 3  # 2GB RAM can handle 3 concurrent sequential audits (~500MB each)

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
import redis.asyncio as aioredis
# Note: requests already imported at line 121
# Note: asyncio already imported at line 123
from jose import jwt, JWTError

# === JINJA2 TEMPLATE SETUP (required for /ui and /auth) ===
templates_dir = "templates"
jinja_env = Environment(
    loader=FileSystemLoader(templates_dir),
    autoescape=True  # Security: Auto-escape HTML to prevent XSS
)

# === FASTAPI APP INSTANCE (MUST COME BEFORE ANY @app ROUTES) ===
app = FastAPI()

# Trust proxy headers from Render (ensures HTTPS URLs are generated correctly)
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts=["*"])

# CRITICAL: Session middleware for Auth0
_secret_key = os.getenv("APP_SECRET_KEY")
_environment = os.getenv("ENVIRONMENT", "development")

if not _secret_key:
    if _environment == "production":
        # SECURITY: In production, APP_SECRET_KEY is mandatory
        logger.critical("FATAL: APP_SECRET_KEY not set in production environment!")
        raise RuntimeError("APP_SECRET_KEY environment variable is required in production")
    else:
        _secret_key = secrets.token_urlsafe(32)
        logger.warning("APP_SECRET_KEY not set; using generated temporary secret (development only)")

app.add_middleware(
    SessionMiddleware,
    secret_key=_secret_key,
    session_cookie="session",
    max_age=4 * 60 * 60,  # 4 hours (reduced from 2 weeks for security)
    same_site="lax",
    https_only=_environment == "production"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# === LEGAL DOCUMENT ROUTES ===
@app.get("/privacy")
async def privacy_page():
    """Serve the privacy policy page."""
    return FileResponse("static/privacy-policy.html", media_type="text/html")

@app.get("/terms")
async def terms_page():
    """Serve the terms of service page."""
    return FileResponse("static/terms-of-service.html", media_type="text/html")

# === CORS CONFIGURATION ===
# Production origins only - localhost origins controlled by ENVIRONMENT
def get_cors_origins() -> list[str]:
    """Get CORS origins based on environment."""
    # Production origins (always allowed)
    origins = [
        "https://defiguard-ai-beta.onrender.com",
    ]

    # Development origins (only in non-production)
    environment = os.getenv("ENVIRONMENT", "development").lower()
    if environment in ("development", "dev", "local", "test"):
        origins.extend([
            "http://127.0.0.1:8000",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://localhost:3000",
        ])
        logger.info("[CORS] Development mode: localhost origins enabled")
    else:
        logger.info("[CORS] Production mode: localhost origins disabled")

    # Allow additional origins from environment variable (comma-separated)
    extra_origins = os.getenv("CORS_EXTRA_ORIGINS", "")
    if extra_origins:
        origins.extend([o.strip() for o in extra_origins.split(",") if o.strip()])

    return origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["Authorization", "Content-Type", "X-CSRFToken", "Accept"],
)

# === SECURITY HEADERS MIDDLEWARE ===
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers including CSP to all responses."""

    # Content Security Policy - restricts resource loading sources
    # Security notes:
    # - 'unsafe-inline' required for inline scripts in templates (WalletConnect, queue monitor)
    #   TODO: Implement nonce-based CSP to remove 'unsafe-inline' entirely
    # - 'unsafe-eval' REMOVED - verified no code uses eval(), new Function(), or string setTimeout
    # - esm.sh added for WalletConnect ES module imports
    CSP_POLICY = "; ".join([
        "default-src 'self'",
        # Script sources - 'unsafe-eval' removed (code verified safe)
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://unpkg.com https://cdnjs.cloudflare.com https://www.googletagmanager.com https://js.stripe.com https://esm.sh",
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net https://cdnjs.cloudflare.com",
        "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com data:",
        "img-src 'self' data: https: blob:",
        "connect-src 'self' wss: ws: https://api.stripe.com https://*.infura.io https://*.walletconnect.com https://*.auth0.com https://esm.sh",
        "frame-src 'self' https://js.stripe.com https://verify.walletconnect.com",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self' https://*.auth0.com https://checkout.stripe.com",
        "upgrade-insecure-requests"
    ])

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Add security headers to all responses
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        # HSTS - Force HTTPS (1 year, include subdomains, preload-ready)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"

        # Add CSP header for HTML responses
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            response.headers["Content-Security-Policy"] = self.CSP_POLICY

        # Add Cache-Control for sensitive API endpoints
        path = request.url.path
        sensitive_paths = ["/tier", "/me", "/api/keys", "/api/regenerate-key", "/api/ws-token"]
        if any(path.startswith(p) for p in sensitive_paths):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
            response.headers["Pragma"] = "no-cache"

        return response

app.add_middleware(SecurityHeadersMiddleware)

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

@app.on_event("startup")
async def startup_redis_pubsub():
    """Initialize Redis pub/sub for cross-worker WebSocket messaging."""
    if await init_redis_pubsub():
        asyncio.create_task(redis_audit_subscriber())
        logger.info("[STARTUP] Redis audit subscriber started")


# Background task to poll pending Certora jobs
async def certora_job_poller():
    """
    Background task that periodically polls pending Certora jobs.
    Updates job status in database when verification completes.

    Performance optimized:
    - Uses asyncio.to_thread() for blocking database/HTTP calls
    - Fetches all job results concurrently using asyncio.gather()
    """
    POLL_INTERVAL = 60  # Check every 60 seconds
    logger.info("[CERTORA] Background job poller started")

    async def fetch_job_result(runner, job_url: str) -> tuple[str, dict]:
        """Fetch job result in thread pool to avoid blocking event loop."""
        try:
            result = await asyncio.to_thread(runner._fetch_job_results, job_url)
            return (job_url, result)
        except Exception as e:
            logger.warning(f"[CERTORA] Error fetching {job_url}: {e}")
            return (job_url, {"status": "running", "error": str(e)})

    def get_pending_jobs():
        """Synchronous database query - run in thread pool."""
        db = SessionLocal()
        try:
            jobs = db.query(CertoraJob).filter(
                CertoraJob.status.in_(["pending", "running"])
            ).all()
            # Detach from session for async use - extract needed data
            return [(j.id, j.job_id, j.job_url, j.status) for j in jobs]
        finally:
            db.close()

    def update_job_status(job_id: int, updates: dict):
        """Synchronous database update - run in thread pool."""
        db = SessionLocal()
        try:
            job = db.query(CertoraJob).filter(CertoraJob.id == job_id).first()
            if job:
                for key, value in updates.items():
                    setattr(job, key, value)
                db.commit()
        finally:
            db.close()

    while True:
        try:
            await asyncio.sleep(POLL_INTERVAL)

            # Get pending jobs (non-blocking via thread pool)
            pending_jobs = await asyncio.to_thread(get_pending_jobs)

            if not pending_jobs:
                continue

            logger.info(f"[CERTORA] Polling {len(pending_jobs)} pending jobs concurrently...")

            from certora import CertoraRunner
            runner = CertoraRunner()

            # Fetch all job results concurrently (major performance improvement)
            fetch_tasks = [
                fetch_job_result(runner, job_url)
                for (_, _, job_url, _) in pending_jobs
            ]
            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            # Map results by job_url
            results_map = {}
            for r in results:
                if isinstance(r, tuple):
                    job_url, result = r
                    results_map[job_url] = result

            # Update database (batch updates via thread pool)
            update_tasks = []
            for (db_id, job_id, job_url, current_status) in pending_jobs:
                result = results_map.get(job_url, {"status": "running"})
                status = result.get("status", "running")

                if status in ["verified", "issues_found"]:
                    updates = {
                        "status": "completed",
                        "completed_at": datetime.now(),
                        "rules_verified": result.get("rules_verified", 0),
                        "rules_violated": result.get("rules_violated", 0),
                        "results_json": json.dumps(result)
                    }
                    update_tasks.append(asyncio.to_thread(update_job_status, db_id, updates))
                    logger.info(f"[CERTORA] Job {job_id} completed: {result.get('rules_verified', 0)} verified, {result.get('rules_violated', 0)} violated")

                elif status == "error":
                    updates = {
                        "status": "error",
                        "results_json": json.dumps(result)
                    }
                    update_tasks.append(asyncio.to_thread(update_job_status, db_id, updates))
                    logger.warning(f"[CERTORA] Job {job_id} failed: {result.get('error', 'Unknown error')}")

                elif current_status != "running":
                    updates = {"status": "running"}
                    update_tasks.append(asyncio.to_thread(update_job_status, db_id, updates))

            # Execute all database updates concurrently
            if update_tasks:
                await asyncio.gather(*update_tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"[CERTORA] Background poller error: {e}")


@app.on_event("startup")
async def startup_certora_poller():
    """Start Certora job poller background task."""
    if os.getenv("CERTORAKEY"):
        asyncio.create_task(certora_job_poller())
        logger.info("[STARTUP] Certora job poller started")
        
@app.get("/debug-files")
async def debug_files(admin_key: str = Header(None, alias="X-Admin-Key")):
    """Debug endpoint - requires admin key in production (via X-Admin-Key header)."""
    expected_key = os.getenv("ADMIN_KEY")
    if not expected_key or not admin_key or not secrets.compare_digest(admin_key, expected_key):
        raise HTTPException(status_code=403, detail="Admin access required")
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
    try:
        # Rate limiting for auth attempts
        client_ip = request.client.host if request.client else "unknown"
        rate_key = f"auth:{client_ip}"
        limit = RATE_LIMITS["auth"]
        if not await rate_limiter.is_allowed(rate_key, limit["max_requests"], limit["window_seconds"]):
            retry_after = await rate_limiter.get_retry_after(rate_key, limit["window_seconds"])
            logger.warning(f"[LOGIN] Rate limited: {client_ip}")
            return RedirectResponse(url=f"/auth?error=rate_limited&retry_after={retry_after}")

        if not AUTH0_DOMAIN:
            logger.error("[LOGIN] AUTH0_DOMAIN not configured")
            return RedirectResponse(url="/auth?error=auth_not_configured")

        redirect_uri = request.url_for("callback")
        logger.info(f"[LOGIN] Initiating Auth0 redirect, callback={redirect_uri}, screen_hint={screen_hint}")

        response = await oauth.auth0.authorize_redirect(request, redirect_uri)

        if screen_hint:
            location = response.headers["Location"]
            parsed = urlparse(location)
            params = parse_qs(parsed.query)
            params["screen_hint"] = [screen_hint]
            new_query = urlencode(params, doseq=True)
            new_location = parsed._replace(query=new_query).geturl()
            response.headers["Location"] = new_location

        logger.info(f"[LOGIN] Redirecting to Auth0: {response.headers.get('Location', 'NO_LOCATION')[:100]}...")
        return response

    except Exception as e:
        logger.error(f"[LOGIN] Failed to redirect to Auth0: {e}")
        return RedirectResponse(url=f"/auth?error=login_failed")

@app.get("/logout")
async def logout(request: Request):
    # Log the logout for debugging
    username = request.session.get("username", "unknown")
    logger.info(f"[LOGOUT] User {username} logging out")
    
    # Clear ALL session data (including cached auth_provider)
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
    response.delete_cookie("session")
    response.delete_cookie("username")  # Also clear username cookie
    logger.info(f"[LOGOUT] Session cleared, redirecting to Auth0 logout")
    return response

# === DATABASE SETUP AND DEPENDENCIES (MUST BE BEFORE ANY ROUTES) ===
from pathlib import Path

# Feature Flag: Enable tier persistence via Stripe sync
# Set ENABLE_TIER_PERSISTENCE=true in production to restore user tiers from Stripe
# Leave unset during testing to allow tier resets on each deploy
ENABLE_TIER_PERSISTENCE = os.getenv("ENABLE_TIER_PERSISTENCE", "").lower() == "true"
if ENABLE_TIER_PERSISTENCE:
    logger.info("[CONFIG] ✓ Tier persistence ENABLED - user tiers will sync with Stripe")
else:
    logger.info("[CONFIG] Tier persistence DISABLED - tiers reset on deploy (testing mode)")

# Database Configuration - Supports PostgreSQL (recommended) or SQLite (dev only)
# Priority: DATABASE_URL env var > Render SQLite > Local SQLite
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    # PostgreSQL from Render or external provider
    # Render uses 'postgres://' but SQLAlchemy needs 'postgresql://'
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    logger.info(f"[DB] Using PostgreSQL database")
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
elif os.getenv("RENDER"):
    # Fallback: SQLite on Render (WARNING: Data lost on redeploy!)
    DATABASE_URL = "sqlite:////tmp/users.db"
    logger.warning("[DB] ⚠️ Using ephemeral SQLite on Render - data will be lost on redeploy!")
    logger.warning("[DB] ⚠️ Set DATABASE_URL env var to use PostgreSQL for persistence")
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    # Local development
    Path("./instance").mkdir(exist_ok=True)
    DATABASE_URL = "sqlite:///./instance/users.db"
    logger.info("[DB] Using local SQLite database for development")
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
    tier: Mapped[str] = mapped_column(String, default="free", index=True)
    has_diamond: Mapped[bool] = mapped_column(Boolean, default=False)
    last_reset: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    api_key: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    auth0_sub: Mapped[Optional[str]] = mapped_column(String, unique=True, nullable=True, index=True)
    stripe_customer_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
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

    # Account tracking
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
   
# Note: ForeignKey and relationship already imported from sqlalchemy at lines 110-111

class APIKey(Base):
    """Multi-key API key management for Pro/Enterprise users"""
    __tablename__ = "api_keys"
    __table_args__ = {'extend_existing': True}
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    key: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    label: Mapped[str] = mapped_column(String, default="API Key", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    
    # Relationship
    user = relationship("User", back_populates="api_keys")
class PendingAudit(Base):
    __tablename__ = "pending_audits"
    __table_args__ = {'extend_existing': True}
    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String, index=True)
    temp_path: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="pending", index=True)
    results: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)


class CertoraJob(Base):
    """
    Stores Certora verification jobs for caching and async polling.

    This allows:
    1. Users to run Certora separately from their main audit
    2. Caching results so same contract doesn't need re-verification
    3. Background polling to update job status
    4. Users to see their Certora job history
    """
    __tablename__ = "certora_jobs"
    __table_args__ = {'extend_existing': True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey('users.id'), nullable=False, index=True)

    # Contract identification - used for cache lookup
    contract_hash: Mapped[str] = mapped_column(String(64), index=True)  # SHA256 of contract content
    contract_name: Mapped[str] = mapped_column(String(255))  # Original filename

    # Certora job details
    job_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)  # Certora's job ID
    job_url: Mapped[str] = mapped_column(String(512))  # Full URL to Certora dashboard

    # Status tracking
    status: Mapped[str] = mapped_column(String(50), default="pending", index=True)
    # Status values: pending, running, completed, error, timeout

    # Results (stored as JSON)
    results_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rules_verified: Mapped[int] = mapped_column(Integer, default=0)
    rules_violated: Mapped[int] = mapped_column(Integer, default=0)
    rules_timeout: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # For user notifications
    user_notified: Mapped[bool] = mapped_column(Boolean, default=False)

    # Track if job was created with full Slither context
    # Jobs from Settings don't have Slither context, jobs from full audit do
    has_slither_context: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationship
    user = relationship("User")


class AuditResult(Base):
    """
    Persistent storage for complete audit results.

    Allows users to:
    1. Start an audit and leave the page
    2. Retrieve results later using their unique audit key
    3. Receive email notification when audit completes

    Each audit gets a unique access key (dga_xxx) that can be used
    to retrieve results without being logged in.
    """
    __tablename__ = "audit_results"
    __table_args__ = {'extend_existing': True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey('users.id'), nullable=True, index=True)

    # Unique access key for this audit (e.g., dga_abc123xyz...)
    audit_key: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)

    # Contract information
    contract_name: Mapped[str] = mapped_column(String(255), default="contract.sol")
    contract_hash: Mapped[str] = mapped_column(String(64), index=True)  # SHA256
    contract_address: Mapped[Optional[str]] = mapped_column(String(42), nullable=True)  # On-chain address

    # Persisted file content for recovery after server restart
    # Stored as binary to preserve exact file content for retry capability
    file_content: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)

    # Queue/Processing status
    status: Mapped[str] = mapped_column(String(50), default="queued", index=True)
    # Status values: queued, processing, completed, failed
    queue_position: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    current_phase: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)

    # Individual tool results (JSON strings)
    slither_results: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    mythril_results: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    echidna_results: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    certora_results: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    onchain_results: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # AI analysis and final report
    ai_analysis: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    full_report: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Complete normalized response JSON

    # PDF report path
    pdf_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    # Summary metrics
    risk_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    issues_count: Mapped[int] = mapped_column(Integer, default=0)
    critical_count: Mapped[int] = mapped_column(Integer, default=0)
    high_count: Mapped[int] = mapped_column(Integer, default=0)
    medium_count: Mapped[int] = mapped_column(Integer, default=0)
    low_count: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Email notification
    notification_email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    email_sent: Mapped[bool] = mapped_column(Boolean, default=False)

    # User tier at time of audit (for feature gating in results)
    user_tier: Mapped[str] = mapped_column(String(50), default="free")

    # Error information if failed
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Link to in-memory job_id for real-time updates
    job_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)

    # API Key assignment (Pro/Enterprise feature)
    # Allows users to organize audits by project/client via named API keys
    api_key_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey('api_keys.id', ondelete='SET NULL'),  # Keep audit if key is revoked
        nullable=True,
        index=True
    )

    # Relationships
    user = relationship("User")
    api_key = relationship("APIKey", backref="audits")


def generate_audit_key() -> str:
    """Generate a unique audit access key."""
    import secrets
    return f"dga_{secrets.token_urlsafe(32)}"


# ============================================================================
# EMAIL NOTIFICATION SYSTEM
# ============================================================================
# Environment variables:
#   SMTP_HOST - SMTP server hostname (default: smtp.gmail.com)
#   SMTP_PORT - SMTP server port (default: 587)
#   SMTP_USER - SMTP username/email
#   SMTP_PASSWORD - SMTP password or app-specific password
#   SMTP_FROM_EMAIL - From email address (default: uses SMTP_USER)
#   SMTP_FROM_NAME - From name (default: DeFiGuard AI)

def get_smtp_config() -> dict:
    """Get SMTP configuration from environment."""
    return {
        "host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
        "port": int(os.getenv("SMTP_PORT", "587")),
        "user": os.getenv("SMTP_USER", ""),
        "password": os.getenv("SMTP_PASSWORD", ""),
        "from_email": os.getenv("SMTP_FROM_EMAIL", os.getenv("SMTP_USER", "")),
        "from_name": os.getenv("SMTP_FROM_NAME", "DeFiGuard AI"),
    }


def is_email_configured() -> bool:
    """Check if email sending is properly configured."""
    config = get_smtp_config()
    return bool(config["user"] and config["password"])


# Email validation pattern (RFC 5322 simplified)
EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

# Ethereum address validation pattern
ETH_ADDRESS_PATTERN = re.compile(r'^0x[a-fA-F0-9]{40}$')


def validate_email(email: str) -> bool:
    """Validate email format."""
    if not email or not isinstance(email, str):
        return False
    if len(email) > 254:  # RFC 5321 limit
        return False
    return bool(EMAIL_PATTERN.match(email))


def validate_eth_address(address: str) -> bool:
    """Validate Ethereum address format."""
    if not address or not isinstance(address, str):
        return False
    return bool(ETH_ADDRESS_PATTERN.match(address))


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and special characters."""
    if not filename or not isinstance(filename, str):
        return "contract.sol"
    # Remove path components
    filename = os.path.basename(filename)
    # Remove special characters except . - _
    filename = re.sub(r'[^\w\s\-.]', '', filename)
    # Remove leading/trailing whitespace and dots
    filename = filename.strip().strip('.')
    # Limit length
    if len(filename) > 240:
        filename = filename[:240]
    return filename or "contract.sol"


def validate_solidity_file(filename: str, content: bytes) -> tuple[bool, str]:
    """
    Validate that the uploaded file is a valid Solidity contract.
    Returns (is_valid, error_message).
    Security: Prevents upload of malicious files disguised as .sol files.
    """
    # Check extension
    if not filename or not filename.lower().endswith('.sol'):
        return False, "Only .sol (Solidity) files are accepted"

    # Try to decode as UTF-8
    try:
        text_content = content.decode('utf-8')
    except UnicodeDecodeError:
        return False, "File must be valid UTF-8 text"

    # Check for Solidity markers (at least one must be present)
    solidity_markers = [
        'pragma solidity',
        'contract ',
        'interface ',
        'library ',
        'abstract contract',
        'function ',
        'event ',
        'mapping(',
        'address ',
        'uint256',
        'uint ',
        'bytes32',
        'SPDX-License-Identifier'
    ]

    content_lower = text_content.lower()
    if not any(marker.lower() in content_lower for marker in solidity_markers):
        return False, "File does not appear to be a Solidity contract"

    # Check for obviously malicious content
    dangerous_patterns = [
        '<?php',
        '<script>',
        '#!/bin/',
        '#!/usr/bin/',
        'eval(',
        'exec(',
    ]
    if any(pattern.lower() in content_lower for pattern in dangerous_patterns):
        return False, "File contains potentially dangerous content"

    return True, ""


def send_audit_completion_email(
    to_email: str,
    audit_key: str,
    contract_name: str,
    risk_score: float,
    issues_count: int,
    critical_count: int,
    high_count: int,
    status: str = "completed"
) -> bool:
    """
    Send email notification when audit completes.

    Returns True if email sent successfully, False otherwise.
    """
    if not is_email_configured():
        logger.warning("Email not configured - skipping audit notification")
        return False

    # Validate email format
    if not validate_email(to_email):
        logger.error(f"Invalid email format: {to_email[:50]}...")
        return False

    # Validate audit key format
    if not audit_key or not audit_key.startswith("dga_"):
        logger.error("Invalid audit key format")
        return False

    config = get_smtp_config()
    base_url = os.getenv("BASE_URL", "https://defiguard.onrender.com")

    try:
        # HTML escape user-controlled content to prevent injection
        import html
        safe_contract_name = html.escape(contract_name or "contract.sol")
        safe_audit_key = html.escape(audit_key)

        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🔒 DeFiGuard Audit Complete: {safe_contract_name}"
        msg["From"] = f"{config['from_name']} <{config['from_email']}>"
        msg["To"] = to_email

        # Determine status emoji and message
        if status == "completed":
            status_emoji = "✅"
            status_text = "Your smart contract audit has completed successfully!"
        else:
            status_emoji = "❌"
            status_text = "Your smart contract audit encountered an issue."

        # Risk level indicator
        if risk_score is not None:
            if risk_score >= 80:
                risk_color = "#22c55e"  # green
                risk_label = "Low Risk"
            elif risk_score >= 60:
                risk_color = "#eab308"  # yellow
                risk_label = "Medium Risk"
            elif risk_score >= 40:
                risk_color = "#f97316"  # orange
                risk_label = "High Risk"
            else:
                risk_color = "#ef4444"  # red
                risk_label = "Critical Risk"
        else:
            risk_color = "#6b7280"
            risk_label = "Unknown"
            risk_score = 0

        # Plain text version (no HTML escaping needed)
        text_content = f"""
DeFiGuard Smart Contract Audit Complete
========================================

{status_text}

Contract: {safe_contract_name}
Security Score: {risk_score:.0f}/100 ({risk_label})
Issues Found: {issues_count} total ({critical_count} critical, {high_count} high)

View your full audit results:
{base_url}/audit/retrieve/{safe_audit_key}

Your Audit Key: {safe_audit_key}
(Save this key to access your results anytime)

---
DeFiGuard AI - Advanced Smart Contract Security
"""

        # HTML version (using escaped values)
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #0a0a0a; color: #ffffff; padding: 20px; margin: 0;">
    <div style="max-width: 600px; margin: 0 auto; background-color: #1a1a2e; border-radius: 12px; overflow: hidden; border: 1px solid #333;">
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #6366f1, #8b5cf6); padding: 30px; text-align: center;">
            <h1 style="margin: 0; font-size: 24px; color: white;">🔒 DeFiGuard AI</h1>
            <p style="margin: 10px 0 0; opacity: 0.9; color: white;">Smart Contract Security Audit</p>
        </div>

        <!-- Content -->
        <div style="padding: 30px;">
            <h2 style="margin: 0 0 20px; color: #fff;">{status_emoji} Audit Complete</h2>
            <p style="color: #a0a0a0; margin-bottom: 25px;">{status_text}</p>

            <!-- Contract Info -->
            <div style="background-color: #252542; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
                <p style="margin: 0 0 10px; color: #a0a0a0; font-size: 14px;">Contract</p>
                <p style="margin: 0; font-size: 18px; font-weight: bold; color: #fff;">{safe_contract_name}</p>
            </div>

            <!-- Security Score -->
            <div style="background-color: #252542; border-radius: 8px; padding: 20px; margin-bottom: 20px; text-align: center;">
                <p style="margin: 0 0 10px; color: #a0a0a0; font-size: 14px;">Security Score</p>
                <p style="margin: 0; font-size: 48px; font-weight: bold; color: {risk_color};">{risk_score:.0f}</p>
                <p style="margin: 5px 0 0; color: {risk_color}; font-weight: bold;">{risk_label}</p>
            </div>

            <!-- Issues Summary -->
            <div style="display: flex; gap: 10px; margin-bottom: 25px;">
                <div style="flex: 1; background-color: #252542; border-radius: 8px; padding: 15px; text-align: center;">
                    <p style="margin: 0; font-size: 24px; font-weight: bold; color: #ef4444;">{critical_count}</p>
                    <p style="margin: 5px 0 0; font-size: 12px; color: #a0a0a0;">Critical</p>
                </div>
                <div style="flex: 1; background-color: #252542; border-radius: 8px; padding: 15px; text-align: center;">
                    <p style="margin: 0; font-size: 24px; font-weight: bold; color: #f97316;">{high_count}</p>
                    <p style="margin: 5px 0 0; font-size: 12px; color: #a0a0a0;">High</p>
                </div>
                <div style="flex: 1; background-color: #252542; border-radius: 8px; padding: 15px; text-align: center;">
                    <p style="margin: 0; font-size: 24px; font-weight: bold; color: #fff;">{issues_count}</p>
                    <p style="margin: 5px 0 0; font-size: 12px; color: #a0a0a0;">Total</p>
                </div>
            </div>

            <!-- CTA Button -->
            <a href="{base_url}/audit/retrieve/{safe_audit_key}" style="display: block; background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; text-decoration: none; padding: 15px 30px; border-radius: 8px; font-weight: bold; text-align: center; margin-bottom: 20px;">
                View Full Audit Report
            </a>

            <!-- Audit Key -->
            <div style="background-color: #252542; border-radius: 8px; padding: 15px; text-align: center;">
                <p style="margin: 0 0 5px; color: #a0a0a0; font-size: 12px;">Your Audit Key (save this)</p>
                <code style="color: #8b5cf6; font-size: 14px; word-break: break-all;">{safe_audit_key}</code>
            </div>
        </div>

        <!-- Footer -->
        <div style="background-color: #0f0f1a; padding: 20px; text-align: center; border-top: 1px solid #333;">
            <p style="margin: 0; color: #666; font-size: 12px;">
                DeFiGuard AI - Advanced Smart Contract Security<br>
                <a href="{base_url}" style="color: #6366f1;">defiguard.onrender.com</a>
            </p>
        </div>
    </div>
</body>
</html>
"""

        msg.attach(MIMEText(text_content, "plain"))
        msg.attach(MIMEText(html_content, "html"))

        # Send email
        with smtplib.SMTP(config["host"], config["port"]) as server:
            server.starttls()
            server.login(config["user"], config["password"])
            server.sendmail(config["from_email"], to_email, msg.as_string())

        logger.info(f"Audit completion email sent to {to_email} for audit {audit_key}")
        return True

    except Exception as e:
        logger.error(f"Failed to send audit email to {to_email}: {e}")
        return False


async def send_audit_email_async(
    to_email: str,
    audit_key: str,
    contract_name: str,
    risk_score: float,
    issues_count: int,
    critical_count: int,
    high_count: int,
    status: str = "completed"
) -> bool:
    """Async wrapper for sending audit completion emails."""
    return await asyncio.to_thread(
        send_audit_completion_email,
        to_email, audit_key, contract_name, risk_score,
        issues_count, critical_count, high_count, status
    )


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
        
        # Check if provider is already cached in session (avoids JWT decode spam)
        cached_provider = request.session.get("auth_provider")
        if cached_provider:
            auth0_user['provider'] = cached_provider
        else:
            # JWKS decode for provider - only done ONCE per session, then cached
            try:
                token = request.session.get("id_token")
                if token:
                    jwks_url = f"https://{os.getenv('AUTH0_DOMAIN')}/.well-known/jwks.json"
                    # Use async httpx instead of blocking requests.get()
                    async with httpx.AsyncClient() as client:
                        jwks_response = await client.get(jwks_url, timeout=10.0)
                        jwks = jwks_response.json()
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
                        # Cache provider in session to avoid future JWT decodes
                        request.session["auth_provider"] = provider
                    else:
                        # Fallback: extract provider from sub (e.g., "google-oauth2|12345")
                        provider = auth0_sub.split('|')[0] if '|' in auth0_sub else 'auth0'
                        auth0_user['provider'] = provider
                        request.session["auth_provider"] = provider
                else:
                    # No token, extract from sub
                    provider = auth0_sub.split('|')[0] if '|' in auth0_sub else 'auth0'
                    auth0_user['provider'] = provider
                    request.session["auth_provider"] = provider
            except JWTError as e:
                # Expected: JWT expired - this is normal, extract provider from sub instead
                # Log at DEBUG to avoid log spam (tokens expire every ~1 hour)
                logger.debug(f"JWT expired (expected): {e}")
                provider = auth0_sub.split('|')[0] if '|' in auth0_sub else 'auth0'
                auth0_user['provider'] = provider
                request.session["auth_provider"] = provider
            except Exception as e:
                # Unexpected error - log at WARNING
                logger.warning(f"JWT decode failed (unexpected): {e}")
                provider = auth0_sub.split('|')[0] if '|' in auth0_sub else 'auth0'
                auth0_user['provider'] = provider
                request.session["auth_provider"] = provider
        
        user = db.query(User).filter(User.auth0_sub == auth0_sub).first()
        if user:
            return user

        # Check if user exists with same email (different auth provider)
        email = auth0_user.get("email", "")
        if email:
            existing_user = db.query(User).filter(User.email == email).first()
            if existing_user:
                # Link this auth0_sub to existing account
                logger.info(f"[AUTH] Linking new auth0_sub {auth0_sub} to existing user {existing_user.username}")
                existing_user.auth0_sub = auth0_sub
                db.commit()
                return existing_user

        # Auto-create user on first Auth0 login
        base_username = auth0_user.get("nickname") or auth0_user.get("email", "auth0_user").split("@")[0]
        username = base_username

        # Handle username collision by appending number
        counter = 1
        while db.query(User).filter(User.username == username).first():
            username = f"{base_username}_{counter}"
            counter += 1
            if counter > 100:  # Safety limit
                username = f"{base_username}_{auth0_sub[-8:]}"
                break

        new_user = User(
            username=username,
            email=email,
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
        "INFURA_PROJECT_ID",
        "STRIPE_API_KEY",
        "STRIPE_WEBHOOK_SECRET",
        "AUTH0_DOMAIN",
        "AUTH0_CLIENT_ID",
        "AUTH0_CLIENT_SECRET",
        "APP_SECRET_KEY",
        "REDIS_URL",
        "DATABASE_URL"
    ]

    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing critical environment variables: {', '.join(missing_vars)}")
        raise RuntimeError(f"Missing critical environment variables: {', '.join(missing_vars)}")

    # Require at least one AI API key (ANTHROPIC preferred, GROK as fallback)
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("GROK_API_KEY"):
        logger.error("No AI API key configured - requires ANTHROPIC_API_KEY or GROK_API_KEY")
        raise RuntimeError("No AI API key configured - requires ANTHROPIC_API_KEY or GROK_API_KEY")
    
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

    # Recover pending jobs from previous server run (CRITICAL for data persistence)
    # This must run BEFORE starting the queue processor to avoid race conditions
    try:
        await recover_pending_jobs()
        logger.info("[STARTUP] Job recovery completed")
    except Exception as e:
        logger.error(f"[STARTUP] Job recovery failed: {e}")
        # Continue startup even if recovery fails - better to start than crash

    # Start background queue processor
    processor_task = asyncio.create_task(process_audit_queue())
    logger.info("[QUEUE] Background audit queue processor started")

    # Start periodic cleanup task
    cleanup_task = asyncio.create_task(periodic_queue_cleanup())
    logger.info("[QUEUE] Periodic cleanup task started")

    # Start stale job detector (prevents orphaned processing jobs)
    timeout_task = asyncio.create_task(detect_stale_processing_jobs())
    logger.info("[QUEUE] Stale job detector started (30 min timeout)")

    yield  # App running

    # Cleanup on shutdown
    processor_task.cancel()
    cleanup_task.cancel()
    timeout_task.cancel()
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

        # Check if email is verified (required for email/password sign-ups)
        email = userinfo.get("email", "")
        email_verified = userinfo.get("email_verified", True)  # Social logins are pre-verified

        if email and not email_verified:
            # Email not verified yet - show friendly message
            logger.info(f"[CALLBACK] Email not verified for {email}, redirecting to verification page")
            return RedirectResponse(url="/auth?verify_email=true")

        if not email:
            logger.warning("[CALLBACK] No email in userinfo, cannot create account")
            return RedirectResponse(url="/auth?error=no_email")

        request.session["userinfo"] = userinfo
        request.session["id_token"] = token.get("id_token")

        db = next(get_db())
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
            # Initialize defaults
            stripe_customer_id = None
            stripe_tier = "free"
            stripe_subscription_id = None

            # Only restore tier from Stripe if persistence is enabled
            if ENABLE_TIER_PERSISTENCE:
                try:
                    customers = stripe.Customer.list(email=email, limit=1)
                    if customers.data:
                        stripe_customer_id = customers.data[0].id
                        logger.info(f"[CALLBACK] Found existing Stripe customer for {email}: {stripe_customer_id}")

                        # Check for active subscriptions
                        subs = stripe.Subscription.list(customer=stripe_customer_id, status='active', limit=1)
                        if subs.data:
                            active_sub = subs.data[0]
                            stripe_subscription_id = active_sub.id
                            stripe_tier = active_sub.metadata.get('tier')

                            if not stripe_tier:
                                # Fallback: determine tier from price ID
                                price_id = active_sub['items']['data'][0]['price']['id'] if active_sub.get('items', {}).get('data') else None
                                price_to_tier = {
                                    os.getenv("STRIPE_PRICE_STARTER"): "starter",
                                    os.getenv("STRIPE_PRICE_PRO"): "pro",
                                    os.getenv("STRIPE_PRICE_ENTERPRISE"): "enterprise",
                                    os.getenv("STRIPE_PRICE_BEGINNER"): "starter",
                                    os.getenv("STRIPE_PRICE_DIAMOND"): "diamond",
                                }
                                stripe_tier = price_to_tier.get(price_id, "free")

                            logger.info(f"[CALLBACK] Restored tier from Stripe for {email}: {stripe_tier}")
                except stripe.error.StripeError as e:
                    logger.warning(f"[CALLBACK] Stripe lookup failed for {email}: {e}")
                except Exception as e:
                    logger.error(f"[CALLBACK] Unexpected error looking up Stripe customer: {e}")

            user = User(
                username=username,
                email=email,
                auth0_sub=sub,
                tier=stripe_tier,
                stripe_customer_id=stripe_customer_id,
                stripe_subscription_id=stripe_subscription_id,
                audit_history="[]",
                last_reset=datetime.now(timezone.utc),
                has_diamond=False
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info(f"New user created: {username} (tier: {stripe_tier})")
        else:
            if not user.auth0_sub:
                user.auth0_sub = sub
                db.commit()

        # Verify tier with Stripe (only if persistence enabled)
        if ENABLE_TIER_PERSISTENCE and user.stripe_customer_id:
            verified_tier = verify_tier_with_stripe(user, db)
            logger.info(f"[CALLBACK] Stripe tier verification for {user.username}: {verified_tier}")

        # Set session
        request.session["user_id"] = user.id
        request.session["username"] = user.username
        request.session["csrf_token"] = secrets.token_urlsafe(32)
        
        # Provider from userinfo - extract and cache immediately
        provider = 'unknown'
        if userinfo:
            identities = userinfo.get('identities', [])
            if identities and len(identities) > 0:
                provider = identities[0].get('provider', 'unknown')
            elif sub and '|' in sub:
                # Fallback: extract from sub (e.g., "google-oauth2|12345")
                provider = sub.split('|')[0]
        
        # Cache provider in session to avoid JWT decode spam on subsequent requests
        request.session["auth_provider"] = provider
        request.session["user"] = {"sub": sub, "email": email, "provider": provider}
        logger.info(f"[CALLBACK] Cached auth_provider={provider} for {user.username}")
        
        response = RedirectResponse(url="/ui")
        response.set_cookie("username", str(user.username), httponly=True, secure=True, samesite="lax", max_age=2592000)
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

        # Periodically verify tier with Stripe (only if persistence enabled)
        if ENABLE_TIER_PERSISTENCE and user.stripe_customer_id:
            verify_tier_with_stripe(user, db)

        provider = getattr(user, 'provider', 'unknown')
        if provider == 'unknown' and 'userinfo' in request.session:
            ui = request.session['userinfo']
            provider = ui.get('identities', [{}])[0].get('provider', 'unknown') if ui.get('identities') else 'unknown'

        # Get member_since date (created_at or fallback to last_reset)
        member_since = None
        if hasattr(user, 'created_at') and user.created_at:
            member_since = user.created_at.isoformat()
        elif user.last_reset:
            member_since = user.last_reset.isoformat()

        return {
            "sub": user.auth0_sub,
            "email": user.email,
            "username": user.username,
            "provider": user.auth0_sub.split("|")[0] if user.auth0_sub and "|" in user.auth0_sub else "auth0",
            "logged_in": True,
            "tier": user.tier,
            "member_since": member_since
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
        
        # Generate new API key (cryptographically secure)
        new_api_key = secrets.token_urlsafe(32)
        
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

        # Build key data with audit counts
        keys_data = []
        for key in keys:
            # Count audits assigned to this key
            audit_count = db.query(AuditResult).filter(
                AuditResult.api_key_id == key.id
            ).count()

            keys_data.append({
                "id": key.id,
                "key_preview": f"{key.key[:12]}...{key.key[-4:]}" if key.key and len(key.key) > 16 else "****",
                "label": key.label,
                "created_at": key.created_at.isoformat(),
                "last_used_at": key.last_used_at.isoformat() if key.last_used_at else None,
                "is_active": key.is_active,
                "audit_count": audit_count  # Number of audits assigned to this key
            })
        
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

        # Rate limiting for API key creation
        client_ip = request.client.host if request.client else "unknown"
        rate_key = f"api_key_create:{client_ip}"
        limit = RATE_LIMITS["api_key_create"]
        if not await rate_limiter.is_allowed(rate_key, limit["max_requests"], limit["window_seconds"]):
            retry_after = await rate_limiter.get_retry_after(rate_key, limit["window_seconds"])
            raise HTTPException(
                status_code=429,
                detail=f"Too many API key creation requests. Try again in {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)}
            )

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
        
        # Generate new API key (cryptographically secure)
        new_key = secrets.token_urlsafe(32)
        
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
# WALLET CONNECTION ENDPOINTS
# ============================================================================

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "")

@app.post("/api/wallet/connect")
async def connect_wallet(
    request: Request,
    body: dict = Body(...),
    db: Session = Depends(get_db)
) -> dict[str, Any]:
    """
    Connect and verify wallet ownership via signature.
    Saves wallet address to user profile after verification.
    """
    try:
        await verify_csrf_token(request)

        # Rate limiting for wallet connection attempts
        client_ip = request.client.host if request.client else "unknown"
        rate_key = f"wallet_connect:{client_ip}"
        limit = RATE_LIMITS["wallet_connect"]
        if not await rate_limiter.is_allowed(rate_key, limit["max_requests"], limit["window_seconds"]):
            retry_after = await rate_limiter.get_retry_after(rate_key, limit["window_seconds"])
            raise HTTPException(
                status_code=429,
                detail=f"Too many wallet connection attempts. Try again in {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)}
            )

        user = await get_authenticated_user(request, db)
        if not user:
            raise HTTPException(status_code=401, detail="Not authenticated")

        address = body.get("address", "").strip()
        message = body.get("message", "")
        signature = body.get("signature", "")

        if not address or not message or not signature:
            raise HTTPException(status_code=400, detail="Missing address, message, or signature")

        # Validate address format
        if not w3.is_address(address):
            raise HTTPException(status_code=400, detail="Invalid Ethereum address")

        # Convert to checksum address
        checksum_address = w3.to_checksum_address(address)

        # Verify signature using eth_account
        from eth_account.messages import encode_defunct
        from eth_account import Account

        try:
            message_hash = encode_defunct(text=message)
            recovered_address = Account.recover_message(message_hash, signature=signature)

            if recovered_address.lower() != checksum_address.lower():
                raise HTTPException(status_code=400, detail="Signature verification failed")

        except Exception as e:
            logger.warning(f"[WALLET] Signature verification error: {e}")
            raise HTTPException(status_code=400, detail="Invalid signature")

        # Save wallet address to user profile
        user.wallet_address = checksum_address
        db.commit()

        logger.info(f"[WALLET] User {user.username} connected wallet: {checksum_address}")

        return {
            "success": True,
            "address": checksum_address,
            "message": "Wallet connected successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[WALLET] Connect error: {e}")
        raise HTTPException(status_code=500, detail="Failed to connect wallet")


@app.get("/api/wallet/contracts")
async def get_wallet_contracts(
    address: str,
    request: Request,
    db: Session = Depends(get_db)
) -> dict[str, Any]:
    """
    Fetch contracts deployed by a wallet address from Etherscan.
    Returns list of verified contracts with their names.
    """
    try:
        user = await get_authenticated_user(request, db)
        if not user:
            raise HTTPException(status_code=401, detail="Not authenticated")

        if not w3.is_address(address):
            raise HTTPException(status_code=400, detail="Invalid Ethereum address")

        checksum_address = w3.to_checksum_address(address)

        if not ETHERSCAN_API_KEY:
            logger.warning("[WALLET] ETHERSCAN_API_KEY not configured")
            return {"contracts": [], "message": "Etherscan API not configured"}

        # Fetch transactions from Etherscan to find contract creations
        async with httpx.AsyncClient() as client:
            # Get normal transactions (contract creations have 'to' = empty)
            response = await client.get(
                "https://api.etherscan.io/api",
                params={
                    "module": "account",
                    "action": "txlist",
                    "address": checksum_address,
                    "startblock": 0,
                    "endblock": 99999999,
                    "sort": "desc",
                    "apikey": ETHERSCAN_API_KEY
                },
                timeout=30.0
            )

            data = response.json()

            if data.get("status") != "1":
                logger.warning(f"[WALLET] Etherscan API error: {data.get('message')}")
                return {"contracts": [], "message": data.get("message", "API error")}

            transactions = data.get("result", [])

            # Find contract creation transactions (to address is empty, contractAddress is set)
            contract_creations = []
            seen_addresses = set()

            for tx in transactions:
                if tx.get("to") == "" and tx.get("contractAddress"):
                    contract_addr = tx.get("contractAddress")
                    if contract_addr and contract_addr not in seen_addresses:
                        seen_addresses.add(contract_addr)
                        contract_creations.append({
                            "address": w3.to_checksum_address(contract_addr),
                            "txHash": tx.get("hash"),
                            "blockNumber": tx.get("blockNumber"),
                            "timestamp": tx.get("timeStamp")
                        })

            # Fetch contract names for verified contracts
            contracts_with_names = []
            for contract in contract_creations[:20]:  # Limit to 20 contracts
                try:
                    # Get contract ABI/source to check if verified and get name
                    source_response = await client.get(
                        "https://api.etherscan.io/api",
                        params={
                            "module": "contract",
                            "action": "getsourcecode",
                            "address": contract["address"],
                            "apikey": ETHERSCAN_API_KEY
                        },
                        timeout=10.0
                    )

                    source_data = source_response.json()
                    if source_data.get("status") == "1" and source_data.get("result"):
                        result = source_data["result"][0]
                        contract_name = result.get("ContractName", "")
                        if contract_name:  # Only include verified contracts
                            contracts_with_names.append({
                                "address": contract["address"],
                                "name": contract_name,
                                "txHash": contract["txHash"],
                                "verified": True
                            })
                except Exception as e:
                    logger.debug(f"[WALLET] Failed to get contract name for {contract['address']}: {e}")
                    continue

            logger.info(f"[WALLET] Found {len(contracts_with_names)} verified contracts for {checksum_address}")

            return {
                "contracts": contracts_with_names,
                "total_found": len(contract_creations),
                "verified_count": len(contracts_with_names)
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[WALLET] Get contracts error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch contracts")


@app.get("/api/wallet/contract-source")
async def get_contract_source(
    address: str,
    request: Request,
    db: Session = Depends(get_db)
) -> dict[str, Any]:
    """
    Fetch verified source code for a contract from Etherscan.
    """
    try:
        user = await get_authenticated_user(request, db)
        if not user:
            raise HTTPException(status_code=401, detail="Not authenticated")

        if not w3.is_address(address):
            raise HTTPException(status_code=400, detail="Invalid contract address")

        checksum_address = w3.to_checksum_address(address)

        if not ETHERSCAN_API_KEY:
            raise HTTPException(status_code=503, detail="Etherscan API not configured")

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.etherscan.io/api",
                params={
                    "module": "contract",
                    "action": "getsourcecode",
                    "address": checksum_address,
                    "apikey": ETHERSCAN_API_KEY
                },
                timeout=30.0
            )

            data = response.json()

            if data.get("status") != "1":
                raise HTTPException(status_code=404, detail="Contract not found or not verified")

            result = data.get("result", [{}])[0]
            source_code = result.get("SourceCode", "")
            contract_name = result.get("ContractName", "")

            if not source_code:
                raise HTTPException(status_code=404, detail="No verified source code found")

            # Handle JSON-formatted source (multi-file contracts)
            if source_code.startswith("{{") or source_code.startswith("{"):
                try:
                    # Remove outer braces if double-wrapped
                    if source_code.startswith("{{"):
                        source_code = source_code[1:-1]

                    source_json = json.loads(source_code)

                    # Combine all source files
                    if "sources" in source_json:
                        combined_source = ""
                        for file_path, file_data in source_json["sources"].items():
                            content = file_data.get("content", "")
                            combined_source += f"// File: {file_path}\n{content}\n\n"
                        source_code = combined_source
                except json.JSONDecodeError:
                    pass  # Keep original source code

            logger.info(f"[WALLET] Fetched source for {contract_name} ({checksum_address})")

            return {
                "source_code": source_code,
                "contract_name": contract_name,
                "address": checksum_address,
                "compiler_version": result.get("CompilerVersion", ""),
                "optimization_used": result.get("OptimizationUsed", ""),
                "runs": result.get("Runs", "")
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[WALLET] Get source error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch source code")


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

    token = request.headers.get("X-CSRFToken") or ""
    expected = request.session.get("csrf_token") or ""

    # Ensure we always have a token to compare against (for new sessions)
    if not expected:
        request.session["csrf_token"] = secrets.token_urlsafe(32)
        expected = request.session["csrf_token"]

    # Security: Use constant-time comparison to prevent timing attacks
    # Always compare even if token is empty to prevent timing differences
    is_valid = secrets.compare_digest(token, expected) if token else False

    if not is_valid:
        logger.error("CSRF validation failed")  # Security: Don't log tokens
        raise HTTPException(status_code=403, detail="CSRF token invalid")

    logger.debug("CSRF valid")

# Sync client init (STABLE)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def initialize_client() -> tuple[Optional[OpenAI], Web3]:
    logger.info("Initializing clients...")
    global client, w3

    # Grok client – safe with fallback
    grok_key = os.getenv("GROK_API_KEY")
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
    for idx, pred in enumerate(predictions):
        if isinstance(pred, dict):
            normalized_pred = {
                "id": pred.get("id") or f"PRED-{idx + 1:03d}",
                "title": pred.get("title") or pred.get("scenario") or pred.get("name") or "Attack Scenario",
                "severity": _normalize_severity(pred.get("severity", "High")),
                "probability": pred.get("probability") or pred.get("likelihood") or "Medium",
                "attack_vector": pred.get("attack_vector") or pred.get("description") or pred.get("scenario") or "Attack vector not specified",
                "preconditions": pred.get("preconditions") or pred.get("prerequisites") or "Standard deployment conditions",
                "financial_impact": pred.get("financial_impact") or pred.get("impact") or pred.get("estimated_loss") or "Impact not quantified",
                "affected_functions": pred.get("affected_functions") or pred.get("functions") or [],
                "time_to_exploit": pred.get("time_to_exploit") or pred.get("exploitation_time") or "Unknown",
                "detection_difficulty": pred.get("detection_difficulty") or pred.get("detectability") or "Medium",
                "mitigation": pred.get("mitigation") or pred.get("fix") or pred.get("prevention") or "See recommendations",
                "real_world_example": pred.get("real_world_example") or pred.get("reference") or None
            }
            normalized_predictions.append(normalized_pred)
        elif isinstance(pred, str):
            # Convert legacy string predictions to enhanced format
            normalized_predictions.append({
                "id": f"PRED-{idx + 1:03d}",
                "title": pred[:50] + "..." if len(pred) > 50 else pred,
                "severity": "Medium",
                "probability": "Medium",
                "attack_vector": pred,
                "preconditions": "Standard deployment conditions",
                "financial_impact": "Impact assessment required",
                "affected_functions": [],
                "time_to_exploit": "Unknown",
                "detection_difficulty": "Medium",
                "mitigation": "See recommendations section",
                "real_world_example": None
            })
    
    # Generate default predictions if none provided but issues exist
    if not normalized_predictions and normalized_issues:
        if audit_json["critical_count"] > 0:
            normalized_predictions.append({
                "id": "PRED-001",
                "title": "Critical Vulnerability Exploitation",
                "severity": "Critical",
                "probability": "Very High",
                "attack_vector": f"This contract contains {audit_json['critical_count']} critical vulnerabilities that could be exploited within hours of deployment. Attackers actively scan for newly deployed contracts with known vulnerability patterns. Once identified, exploitation typically follows a predictable pattern: (1) Attacker identifies vulnerable function, (2) Prepares exploit transaction, (3) Executes attack potentially using flashloans for capital, (4) Drains vulnerable funds, (5) Launders proceeds through mixers.",
                "preconditions": "Contract deployed to mainnet with accessible funds",
                "financial_impact": "Potential total loss of contract funds - 100% of TVL at risk",
                "affected_functions": [issue.get("function_name", "Unknown") for issue in normalized_issues if issue.get("severity") == "Critical"][:3],
                "time_to_exploit": "< 1 hour after deployment",
                "detection_difficulty": "Hard - attacks often complete before detection",
                "mitigation": "Address all critical issues before deployment; implement monitoring",
                "real_world_example": "Euler Finance (Mar 2023) - $197M lost to vulnerability exploitation"
            })
        if audit_json["high_count"] > 0:
            normalized_predictions.append({
                "id": "PRED-002",
                "title": "High-Severity Issue Exploitation",
                "severity": "High",
                "probability": "High",
                "attack_vector": f"The {audit_json['high_count']} high-severity vulnerabilities present significant risk for sophisticated attackers. These issues typically require more complex exploitation but offer substantial rewards. Attack pattern: (1) Attacker analyzes contract for profitable exploit paths, (2) Develops custom attack contract, (3) Tests on fork, (4) Executes in favorable market conditions, (5) Extracts maximum value.",
                "preconditions": "Contract deployed with sufficient TVL to justify attack costs",
                "financial_impact": "Estimated 30-70% of TVL at risk depending on vulnerability type",
                "affected_functions": [issue.get("function_name", "Unknown") for issue in normalized_issues if issue.get("severity") == "High"][:3],
                "time_to_exploit": "1-24 hours for sophisticated attackers",
                "detection_difficulty": "Medium - may be detected during execution",
                "mitigation": "Fix high-severity issues; implement circuit breakers",
                "real_world_example": "Cream Finance (Oct 2021) - $130M flashloan attack"
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
    """Send audit log message via Redis pub/sub AND local WebSocket (reliable delivery)."""
    global redis_pubsub_client

    delivered_locally = False

    # ALWAYS try local delivery first (immediate for same-worker connections)
    ws = active_audit_websockets.get(username)
    if ws and ws.application_state == WebSocketState.CONNECTED:
        try:
            await ws.send_json({"type": "audit_log", "message": message})
            logger.info(f"[AUDIT_LOG] ✅ Local send to '{username}': {message}")
            delivered_locally = True
        except Exception as e:
            logger.error(f"[AUDIT_LOG] ❌ Local send failed for '{username}': {e}")
            active_audit_websockets.pop(username, None)

    # ALSO use Redis pub/sub for cross-worker messaging (if available)
    if redis_pubsub_client:
        try:
            await redis_pubsub_client.publish("audit_log_broadcast", json.dumps({
                "username": username,
                "message": message,
                "source_worker": WORKER_ID  # For deduplication on same worker
            }))
            logger.debug(f"[AUDIT_LOG] 📤 Published via Redis for '{username}': {message}")
        except Exception as e:
            logger.warning(f"[AUDIT_LOG] Redis publish failed for '{username}': {e}")

    if not delivered_locally and not redis_pubsub_client:
        logger.warning(f"[AUDIT_LOG] ⚠️ No local WebSocket and no Redis for '{username}' - message may be lost")

    # Yield to event loop to allow message delivery
    await asyncio.sleep(0)

# Stripe setup
STRIPE_API_KEY = os.getenv("STRIPE_API_KEY")
stripe.api_key = STRIPE_API_KEY
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")


def verify_tier_with_stripe(user, db: Session) -> str:
    """
    Verify user's tier with Stripe subscription status.
    This ensures tier data persists even if local database is reset.

    Returns the verified tier (updates user if changed).
    """
    if not user or not user.stripe_customer_id:
        return user.tier if user else "free"

    try:
        # Get active subscriptions for this customer
        subscriptions = stripe.Subscription.list(
            customer=user.stripe_customer_id,
            status='active',
            limit=1
        )

        if not subscriptions.data:
            # No active subscription - check for canceled/past_due
            all_subs = stripe.Subscription.list(
                customer=user.stripe_customer_id,
                limit=5
            )

            # If they had subscriptions but none active, they've churned
            if all_subs.data and user.tier != "free":
                logger.info(f"[STRIPE_VERIFY] User {user.username} has no active subscription, downgrading to free")
                user.tier = "free"
                db.commit()
            return user.tier

        # Get tier from subscription metadata or price ID
        active_sub = subscriptions.data[0]
        stripe_tier = active_sub.metadata.get('tier')

        if not stripe_tier:
            # Fallback: determine tier from price ID
            price_id = active_sub['items']['data'][0]['price']['id'] if active_sub.get('items', {}).get('data') else None

            price_to_tier = {
                os.getenv("STRIPE_PRICE_STARTER"): "starter",
                os.getenv("STRIPE_PRICE_PRO"): "pro",
                os.getenv("STRIPE_PRICE_ENTERPRISE"): "enterprise",
                os.getenv("STRIPE_PRICE_BEGINNER"): "starter",  # Legacy
                os.getenv("STRIPE_PRICE_DIAMOND"): "diamond",
            }
            stripe_tier = price_to_tier.get(price_id, "free")

        # Update user if tier doesn't match
        if user.tier != stripe_tier:
            logger.info(f"[STRIPE_VERIFY] Updating {user.username} tier: {user.tier} -> {stripe_tier}")
            user.tier = stripe_tier
            user.stripe_subscription_id = active_sub.id
            db.commit()

        return stripe_tier

    except stripe.error.StripeError as e:
        logger.warning(f"[STRIPE_VERIFY] Stripe API error for {user.username}: {e}")
        return user.tier  # Keep existing tier on API error
    except Exception as e:
        logger.error(f"[STRIPE_VERIFY] Unexpected error for {user.username}: {e}")
        return user.tier


# Base URL fallback for queue-originated requests (when request object is None)
# This should match your current Render deployment
APP_BASE_URL = os.getenv("APP_BASE_URL", "https://defiguard-ai-beta.onrender.com")

# ============================================================================
# NEW PRICING STRUCTURE (December 7, 2025)
# ============================================================================
# Env fallback: Parse prices.txt if env missing
import csv

# ============================================================================
# STRIPE PRICE IDS (Required environment variables - no hardcoded fallbacks)
# ============================================================================
STRIPE_PRICE_STARTER = os.getenv("STRIPE_PRICE_STARTER")
STRIPE_PRICE_PRO = os.getenv("STRIPE_PRICE_PRO")
STRIPE_PRICE_ENTERPRISE = os.getenv("STRIPE_PRICE_ENTERPRISE")

# Legacy tier support (optional - only needed if supporting old subscriptions)
STRIPE_PRICE_BEGINNER = os.getenv("STRIPE_PRICE_BEGINNER")  # Maps to starter
STRIPE_PRICE_DIAMOND = os.getenv("STRIPE_PRICE_DIAMOND")    # Diamond add-on
STRIPE_METERED_PRICE_DIAMOND = os.getenv("STRIPE_METERED_PRICE_DIAMOND")  # Metered billing

# Log warnings for missing required price IDs
_missing_prices = []
if not STRIPE_PRICE_STARTER:
    _missing_prices.append("STRIPE_PRICE_STARTER")
if not STRIPE_PRICE_PRO:
    _missing_prices.append("STRIPE_PRICE_PRO")
if not STRIPE_PRICE_ENTERPRISE:
    _missing_prices.append("STRIPE_PRICE_ENTERPRISE")

if _missing_prices:
    logger.warning(f"[STRIPE] Missing price IDs (checkout will fail): {', '.join(_missing_prices)}")

# ============================================================================
# TIER LIMITS AND MAPPING
# ============================================================================
FREE_LIMIT = 1       # 1 audit/month - prove value, drive upgrade
STARTER_LIMIT = 25   # Developer tier: 25 audits/month
PRO_LIMIT = 9999     # Team tier: Unlimited
ENTERPRISE_LIMIT = 9999  # Enterprise: Unlimited

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
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "severity": {"type": "string", "enum": ["Critical", "High", "Medium", "Low"]},
                    "probability": {"type": "string", "enum": ["Very High", "High", "Medium", "Low"]},
                    "attack_vector": {"type": "string"},
                    "preconditions": {"type": "string"},
                    "financial_impact": {"type": "string"},
                    "affected_functions": {"type": "array", "items": {"type": "string"}},
                    "time_to_exploit": {"type": "string"},
                    "detection_difficulty": {"type": "string", "enum": ["Easy", "Medium", "Hard"]},
                    "mitigation": {"type": "string"},
                    "real_world_example": {"type": ["string", "null"]}
                },
                "required": ["title", "severity", "attack_vector", "financial_impact"]
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
        },
        "certora_results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rule": {"type": "string"},
                    "status": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["rule", "status"]
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
FORMAL VERIFICATION: {certora_results}
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

PHASE 2.5: FORMAL VERIFICATION INTEGRATION (if available)
═══════════════════════════════════════════════════════════════════
Use the FORMAL VERIFICATION section above to enhance your analysis:

1. VERIFIED PROPERTIES: These have mathematical proofs of correctness.
   - You can have HIGH CONFIDENCE in these areas
   - Still check for edge cases not covered by the verification

2. VIOLATED PROPERTIES: These are PROVEN BUGS - treat as CRITICAL.
   - Formal verification found actual issues
   - Report these as HIGH or CRITICAL severity
   - These are not false positives - they are mathematically proven

3. UNVERIFIED PROPERTIES: These could NOT be proven safe.
   - The verification engine timed out, had errors, or is pending
   - YOU MUST give these areas EXTRA manual scrutiny
   - Assume these may contain vulnerabilities until proven otherwise
   - Check for: reentrancy, overflow, access control, state corruption

4. NO VERIFICATION: If formal verification was not run:
   - Proceed with standard thorough analysis
   - Consider recommending formal verification for critical contracts
═══════════════════════════════════════════════════════════════════

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
SEVERITY CLASSIFICATION CRITERIA (FOLLOW STRICTLY - NO EXCEPTIONS):
═══════════════════════════════════════════════════════════════════

CRITICAL (Score: 25 points each) - MUST meet ALL criteria:
- Direct loss of user funds possible (theft, permanent lock)
- Exploitable without special permissions/conditions
- Examples: Reentrancy with external calls before state updates,
  unprotected selfdestruct, arbitrary external call injection,
  uninitialized proxy implementation, signature replay attacks

HIGH (Score: 15 points each) - MUST meet criteria:
- Potential fund loss under specific conditions OR
- Protocol functionality can be permanently broken OR
- Admin/privileged role abuse possible
- Examples: Missing access control on sensitive functions,
  oracle manipulation without TWAP, flash loan attack vectors,
  unchecked return values on transfers, front-running with MEV

MEDIUM (Score: 8 points each) - MUST meet criteria:
- Temporary disruption to protocol OR
- Minor fund loss (gas griefing, dust amounts) OR
- Requires unlikely conditions to exploit
- Examples: DoS via gas limits, timestamp manipulation,
  missing event emissions, unsafe ERC20 assumptions,
  integer truncation (non-critical paths)

LOW (Score: 3 points each) - MUST meet criteria:
- Best practice violations with no direct security impact
- Gas inefficiencies
- Code quality/style issues
- Examples: Unused variables, missing zero-address checks (non-critical),
  suboptimal gas usage, missing NatSpec comments, magic numbers

═══════════════════════════════════════════════════════════════════
RISK SCORE CALCULATION (MANDATORY FORMULA - USE EXACTLY):
═══════════════════════════════════════════════════════════════════

risk_score = min(100, (critical_count × 25) + (high_count × 15) + (medium_count × 8) + (low_count × 3))

Examples:
- 2 Critical + 1 High + 3 Medium + 5 Low = (2×25)+(1×15)+(3×8)+(5×3) = 50+15+24+15 = 104 → 100
- 0 Critical + 2 High + 4 Medium + 8 Low = (0×25)+(2×15)+(4×8)+(8×3) = 0+30+32+24 = 86
- 0 Critical + 0 High + 2 Medium + 4 Low = (0×25)+(0×15)+(2×8)+(4×3) = 0+0+16+12 = 28

IMPORTANT: The risk_score MUST be calculated using this formula. Do NOT subjectively adjust.
Count ALL issues found, then apply the formula. Be consistent.

═══════════════════════════════════════════════════════════════════
ANALYSIS REQUIREMENTS BY TIER:
═══════════════════════════════════════════════════════════════════

FREE TIER:
- Risk score calculated using the MANDATORY FORMULA above (no subjective adjustment)
- Top 3 CRITICAL or HIGH severity issues ONLY
- For each issue: type, severity, 2-3 sentence description
- NO fix recommendations (upgrade required)
- Calculate exact counts: critical_count, high_count, medium_count, low_count
- Set upgrade_prompt: "⚠️ [X] critical and [Y] high-severity issues detected. [Z] total issues found. Upgrade to Developer ($99/mo) to get AI-powered fix recommendations and see all vulnerabilities."
- Executive summary under 100 words focusing on most critical risk

DEVELOPER PLAN ($99/mo):
- Full executive summary (2-3 paragraphs with regulatory context)
- ALL issues with: type, severity, detailed description (4-5 sentences), basic fix
- Fix recommendations must be SPECIFIC and ACTIONABLE:
  ✅ GOOD: "Use OpenZeppelin's ReentrancyGuard by importing '@openzeppelin/contracts/security/ReentrancyGuard.sol' and adding 'nonReentrant' modifier to withdraw()"
  ❌ BAD: "Add reentrancy protection"
- PREDICTIONS (3-5 attack scenarios) - EACH prediction MUST include:
  * id: STRING - Unique identifier (PRED-001, PRED-002, etc.)
  * title: STRING - Short attack name (e.g., "Parameter Manipulation Attack")
  * severity: STRING - "Critical", "High", "Medium", or "Low"
  * probability: STRING - "Very High", "High", "Medium", or "Low" based on attack complexity
  * attack_vector: STRING - DETAILED step-by-step attack narrative (MINIMUM 100 words):
    - Step 1: How attacker discovers the vulnerability
    - Step 2: What tools/transactions they use
    - Step 3: How they execute the exploit
    - Step 4: How they extract value
    - Step 5: How they cover their tracks (if applicable)
  * preconditions: STRING - What conditions must exist for attack to succeed
  * financial_impact: STRING - Quantified dollar range with rationale (e.g., "$50K-500K based on average TVL")
  * affected_functions: ARRAY - List of function names targeted
  * time_to_exploit: STRING - How long to execute (e.g., "< 1 hour", "1-24 hours", "Days")
  * detection_difficulty: STRING - "Easy", "Medium", or "Hard" with explanation
  * mitigation: STRING - Specific fix to prevent this attack
  * real_world_example: STRING (optional) - Reference to similar historical exploit with losses

- Recommendations in THREE categories:
  * immediate: Actions before deployment (fix critical bugs)
  * short_term: Actions for next 7-30 days (audits, testing)
  * long_term: Strategic improvements (formal verification, monitoring)
- Basic MiCA/SEC compliance analysis (high-level only)
- NO line numbers, code snippets, or PoC exploits (Pro+ features)

TEAM PLAN ($349/mo):
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

ENTERPRISE PLAN ($1,499/mo):
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
    """Thread-safe usage tracker with asyncio.Lock for concurrent request safety."""

    # Class-level lock for thread safety across all async operations
    _lock = asyncio.Lock()

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
            "free": 250 * 1024,              # 250KB - tight limit to encourage upgrades
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

            # Reset usage counter monthly for free tier only
            # Paid tiers are managed via Stripe webhooks, not time-based downgrade
            if user.tier == "free" and elapsed_days >= 30:
                self.count = 0
                user.last_reset = current_time
                logger.info(f"Reset usage for {username} on free tier after 30 days")
            # NOTE: Paid tier downgrades are handled via Stripe webhook (invoice.payment_failed)
            # We do NOT auto-downgrade subscribers based on elapsed time - this punishes users
            # who take breaks but maintain active subscriptions
            
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
                    detail=f"File size exceeds tier limit. Upgrade to Team ($349/mo) or Enterprise ($1,499/mo) for larger files."
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
            logger.error(f"Reset usage error for {username or 'anonymous'}: {e}")
            raise HTTPException(status_code=500, detail="Failed to reset usage")

    async def async_increment(self, file_size: int, username: Optional[str] = None, db: Optional[Session] = None, commit: bool = True):
        """Thread-safe async wrapper for increment method using asyncio.Lock."""
        async with self._lock:
            return self.increment(file_size, username, db, commit)

    async def async_reset_usage(self, username: Optional[str] = None, db: Optional[Session] = None):
        """Thread-safe async wrapper for reset_usage method using asyncio.Lock."""
        async with self._lock:
            return self.reset_usage(username, db)

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
            timeout=180  # 3 minutes - generous for complex contracts
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
        logger.warning("[ECHIDNA] Timed out after 180 seconds")
        return [{"vulnerability": "Echidna timeout", "description": "Fuzzing exceeded 3 minute limit - contract may be complex. Partial results may be available."}]
    
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
        
        # Mythril command with execution limits for thorough but bounded analysis
        result = subprocess.run(
            [
                "myth", "analyze", temp_path,
                "-o", "json",
                "--execution-timeout", "180",     # 3 minutes for symbolic execution
                "--max-depth", "30",              # Increased search depth for better coverage
                "--solver-timeout", "30000"       # 30s solver timeout
            ],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes total - generous for complex contracts
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
        logger.warning("[MYTHRIL] Timed out after 300 seconds")
        return [{"vulnerability": "Mythril timeout", "description": "Analysis exceeded 5 minute limit - contract may be highly complex. Try running a focused audit."}]
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
    
    # Count severity of hidden issues for psychological impact
    hidden_issues = sorted_issues[3:]
    hidden_critical = sum(1 for i in hidden_issues if i.get("severity", "").upper() == "CRITICAL")
    hidden_high = sum(1 for i in hidden_issues if i.get("severity", "").upper() == "HIGH")

    # Add upgrade message with loss aversion and urgency triggers
    filtered_report = report.copy()
    filtered_report["issues"] = top_3

    # Craft message based on hidden severity (psychological: loss aversion)
    if hidden_critical > 0:
        severity_warning = f"⚠️ {hidden_critical} CRITICAL"
        if hidden_high > 0:
            severity_warning += f" and {hidden_high} HIGH severity"
        severity_warning += f" issue{'s' if hidden_critical + hidden_high > 1 else ''} not shown"
        urgency = " These require immediate attention."
    elif hidden_high > 0:
        severity_warning = f"🔴 {hidden_high} HIGH severity issue{'s' if hidden_high > 1 else ''} hidden"
        urgency = " Don't leave your contract exposed."
    else:
        severity_warning = f"🔒 {hidden_count} additional issue{'s' if hidden_count > 1 else ''} found"
        urgency = ""

    filtered_report["upgrade_message"] = (
        f"{severity_warning}.{urgency} "
        f"Upgrade to Developer ($99/mo) to reveal all {len(issues)} vulnerabilities with AI-powered fix recommendations."
    )

    # Dynamic upgrade prompt based on findings (creates specificity + urgency)
    filtered_report["upgrade_prompt"] = (
        f"We detected {len(issues)} total vulnerabilities in your contract. "
        f"The free tier shows only the top 3 most critical. "
        f"{'⚠️ ' + str(hidden_critical) + ' CRITICAL issues remain hidden that could result in fund loss. ' if hidden_critical > 0 else ''}"
        f"Upgrade to Developer ($99/mo) to see the complete analysis with actionable fix recommendations."
    )

    filtered_report["watermark"] = "Limited Preview - Upgrade for Complete Analysis"

    # Strategic placeholder for hidden recommendations (creates desire)
    filtered_report["recommendations"] = [
        "🔒 AI-powered fix recommendations available with Developer plan ($99/mo)",
        "🔒 Code-level remediation steps hidden - Upgrade to unlock",
        "🔒 Exploit prevention strategies require paid tier"
    ]
    
    return filtered_report

# =============================================================================
# PDF GENERATION - Comprehensive Tier-Based Reports
# =============================================================================

# Logo path for PDF branding (used for all tiers except white-label Enterprise)
PDF_LOGO_PATH = os.path.join("static", "images", "defiguard-logo.png")
PDF_LOGO_SMALL_PATH = os.path.join("static", "images", "DeFiguard Logo 127 x127.png")

# Professional color palette for PDFs
PDF_COLORS = {
    "primary": colors.Color(0.0, 0.82, 0.70),      # Teal (#00d1b2)
    "primary_dark": colors.Color(0.0, 0.65, 0.55), # Dark teal
    "secondary": colors.Color(0.15, 0.15, 0.25),   # Dark navy
    "accent": colors.Color(0.55, 0.35, 0.95),      # Purple accent
    "success": colors.Color(0.2, 0.7, 0.35),       # Green
    "warning": colors.Color(0.95, 0.65, 0.15),     # Orange
    "error": colors.Color(0.85, 0.2, 0.2),         # Red
    "text_primary": colors.Color(0.15, 0.15, 0.2),
    "text_secondary": colors.Color(0.4, 0.4, 0.45),
    "border": colors.Color(0.85, 0.85, 0.88),
    "bg_light": colors.Color(0.97, 0.98, 0.99),
}


class ProfessionalPDFTemplate:
    """
    Custom PDF page template with professional header/footer branding.
    Handles tier-specific branding (logo watermark for non-Enterprise).
    """

    def __init__(self, tier: str, username: str, report_id: str):
        self.tier = tier
        self.username = username
        self.report_id = report_id
        self.is_whitelabel = tier in ["enterprise", "diamond"]

    def _draw_header(self, canvas: pdf_canvas.Canvas, doc):
        """Draw professional header on each page."""
        canvas.saveState()
        page_width = letter[0]

        # Header line
        canvas.setStrokeColor(PDF_COLORS["primary"])
        canvas.setLineWidth(2)
        canvas.line(0.5*inch, letter[1] - 0.5*inch, page_width - 0.5*inch, letter[1] - 0.5*inch)

        # Report ID on left (small, professional)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(PDF_COLORS["text_secondary"])
        canvas.drawString(0.75*inch, letter[1] - 0.4*inch, f"Report ID: {self.report_id}")

        # Date on right
        canvas.drawRightString(page_width - 0.75*inch, letter[1] - 0.4*inch,
                               datetime.now().strftime("%Y-%m-%d"))

        canvas.restoreState()

    def _draw_footer(self, canvas: pdf_canvas.Canvas, doc):
        """Draw professional footer with logo watermark (non-Enterprise) and page numbers."""
        canvas.saveState()
        page_width = letter[0]

        # Footer line
        canvas.setStrokeColor(PDF_COLORS["border"])
        canvas.setLineWidth(0.5)
        canvas.line(0.75*inch, 0.6*inch, page_width - 0.75*inch, 0.6*inch)

        # Page number (centered)
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(PDF_COLORS["text_secondary"])
        page_num = canvas.getPageNumber()
        canvas.drawCentredString(page_width / 2, 0.4*inch, f"Page {page_num}")

        if not self.is_whitelabel:
            # Logo watermark for non-Enterprise (subtle, professional)
            try:
                if os.path.exists(PDF_LOGO_SMALL_PATH):
                    # Draw small logo in footer
                    canvas.drawImage(PDF_LOGO_SMALL_PATH, 0.75*inch, 0.25*inch,
                                    width=0.4*inch, height=0.4*inch,
                                    preserveAspectRatio=True, mask='auto')
            except Exception:
                pass  # Gracefully handle missing logo

            # Branding text
            canvas.setFont("Helvetica-Oblique", 7)
            canvas.setFillColor(PDF_COLORS["text_secondary"])
            if self.tier == "pro":
                canvas.drawString(1.25*inch, 0.35*inch, "Powered by DeFiGuard AI")
            else:
                canvas.drawString(1.25*inch, 0.35*inch, "Generated by DeFiGuard AI")
        else:
            # Enterprise: Clean, white-label footer
            canvas.setFont("Helvetica-Oblique", 7)
            canvas.setFillColor(PDF_COLORS["text_secondary"])
            canvas.drawString(0.75*inch, 0.35*inch, "Enterprise Security Report • Confidential")

        # Disclaimer on right
        canvas.setFont("Helvetica", 6)
        canvas.drawRightString(page_width - 0.75*inch, 0.35*inch,
                              "For informational purposes only")

        canvas.restoreState()

    def on_page(self, canvas: pdf_canvas.Canvas, doc):
        """Called on each page to draw header/footer."""
        self._draw_header(canvas, doc)
        self._draw_footer(canvas, doc)

    def on_first_page(self, canvas: pdf_canvas.Canvas, doc):
        """Called on cover page - no header/footer needed."""
        pass


def _build_cover_page(story: list, styles: dict, username: str, file_size: int,
                      tier: str, report: dict, report_id: str) -> None:
    """
    Build professional cover page with logo (non-Enterprise) and key metrics.
    """
    is_whitelabel = tier in ["enterprise", "diamond"]

    story.append(Spacer(1, 1.5*inch))

    # Logo for non-Enterprise tiers - preserve aspect ratio
    if not is_whitelabel and os.path.exists(PDF_LOGO_PATH):
        try:
            from PIL import Image as PILImage
            # Get original image dimensions to preserve aspect ratio
            with PILImage.open(PDF_LOGO_PATH) as img:
                orig_width, orig_height = img.size
                aspect_ratio = orig_width / orig_height

            # Set max dimensions while preserving aspect ratio
            max_width = 2.0 * inch
            max_height = 2.0 * inch

            if aspect_ratio >= 1:  # Landscape or square
                logo_width = min(max_width, max_height * aspect_ratio)
                logo_height = logo_width / aspect_ratio
            else:  # Portrait
                logo_height = min(max_height, max_width / aspect_ratio)
                logo_width = logo_height * aspect_ratio

            logo = Image(PDF_LOGO_PATH, width=logo_width, height=logo_height)
            logo.hAlign = 'CENTER'
            story.append(logo)
            story.append(Spacer(1, 0.5*inch))
        except Exception as e:
            logger.debug(f"Logo loading failed: {e}")
            pass  # Gracefully handle missing logo

    # Main title with tier-appropriate styling
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['base']['Title'],
        fontSize=32,
        textColor=PDF_COLORS["secondary"],
        alignment=TA_CENTER,
        spaceAfter=10,
        fontName='Helvetica-Bold',
    )

    if is_whitelabel:
        story.append(Paragraph("SMART CONTRACT", title_style))
        story.append(Paragraph("SECURITY AUDIT", title_style))
    else:
        story.append(Paragraph("SMART CONTRACT", title_style))
        story.append(Paragraph("SECURITY AUDIT", title_style))

    story.append(Spacer(1, 0.3*inch))

    # Subtitle
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['base']['Normal'],
        fontSize=14,
        textColor=PDF_COLORS["text_secondary"],
        alignment=TA_CENTER,
        spaceAfter=30,
    )

    tier_names = {
        "free": "Trial Analysis",
        "starter": "Developer Report",
        "pro": "Professional Analysis",
        "enterprise": "Enterprise Security Assessment",
        "diamond": "Enterprise Security Assessment"
    }
    story.append(Paragraph(tier_names.get(tier, "Security Analysis"), subtitle_style))

    story.append(Spacer(1, 0.5*inch))

    # Horizontal separator line
    separator_style = ParagraphStyle('Separator', alignment=TA_CENTER, textColor=PDF_COLORS["primary"])
    story.append(Paragraph("━" * 40, separator_style))

    story.append(Spacer(1, 0.5*inch))

    # Risk score display (large, centered)
    try:
        risk_score = float(report.get("risk_score", 50))
    except (ValueError, TypeError):
        risk_score = 50.0

    risk_color = _get_risk_color(risk_score)

    score_style = ParagraphStyle(
        'CoverScore',
        parent=styles['base']['Normal'],
        fontSize=72,
        textColor=risk_color,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        spaceAfter=10,  # Add spacing after score to prevent overlap
    )
    story.append(Paragraph(f"{risk_score:.0f}", score_style))

    # Add explicit spacer for clear separation
    story.append(Spacer(1, 10))

    score_label_style = ParagraphStyle(
        'ScoreLabel',
        parent=styles['base']['Normal'],
        fontSize=14,
        textColor=PDF_COLORS["text_secondary"],
        alignment=TA_CENTER,
        spaceBefore=5,  # Additional spacing before label
    )
    story.append(Paragraph("SECURITY SCORE", score_label_style))

    story.append(Spacer(1, 0.5*inch))

    # Key metrics summary box
    critical = report.get("critical_count", 0)
    high = report.get("high_count", 0)
    medium = report.get("medium_count", 0)
    low = report.get("low_count", 0)

    metrics_data = [
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        [str(critical), str(high), str(medium), str(low)],
    ]

    metrics_table = Table(metrics_data, colWidths=[1.3*inch, 1.3*inch, 1.3*inch, 1.3*inch])
    metrics_table.setStyle(TableStyle([
        # Header row
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('TEXTCOLOR', (0, 0), (0, 0), PDF_COLORS["error"]),
        ('TEXTCOLOR', (1, 0), (1, 0), PDF_COLORS["warning"]),
        ('TEXTCOLOR', (2, 0), (2, 0), colors.Color(0.7, 0.55, 0)),
        ('TEXTCOLOR', (3, 0), (3, 0), PDF_COLORS["success"]),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        # Values row
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, 1), 24),
        ('TEXTCOLOR', (0, 1), (0, 1), PDF_COLORS["error"]),
        ('TEXTCOLOR', (1, 1), (1, 1), PDF_COLORS["warning"]),
        ('TEXTCOLOR', (2, 1), (2, 1), colors.Color(0.7, 0.55, 0)),
        ('TEXTCOLOR', (3, 1), (3, 1), PDF_COLORS["success"]),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(metrics_table)

    story.append(Spacer(1, 0.75*inch))

    # Report metadata
    meta_style = ParagraphStyle(
        'CoverMeta',
        parent=styles['base']['Normal'],
        fontSize=10,
        textColor=PDF_COLORS["text_secondary"],
        alignment=TA_CENTER,
        leading=16,
    )

    file_size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / 1024 / 1024:.2f} MB"

    story.append(Paragraph(f"<b>Prepared For:</b> {username}", meta_style))
    story.append(Paragraph(f"<b>Report ID:</b> {report_id}", meta_style))
    story.append(Paragraph(f"<b>Contract Size:</b> {file_size_str}", meta_style))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %H:%M UTC')}", meta_style))

    story.append(Spacer(1, 0.5*inch))

    # Confidential stamp for paid tiers
    if tier not in ["free"]:
        conf_style = ParagraphStyle(
            'Confidential',
            parent=styles['base']['Normal'],
            fontSize=11,
            textColor=PDF_COLORS["text_secondary"],
            alignment=TA_CENTER,
            borderColor=PDF_COLORS["border"],
            borderWidth=1,
            borderPadding=8,
        )
        story.append(Paragraph("─ CONFIDENTIAL ─", conf_style))

    story.append(PageBreak())


def _get_severity_color(severity: str) -> colors.Color:
    """Return color based on severity level."""
    severity_lower = severity.lower() if severity else "medium"
    return {
        "critical": colors.Color(0.8, 0, 0),      # Dark red
        "high": colors.Color(0.9, 0.4, 0),        # Orange
        "medium": colors.Color(0.9, 0.7, 0),      # Yellow/Gold
        "low": colors.Color(0.2, 0.6, 0.2),       # Green
        "info": colors.Color(0.3, 0.5, 0.7),      # Blue
    }.get(severity_lower, colors.gray)

def _get_risk_color(score: float) -> colors.Color:
    """Return color based on risk score (0-100)."""
    if score >= 70:
        return colors.Color(0.8, 0, 0)      # Red - Critical risk
    elif score >= 50:
        return colors.Color(0.9, 0.4, 0)    # Orange - High risk
    elif score >= 30:
        return colors.Color(0.9, 0.7, 0)    # Yellow - Medium risk
    else:
        return colors.Color(0.2, 0.6, 0.2)  # Green - Low risk

def _create_pdf_styles() -> dict:
    """Create custom paragraph styles for PDF."""
    base_styles = getSampleStyleSheet()

    custom_styles = {
        'base': base_styles,
        'title': ParagraphStyle(
            'CustomTitle',
            parent=base_styles['Title'],
            fontSize=24,
            spaceAfter=20,
            textColor=colors.Color(0.1, 0.1, 0.3),
        ),
        'heading1': ParagraphStyle(
            'CustomH1',
            parent=base_styles['Heading1'],
            fontSize=16,
            spaceBefore=15,
            spaceAfter=10,
            textColor=colors.Color(0.15, 0.15, 0.35),
        ),
        'heading2': ParagraphStyle(
            'CustomH2',
            parent=base_styles['Heading2'],
            fontSize=13,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.Color(0.2, 0.2, 0.4),
        ),
        'normal': base_styles['Normal'],
        'code': ParagraphStyle(
            'Code',
            parent=base_styles['Normal'],
            fontName='Courier',
            fontSize=8,
            backColor=colors.Color(0.95, 0.95, 0.95),
            leftIndent=10,
            rightIndent=10,
        ),
        'footer': ParagraphStyle(
            'Footer',
            parent=base_styles['Normal'],
            fontSize=8,
            textColor=colors.gray,
            alignment=TA_CENTER,
        ),
    }
    return custom_styles

def _build_header_section(story: list, styles: dict, username: str, file_size: int, tier: str) -> None:
    """Build PDF header with metadata - tier-aware branding."""
    # White-label for Enterprise: No DeFiGuard branding
    if tier in ["enterprise", "diamond"]:
        story.append(Paragraph("Smart Contract Security Audit", styles['title']))
        story.append(Paragraph("<b>Comprehensive Security Analysis Report</b>", styles['heading1']))
        report_type = "Enterprise Security Report"
    elif tier == "pro":
        # Branded for Pro: Subtle branding
        story.append(Paragraph("Security Audit Report", styles['title']))
        story.append(Paragraph("<b>Professional Smart Contract Analysis</b>", styles['heading1']))
        report_type = "Professional Tier"
    else:
        # Full branding for Starter/Free
        story.append(Paragraph("DeFiGuard AI", styles['title']))
        story.append(Paragraph("<b>Smart Contract Security Audit Report</b>", styles['heading1']))
        report_type = tier.title() + " Tier"

    story.append(Spacer(1, 10))

    # Metadata table
    meta_data = [
        ["Audit Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")],
        ["Prepared For:", username],
        ["Contract Size:", f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / 1024 / 1024:.2f} MB"],
        ["Report Type:", report_type],
    ]

    meta_table = Table(meta_data, colWidths=[1.5*inch, 4*inch])
    meta_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.Color(0.3, 0.3, 0.3)),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))

def _build_executive_summary(story: list, styles: dict, report: dict) -> None:
    """Build executive summary section."""
    story.append(Paragraph("Executive Summary", styles['heading1']))

    summary = report.get("executive_summary", "AI analysis completed. Review the identified issues below for detailed findings.")
    story.append(Paragraph(summary, styles['normal']))
    story.append(Spacer(1, 10))

def _build_risk_score_section(story: list, styles: dict, report: dict) -> None:
    """Build risk score section with visual indicator."""
    story.append(Paragraph("Risk Assessment", styles['heading1']))

    risk_score = float(report.get("risk_score", 50))
    risk_color = _get_risk_color(risk_score)

    # Determine risk level text
    if risk_score >= 70:
        risk_level = "CRITICAL RISK"
        risk_desc = "This contract has severe vulnerabilities that must be addressed before deployment."
    elif risk_score >= 50:
        risk_level = "HIGH RISK"
        risk_desc = "Significant security issues detected. Remediation strongly recommended."
    elif risk_score >= 30:
        risk_level = "MEDIUM RISK"
        risk_desc = "Some security concerns identified. Review and address before production."
    else:
        risk_level = "LOW RISK"
        risk_desc = "Contract appears relatively secure. Minor improvements may still be beneficial."

    # Risk score display - with proper spacing between elements
    score_style = ParagraphStyle('RiskScore', parent=styles['normal'], fontSize=36, textColor=risk_color, alignment=TA_CENTER, spaceAfter=15)
    story.append(Paragraph(f"<b>{risk_score:.0f}/100</b>", score_style))

    # Add spacer between score and risk level to prevent overlap
    story.append(Spacer(1, 20))

    level_style = ParagraphStyle('RiskLevel', parent=styles['normal'], fontSize=14, textColor=risk_color, alignment=TA_CENTER, spaceAfter=10)
    story.append(Paragraph(f"<b>{risk_level}</b>", level_style))

    story.append(Spacer(1, 15))
    story.append(Paragraph(risk_desc, styles['normal']))
    story.append(Spacer(1, 20))

def _build_severity_breakdown(story: list, styles: dict, report: dict) -> None:
    """Build severity breakdown table."""
    story.append(Paragraph("Findings Overview", styles['heading2']))

    critical = report.get("critical_count", 0)
    high = report.get("high_count", 0)
    medium = report.get("medium_count", 0)
    low = report.get("low_count", 0)
    total = critical + high + medium + low

    severity_data = [
        ["Severity", "Count", "Description"],
        ["CRITICAL", str(critical), "Immediate exploitation risk, potential total fund loss"],
        ["HIGH", str(high), "Significant vulnerability, likely exploitable"],
        ["MEDIUM", str(medium), "Security concern, should be addressed"],
        ["LOW", str(low), "Minor issue or best practice recommendation"],
        ["TOTAL", str(total), ""],
    ]

    severity_table = Table(severity_data, colWidths=[1.2*inch, 0.8*inch, 4*inch])
    severity_table.setStyle(TableStyle([
        # Header row
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.2, 0.2, 0.3)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        # Severity colors
        ('TEXTCOLOR', (0, 1), (0, 1), colors.Color(0.8, 0, 0)),       # Critical
        ('TEXTCOLOR', (0, 2), (0, 2), colors.Color(0.9, 0.4, 0)),     # High
        ('TEXTCOLOR', (0, 3), (0, 3), colors.Color(0.7, 0.5, 0)),     # Medium
        ('TEXTCOLOR', (0, 4), (0, 4), colors.Color(0.2, 0.6, 0.2)),   # Low
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        # Total row
        ('BACKGROUND', (0, -1), (-1, -1), colors.Color(0.9, 0.9, 0.9)),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        # Grid
        ('GRID', (0, 0), (-1, -1), 0.5, colors.Color(0.7, 0.7, 0.7)),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(severity_table)
    story.append(Spacer(1, 15))

def _build_issues_section(story: list, styles: dict, report: dict, tier: str) -> None:
    """Build detailed issues section based on tier."""
    issues = report.get("issues", [])
    if not issues:
        story.append(Paragraph("Detailed Findings", styles['heading1']))
        story.append(Paragraph("No security issues were identified in this contract.", styles['normal']))
        story.append(Spacer(1, 10))
        return

    story.append(Paragraph("Detailed Findings", styles['heading1']))

    for idx, issue in enumerate(issues, 1):
        severity = issue.get("severity", "Medium")
        severity_color = _get_severity_color(severity)

        # Issue header
        header_style = ParagraphStyle('IssueHeader', parent=styles['heading2'], textColor=severity_color)
        issue_type = issue.get("type", "Security Issue")
        story.append(Paragraph(f"[{severity.upper()}] {idx}. {issue_type}", header_style))

        # Issue ID
        issue_id = issue.get("id", f"ISSUE-{idx:03d}")
        story.append(Paragraph(f"<i>ID: {issue_id}</i>", styles['normal']))

        # Description
        description = issue.get("description", "No description provided.")
        story.append(Paragraph(f"<b>Description:</b> {description}", styles['normal']))

        # Fix recommendation (basic for all tiers)
        fix = issue.get("fix", "Manual review recommended.")
        story.append(Paragraph(f"<b>Recommendation:</b> {fix}", styles['normal']))

        # Pro tier: Add location information
        if tier in ["pro", "enterprise", "diamond"]:
            line_num = issue.get("line_number")
            func_name = issue.get("function_name")
            if line_num or func_name:
                location = []
                if func_name:
                    location.append(f"Function: {func_name}")
                if line_num:
                    location.append(f"Line: {line_num}")
                story.append(Paragraph(f"<b>Location:</b> {' | '.join(location)}", styles['normal']))

            # Vulnerable code snippet
            vuln_code = issue.get("vulnerable_code")
            if vuln_code:
                story.append(Paragraph("<b>Vulnerable Code:</b>", styles['normal']))
                # Escape HTML and format code
                code_text = str(vuln_code).replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(f"<font face='Courier' size='8'>{code_text}</font>", styles['code']))

            # Exploit scenario
            exploit = issue.get("exploit_scenario")
            if exploit:
                story.append(Paragraph(f"<b>Exploit Scenario:</b> {exploit}", styles['normal']))

            # Impact estimate
            impact = issue.get("estimated_impact")
            if impact:
                story.append(Paragraph(f"<b>Estimated Impact:</b> {impact}", styles['normal']))

            # Code fix (before/after)
            code_fix = issue.get("code_fix")
            if code_fix and isinstance(code_fix, dict):
                before = code_fix.get("before", "")
                after = code_fix.get("after", "")
                explanation = code_fix.get("explanation", "")

                if before or after:
                    story.append(Paragraph("<b>Suggested Fix:</b>", styles['normal']))
                    if before:
                        before_text = str(before).replace('<', '&lt;').replace('>', '&gt;')
                        story.append(Paragraph(f"<i>Before:</i>", styles['normal']))
                        story.append(Paragraph(f"<font face='Courier' size='8'>{before_text}</font>", styles['code']))
                    if after:
                        after_text = str(after).replace('<', '&lt;').replace('>', '&gt;')
                        story.append(Paragraph(f"<i>After:</i>", styles['normal']))
                        story.append(Paragraph(f"<font face='Courier' size='8'>{after_text}</font>", styles['code']))
                    if explanation:
                        story.append(Paragraph(f"<i>Explanation:</i> {explanation}", styles['normal']))

        # Enterprise tier: Add POC and references
        if tier in ["enterprise", "diamond"]:
            poc = issue.get("proof_of_concept")
            if poc:
                story.append(Paragraph("<b>Proof of Concept:</b>", styles['normal']))
                poc_text = str(poc).replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(f"<font face='Courier' size='8'>{poc_text}</font>", styles['code']))

            refs = issue.get("references", [])
            if refs:
                story.append(Paragraph("<b>References:</b>", styles['normal']))
                for ref in refs:
                    if isinstance(ref, dict):
                        title = ref.get("title", "Reference")
                        url = ref.get("url", "#")
                        story.append(Paragraph(f"• {title}: {url}", styles['normal']))

        story.append(Spacer(1, 10))

def _build_predictions_section(story: list, styles: dict, report: dict, tier: str) -> None:
    """Build attack predictions section."""
    predictions = report.get("predictions", [])
    if not predictions:
        return

    story.append(PageBreak())
    story.append(Paragraph("Attack Scenario Predictions", styles['heading1']))
    story.append(Paragraph(
        "Based on the identified vulnerabilities, the following attack scenarios are possible:",
        styles['normal']
    ))
    story.append(Spacer(1, 10))

    for idx, pred in enumerate(predictions, 1):
        if not isinstance(pred, dict):
            continue

        title = pred.get("title", f"Scenario {idx}")
        severity = pred.get("severity", "Medium")
        severity_color = _get_severity_color(severity)

        header_style = ParagraphStyle('PredHeader', parent=styles['heading2'], textColor=severity_color)
        story.append(Paragraph(f"{idx}. {title}", header_style))

        # Core prediction info
        probability = pred.get("probability", "Unknown")
        story.append(Paragraph(f"<b>Probability:</b> {probability}", styles['normal']))

        attack_vector = pred.get("attack_vector", "")
        if attack_vector:
            story.append(Paragraph(f"<b>Attack Vector:</b> {attack_vector}", styles['normal']))

        financial_impact = pred.get("financial_impact", "")
        if financial_impact:
            story.append(Paragraph(f"<b>Financial Impact:</b> {financial_impact}", styles['normal']))

        time_to_exploit = pred.get("time_to_exploit", "")
        if time_to_exploit:
            story.append(Paragraph(f"<b>Time to Exploit:</b> {time_to_exploit}", styles['normal']))

        mitigation = pred.get("mitigation", "")
        if mitigation:
            story.append(Paragraph(f"<b>Mitigation:</b> {mitigation}", styles['normal']))

        # Real world example
        example = pred.get("real_world_example")
        if example:
            story.append(Paragraph(f"<b>Real-World Example:</b> {example}", styles['normal']))

        story.append(Spacer(1, 8))

def _build_recommendations_section(story: list, styles: dict, report: dict) -> None:
    """Build recommendations section."""
    recommendations = report.get("recommendations", {})
    if not recommendations:
        return

    story.append(Paragraph("Recommendations", styles['heading1']))

    # Immediate actions
    immediate = recommendations.get("immediate", [])
    if immediate:
        story.append(Paragraph("Immediate Actions (Before Deployment)", styles['heading2']))
        for rec in immediate:
            story.append(Paragraph(f"• {rec}", styles['normal']))

    # Short-term
    short_term = recommendations.get("short_term", [])
    if short_term:
        story.append(Paragraph("Short-Term (Within 1 Week)", styles['heading2']))
        for rec in short_term:
            story.append(Paragraph(f"• {rec}", styles['normal']))

    # Long-term
    long_term = recommendations.get("long_term", [])
    if long_term:
        story.append(Paragraph("Long-Term (Ongoing)", styles['heading2']))
        for rec in long_term:
            story.append(Paragraph(f"• {rec}", styles['normal']))

    story.append(Spacer(1, 10))

def _build_compliance_section(story: list, styles: dict, compliance_data: dict) -> None:
    """Build regulatory compliance section."""
    if not compliance_data:
        return

    story.append(PageBreak())
    story.append(Paragraph("Regulatory Compliance Analysis", styles['heading1']))

    # MiCA Compliance
    mica = compliance_data.get("mica_compliance", {})
    if mica:
        story.append(Paragraph("EU MiCA Compliance", styles['heading2']))
        compliant = mica.get("compliant", True)
        status = "COMPLIANT" if compliant else "NON-COMPLIANT"
        status_color = colors.Color(0.2, 0.6, 0.2) if compliant else colors.Color(0.8, 0, 0)
        status_style = ParagraphStyle('Status', parent=styles['normal'], textColor=status_color)
        story.append(Paragraph(f"<b>Status: {status}</b>", status_style))

        violations = mica.get("violations", [])
        if violations:
            story.append(Paragraph("Violations:", styles['normal']))
            for v in violations:
                if isinstance(v, dict):
                    article = v.get("article", "Unknown")
                    issue = v.get("issue", "")
                    story.append(Paragraph(f"• {article}: {issue}", styles['normal']))

    # SEC Compliance
    sec = compliance_data.get("sec_compliance", {})
    if sec:
        story.append(Paragraph("SEC / Howey Test Analysis", styles['heading2']))
        is_security = sec.get("is_security", False)
        if is_security:
            story.append(Paragraph("<b>Warning:</b> Token may be classified as a security.",
                                   ParagraphStyle('Warning', parent=styles['normal'], textColor=colors.Color(0.8, 0, 0))))

        howey_factors = sec.get("howey_factors", [])
        if howey_factors:
            story.append(Paragraph("Howey Test Factors Present:", styles['normal']))
            for factor in howey_factors:
                story.append(Paragraph(f"• {factor}", styles['normal']))

    story.append(Spacer(1, 10))

def _build_remediation_roadmap(story: list, styles: dict, report: dict) -> None:
    """Build remediation roadmap section (Enterprise only)."""
    roadmap = report.get("remediation_roadmap")
    if not roadmap:
        return

    story.append(Paragraph("Remediation Roadmap", styles['heading1']))

    # Split roadmap into lines and display
    lines = str(roadmap).strip().split('\n')
    for line in lines:
        if line.strip():
            story.append(Paragraph(f"• {line.strip()}", styles['normal']))

    story.append(Spacer(1, 10))

def _build_tool_results_section(story: list, styles: dict, report: dict, tier: str) -> None:
    """Build tool results section (Pro+ only)."""
    if tier not in ["pro", "enterprise", "diamond"]:
        return

    fuzzing = report.get("fuzzing_results", [])
    mythril = report.get("mythril_results", [])
    certora = report.get("certora_results", [])

    if not fuzzing and not mythril and not certora:
        return

    story.append(Paragraph("Static & Dynamic Analysis Results", styles['heading1']))

    # Echidna Fuzzing Results
    if fuzzing:
        story.append(Paragraph("Echidna Fuzzing Results", styles['heading2']))
        for result in fuzzing[:5]:  # Limit to first 5
            if isinstance(result, dict):
                # Handle the actual Echidna data structure
                parsed = result.get("parsed", {})
                if parsed:
                    tests_passed = parsed.get("tests_passed", 0)
                    tests_total = parsed.get("tests_total", 0)
                    iterations = parsed.get("fuzzing_iterations", 0)
                    story.append(Paragraph(f"• Tests: {tests_passed}/{tests_total} passed", styles['normal']))
                    story.append(Paragraph(f"• Fuzzing iterations: {iterations}", styles['normal']))
                    if parsed.get("execution_summary"):
                        story.append(Paragraph(f"• {parsed['execution_summary'][:200]}", styles['normal']))
                else:
                    # Fallback for error/timeout results
                    vuln = result.get("vulnerability", result.get("test", "Fuzzing"))
                    desc = result.get("description", result.get("status", ""))
                    story.append(Paragraph(f"• {vuln}: {desc[:200]}", styles['normal']))
            elif isinstance(result, str):
                story.append(Paragraph(f"• {result}", styles['normal']))

    # Mythril Symbolic Analysis Results
    if mythril:
        story.append(Paragraph("Mythril Symbolic Analysis", styles['heading2']))
        for result in mythril[:5]:  # Limit to first 5
            if isinstance(result, dict):
                # Handle actual Mythril data structure: vulnerability, description
                vuln = result.get("vulnerability", result.get("title", "Finding"))
                desc = result.get("description", "")
                # Mythril severity comes from title patterns
                severity = "Medium"
                if "Ether" in vuln or "selfdestruct" in vuln.lower():
                    severity = "High"
                elif "Integer" in vuln or "Overflow" in vuln:
                    severity = "Medium"
                story.append(Paragraph(f"• [{severity}] <b>{vuln}</b>", styles['normal']))
                if desc and desc != "No description":
                    story.append(Paragraph(f"  {desc[:150]}", styles['normal']))
            elif isinstance(result, str):
                story.append(Paragraph(f"• {result}", styles['normal']))

    # Certora Formal Verification (Enterprise only)
    if certora and tier in ["enterprise", "diamond"]:
        story.append(Paragraph("Certora Formal Verification", styles['heading2']))
        verified_count = sum(1 for r in certora if isinstance(r, dict) and r.get("status") == "verified")
        violated_count = sum(1 for r in certora if isinstance(r, dict) and r.get("status") in ["violated", "issues_found"])
        error_count = sum(1 for r in certora if isinstance(r, dict) and r.get("status") in ["error", "incomplete", "timeout"])

        story.append(Paragraph(f"<b>Summary:</b> {verified_count} rules verified, {violated_count} violations found", styles['normal']))

        for result in certora[:10]:  # Limit to first 10
            if isinstance(result, dict):
                rule = result.get("rule", "Verification Check")
                status = result.get("status", "unknown")
                description = result.get("description", result.get("reason", ""))

                # Better status icons
                if status == "verified":
                    status_icon = "✓"
                elif status in ["violated", "issues_found"]:
                    status_icon = "✗"
                elif status in ["error", "incomplete"]:
                    status_icon = "⚠"
                elif status == "timeout":
                    status_icon = "⏱"
                elif status == "skipped":
                    status_icon = "○"
                else:
                    status_icon = "•"

                story.append(Paragraph(f"• {status_icon} <b>{rule}</b>: {description[:150]}", styles['normal']))

    story.append(Spacer(1, 10))


def _build_onchain_section(story: list, styles: dict, report: dict) -> None:
    """Build on-chain analysis section for PDF (Pro/Enterprise)."""
    onchain = report.get("onchain_analysis")
    if not onchain:
        return

    story.append(Paragraph("⛓️ On-Chain Analysis", styles['heading1']))

    # Contract address and chain
    address = onchain.get("address", "Unknown")
    chain = onchain.get("chain", "Ethereum")
    story.append(Paragraph(f"<b>Contract:</b> {address}", styles['normal']))
    story.append(Paragraph(f"<b>Chain:</b> {chain}", styles['normal']))
    story.append(Spacer(1, 8))

    # Proxy Detection
    proxy = onchain.get("proxy", {})
    if proxy:
        story.append(Paragraph("🔄 Proxy Status", styles['heading2']))
        is_proxy = proxy.get("is_proxy", False)
        if is_proxy:
            proxy_type = proxy.get("proxy_type", "Unknown")
            upgrade_risk = proxy.get("upgrade_risk", "N/A")
            impl = proxy.get("implementation", "N/A")
            story.append(Paragraph(f"• Type: {proxy_type} Proxy", styles['normal']))
            story.append(Paragraph(f"• Upgrade Risk: {upgrade_risk}", styles['normal']))
            if impl and impl != "N/A":
                story.append(Paragraph(f"• Implementation: {impl[:20]}...{impl[-8:]}", styles['normal']))
        else:
            story.append(Paragraph("• Not a proxy contract (code is immutable)", styles['normal']))
        story.append(Spacer(1, 6))

    # Storage/Ownership
    storage = onchain.get("storage", {})
    if storage:
        story.append(Paragraph("👤 Ownership & Access Control", styles['heading2']))
        owner = storage.get("owner")
        if owner:
            story.append(Paragraph(f"• Owner: {owner[:20]}...{owner[-8:]}", styles['normal']))
        central_risk = storage.get("centralization_risk", "LOW")
        story.append(Paragraph(f"• Centralization Risk: {central_risk}", styles['normal']))
        if storage.get("is_pausable"):
            paused_status = "PAUSED" if storage.get("is_paused") else "Active"
            story.append(Paragraph(f"• Pausable: Yes ({paused_status})", styles['normal']))
        story.append(Spacer(1, 6))

    # Backdoor Detection
    backdoors = onchain.get("backdoors", {})
    if backdoors:
        story.append(Paragraph("🚨 Backdoor Scan", styles['heading2']))
        has_backdoors = backdoors.get("has_backdoors", False)
        risk_level = backdoors.get("risk_level", "LOW")
        if has_backdoors:
            story.append(Paragraph(f"• Risk Level: {risk_level}", styles['normal']))
            summary = backdoors.get("summary", "")
            if summary:
                story.append(Paragraph(f"• {summary}", styles['normal']))
            dangerous_funcs = backdoors.get("dangerous_functions", [])
            if dangerous_funcs:
                story.append(Paragraph("• Dangerous Functions Detected:", styles['normal']))
                for func in dangerous_funcs[:5]:
                    name = func.get("name", func.get("selector", "Unknown"))
                    category = func.get("category", "unknown")
                    story.append(Paragraph(f"    - {name} ({category})", styles['normal']))
        else:
            story.append(Paragraph("• No backdoor patterns detected", styles['normal']))
        story.append(Spacer(1, 6))

    # Honeypot Detection
    honeypot = onchain.get("honeypot", {})
    if honeypot:
        story.append(Paragraph("🍯 Honeypot Analysis", styles['heading2']))
        is_honeypot = honeypot.get("is_honeypot", False)
        if is_honeypot:
            confidence = honeypot.get("confidence", "LOW")
            story.append(Paragraph(f"• ⚠️ HONEYPOT DETECTED (Confidence: {confidence})", styles['normal']))
            recommendation = honeypot.get("recommendation", "")
            if recommendation:
                story.append(Paragraph(f"• {recommendation}", styles['normal']))
        else:
            story.append(Paragraph("• ✅ No honeypot indicators detected", styles['normal']))
        story.append(Spacer(1, 6))

    # Overall On-Chain Risk
    overall = onchain.get("overall_risk", {})
    if overall:
        story.append(Paragraph("📊 Overall On-Chain Risk", styles['heading2']))
        level = overall.get("level", "N/A")
        score = overall.get("score", 0)
        story.append(Paragraph(f"• Risk Level: {level}", styles['normal']))
        story.append(Paragraph(f"• Risk Score: {score}/100", styles['normal']))
        summary = overall.get("summary", "")
        if summary:
            story.append(Paragraph(f"• {summary}", styles['normal']))

    story.append(Spacer(1, 10))


def _build_footer(story: list, styles: dict, tier: str) -> None:
    """Build PDF footer - tier-aware branding."""
    story.append(Spacer(1, 20))
    story.append(Paragraph("─" * 60, styles['footer']))

    # White-label for Enterprise: No DeFiGuard branding in footer
    if tier in ["enterprise", "diamond"]:
        story.append(Paragraph(
            f"Enterprise Security Report • {datetime.now().strftime('%Y-%m-%d')}",
            styles['footer']
        ))
        story.append(Paragraph(
            "This report is provided for informational purposes. Always conduct additional security reviews.",
            styles['footer']
        ))
    elif tier == "pro":
        # Subtle branding for Pro
        story.append(Paragraph(
            f"Professional Security Report • {datetime.now().strftime('%Y-%m-%d')} • Powered by DeFiGuard AI",
            styles['footer']
        ))
        story.append(Paragraph(
            "This report is provided for informational purposes. Always conduct additional security reviews.",
            styles['footer']
        ))
    else:
        # Full branding for Starter/Free
        story.append(Paragraph(
            f"Generated by DeFiGuard AI • {tier.title()} Tier Report • {datetime.now().strftime('%Y-%m-%d')}",
            styles['footer']
        ))
        story.append(Paragraph(
            "This report is provided for informational purposes. Always conduct additional security reviews.",
            styles['footer']
        ))

def generate_compliance_pdf(
    report: dict[str, Any],
    username: str,
    file_size: int,
    tier: str = "starter",
    compliance_data: dict[str, Any] | None = None
) -> str | None:
    """
    Generate comprehensive, professional-grade PDF audit report based on user tier.

    Features:
    - Professional cover page with logo (non-Enterprise) and key metrics
    - Tier-aware branding (white-label for Enterprise)
    - Logo watermark in footer for Starter/Pro
    - Page numbers on all pages
    - Report ID for traceability
    - CONFIDENTIAL marking for paid tiers

    Tier capabilities:
    - starter: Basic report with issues, risk score, recommendations, logo branding
    - pro: Adds code snippets, exploit scenarios, tool results, subtle branding
    - enterprise: White-label, POC, references, remediation roadmap, compliance
    """
    try:
        timestamp = int(time.time())
        report_id = f"DFG-{datetime.now().strftime('%Y%m%d')}-{secrets.token_hex(4).upper()}"
        pdf_path = os.path.join(DATA_DIR, f"audit_report_{username}_{timestamp}.pdf")

        # Create professional page template with header/footer
        pdf_template = ProfessionalPDFTemplate(tier, username, report_id)

        # Use BaseDocTemplate for more control over page templates
        doc = BaseDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.85*inch,   # Extra space for header
            bottomMargin=0.85*inch  # Extra space for footer
        )

        # Define frame for content (slightly reduced to account for header/footer)
        content_frame = Frame(
            0.75*inch,
            0.85*inch,
            letter[0] - 1.5*inch,
            letter[1] - 1.7*inch,
            id='content'
        )

        # Cover page template (no header/footer)
        cover_template = PageTemplate(
            id='cover',
            frames=[Frame(0.75*inch, 0.75*inch, letter[0] - 1.5*inch, letter[1] - 1.5*inch, id='cover_frame')],
            onPage=pdf_template.on_first_page
        )

        # Content pages template (with header/footer and logo watermark)
        content_template = PageTemplate(
            id='content',
            frames=[content_frame],
            onPage=pdf_template.on_page
        )

        doc.addPageTemplates([cover_template, content_template])

        styles = _create_pdf_styles()
        story: list[Flowable] = []

        # ═══════════════════════════════════════════════════════════════════════
        # COVER PAGE - Professional first impression
        # ═══════════════════════════════════════════════════════════════════════
        _build_cover_page(story, styles, username, file_size, tier, report, report_id)

        # Switch to content template after cover page
        story.append(NextPageTemplate('content'))

        # ═══════════════════════════════════════════════════════════════════════
        # TABLE OF CONTENTS placeholder (for longer reports)
        # ═══════════════════════════════════════════════════════════════════════
        story.append(Paragraph("Table of Contents", styles['heading1']))
        toc_style = ParagraphStyle('TOC', parent=styles['normal'], fontSize=10, leftIndent=20)
        story.append(Paragraph("1. Executive Summary", toc_style))
        story.append(Paragraph("2. Risk Assessment", toc_style))
        story.append(Paragraph("3. Findings Overview", toc_style))
        story.append(Paragraph("4. Detailed Findings", toc_style))
        story.append(Paragraph("5. Attack Scenario Predictions", toc_style))
        story.append(Paragraph("6. Recommendations", toc_style))
        if tier in ["pro", "enterprise", "diamond"]:
            story.append(Paragraph("7. Static & Dynamic Analysis Results", toc_style))
            if report.get("onchain_analysis"):
                story.append(Paragraph("8. On-Chain Analysis", toc_style))
        if tier in ["enterprise", "diamond"] and compliance_data:
            story.append(Paragraph("9. Regulatory Compliance Analysis", toc_style))
        story.append(PageBreak())

        # ═══════════════════════════════════════════════════════════════════════
        # MAIN REPORT CONTENT
        # ═══════════════════════════════════════════════════════════════════════
        _build_header_section(story, styles, username, file_size, tier)
        _build_executive_summary(story, styles, report)
        _build_risk_score_section(story, styles, report)
        _build_severity_breakdown(story, styles, report)
        _build_issues_section(story, styles, report, tier)
        _build_predictions_section(story, styles, report, tier)
        _build_recommendations_section(story, styles, report)

        # Pro+ tier sections
        if tier in ["pro", "enterprise", "diamond"]:
            _build_tool_results_section(story, styles, report, tier)
            _build_onchain_section(story, styles, report)

        # Enterprise tier sections
        if tier in ["enterprise", "diamond"]:
            if compliance_data:
                _build_compliance_section(story, styles, compliance_data)
            _build_remediation_roadmap(story, styles, report)

        # ═══════════════════════════════════════════════════════════════════════
        # FINAL PAGE - Disclaimer and contact
        # ═══════════════════════════════════════════════════════════════════════
        story.append(PageBreak())
        story.append(Spacer(1, 1*inch))

        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['normal'],
            fontSize=9,
            textColor=PDF_COLORS["text_secondary"],
            alignment=TA_CENTER,
            leading=14,
        )

        story.append(Paragraph("─" * 50, disclaimer_style))
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("<b>IMPORTANT DISCLAIMER</b>", disclaimer_style))
        story.append(Spacer(1, 0.15*inch))
        story.append(Paragraph(
            "This security audit report is provided for informational purposes only and does not constitute "
            "financial, legal, or investment advice. While this analysis uses industry-standard tools "
            "(Slither, Mythril, Echidna) and AI-powered analysis, no automated audit can guarantee "
            "the complete absence of vulnerabilities.",
            disclaimer_style
        ))
        story.append(Spacer(1, 0.15*inch))
        story.append(Paragraph(
            "We strongly recommend conducting additional manual security reviews, formal verification, "
            "and obtaining professional audits before deploying smart contracts to production environments.",
            disclaimer_style
        ))
        story.append(Spacer(1, 0.3*inch))

        if tier not in ["enterprise", "diamond"]:
            # Branding for non-Enterprise
            story.append(Paragraph("─" * 30, disclaimer_style))
            story.append(Spacer(1, 0.2*inch))

            if os.path.exists(PDF_LOGO_SMALL_PATH):
                try:
                    logo = Image(PDF_LOGO_SMALL_PATH, width=0.5*inch, height=0.5*inch)
                    logo.hAlign = 'CENTER'
                    story.append(logo)
                except Exception:
                    pass

            story.append(Paragraph(
                "<b>DeFiGuard AI</b> — Enterprise Smart Contract Security",
                disclaimer_style
            ))
            story.append(Paragraph(
                "https://defiguard-ai-beta.onrender.com",
                disclaimer_style
            ))

        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph(f"Report ID: {report_id}", disclaimer_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}", disclaimer_style))

        # Build the PDF
        doc.build(story)
        logger.info(f"[PDF] ✅ Professional audit report generated: {pdf_path} (tier={tier}, id={report_id})")
        return pdf_path

    except Exception as e:
        logger.error(f"[PDF] ❌ Generation failed: {str(e)}", exc_info=True)
        return None

# Celery/Redis for scale
celery = Celery(__name__, broker=os.getenv("REDIS_URL"), backend=os.getenv("REDIS_URL"))

# Redis pub/sub for cross-worker WebSocket messaging
redis_pubsub_client = None

async def init_redis_pubsub():
    """Initialize Redis pub/sub for cross-worker WebSocket messaging."""
    global redis_pubsub_client
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            redis_pubsub_client = await aioredis.from_url(redis_url)
            logger.info("[REDIS] ✅ Pub/sub client connected for WebSocket messaging")
            return True
        except Exception as e:
            logger.warning(f"[REDIS] ⚠️ Pub/sub connection failed: {e}")
    return False

async def redis_audit_subscriber():
    """Subscribe to audit log messages and deliver to local WebSockets (for cross-worker messaging)."""
    global redis_pubsub_client
    if not redis_pubsub_client:
        return

    try:
        pubsub = redis_pubsub_client.pubsub()
        await pubsub.subscribe("audit_log_broadcast")
        logger.info("[REDIS] 📡 Subscribed to audit_log_broadcast channel")

        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    username = data.get("username")
                    msg = data.get("message")
                    source_worker = data.get("source_worker")

                    # Skip if this message came from THIS worker (already delivered locally)
                    if source_worker == WORKER_ID:
                        logger.debug(f"[REDIS_SUB] Skipping own message for '{username}': {msg}")
                        continue

                    # Deliver to WebSocket if on THIS worker
                    ws = active_audit_websockets.get(username)
                    if ws and ws.application_state == WebSocketState.CONNECTED:
                        await ws.send_json({"type": "audit_log", "message": msg})
                        logger.info(f"[REDIS_SUB] ✅ Delivered to '{username}': {msg}")
                except Exception as e:
                    logger.error(f"[REDIS_SUB] Message handling error: {e}")
    except asyncio.CancelledError:
        logger.info("[REDIS] Subscriber cancelled")
    except Exception as e:
        logger.error(f"[REDIS] Subscriber error: {e}")
        # Try to restart subscriber after a delay
        await asyncio.sleep(5)
        asyncio.create_task(redis_audit_subscriber())

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

# Global tracking for ws-alerts connections
active_alerts_websockets: set = set()
MAX_ALERTS_CONNECTIONS = 100

# /ws-alerts for scale - requires authentication, connection limits, error handling
@app.websocket("/ws-alerts")
async def ws_alerts(websocket: WebSocket):
    """WebSocket for DeFi exploit alerts - authenticated with connection limits."""
    # Connection limit check
    if len(active_alerts_websockets) >= MAX_ALERTS_CONNECTIONS:
        await websocket.close(code=4029, reason="Connection limit exceeded")
        return

    # Authentication: require valid token
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4001, reason="Authentication required")
        return

    validated_username = verify_ws_token(token, _secret_key)
    if not validated_username:
        await websocket.close(code=4001, reason="Invalid or expired token")
        return

    await websocket.accept()
    active_alerts_websockets.add(websocket)
    logger.info(f"[WS_ALERTS] Connected: {validated_username} (total: {len(active_alerts_websockets)})")

    try:
        while True:
            try:
                # Check for client disconnect or ping
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send alerts on timeout (every 30s check, 5min alert cycle)
                pass

            # Only send alerts every 5 minutes (track with connection state)
            await asyncio.sleep(270)  # 4.5 min to complete 5 min cycle with 30s timeout
            try:
                search_result = x_semantic_search(query="latest DeFi exploits", limit=5)
                await websocket.send_json(search_result)
            except Exception as e:
                logger.error(f"[WS_ALERTS] Send failed for {validated_username}: {e}")
                break
    except WebSocketDisconnect:
        logger.info(f"[WS_ALERTS] Disconnected: {validated_username}")
    except Exception as e:
        logger.error(f"[WS_ALERTS] Error for {validated_username}: {e}")
    finally:
        active_alerts_websockets.discard(websocket)
        logger.info(f"[WS_ALERTS] Cleaned up: {validated_username} (remaining: {len(active_alerts_websockets)})")

# /overage for pre-calc
@app.post("/overage")
async def overage(request: Request, file_size: int = Body(...)):
    """Calculate overage cost. Rate limited and requires authentication."""
    # Get client IP for rate limiting
    client_ip = request.client.host if request.client else "unknown"

    # Rate limit check
    limit = RATE_LIMITS["overage"]
    if not await rate_limiter.is_allowed(f"overage:{client_ip}", limit["max_requests"], limit["window_seconds"]):
        retry_after = await rate_limiter.get_retry_after(f"overage:{client_ip}", limit["window_seconds"])
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)}
        )

    # Input validation
    if file_size < 0 or file_size > 1024 * 1024 * 1024:  # Max 1GB
        raise HTTPException(status_code=400, detail="Invalid file size")

    cost = usage_tracker.calculate_diamond_overage(file_size) / 100
    return {"cost": cost}

# /push for mobile - requires authentication
@app.post("/push")
async def push(request: Request, msg: str = Body(...)):
    """Push notification endpoint. Requires authentication, CSRF, and rate limited."""
    # Security: Verify CSRF token
    await verify_csrf_token(request)

    # Require authentication
    session_username = request.session.get("username")
    if not session_username:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Rate limit check
    limit = RATE_LIMITS["push"]
    if not await rate_limiter.is_allowed(f"push:{session_username}", limit["max_requests"], limit["window_seconds"]):
        retry_after = await rate_limiter.get_retry_after(f"push:{session_username}", limit["window_seconds"])
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)}
        )

    # Input validation - limit message size
    if len(msg) > 1000:
        raise HTTPException(status_code=400, detail="Message too long (max 1000 characters)")

    logger.info(f"Push sent by {session_username}: {msg[:100]}...")
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
async def debug_log(admin_key: str = Header(None, alias="X-Admin-Key")):
    """Debug endpoint - requires admin key in production (via X-Admin-Key header)."""
    expected_key = os.getenv("ADMIN_KEY")
    if not expected_key or not admin_key or not secrets.compare_digest(admin_key, expected_key):
        raise HTTPException(status_code=403, detail="Admin access required")
    logger.debug("Debug endpoint called")
    logger.info("Test INFO log")
    logger.warning("Test WARNING log")
    logger.error("Test ERROR log")
    return {"message": "Debug logs written to debug.log and console"}

@app.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    """Serve static files with path traversal protection."""
    logger.info(f"Serving static file: /static/{file_path}")

    # Security: Reject path traversal attempts
    if ".." in file_path or file_path.startswith("/") or file_path.startswith("\\"):
        logger.warning(f"Path traversal attempt blocked: {file_path}")
        raise HTTPException(status_code=403, detail="Invalid file path")

    # Security: Reject URL-encoded traversal attempts
    from urllib.parse import unquote
    decoded_path = unquote(file_path)
    if ".." in decoded_path or decoded_path.startswith("/") or decoded_path.startswith("\\"):
        logger.warning(f"URL-encoded path traversal attempt blocked: {file_path}")
        raise HTTPException(status_code=403, detail="Invalid file path")

    # Security: Validate path is within static directory using realpath to follow symlinks
    static_dir = os.path.realpath("static")
    file_full_path = os.path.realpath(os.path.join("static", file_path))

    # Ensure the resolved path is within static directory (prevents symlink attacks)
    if not file_full_path.startswith(static_dir + os.sep) and file_full_path != static_dir:
        logger.warning(f"Path escape attempt blocked: {file_path} -> {file_full_path}")
        raise HTTPException(status_code=403, detail="Access denied")

    # Reject symlinks pointing outside static directory
    if os.path.islink(os.path.join("static", file_path)):
        logger.warning(f"Symlink access blocked: {file_path}")
        raise HTTPException(status_code=403, detail="Access denied")

    if not os.path.isfile(file_full_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_full_path)

@app.get("/api/reports/{report_filename}")
async def download_report(
    report_filename: str,
    request: Request,
    current_user: Optional[User] = Depends(get_authenticated_user)
):
    """
    Protected PDF report download endpoint.
    Only allows authenticated users to download their own reports.
    """
    # Require authentication
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required to download reports")

    # Validate filename format (prevent path traversal)
    if not report_filename.endswith('.pdf') or '..' in report_filename or '/' in report_filename:
        raise HTTPException(status_code=400, detail="Invalid report filename")

    # Check if the report belongs to this user (filename contains username)
    # Format: audit_report_{username}_{timestamp}.pdf
    if f"_{current_user.username}_" not in report_filename:
        logger.warning(f"[REPORT] User {current_user.username} attempted to access report: {report_filename}")
        raise HTTPException(status_code=403, detail="You can only download your own reports")

    # Build full path
    report_path = os.path.join(DATA_DIR, report_filename)

    if not os.path.isfile(report_path):
        raise HTTPException(status_code=404, detail="Report not found")

    logger.info(f"[REPORT] Serving PDF report: {report_filename} to {current_user.username}")
    return FileResponse(
        report_path,
        media_type="application/pdf",
        filename=report_filename,
        headers={"Content-Disposition": f"attachment; filename={report_filename}"}
    )

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
            # Guest mode - not authenticated
            template = jinja_env.get_template("index.html")
            html_content = template.render(
                session=request.session,
                userinfo={},
                username=None,  # None = not authenticated, template shows generic welcome
                upgrade=upgrade,
                message=message or "Please sign in for full access"
            )
            return HTMLResponse(content=html_content)
        
        # Logged-in user flow
        if "csrf_token" not in request.session:
            request.session["csrf_token"] = secrets.token_urlsafe(32)
        
        session_username = request.session.get("username")
        # Security: Don't log session object - contains sensitive auth data
        logger.debug(f"UI request: tier={tier}, has_diamond={has_diamond}, session_username={session_username}")
        
        if session_id:
            effective_username = username or session_username
            if not effective_username:
                logger.error("No username provided for post-payment redirect; redirecting to login")
                return RedirectResponse(url="/auth?redirect_reason=no_username")
            
            if temp_id:
                logger.info(f"Processing post-payment redirect for Diamond audit, username={effective_username}, session_id={session_id}, temp_id={temp_id}")
                # URL-encode username to prevent parameter injection
                safe_username = urllib.parse.quote(effective_username, safe='')
                return RedirectResponse(url=f"/complete-diamond-audit?session_id={session_id}&temp_id={temp_id}&username={safe_username}")

            if tier:
                logger.info(f"Processing post-payment redirect for tier upgrade, username={effective_username}, session_id={session_id}, tier={tier}, has_diamond={has_diamond}")
                # URL-encode username to prevent parameter injection
                safe_username = urllib.parse.quote(effective_username, safe='')
                return RedirectResponse(url=f"/complete-tier-checkout?session_id={session_id}&tier={tier}&has_diamond={has_diamond}&username={safe_username}")
        
        template = jinja_env.get_template("index.html")
        userinfo = request.session.get("userinfo", {})
        init_script = ""
        init_js_cookie = request.cookies.get("init_js")
        if init_js_cookie:
            init_script = f"<script>{init_js_cookie}</script>"

        # Get display name from Auth0 userinfo with fallback chain
        # Priority: preferred_username → name → nickname → email → database username → None
        # Never default to "Guest" - authenticated users without display info get generic welcome
        display_name = (
            userinfo.get("preferred_username")
            or userinfo.get("name")
            or userinfo.get("nickname")
            or userinfo.get("email")
            or getattr(current_user, "username", None)
        )
        # Don't pass "Guest" for authenticated users - let template show generic welcome
        username = display_name if display_name else None
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
        return HTMLResponse(content="<h1>Internal server error</h1>", status_code=500)

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
        return HTMLResponse(content="<h1>Internal server error</h1>", status_code=500)

@app.get("/auth", response_class=HTMLResponse)
async def read_auth(request: Request):
    try:
        logger.debug(f"Auth page accessed")
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
async def get_tier(request: Request, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Get user tier information. Uses authenticated session only - no query param bypass."""
    session_username = request.session.get("username")
    logger.debug(f"Tier request: Session username={session_username}")

    # Security: Only use authenticated session username, never trust query params
    if not session_username:
        logger.debug("No session username for /tier; returning free tier defaults")
        return {
            "tier": "free",
            "size_limit": "250KB",
            "feature_flags": usage_tracker.feature_flags["free"],
            "api_key": None,
            "audit_count": usage_tracker.count,
            "audit_limit": FREE_LIMIT,
            "has_diamond": False,
            "username": None
        }

    user = db.query(User).filter(User.username == session_username).first()
    if not user:
        # Auto-recover: User exists in session but not in DB (ephemeral DB was wiped)
        # Create user with free tier - they can login again to sync with Stripe
        logger.warning(f"Tier fetch: User {session_username} not found in database - auto-creating with free tier")

        # Extract email from session or generate placeholder
        session_email = request.session.get("user", {}).get("email")
        if not session_email:
            # Fallback: convert username to email format if needed
            session_email = session_username if "@" in session_username else f"{session_username}@placeholder.local"

        # Create minimal user record to prevent loading failures
        user = User(
            username=session_username,
            email=session_email,
            tier="free",
            audit_history="[]",
            last_reset=datetime.now(timezone.utc),
            has_diamond=False
        )

        try:
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info(f"[AUTO-RECOVERY] Created user {session_username} with free tier - login again to sync with Stripe")
        except Exception as e:
            db.rollback()
            logger.error(f"[AUTO-RECOVERY] Failed to create user {session_username}: {e}")
            # Return free tier defaults instead of 404
            return {
                "tier": "free",
                "size_limit": "250KB",
                "feature_flags": usage_tracker.feature_flags["free"],
                "api_key": None,
                "audit_count": 0,
                "audit_limit": FREE_LIMIT,
                "has_diamond": False,
                "username": session_username,
                "recovery_note": "Session restored with free tier - login again to sync subscription"
            }
    
    user_tier = user.tier
    
    # Map size limits based on tier
    size_limit_map = {
        "free": "250KB",
        "starter": "1MB",
        "beginner": "1MB",  # Legacy
        "pro": "5MB",
        "enterprise": "Unlimited",
        "diamond": "Unlimited"  # Legacy
    }
    size_limit = size_limit_map.get(user_tier, "250KB")

    # Map tier for feature flags (diamond users get enterprise features)
    has_diamond = getattr(user, "has_diamond", False)
    flags_tier = "enterprise" if user_tier == "enterprise" else ("enterprise" if has_diamond else user_tier)
    tier_flags_map = {"beginner": "starter", "diamond": "enterprise"}
    flags_tier = tier_flags_map.get(flags_tier, flags_tier)

    feature_flags = usage_tracker.feature_flags.get(flags_tier, usage_tracker.feature_flags["free"])
    api_key = user.api_key if user.tier in ["pro", "enterprise"] or has_diamond else None

    # Get per-user audit count from database (not global counter)
    audit_count = user.total_audits if hasattr(user, 'total_audits') else 0
    
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

    logger.debug(f"Retrieved tier for {session_username}: {user_tier}, audit count: {audit_count}, has_diamond: {has_diamond}")
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

    # Security: Validate session user matches path username to prevent privilege escalation
    session_username = request.session.get("username")
    if not session_username:
        raise HTTPException(status_code=401, detail="Authentication required")
    if session_username != username:
        logger.warning(f"Tier bypass attempt: session user {session_username} tried to modify {username}")
        raise HTTPException(status_code=403, detail="Cannot modify another user's tier")

    logger.debug(f"Set-tier request for {username}, tier: {tier}, has_diamond: {has_diamond}")

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
        
        # Use dynamic base URL from request to ensure redirects go to current instance
        base_url = f"{request.url.scheme}://{request.url.netloc}"

        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=line_items,
            mode="subscription",
            success_url=f"{base_url}/complete-tier-checkout?session_id={{CHECKOUT_SESSION_ID}}&tier={urllib.parse.quote(tier)}&has_diamond={urllib.parse.quote(str(has_diamond).lower())}&username={urllib.parse.quote(username)}",
            cancel_url=f"{base_url}/ui?username={urllib.parse.quote(username)}",
            metadata={"username": username, "tier": tier, "has_diamond": str(has_diamond).lower()}
        )
        
        logger.info(f"Redirecting {username} to Stripe checkout for {tier} tier, has_diamond: {has_diamond}")
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
        raise HTTPException(status_code=400, detail="Invalid payment request. Please try again.")
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

    # Security: Only use session username - ignore any username from request body
    # This prevents privilege escalation where an attacker could create checkout sessions for other users
    if not session_username:
        raise HTTPException(status_code=401, detail="Login required")
    effective_username = session_username
    
    user = db.query(User).filter(User.username == effective_username).first()
    if not user:
        # Auto-recover: User exists in session but not in DB (ephemeral DB was wiped)
        logger.warning(f"Checkout: User {effective_username} not found in database - auto-creating")
        session_email = request.session.get("user", {}).get("email")
        if not session_email:
            session_email = effective_username if "@" in effective_username else f"{effective_username}@placeholder.local"

        user = User(
            username=effective_username,
            email=session_email,
            tier="free",
            audit_history="[]",
            last_reset=datetime.now(timezone.utc),
            has_diamond=False
        )
        try:
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info(f"[AUTO-RECOVERY] Created user {effective_username} during checkout")
        except Exception as e:
            db.rollback()
            logger.error(f"[AUTO-RECOVERY] Failed to create user {effective_username}: {e}")
            raise HTTPException(status_code=500, detail="Unable to process upgrade - please try logging in again")

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
    # URL-encode username to prevent parameter injection
    safe_username = urllib.parse.quote(effective_username, safe='')
    success_url = f"{base_url}/complete-tier-checkout?session_id={{CHECKOUT_SESSION_ID}}&tier={tier}&has_diamond={has_diamond}&username={safe_username}"
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
                "has_diamond": "true" if has_diamond else "false",  # JSON-compatible boolean
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
    """
    Complete tier checkout after Stripe payment.

    Security model:
    1. Validate Stripe session exists and is paid
    2. Validate URL username matches Stripe metadata (prevents redirect hijacking)
    3. Re-establish session for the user (handles session expiration during checkout)
    4. Generate new CSRF token for security
    """
    logger.debug(f"Complete-tier-checkout request: session_id={session_id}, tier={tier}, has_diamond={has_diamond}, username={username}")

    try:
        # Validate Stripe session
        stripe_session = stripe.checkout.Session.retrieve(session_id)
        logger.info(f"Retrieved Stripe session: payment_status={stripe_session.payment_status}")

        # Security: Validate username matches Stripe session metadata to prevent redirect hijacking
        metadata_username = stripe_session.metadata.get("username") if stripe_session.metadata else None
        if not metadata_username or metadata_username != username:
            logger.warning(f"Checkout hijack attempt: URL username {username} != metadata username {metadata_username}")
            return RedirectResponse(url="/ui?upgrade=failed&message=Security%20validation%20failed")

        if stripe_session.payment_status != "paid":
            logger.error(f"Payment not completed for {username}, status={stripe_session.payment_status}")
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

        # Re-establish session completely (handles session expiration during checkout)
        # This is critical: user's session may have expired during the Stripe checkout flow
        if request is not None:
            # Clear any stale session data first
            request.session.clear()

            # Re-establish full session (same as Auth0 callback)
            request.session["user_id"] = user.id
            request.session["username"] = user.username
            request.session["csrf_token"] = secrets.token_urlsafe(32)
            request.session["csrf_last_refresh"] = datetime.now().isoformat()

            # Preserve auth provider info if available from Stripe metadata
            if stripe_session.metadata:
                provider = stripe_session.metadata.get("provider", "unknown")
                request.session["auth_provider"] = provider
                request.session["user"] = {
                    "sub": user.auth0_sub or f"email|{user.id}",
                    "email": user.email,
                    "provider": provider
                }

            logger.info(f"Session fully re-established for {username} after successful Stripe payment (tier: {normalized_tier})")

        # Include tier in redirect so frontend can show it immediately
        return RedirectResponse(url=f"/ui?upgrade=success&tier={urllib.parse.quote(normalized_tier)}&message=Tier%20upgrade%20completed")
    
    except Exception as e:
        logger.error(f"Complete-tier-checkout error: {str(e)}")
        return RedirectResponse(url="/ui?upgrade=error&message=Checkout%20processing%20error")

## Section 4.4: Webhook Endpoint

@app.post("/webhook")
async def webhook(request: Request, db: Session = Depends(get_db)):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    logger.debug(f"Webhook received, payload: {payload[:100]}, sig_header: {sig_header}")
    
    if not STRIPE_API_KEY or not STRIPE_WEBHOOK_SECRET:
        logger.error("Stripe webhook processing failed: STRIPE_API_KEY or STRIPE_WEBHOOK_SECRET not set")
        return Response(status_code=503, content="Webhook processing unavailable: Please set STRIPE_API_KEY and STRIPE_WEBHOOK_SECRET in environment variables.")
    
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
        logger.info(f"Webhook event received: type={event['type']}, id={event['id']}")
    except ValueError as e:
        logger.error(f"Stripe webhook error: Invalid payload - {str(e)}, payload={payload[:200]}")
        return Response(status_code=400, content="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Stripe webhook error: Invalid signature - {str(e)}, sig_header={sig_header}")
        return Response(status_code=400, content="Invalid signature")
    except Exception as e:
        logger.error(f"Stripe webhook unexpected error: {str(e)}, payload={payload[:200]}")
        return Response(status_code=500, content="Webhook processing failed")
    
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
        # Return 200 to acknowledge receipt - Stripe will retry infinitely on 500s
        # Log the error but don't cause retry storms
        return Response(status_code=200, content="Webhook acknowledged (processing error logged)")

# WebSocket for real-time audit logging
@app.websocket("/ws-audit-log")
async def websocket_audit_log(websocket: WebSocket, token: str = Query(None)):
    # Security: Validate token for authenticated users
    # If token is provided but invalid, reject connection (prevent token probing)
    # If no token provided, allow guest access for backward compatibility
    effective_username = "guest"
    if token:
        validated_username = verify_ws_token(token, _secret_key)
        if validated_username:
            effective_username = validated_username
        else:
            logger.warning(f"[WS_AUDIT] Invalid token rejected: {token[:20]}...")
            # Security: Reject connection if token was provided but is invalid
            # This prevents attackers from probing tokens
            await websocket.close(code=4001, reason="Invalid authentication token")
            return

    await websocket.accept()
    active_audit_websockets[effective_username] = websocket
    logger.info(f"[WS_AUDIT] ✅ Connected: '{effective_username}' (total connections: {len(active_audit_websockets)})")
    
    try:
        # Send confirmation message to client
        await websocket.send_json({
            "type": "audit_log",
            "message": f"Audit log connected for {effective_username}"
        })
        
        while True:
            # Handle incoming messages (ping/pong or status requests)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send keepalive
                try:
                    await websocket.send_json({"type": "keepalive"})
                except Exception:
                    break
    except Exception as e:
        logger.info(f"[WS_AUDIT] Disconnected: '{effective_username}' - {str(e)}")
    finally:
        active_audit_websockets.pop(effective_username, None)
        logger.info(f"[WS_AUDIT] Cleaned up: '{effective_username}' (remaining: {len(active_audit_websockets)})")

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
            raw_result: dict[str, Any] | None = await audit_contract(
                file=file, contract_address="", db=db, request=None,
                _from_queue=True, _queue_username=pending_audit.username
            )
        
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
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Upload temp file. Security: Uses authenticated session only."""
    if request is None:
        raise HTTPException(status_code=400, detail="Request object is required")

    await verify_csrf_token(request)

    # Security: Only use authenticated session username
    session_username = request.session.get("username")
    logger.debug(f"Upload-temp request: Session username={session_username}")

    if not session_username:
        logger.error("No session username for /upload-temp; authentication required")
        raise HTTPException(status_code=401, detail="Please login to continue")

    user = db.query(User).filter(User.username == session_username).first()
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
        logger.error(f"Failed to write temp file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save temporary file")
    except Exception as e:
        logger.error(f"Upload temp file failed for {session_username}: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload temporary file")

    logger.info(f"Temporary file uploaded for {session_username}: {temp_id}, size: {file_size / 1024 / 1024:.2f}MB")
    return {"temp_id": temp_id, "file_size": file_size}

@app.post("/diamond-audit")
async def diamond_audit(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Diamond audit endpoint. Security: Uses authenticated session only."""
    await verify_csrf_token(request)

    # Security: Only use authenticated session username, never trust query params
    session_username = request.session.get("username") if request is not None else None
    logger.debug(f"Diamond-audit request: Session username={session_username}")

    if not session_username:
        logger.error("No session username for /diamond-audit; authentication required")
        raise HTTPException(status_code=401, detail="Please login to continue")

    user = db.query(User).filter(User.username == session_username).first()
    if not user:
        # Auto-recover: User exists in session but not in DB (ephemeral DB was wiped)
        logger.warning(f"Diamond-audit: User {session_username} not found in database - auto-creating")
        session_email = request.session.get("user", {}).get("email")
        if not session_email:
            session_email = session_username if "@" in session_username else f"{session_username}@placeholder.local"

        user = User(
            username=session_username,
            email=session_email,
            tier="free",
            audit_history="[]",
            last_reset=datetime.now(timezone.utc),
            has_diamond=False
        )
        try:
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info(f"[AUTO-RECOVERY] Created user {session_username} during diamond-audit")
        except Exception as e:
            db.rollback()
            logger.error(f"[AUTO-RECOVERY] Failed to create user {session_username}: {e}")
            raise HTTPException(status_code=500, detail="Unable to process audit - please try logging in again")

    try:
        code_bytes = await file.read()
        file_size = len(code_bytes)

        if file_size == 0:
            logger.error(f"Empty file uploaded for {session_username}")
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        if file_size > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 50MB limit")

        overage_cost = usage_tracker.calculate_diamond_overage(file_size)
        logger.info(f"Preparing Diamond audit for {session_username} with overage ${overage_cost / 100:.2f} for file size {file_size / 1024 / 1024:.2f}MB")

        if user.has_diamond:
            # Process audit directly
            new_file = UploadFile(filename=file.filename, file=BytesIO(code_bytes), size=file_size)
            result = cast(dict[str, Any], await audit_contract(
                file=new_file, contract_address="", db=db, request=request
            ) or {})

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
                    logger.info(f"Reported {overage_mb:.2f}MB overage for {session_username} to Stripe post-audit")
                except Exception as e:
                    logger.error(f"Failed to report overage for {session_username}: {e}")

            return result
        else:
            # Persist to PendingAudit
            pending_id = str(uuid.uuid4())
            temp_dir = os.path.join(DATA_DIR, "temp_files")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"{pending_id}.sol")

            with open(temp_path, "wb") as f:
                f.write(code_bytes)

            pending_audit = PendingAudit(id=pending_id, username=session_username, temp_path=temp_path)
            db.add(pending_audit)
            db.commit()

            if user.tier == "pro":
                line_items = [{"price": STRIPE_PRICE_DIAMOND, "quantity": 1}]
            else:
                line_items = [{"price": STRIPE_PRICE_PRO, "quantity": 1}, {"price": STRIPE_PRICE_DIAMOND, "quantity": 1}]

            if not STRIPE_API_KEY:
                logger.error(f"Stripe checkout creation failed for {session_username} Diamond add-on: STRIPE_API_KEY not set")
                os.unlink(temp_path)
                db.delete(pending_audit)
                db.commit()
                raise HTTPException(status_code=503, detail="Payment processing unavailable")

            if request is not None:
                base_url = f"{request.url.scheme}://{request.url.netloc}"
            else:
                base_url = APP_BASE_URL

            success_url = f"{base_url}/ui?upgrade=success&audit=complete&pending_id={urllib.parse.quote(pending_id)}"
            cancel_url = f"{base_url}/ui"

            session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=line_items,
                mode="subscription",
                success_url=success_url,
                cancel_url=cancel_url,
                metadata={"pending_id": pending_id, "username": session_username, "audit_type": "diamond_overage"}
            )

            logger.info(f"Redirecting {session_username} to Stripe checkout for Diamond add-on")
            return {"session_url": session.url}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Diamond audit error for {session_username}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred processing your request")

@app.get("/pending-status/{pending_id}")
async def pending_status(pending_id: str, request: Request, db: Session = Depends(get_db)):
    # Security: Validate ownership - user can only check their own pending audits
    session_username = request.session.get("username")
    if not session_username:
        raise HTTPException(status_code=401, detail="Authentication required")

    pending_audit = db.query(PendingAudit).filter(
        PendingAudit.id == pending_id,
        PendingAudit.username == session_username  # Ownership check prevents IDOR
    ).first()
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
):
    """Complete diamond audit after Stripe payment. Security: Uses session + Stripe metadata verification."""
    session_username = request.session.get("username") if request is not None else None
    logger.debug(f"Complete-diamond-audit: Session username={session_username}, session_id={session_id}, temp_id={temp_id}")

    if not session_username:
        logger.error("No session username for /complete-diamond-audit; authentication required")
        return RedirectResponse(url="/auth?redirect_reason=no_username")

    user = db.query(User).filter(User.username == session_username).first()
    if not user:
        logger.error(f"User {session_username} not found for /complete-diamond-audit")
        return RedirectResponse(url="/auth?redirect_reason=user_not_found")

    if not STRIPE_API_KEY:
        logger.error(f"Complete diamond audit failed for {session_username}: STRIPE_API_KEY not set")
        return RedirectResponse(url="/ui?upgrade=error&message=Payment%20processing%20unavailable")

    try:
        session = stripe.checkout.Session.retrieve(session_id)

        # Security: Verify the Stripe session belongs to this user
        stripe_username = session.metadata.get("username") if session.metadata else None
        if stripe_username and stripe_username != session_username:
            logger.warning(f"Stripe session username mismatch: session={session_username}, stripe={stripe_username}")
            return RedirectResponse(url="/ui?upgrade=error&message=Session%20mismatch")

        if session.payment_status == "paid":
            # Security: Validate temp_id is a valid UUID to prevent path traversal
            try:
                uuid.UUID(temp_id)  # Raises ValueError if not valid UUID
            except ValueError:
                logger.warning(f"Invalid temp_id format attempted by {session_username}: {temp_id[:50]}")
                raise HTTPException(status_code=400, detail="Invalid temp_id format")

            # Use basename to strip any directory components
            safe_temp_id = os.path.basename(temp_id)
            temp_path = os.path.join(DATA_DIR, "temp_files", f"{safe_temp_id}.sol")

            # Verify resolved path is within temp_files directory (prevent symlink attacks)
            real_path = os.path.realpath(temp_path)
            allowed_dir = os.path.realpath(os.path.join(DATA_DIR, "temp_files"))
            if not real_path.startswith(allowed_dir + os.sep):
                logger.warning(f"Path traversal attempt by {session_username}: {temp_id}")
                raise HTTPException(status_code=403, detail="Access denied")

            if not os.path.exists(real_path):
                raise HTTPException(status_code=404, detail="Temporary file not found")

            # Process audit with guaranteed temp file cleanup
            try:
                file_size = os.path.getsize(real_path)
                with open(real_path, "rb") as f:
                    file = UploadFile(filename="temp.sol", file=f, size=file_size)
                    _result: dict[str, Any] | None = await audit_contract(
                        file=file, contract_address=None, db=db, request=request
                    )
                logger.info(f"Diamond audit completed for {session_username} after payment")
            finally:
                # Always clean up temp file, even if audit fails
                if os.path.exists(real_path):
                    try:
                        os.unlink(real_path)
                    except Exception as cleanup_err:
                        logger.warning(f"Failed to cleanup temp file {real_path}: {cleanup_err}")
            return RedirectResponse(url="/ui?upgrade=success")
        else:
            logger.error(f"Payment not completed for {session_username}, session_id={session_id}")
            return RedirectResponse(url="/ui?upgrade=failed")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Complete diamond audit failed for {session_username}: {e}")
        return RedirectResponse(url="/ui?upgrade=error&message=Processing%20error")

@app.get("/api/audit")
async def api_audit(
    api_key: str = Header(..., alias="X-API-Key"),
    db: Session = Depends(get_db)
):
    """
    API audit endpoint for Pro/Enterprise users.
    Security: Uses header-based API key authentication with new APIKey table.
    """
    try:
        # Check new APIKey table (supports multiple keys per user)
        api_key_obj = db.query(APIKey).filter(
            APIKey.key == api_key,
            APIKey.is_active == True
        ).first()

        if api_key_obj:
            user = db.query(User).filter(User.id == api_key_obj.user_id).first()
            if user and user.tier in ["pro", "enterprise"]:
                # Update last_used_at for tracking
                api_key_obj.last_used_at = datetime.now()
                db.commit()
                logger.info(f"API audit endpoint accessed by {user.username} via APIKey")
                return {"message": "API audit endpoint (Pro/Enterprise tier)", "user": user.username}

        # Fallback: check legacy user.api_key field for backward compatibility
        user = db.query(User).filter(User.api_key == api_key).first()
        if user and user.tier in ["pro", "enterprise"]:
            logger.info(f"API audit endpoint accessed by {user.username} via legacy api_key")
            return {"message": "API audit endpoint (Pro/Enterprise tier)", "user": user.username}

        raise HTTPException(status_code=403, detail="API access requires Pro/Enterprise tier and valid API key")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API audit error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/ws-token")
async def get_ws_token(request: Request):
    """Get a signed WebSocket token for authenticated users."""
    session_username = request.session.get("username")
    if not session_username:
        raise HTTPException(status_code=401, detail="Authentication required")

    token = generate_ws_token(session_username, _secret_key)
    return {"token": token, "expires_in": WS_TOKEN_EXPIRY_SECONDS}

@app.post("/mint-nft")
async def mint_nft(request: Request, db: Session = Depends(get_db)):
    """Mint NFT for enterprise users. Security: Uses authenticated session only."""
    await verify_csrf_token(request)

    # Security: Only use authenticated session username
    session_username = request.session.get("username")
    if not session_username:
        raise HTTPException(status_code=401, detail="Authentication required")

    user = db.query(User).filter(User.username == session_username).first()
    if not user or user.tier != "enterprise":
        raise HTTPException(status_code=403, detail="NFT mint requires Enterprise tier")

    token_id = secrets.token_hex(8)
    logger.info(f"Minted NFT for {session_username}: token_id={token_id}")
    return {"token_id": token_id}

# Removed: /oauth-google stub endpoint (was using placeholder client ID)

@app.post("/refer")
async def refer(request: Request, link: str = Query(...), db: Session = Depends(get_db)):
    await verify_csrf_token(request)
    logger.info(f"Referral tracked for link: {link}")
    return {"message": "Referral tracked"}

@app.get("/upgrade")
async def upgrade_page():
    try:
        logger.debug("Upgrade page accessed")
        return {"message": "Upgrade at /ui for Developer ($99/mo), Team ($349/mo), or Enterprise ($1,499/mo)."}
    except Exception as e:
        logger.error(f"Upgrade page error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/facets/{contract_address}")
async def get_facets(contract_address: str, request: Request, api_key: str = Header(None, alias="X-API-Key"), db: Session = Depends(get_db)) -> dict[str, Any]:
    """
    Get diamond proxy facets. Supports both session auth (web UI) and API key auth.
    Security: No username query param - uses session or API key only.
    """
    try:
        if w3 is None or not w3.is_address(contract_address):
            logger.error(f"Invalid Ethereum address or Web3 not initialized: {contract_address}")
            raise HTTPException(status_code=400, detail="Invalid Ethereum address or Web3 not initialized")

        # Security: Only use authenticated session or API key - no query param bypass
        session_username = request.session.get("username")
        user = None

        # Try session auth first
        if session_username:
            user = db.query(User).filter(User.username == session_username).first()
            logger.debug(f"Facets request for {contract_address} by session user {session_username}")

        # If no session user, try API key auth
        if not user and api_key:
            api_key_obj = db.query(APIKey).filter(APIKey.key == api_key, APIKey.is_active == True).first()
            if api_key_obj:
                user = db.query(User).filter(User.id == api_key_obj.user_id).first()
                if user:
                    api_key_obj.last_used_at = datetime.now()
                    db.commit()
                    logger.debug(f"Facets request for {contract_address} by API key user {user.username}")

        if not user:
            raise HTTPException(status_code=401, detail="Authentication required. Please sign in or provide API key.")

        current_tier = user.tier
        has_diamond = user.has_diamond

        if current_tier not in ["pro", "enterprise", "diamond"] and not has_diamond:
            logger.warning(f"Facet preview denied for {user.username} (tier: {current_tier}, has_diamond: {has_diamond})")
            raise HTTPException(status_code=403, detail="Facet preview requires Pro/Enterprise tier. Upgrade at /ui.")

        if not os.getenv("INFURA_PROJECT_ID"):
            logger.error(f"Facet fetch failed for {user.username}: INFURA_PROJECT_ID not set")
            raise HTTPException(status_code=503, detail="On-chain analysis unavailable")
        
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
        raise HTTPException(status_code=500, detail="Internal server error")

## Section 4.6 Main Audit Endpoint

# ============================================================================
# CERTORA JOB MANAGEMENT API
# ============================================================================
# These endpoints allow users to:
# 1. Start Certora verification separately from their main audit
# 2. View their Certora job history
# 3. Use cached Certora results in audits
# 4. Delete cached results to force re-verification

import hashlib

def compute_contract_hash(content: str) -> str:
    """Compute SHA256 hash of contract content for cache lookup."""
    return hashlib.sha256(content.encode()).hexdigest()


def get_cached_certora_result(db: Session, user_id: int, contract_hash: str) -> Optional[CertoraJob]:
    """
    Look up cached Certora result for a user + contract combination.
    Returns the most recent completed job, or None if not found.
    """
    job = db.query(CertoraJob).filter(
        CertoraJob.user_id == user_id,
        CertoraJob.contract_hash == contract_hash,
        CertoraJob.status == "completed"
    ).order_by(CertoraJob.completed_at.desc()).first()
    return job


def get_pending_certora_job(db: Session, user_id: int, contract_hash: str) -> Optional[CertoraJob]:
    """
    Check if there's a pending/running Certora job for this contract.
    Prevents duplicate job submissions.
    """
    job = db.query(CertoraJob).filter(
        CertoraJob.user_id == user_id,
        CertoraJob.contract_hash == contract_hash,
        CertoraJob.status.in_(["pending", "running"])
    ).first()
    return job


@app.get("/api/certora/jobs")
async def get_certora_jobs(
    request: Request,
    db: Session = Depends(get_db)
) -> dict[str, Any]:
    """
    Get user's Certora job history.
    Shows all verification jobs with their status.
    """
    user = await get_authenticated_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    jobs = db.query(CertoraJob).filter(
        CertoraJob.user_id == user.id
    ).order_by(CertoraJob.created_at.desc()).limit(20).all()

    return {
        "jobs": [
            {
                "id": job.id,
                "job_id": job.job_id,
                "job_url": job.job_url,
                "contract_name": job.contract_name,
                "contract_hash": job.contract_hash[:16] + "...",  # Truncated for display
                "status": job.status,
                "rules_verified": job.rules_verified,
                "rules_violated": job.rules_violated,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None
            }
            for job in jobs
        ]
    }


@app.get("/api/certora/job/{job_id}")
async def get_certora_job_status(
    job_id: str,
    request: Request,
    db: Session = Depends(get_db)
) -> dict[str, Any]:
    """
    Get status of a specific Certora job.
    Returns full results if completed.
    """
    user = await get_authenticated_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    job = db.query(CertoraJob).filter(
        CertoraJob.job_id == job_id,
        CertoraJob.user_id == user.id
    ).first()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = {
        "id": job.id,
        "job_id": job.job_id,
        "job_url": job.job_url,
        "contract_name": job.contract_name,
        "status": job.status,
        "rules_verified": job.rules_verified,
        "rules_violated": job.rules_violated,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None
    }

    # Include full results if completed
    if job.status == "completed" and job.results_json:
        try:
            result["results"] = json.loads(job.results_json)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning(f"Failed to parse results_json for job {job.id}: {e}")
            result["results"] = []

    return result


@app.delete("/api/certora/job/{job_id}")
async def delete_certora_job(
    job_id: str,
    request: Request,
    db: Session = Depends(get_db)
) -> dict[str, Any]:
    """
    Delete a cached Certora job.
    This forces a fresh verification on next audit.
    """
    user = await get_authenticated_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    job = db.query(CertoraJob).filter(
        CertoraJob.job_id == job_id,
        CertoraJob.user_id == user.id
    ).first()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    db.delete(job)
    db.commit()

    return {"success": True, "message": "Certora job deleted. Fresh verification will run on next audit."}


@app.post("/api/certora/start")
async def start_certora_verification_deprecated(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> dict[str, Any]:
    """
    DEPRECATED: Manual Certora submission has been replaced by the integrated audit system.

    Certora formal verification now runs automatically as part of the full audit flow.
    Use /audit/submit to start an audit - you'll receive an audit_key that can be used
    to retrieve results (including Certora verification) once the audit completes.

    Benefits of the new system:
    - Certora runs with full Slither context for better specs
    - Results are permanently stored and accessible via audit_key
    - Email notifications when audit completes
    - No need to manually poll for status
    """
    return {
        "status": "deprecated",
        "message": "Manual Certora submission has been replaced. Please use the main audit flow instead.",
        "instructions": {
            "step1": "Submit your contract via POST /audit/submit",
            "step2": "Save the audit_key returned in the response",
            "step3": "Your audit (including Certora verification) will run in the background",
            "step4": "Retrieve results anytime via GET /audit/retrieve/{audit_key}",
            "step5": "Optional: Provide notify_email parameter to receive email when complete"
        },
        "benefits": [
            "Certora runs with full Slither context for better specifications",
            "Results are permanently stored and accessible anytime",
            "Email notifications when audit completes",
            "No need to manually track job IDs"
        ]
    }


@app.post("/api/certora/poll/{job_id}")
async def poll_certora_job(
    job_id: str,
    request: Request,
    db: Session = Depends(get_db)
) -> dict[str, Any]:
    """
    Poll for Certora job results.
    This is called by the frontend or background task to update job status.
    """
    # Security: Verify CSRF token
    await verify_csrf_token(request)

    user = await get_authenticated_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    job = db.query(CertoraJob).filter(
        CertoraJob.job_id == job_id,
        CertoraJob.user_id == user.id
    ).first()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # If already completed, just return status
    if job.status == "completed":
        return {
            "status": "completed",
            "rules_verified": job.rules_verified,
            "rules_violated": job.rules_violated
        }

    # Try to fetch results from Certora
    try:
        from certora import CertoraRunner
        runner = CertoraRunner()
        result = runner._fetch_job_results(job.job_url)

        status = result.get("status", "running")

        if status in ["verified", "issues_found"]:
            # Job completed!
            job.status = "completed"
            job.completed_at = datetime.now()
            job.rules_verified = result.get("rules_verified", 0)
            job.rules_violated = result.get("rules_violated", 0)
            job.results_json = json.dumps(result)
            db.commit()

            logger.info(f"Certora job {job_id} completed: {job.rules_verified} verified, {job.rules_violated} violated")

            return {
                "status": "completed",
                "rules_verified": job.rules_verified,
                "rules_violated": job.rules_violated,
                "message": "Verification complete! Results will be used in your next audit."
            }

        elif status == "error":
            job.status = "error"
            job.results_json = json.dumps(result)
            db.commit()

            return {
                "status": "error",
                "message": result.get("error", "Verification failed")
            }

        else:
            # Still running
            job.status = "running"
            db.commit()

            return {
                "status": "running",
                "message": "Verification still in progress on Certora cloud..."
            }

    except Exception as e:
        logger.error(f"Error polling Certora job {job_id}: {e}")
        return {
            "status": "running",
            "message": "Could not check status. Job may still be running."
        }


@app.get("/api/certora/notifications")
async def get_certora_notifications(
    request: Request,
    db: Session = Depends(get_db)
) -> dict[str, Any]:
    """
    Get completed Certora jobs that haven't been notified to the user yet.
    This allows the frontend to show notifications for jobs that completed in background.
    """
    user = await get_authenticated_user(request, db)
    if not user:
        return {"notifications": []}

    # Find completed jobs that haven't been notified
    unnotified_jobs = db.query(CertoraJob).filter(
        CertoraJob.user_id == user.id,
        CertoraJob.status == "completed",
        CertoraJob.user_notified == False
    ).order_by(CertoraJob.completed_at.desc()).all()

    return {
        "notifications": [
            {
                "job_id": job.job_id,
                "contract_name": job.contract_name,
                "rules_verified": job.rules_verified,
                "rules_violated": job.rules_violated,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None
            }
            for job in unnotified_jobs
        ]
    }


@app.post("/api/certora/notifications/dismiss")
async def dismiss_certora_notifications(
    request: Request,
    body: dict = Body(...),
    db: Session = Depends(get_db)
) -> dict[str, Any]:
    """
    Mark Certora job notifications as dismissed (user has seen them).
    """
    # Security: Verify CSRF token
    await verify_csrf_token(request)

    user = await get_authenticated_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    job_ids = body.get("job_ids", [])
    if not job_ids:
        return {"success": True, "dismissed": 0}

    # Mark jobs as notified
    updated = db.query(CertoraJob).filter(
        CertoraJob.user_id == user.id,
        CertoraJob.job_id.in_(job_ids)
    ).update({CertoraJob.user_notified: True}, synchronize_session=False)

    db.commit()

    return {"success": True, "dismissed": updated}


def run_certora(temp_path: str, slither_findings: list = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Run Certora formal verification via cloud API.

    Generates CVL specifications using AI and submits to Certora Prover.
    Returns tuple of (formatted_results, raw_results) where raw_results contains job_url.

    Returns:
        tuple: (formatted_results for display, raw_results for job caching)
    """
    try:
        # Check if Certora is configured
        if not os.getenv("CERTORAKEY"):
            logger.info("Certora: CERTORAKEY not configured, skipping formal verification")
            return [{"status": "skipped", "reason": "API key not configured"}], {}

        # Import certora module (lazy import to avoid startup cost)
        from certora import CVLGenerator, CertoraRunner

        # Read contract code
        with open(temp_path, 'r') as f:
            contract_code = f.read()

        logger.info(f"Certora: Generating CVL specifications for {temp_path}")

        # Generate CVL specs using AI
        generator = CVLGenerator()
        cvl_specs = generator.generate_specs_sync(contract_code, slither_findings)

        if not cvl_specs:
            logger.warning("Certora: Could not generate CVL specifications")
            return [{"status": "skipped", "reason": "Could not generate specifications"}], {}

        logger.info(f"Certora: Generated {len(cvl_specs)} chars of CVL specs")

        # Run verification
        runner = CertoraRunner()
        results = runner.run_verification_sync(temp_path, cvl_specs)

        # Log job URL internally for debugging (not exposed to users)
        job_url = results.get("job_url")
        logger.info(f"Certora: Verification complete - {results.get('rules_verified', 0)} verified, {results.get('rules_violated', 0)} violated")
        if job_url:
            logger.info(f"Certora: Internal job URL (admin only): {job_url}")

        # Format results for report - job_url is NOT exposed to end users
        # Users see verification results but can't access Certora dashboard
        formatted_results = []

        status = results.get("status", "complete")

        # Handle "pending" status - job submitted but results not ready yet
        # This happens when verification takes longer than expected
        if status == "pending":
            logger.info("Certora: Job submitted, results pending on Certora cloud")
            formatted_results.append({
                "rule": "Formal Verification",
                "status": "pending",
                "description": "Verification job submitted to Certora cloud. Comprehensive analysis is in progress - results will be included in future audits of this contract."
            })
            # Still return success since job was submitted, include raw results for caching
            return formatted_results, results

        # Map rule names to user-friendly descriptions
        rule_descriptions = {
            "sanitycheck": "Contract functions are callable and specification is valid",
            "viewfunctionsarereadonly": "View/pure functions do not modify contract state",
            "revertpreservesstate": "Reverted transactions preserve original state",
            "nounexpectedstatechanges": "State changes are consistent and predictable",
            "transferintegrity": "Token transfers move exact amounts between accounts",
            "balancenotexceedsupply": "Individual balances never exceed total supply",
            "supplychangetracking": "Total supply changes only through authorized operations",
            "envfreefuncsstaticcheck": "Environment-free functions have correct access patterns",
            "rule_not_vacuous": "Verification rules are meaningful (not trivially satisfied)"
        }

        # Map rule names to actionable fixes when VIOLATED
        # These are mathematically proven bugs - provide specific remediation
        rule_fixes = {
            "sanitycheck": {
                "severity": "HIGH",
                "fix": "Review function visibility and ensure all public/external functions can be called without reverting unexpectedly. Check for missing initializers or invalid state.",
                "category": "Contract Initialization"
            },
            "viewfunctionsarereadonly": {
                "severity": "CRITICAL",
                "fix": "CRITICAL: View/pure function is modifying state! Remove state-changing operations from view/pure functions or change the function visibility. This violates Solidity semantics.",
                "category": "State Mutation Bug"
            },
            "revertpreservesstate": {
                "severity": "HIGH",
                "fix": "Reverted transactions are not preserving original state. Check for state changes before require/revert statements. Apply Checks-Effects-Interactions pattern.",
                "category": "State Consistency"
            },
            "nounexpectedstatechanges": {
                "severity": "HIGH",
                "fix": "Unexpected state changes detected. Audit all state-modifying operations and ensure they follow expected patterns. Check for reentrancy or storage collisions.",
                "category": "State Integrity"
            },
            "transferintegrity": {
                "severity": "CRITICAL",
                "fix": "CRITICAL: Transfer amounts are incorrect! Audit transfer logic - tokens may be created, destroyed, or misdirected. Verify: sender balance decreases exactly by amount sent, receiver increases exactly by amount received.",
                "category": "Token Safety"
            },
            "balancenotexceedsupply": {
                "severity": "CRITICAL",
                "fix": "CRITICAL: Individual balance can exceed total supply! This is a token minting/accounting bug. Review mint(), burn(), and transfer() functions for overflow or incorrect arithmetic.",
                "category": "Token Accounting"
            },
            "supplychangetracking": {
                "severity": "HIGH",
                "fix": "Total supply is changing outside of mint/burn operations. Audit all functions that modify totalSupply and ensure it's only changed through authorized mint/burn paths.",
                "category": "Token Supply"
            },
            "envfreefuncsstaticcheck": {
                "severity": "MEDIUM",
                "fix": "Environment-free function has unexpected dependencies. Review the function to ensure it doesn't rely on msg.sender, block.timestamp, or other environmental variables if marked as envfree.",
                "category": "Function Purity"
            },
            "rule_not_vacuous": {
                "severity": "LOW",
                "fix": "The verification rule may be trivially satisfied (always true). Review the specification to ensure it actually tests meaningful properties.",
                "category": "Specification Quality"
            },
            # Common CVL rule patterns
            "reentrancy": {
                "severity": "CRITICAL",
                "fix": "CRITICAL: Reentrancy vulnerability detected! Apply nonReentrant modifier, use Checks-Effects-Interactions pattern, or implement a reentrancy guard.",
                "category": "Reentrancy"
            },
            "overflow": {
                "severity": "CRITICAL",
                "fix": "CRITICAL: Arithmetic overflow/underflow possible! Use Solidity 0.8+ checked arithmetic or OpenZeppelin SafeMath for older versions.",
                "category": "Arithmetic"
            },
            "accesscontrol": {
                "severity": "HIGH",
                "fix": "Access control can be bypassed. Review onlyOwner, onlyAdmin, and role-based modifiers. Ensure authorization checks cannot be circumvented.",
                "category": "Access Control"
            },
            "ownable": {
                "severity": "HIGH",
                "fix": "Ownership-related property violated. Review transferOwnership(), renounceOwnership(), and ensure owner checks are properly enforced.",
                "category": "Ownership"
            },
            "pausable": {
                "severity": "MEDIUM",
                "fix": "Pausable mechanism can be bypassed. Ensure whenNotPaused modifier is applied to all sensitive functions and pause state is properly checked.",
                "category": "Emergency Controls"
            }
        }

        # Add verified rules with meaningful descriptions
        for rule in results.get("verified_rules", []):
            rule_name = rule.get("rule", "Property Check")
            rule_key = rule_name.lower().replace(" ", "").replace("_", "")
            description = rule_descriptions.get(rule_key, rule.get("description", f"Property '{rule_name}' mathematically proven to hold"))
            formatted_results.append({
                "rule": rule_name,
                "status": "verified",
                "description": description
            })

        # Add violations with actionable fixes
        for violation in results.get("violations", []):
            rule_name = violation.get("rule", "Property Check")
            rule_key = rule_name.lower().replace(" ", "").replace("_", "")

            # Look up fix info, with fallback for unknown rules
            fix_info = None
            for key, info in rule_fixes.items():
                if key in rule_key or rule_key in key:
                    fix_info = info
                    break

            if not fix_info:
                # Default for unknown violations
                fix_info = {
                    "severity": "HIGH",
                    "fix": f"Formal verification proved '{rule_name}' is violated. Review the related code section, audit state changes, and ensure the expected property holds.",
                    "category": "Formal Verification"
                }

            formatted_results.append({
                "rule": rule_name,
                "status": violation.get("status", "violated"),
                "severity": fix_info["severity"],
                "category": fix_info["category"],
                "description": violation.get("description", f"Verification of '{rule_name}' found potential issues"),
                "fix": fix_info["fix"],
                "proven": True  # Mark as mathematically proven (not just suspected)
            })

        # Add summary if no specific results
        if not formatted_results:
            if status == "verified":
                formatted_results.append({
                    "rule": "Contract Verification",
                    "status": "verified",
                    "description": "All formal verification checks passed successfully"
                })
            elif status in ["error", "skipped"]:
                formatted_results.append({
                    "rule": "Verification Status",
                    "status": status,
                    "description": results.get("error", "Verification could not be completed")
                })
            elif status == "timeout":
                # Timeout during submission (not polling) - actual error
                formatted_results.append({
                    "rule": "Verification Status",
                    "status": "timeout",
                    "description": "Could not connect to Certora cloud service. Please try again later."
                })
            else:
                formatted_results.append({
                    "rule": "Contract Analysis",
                    "status": status,
                    "description": f"Formal verification completed with status: {status}"
                })

        return formatted_results, results

    except Exception as e:
        logger.error(f"Certora: Verification failed with error: {e}")
        return [{"status": "error", "reason": str(e)}], {}


def format_certora_for_ai(certora_results: list) -> str:
    """
    Format Certora results with context that helps the AI understand
    what was verified, what failed, and what needs extra scrutiny.

    This is critical because:
    - Verified properties = mathematically proven safe (high confidence)
    - Violated properties = issues found by formal verification (CRITICAL)
    - Timeout/pending = couldn't verify (NEEDS MANUAL REVIEW)
    - Errors = verification failed (treat as unverified)
    """
    if not certora_results:
        return "FORMAL VERIFICATION: Not available for this tier."

    verified = []
    violated = []
    unverified = []  # timeouts, errors, pending, skipped

    for r in certora_results:
        if not isinstance(r, dict):
            continue

        status = r.get("status", "unknown")
        rule = r.get("rule", "Unknown property")
        desc = r.get("description", r.get("reason", ""))

        if status == "verified":
            verified.append(f"  ✓ {rule}: {desc}")
        elif status in ["violated", "issues_found"]:
            violated.append(f"  ✗ {rule}: {desc}")
        else:
            # timeout, pending, error, skipped, incomplete
            unverified.append(f"  ? {rule} ({status}): {desc}")

    # Build comprehensive context for AI
    lines = ["═══════════════════════════════════════════════════════════════════"]
    lines.append("FORMAL VERIFICATION ANALYSIS (Certora Prover)")
    lines.append("═══════════════════════════════════════════════════════════════════")

    if verified:
        lines.append("")
        lines.append(f"MATHEMATICALLY PROVEN SAFE ({len(verified)} properties):")
        lines.append("These properties have been formally verified and are guaranteed to hold:")
        lines.extend(verified)

    if violated:
        lines.append("")
        lines.append(f"⚠️ VIOLATIONS DETECTED ({len(violated)} issues) - CRITICAL:")
        lines.append("Formal verification found these properties are VIOLATED:")
        lines.extend(violated)
        lines.append("")
        lines.append("ACTION: These are mathematically proven bugs. Address immediately.")

    if unverified:
        lines.append("")
        lines.append(f"⚠️ UNVERIFIED PROPERTIES ({len(unverified)} properties) - NEEDS MANUAL REVIEW:")
        lines.append("These could not be formally verified (too complex, timeout, or pending):")
        lines.extend(unverified)
        lines.append("")
        lines.append("ACTION: Since formal verification could not prove safety, you MUST")
        lines.append("        analyze these areas manually with extra scrutiny. Assume")
        lines.append("        they may contain vulnerabilities until proven otherwise.")

    # Summary for AI decision making
    lines.append("")
    lines.append("═══════════════════════════════════════════════════════════════════")
    lines.append("AI GUIDANCE:")
    if violated:
        lines.append("- CRITICAL: Formal verification found actual bugs. These MUST be")
        lines.append("  reported as HIGH/CRITICAL severity vulnerabilities.")
    if unverified:
        lines.append("- WARNING: Unverified properties indicate areas that couldn't be")
        lines.append("  mathematically proven safe. Give these EXTRA scrutiny in your")
        lines.append("  manual analysis. Consider them potential attack surfaces.")
    if verified and not violated and not unverified:
        lines.append("- All analyzed properties were verified. This is strong evidence")
        lines.append("  of correctness, but continue with standard vulnerability analysis.")
    lines.append("═══════════════════════════════════════════════════════════════════")

    return "\n".join(lines)


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
                        # Update AuditResult status to processing
                        audit_result = queue_db.query(AuditResult).filter(
                            AuditResult.job_id == job.job_id
                        ).first()
                        if audit_result:
                            audit_result.status = "processing"
                            audit_result.started_at = datetime.now()
                            queue_db.commit()

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
                            db=queue_db,
                            request=None,
                            _from_queue=True,
                            _queue_username=job.username,
                            _job_id=job.job_id  # Pass job_id for phase updates
                        )

                        await audit_queue.complete(job.job_id, result)

                        # Save results to AuditResult for persistent storage
                        if audit_result:
                            audit_result.status = "completed"
                            audit_result.completed_at = datetime.now()
                            audit_result.full_report = json.dumps(result) if result else None
                            audit_result.risk_score = result.get("risk_score") if result else None

                            # Count issues by severity
                            issues = result.get("issues", []) if result else []
                            audit_result.issues_count = len(issues)
                            audit_result.critical_count = sum(1 for i in issues if i.get("severity", "").lower() == "critical")
                            audit_result.high_count = sum(1 for i in issues if i.get("severity", "").lower() == "high")
                            audit_result.medium_count = sum(1 for i in issues if i.get("severity", "").lower() == "medium")
                            audit_result.low_count = sum(1 for i in issues if i.get("severity", "").lower() == "low")

                            # Store individual tool results
                            audit_result.slither_results = json.dumps(result.get("slither_results", [])) if result else None
                            audit_result.mythril_results = json.dumps(result.get("mythril_results", [])) if result else None
                            audit_result.echidna_results = json.dumps(result.get("fuzzing_results", [])) if result else None
                            audit_result.certora_results = json.dumps(result.get("certora_results", [])) if result else None
                            audit_result.pdf_path = result.get("pdf_path") if result else None

                            # Clear file_content to save storage (no longer needed after successful completion)
                            # The full_report contains all results, so we don't need to keep the original file
                            audit_result.file_content = None

                            queue_db.commit()
                            logger.info(f"[AUDIT_RESULT] Saved completed results for audit {audit_result.audit_key[:20]}... (file_content cleared)")

                            # Send email notification if configured
                            if audit_result.notification_email and not audit_result.email_sent:
                                try:
                                    email_sent = await send_audit_email_async(
                                        to_email=audit_result.notification_email,
                                        audit_key=audit_result.audit_key,
                                        contract_name=audit_result.contract_name,
                                        risk_score=audit_result.risk_score or 0,
                                        issues_count=audit_result.issues_count,
                                        critical_count=audit_result.critical_count,
                                        high_count=audit_result.high_count,
                                        status="completed"
                                    )
                                    if email_sent:
                                        audit_result.email_sent = True
                                        queue_db.commit()
                                except Exception as email_err:
                                    logger.error(f"[AUDIT_EMAIL] Failed to send notification: {email_err}")

                        await notify_job_subscribers(job.job_id, {
                            "status": "completed",
                            "result": result,
                            "audit_key": audit_result.audit_key if audit_result else None
                        })

                    finally:
                        queue_db.close()

                except Exception as e:
                    error_msg = str(e)
                    logger.exception(f"[QUEUE_PROCESSOR] Audit failed for job {job.job_id[:8]}...")
                    await audit_queue.fail(job.job_id, error_msg)

                    # Update AuditResult with failure (use separate session with proper cleanup)
                    fail_db = None
                    try:
                        fail_db = SessionLocal()
                        audit_result_fail = fail_db.query(AuditResult).filter(
                            AuditResult.job_id == job.job_id
                        ).first()
                        if audit_result_fail:
                            audit_result_fail.status = "failed"
                            audit_result_fail.error_message = error_msg[:2000] if error_msg else "Unknown error"  # Limit error message length
                            audit_result_fail.completed_at = datetime.now()
                            fail_db.commit()

                            # Send failure notification email (separate try block)
                            if audit_result_fail.notification_email and not audit_result_fail.email_sent:
                                try:
                                    email_sent = await send_audit_email_async(
                                        to_email=audit_result_fail.notification_email,
                                        audit_key=audit_result_fail.audit_key,
                                        contract_name=audit_result_fail.contract_name,
                                        risk_score=0,
                                        issues_count=0,
                                        critical_count=0,
                                        high_count=0,
                                        status="failed"
                                    )
                                    if email_sent:
                                        audit_result_fail.email_sent = True
                                        fail_db.commit()
                                except Exception as email_err:
                                    logger.error(f"[QUEUE_PROCESSOR] Failed to send failure email: {email_err}")
                    except Exception as db_err:
                        logger.error(f"[QUEUE_PROCESSOR] Failed to update AuditResult: {db_err}")
                        if fail_db:
                            fail_db.rollback()
                    finally:
                        if fail_db:
                            fail_db.close()

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


async def recover_pending_jobs():
    """
    Recover jobs that were pending when server crashed/restarted.

    This function:
    1. Finds all AuditResult records with status 'queued' or 'processing'
    2. Resets 'processing' jobs back to 'queued' (they were interrupted)
    3. Re-adds jobs with persisted file_content back to the in-memory queue
    4. Marks jobs without file_content as failed (unrecoverable)

    Called once at startup in lifespan().
    """
    db = SessionLocal()
    recovered_count = 0
    failed_count = 0

    try:
        # Find all pending jobs (queued or processing with file content)
        pending_jobs = db.query(AuditResult).filter(
            AuditResult.status.in_(["queued", "processing"]),
        ).all()

        if not pending_jobs:
            logger.info("[RECOVERY] No pending jobs to recover")
            return

        logger.info(f"[RECOVERY] Found {len(pending_jobs)} pending jobs to recover")

        for result in pending_jobs:
            try:
                # If job was processing, reset to queued
                if result.status == "processing":
                    result.status = "queued"
                    result.started_at = None
                    result.current_phase = None
                    result.progress = 0
                    logger.info(f"[RECOVERY] Reset processing job {result.audit_key[:20]}... to queued")

                # Check if we have file content to recover
                if result.file_content:
                    # Get username from user relationship or default to guest
                    username = "guest"
                    if result.user_id:
                        user = db.query(User).filter(User.id == result.user_id).first()
                        if user:
                            username = user.username or user.email or "guest"

                    # Generate new job_id for the recovered job
                    new_job_id = str(uuid.uuid4())

                    # Submit to queue
                    job = await audit_queue.submit(
                        username=username,
                        file_content=result.file_content,
                        filename=result.contract_name,
                        tier=result.user_tier,
                        contract_address=result.contract_address
                    )

                    # Update AuditResult with new job_id
                    result.job_id = job.job_id
                    recovered_count += 1
                    logger.info(f"[RECOVERY] Recovered job {result.audit_key[:20]}... with new job_id {job.job_id[:8]}...")
                else:
                    # No file content - cannot recover, mark as failed
                    result.status = "failed"
                    result.error_message = "Recovery failed: File content was not persisted. Please re-submit your audit."
                    result.completed_at = datetime.now()
                    failed_count += 1
                    logger.warning(f"[RECOVERY] Cannot recover job {result.audit_key[:20]}... - no file content")

            except Exception as e:
                logger.error(f"[RECOVERY] Error recovering job {result.audit_key[:20]}...: {e}")
                result.status = "failed"
                result.error_message = f"Recovery error: {str(e)[:500]}"
                result.completed_at = datetime.now()
                failed_count += 1

        db.commit()
        logger.info(f"[RECOVERY] Recovery complete: {recovered_count} recovered, {failed_count} failed")

    except Exception as e:
        logger.error(f"[RECOVERY] Critical error during recovery: {e}")
        db.rollback()
    finally:
        db.close()


async def detect_stale_processing_jobs():
    """
    Background task to detect and recover jobs stuck in 'processing' state.

    Jobs that have been processing for more than 30 minutes are considered stale
    and are marked as failed. This prevents resource leaks and gives users
    clear feedback about failed audits.

    Runs every 5 minutes.
    """
    PROCESSING_TIMEOUT_MINUTES = 30
    CHECK_INTERVAL_SECONDS = 300  # 5 minutes

    while True:
        try:
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)

            db = SessionLocal()
            try:
                timeout_threshold = datetime.now() - timedelta(minutes=PROCESSING_TIMEOUT_MINUTES)

                # Find stale processing jobs
                stale_jobs = db.query(AuditResult).filter(
                    AuditResult.status == "processing",
                    AuditResult.started_at < timeout_threshold
                ).all()

                if stale_jobs:
                    logger.warning(f"[TIMEOUT] Found {len(stale_jobs)} stale processing jobs")

                    for result in stale_jobs:
                        processing_time = (datetime.now() - result.started_at).total_seconds() / 60
                        result.status = "failed"
                        result.error_message = (
                            f"Processing timeout: Job exceeded {PROCESSING_TIMEOUT_MINUTES} minute limit "
                            f"(ran for {processing_time:.1f} minutes). This may indicate a system issue. "
                            "Please try re-submitting your audit."
                        )
                        result.completed_at = datetime.now()

                        # Remove from in-memory queue processing set if present
                        if result.job_id:
                            audit_queue.processing.discard(result.job_id)
                            if result.job_id in audit_queue.jobs:
                                audit_queue.jobs[result.job_id].status = AuditStatus.FAILED
                                audit_queue.jobs[result.job_id].error = result.error_message

                        logger.warning(
                            f"[TIMEOUT] Marked job {result.audit_key[:20]}... as failed "
                            f"(processing for {processing_time:.1f} minutes)"
                        )

                    db.commit()

            finally:
                db.close()

        except asyncio.CancelledError:
            logger.info("[TIMEOUT] Stale job detector cancelled")
            break
        except Exception as e:
            logger.error(f"[TIMEOUT] Error detecting stale jobs: {e}")


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


# ============================================================================
# QUEUE API ENDPOINTS
# ============================================================================

@app.post("/audit/submit")
async def submit_audit_to_queue(
    request: Request,
    file: UploadFile = File(...),
    contract_address: str = Query(None),
    notify_email: str = Query(None),
    generate_key: bool = Query(True),
    api_key_id: int = Query(None, description="Assign audit to a specific API key (Pro/Enterprise)"),
    db: Session = Depends(get_db)
) -> dict:
    """
    Submit an audit to the queue. Returns immediately with job_id and audit_key.

    The audit_key can be used to retrieve results later without being logged in.
    This allows users to start an audit, leave the page, and return later.

    Security: Uses authenticated session only - no username query param bypass.
    Unauthenticated users can still submit as "guest" with stricter rate limits.

    Parameters:
        file: The Solidity contract file
        contract_address: Optional on-chain address for verification
        notify_email: Optional email to notify when audit completes
        generate_key: Whether to generate an audit access key (default: True)
        api_key_id: Optional API key ID to assign this audit to (Pro/Enterprise)
    """
    await verify_csrf_token(request)

    # Security: Only use authenticated session username, never trust query params
    session_username = request.session.get("username")
    userinfo = request.session.get("userinfo")
    session_email = userinfo.get("email") if userinfo else None

    # Use session auth or fall back to guest (no query param override)
    effective_username = session_username or session_email or "guest"

    # Rate limiting - stricter for guests
    client_ip = request.client.host if request.client else "unknown"
    rate_key = f"audit:{effective_username}" if effective_username != "guest" else f"audit:guest:{client_ip}"
    limit_config = RATE_LIMITS["audit_submit"] if effective_username != "guest" else RATE_LIMITS["audit_submit_guest"]

    if not await rate_limiter.is_allowed(rate_key, limit_config["max_requests"], limit_config["window_seconds"]):
        retry_after = await rate_limiter.get_retry_after(rate_key, limit_config["window_seconds"])
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)}
        )

    # Get user and tier
    user = None
    tier = "free"
    user_email = notify_email or session_email
    if effective_username != "guest":
        user = db.query(User).filter(
            (User.username == effective_username) |
            (User.email == effective_username)
        ).first()
        if user:
            tier = user.tier
            if not user_email:
                user_email = user.email

    # Validate API key assignment (Pro/Enterprise feature)
    validated_api_key_id = None
    if api_key_id is not None:
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Authentication required to assign audits to API keys"
            )
        if tier not in ["pro", "enterprise"]:
            raise HTTPException(
                status_code=403,
                detail="API key assignment requires Pro or Enterprise tier"
            )
        # Verify the API key belongs to this user and is active
        api_key = db.query(APIKey).filter(
            APIKey.id == api_key_id,
            APIKey.user_id == user.id,
            APIKey.is_active == True
        ).first()
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="Invalid API key. Key must be active and owned by you."
            )
        validated_api_key_id = api_key_id
        logger.info(f"[AUDIT_SUBMIT] Assigning audit to API key '{api_key.label}' (id={api_key_id})")

    # Pre-check file size via Content-Length to prevent DoS
    # Check Content-Length header before reading the file into memory
    content_length = request.headers.get("content-length")
    size_limit = usage_tracker.size_limits.get(tier, 250 * 1024)
    max_allowed = size_limit if tier not in ["enterprise", "diamond"] else 50 * 1024 * 1024  # 50MB max for enterprise

    if content_length:
        try:
            declared_size = int(content_length)
            # Content-Length includes form boundaries, so allow some overhead
            if declared_size > max_allowed + 10 * 1024:  # 10KB overhead for form data
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum allowed: {max_allowed / 1024:.1f}KB for {tier} tier."
                )
        except ValueError:
            pass  # Invalid Content-Length, will be caught when reading

    # Read file with size limit enforcement
    code_bytes = await file.read()
    file_size = len(code_bytes)

    if file_size == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Enforce size limit after reading (in case Content-Length was spoofed)
    if file_size > max_allowed:
        raise HTTPException(
            status_code=413,
            detail=f"File size ({file_size / 1024:.1f}KB) exceeds maximum allowed ({max_allowed / 1024:.1f}KB) for {tier} tier."
        )

    # Security: Validate file type
    is_valid, error_msg = validate_solidity_file(file.filename, code_bytes)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    # Check file size limits for tier restrictions
    if file_size > size_limit and tier not in ["enterprise", "diamond"]:
        raise HTTPException(
            status_code=400,
            detail=f"File size ({file_size / 1024:.1f}KB) exceeds {tier} tier limit ({size_limit / 1024:.1f}KB). Upgrade to continue."
        )

    # Validate contract_address if provided
    if contract_address:
        if not validate_eth_address(contract_address):
            raise HTTPException(
                status_code=400,
                detail="Invalid Ethereum address format. Must be 0x followed by 40 hex characters."
            )

    # Validate notify_email if provided
    if notify_email and not validate_email(notify_email):
        raise HTTPException(
            status_code=400,
            detail="Invalid email format for notification address."
        )

    # Sanitize filename to prevent path traversal and special characters
    safe_filename = sanitize_filename(file.filename)

    # Compute contract hash
    contract_hash = compute_contract_hash(code_bytes.decode('utf-8', errors='replace'))

    # One-File-Per-Key Policy Enforcement
    # Each API key can only be associated with ONE file (identified by contract_hash).
    # Users can re-audit the SAME file multiple times with the same key.
    # Users CANNOT audit DIFFERENT files with the same key.
    if validated_api_key_id is not None:
        existing_different_file = db.query(AuditResult).filter(
            AuditResult.api_key_id == validated_api_key_id,
            AuditResult.contract_hash != contract_hash,  # Different file
            AuditResult.status.in_(["queued", "processing", "completed"])  # Only count real audits
        ).first()

        if existing_different_file:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "one_file_per_key",
                    "message": f"This API key is already assigned to a different file: '{existing_different_file.contract_name}'. "
                               f"Each API key can only audit one file. You can re-audit the same file, or create a new API key for different files.",
                    "existing_file": existing_different_file.contract_name,
                    "existing_audit_key": existing_different_file.audit_key,
                    "api_key_label": api_key.label
                }
            )
        logger.debug(f"[AUDIT_SUBMIT] One-file-per-key check passed for api_key_id={validated_api_key_id}")

    # Submit to queue first
    job = await audit_queue.submit(
        username=effective_username,
        file_content=code_bytes,
        filename=safe_filename,
        tier=tier,
        contract_address=contract_address,
        api_key_id=validated_api_key_id  # Persist API key assignment through queue
    )

    # Generate unique audit key with retry logic for race condition safety
    audit_key = None
    if generate_key:
        from sqlalchemy.exc import IntegrityError
        max_retries = 5
        for attempt in range(max_retries):
            try:
                audit_key = generate_audit_key()
                audit_result = AuditResult(
                    user_id=user.id if user else None,
                    audit_key=audit_key,
                    contract_name=safe_filename,
                    contract_hash=contract_hash,
                    contract_address=contract_address,
                    file_content=code_bytes,  # Persist for crash recovery
                    status="queued",
                    user_tier=tier,
                    notification_email=user_email if tier in ["pro", "enterprise"] else None,
                    job_id=job.job_id,
                    api_key_id=validated_api_key_id  # Assign to API key if specified
                )
                db.add(audit_result)
                db.commit()
                logger.info(f"[AUDIT_KEY] Created audit key {audit_key[:20]}... for job {job.job_id[:8]}...")
                break
            except IntegrityError:
                db.rollback()
                audit_key = None
                logger.warning(f"[AUDIT_KEY] Key collision on attempt {attempt + 1}, retrying...")
                if attempt == max_retries - 1:
                    logger.error("[AUDIT_KEY] Failed to generate unique key after max retries")
                    # Continue without audit key - job is already in queue
            except Exception as e:
                db.rollback()
                logger.error(f"[AUDIT_KEY] Failed to save audit result: {e}")
                audit_key = None
                break  # Don't retry on other errors

    # Get initial status
    status = await audit_queue.get_status(job.job_id)

    logger.info(f"[QUEUE_SUBMIT] Job {job.job_id[:8]}... submitted by {effective_username}, position: {status['position']}")

    response = {
        "job_id": job.job_id,
        "status": "queued",
        "position": status["position"],
        "queue_length": status["queue_length"],
        "estimated_wait_seconds": status["estimated_wait_seconds"],
        "message": f"Audit queued successfully. Position: {status['position']}"
    }

    if audit_key:
        response["audit_key"] = audit_key
        response["retrieve_url"] = f"/audit/retrieve/{audit_key}"
        response["message"] += f"\n\nSave your audit key to access results later: {audit_key}"

    # Include API key assignment info in response
    if validated_api_key_id:
        response["api_key_id"] = validated_api_key_id

    return response


@app.get("/audit/status/{job_id}")
async def get_audit_status(job_id: str, request: Request) -> dict:
    """Get the current status of a queued audit job. Security: Only job owner can check status."""
    # Security: Verify ownership to prevent IDOR
    session_username = request.session.get("username")
    userinfo = request.session.get("userinfo")
    session_email = userinfo.get("email") if userinfo else None

    job = audit_queue.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Verify the job belongs to the requesting user (check username or email)
    if job.username not in [session_username, session_email, "guest"]:
        # Allow guest jobs to be checked by anyone (they're anonymous anyway)
        if job.username != "guest":
            logger.warning(f"IDOR attempt: {session_username} tried to access job {job_id} owned by {job.username}")
            raise HTTPException(status_code=403, detail="Access denied")

    status = await audit_queue.get_status(job_id)
    return status


@app.get("/audit/queue-stats")
async def get_queue_stats() -> dict:
    """Get current queue statistics (for monitoring/debugging)."""
    return audit_queue.get_stats()


# ============================================================================
# AUDIT KEY RETRIEVAL ENDPOINTS
# ============================================================================

@app.get("/audit/retrieve/{audit_key}")
async def retrieve_audit_by_key(
    audit_key: str,
    db: Session = Depends(get_db)
) -> dict:
    """
    Retrieve audit results using an audit access key.

    This endpoint does NOT require authentication - the audit key itself
    serves as the access credential. Users can bookmark this URL to
    return to their results later.

    Returns the full audit report if completed, or status if still processing.
    """
    # Validate audit key format
    if not audit_key.startswith("dga_"):
        raise HTTPException(status_code=400, detail="Invalid audit key format")

    # Look up the audit
    audit_result = db.query(AuditResult).filter(
        AuditResult.audit_key == audit_key
    ).first()

    if not audit_result:
        raise HTTPException(status_code=404, detail="Audit not found. The key may be invalid or expired.")

    # Build response based on status
    response = {
        "audit_key": audit_result.audit_key,
        "status": audit_result.status,
        "contract_name": audit_result.contract_name,
        "created_at": audit_result.created_at.isoformat() if audit_result.created_at else None,
        "user_tier": audit_result.user_tier
    }

    if audit_result.status == "queued":
        # Still in queue - provide position if available
        if audit_result.job_id:
            queue_status = await audit_queue.get_status(audit_result.job_id)
            response["queue_position"] = queue_status.get("position")
            response["estimated_wait_seconds"] = queue_status.get("estimated_wait_seconds")
        response["message"] = "Your audit is queued and will begin shortly."

    elif audit_result.status == "processing":
        # Currently being processed
        response["started_at"] = audit_result.started_at.isoformat() if audit_result.started_at else None
        response["current_phase"] = audit_result.current_phase
        response["progress"] = audit_result.progress
        response["message"] = "Your audit is in progress."

    elif audit_result.status == "completed":
        # Completed - return full results
        response["completed_at"] = audit_result.completed_at.isoformat() if audit_result.completed_at else None
        response["risk_score"] = audit_result.risk_score
        response["issues_count"] = audit_result.issues_count
        response["critical_count"] = audit_result.critical_count
        response["high_count"] = audit_result.high_count
        response["medium_count"] = audit_result.medium_count
        response["low_count"] = audit_result.low_count

        # Parse and return full report
        if audit_result.full_report:
            try:
                response["report"] = json.loads(audit_result.full_report)
            except json.JSONDecodeError:
                response["report"] = None

        # Include PDF path if available
        if audit_result.pdf_path:
            response["pdf_url"] = f"/api/reports/{os.path.basename(audit_result.pdf_path)}"

        response["message"] = "Audit completed successfully."

    elif audit_result.status == "failed":
        # Failed - return error info
        response["completed_at"] = audit_result.completed_at.isoformat() if audit_result.completed_at else None
        response["error"] = audit_result.error_message
        response["message"] = "Audit failed. Please try again or contact support."

    return response


@app.get("/api/user/audits")
async def list_user_audits(
    request: Request,
    db: Session = Depends(get_db),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
) -> dict:
    """
    List all audits for the authenticated user.

    Returns a paginated list of the user's audits with their keys and status.
    Requires authentication.
    """
    user = await get_authenticated_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Query user's audits
    query = db.query(AuditResult).filter(
        AuditResult.user_id == user.id
    ).order_by(AuditResult.created_at.desc())

    total = query.count()
    audits = query.offset(offset).limit(limit).all()

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "audits": [
            {
                "audit_key": a.audit_key,
                "contract_name": a.contract_name,
                "status": a.status,
                "risk_score": a.risk_score,
                "issues_count": a.issues_count,
                "critical_count": a.critical_count,
                "high_count": a.high_count,
                "created_at": a.created_at.isoformat() if a.created_at else None,
                "completed_at": a.completed_at.isoformat() if a.completed_at else None
            }
            for a in audits
        ]
    }


@app.post("/api/audit/resend-email/{audit_key}")
async def resend_audit_email(
    audit_key: str,
    request: Request,
    email: str = Query(..., description="Email address to send notification to"),
    db: Session = Depends(get_db)
) -> dict:
    """
    Resend completion email for a completed audit.

    Requires authentication, the audit key, and a valid email address.
    Only works for completed audits owned by the authenticated user.
    """
    # Security: Require authentication
    session_username = request.session.get("username")
    if not session_username:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Security: Verify CSRF token
    await verify_csrf_token(request)

    # Rate limiting for email resend (prevent email bombing)
    client_ip = request.client.host if request.client else "unknown"
    rate_key = f"email_resend:{session_username}:{client_ip}"
    limit = RATE_LIMITS["email_resend"]
    if not await rate_limiter.is_allowed(rate_key, limit["max_requests"], limit["window_seconds"]):
        retry_after = await rate_limiter.get_retry_after(rate_key, limit["window_seconds"])
        raise HTTPException(
            status_code=429,
            detail=f"Too many email resend requests. Try again in {retry_after // 60} minutes.",
            headers={"Retry-After": str(retry_after)}
        )

    # Validate audit key format and length
    if not audit_key or not audit_key.startswith("dga_") or len(audit_key) > 64:
        raise HTTPException(status_code=400, detail="Invalid audit key format")

    # Validate email format
    if not validate_email(email):
        raise HTTPException(status_code=400, detail="Invalid email format")

    # Get user for ownership verification
    user = db.query(User).filter(User.username == session_username).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    audit_result = db.query(AuditResult).filter(
        AuditResult.audit_key == audit_key
    ).first()

    if not audit_result:
        raise HTTPException(status_code=404, detail="Audit not found")

    # Security: Verify user owns this audit
    if audit_result.user_id != user.id:
        logger.warning(f"[AUDIT_EMAIL] User {session_username} attempted to resend email for audit owned by user_id {audit_result.user_id}")
        raise HTTPException(status_code=403, detail="Not authorized to access this audit")

    if audit_result.status != "completed":
        raise HTTPException(status_code=400, detail="Audit is not yet completed")

    # Send email
    success = await send_audit_email_async(
        to_email=email,
        audit_key=audit_result.audit_key,
        contract_name=audit_result.contract_name,
        risk_score=audit_result.risk_score or 0,
        issues_count=audit_result.issues_count,
        critical_count=audit_result.critical_count,
        high_count=audit_result.high_count,
        status="completed"
    )

    if success:
        logger.info(f"[AUDIT_EMAIL] Resent completion email to {email} for audit {audit_key[:20]}...")
        return {"success": True, "message": f"Email sent to {email}"}
    else:
        raise HTTPException(status_code=500, detail="Failed to send email. Check email configuration.")


@app.websocket("/ws/job/{job_id}")
async def websocket_job_status(websocket: WebSocket, job_id: str, token: str = Query(None)):
    """
    WebSocket endpoint for real-time job status updates.
    Client connects here after submitting an audit to receive live updates.
    Requires valid authentication token AND job ownership.
    """
    # Security: Validate token before accepting connection
    if not token:
        logger.warning(f"[WS_JOB] Connection rejected: No token provided for job {job_id[:8]}...")
        await websocket.close(code=4001, reason="Authentication required")
        return

    validated_username = verify_ws_token(token, _secret_key)
    if not validated_username:
        logger.warning(f"[WS_JOB] Connection rejected: Invalid token for job {job_id[:8]}...")
        await websocket.close(code=4003, reason="Invalid authentication token")
        return

    # Security: Verify job ownership - user can only connect to their own jobs
    job = audit_queue.jobs.get(job_id)
    if job and job.username != validated_username:
        logger.warning(f"[WS_JOB] Authorization denied: User {validated_username} attempted to access job owned by {job.username}")
        await websocket.close(code=4003, reason="Not authorized to access this job")
        return

    await websocket.accept()

    # Register this WebSocket for the job
    if job_id not in active_job_websockets:
        active_job_websockets[job_id] = []
    active_job_websockets[job_id].append(websocket)

    logger.debug(f"[WS_JOB] Client {validated_username} connected for job {job_id[:8]}...")
    
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
    api_key_id: int = Query(None, description="Assign audit to a specific API key (Pro/Enterprise)"),
    db: Session = Depends(get_db),
    request: Request = None,
    _from_queue: bool = False,
    _queue_username: str = None,  # Internal param for queue processor only
    _job_id: str = None  # Internal param for job tracking
):
    """
    Main audit endpoint. Security: Uses authenticated session only.
    Queue processor passes username via _queue_username internal param.
    """
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

    # Security: Only use session auth or internal queue username (no query param)
    # _queue_username is only set by internal queue processor
    if _from_queue and _queue_username:
        effective_username = _queue_username
    else:
        effective_username = session_username or session_email or "guest"

    logger.debug(f"Audit request: Session username={session_username}, effective={effective_username}, from_queue={_from_queue}")
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

    # Validate API key assignment (Pro/Enterprise feature)
    validated_api_key_id = None
    if api_key_id is not None and not _from_queue:
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Authentication required to assign audits to API keys"
            )
        if current_tier not in ["pro", "enterprise"]:
            raise HTTPException(
                status_code=403,
                detail="API key assignment requires Pro or Enterprise tier"
            )
        # Verify the API key belongs to this user and is active
        api_key_obj = db.query(APIKey).filter(
            APIKey.id == api_key_id,
            APIKey.user_id == user.id,
            APIKey.is_active == True
        ).first()
        if not api_key_obj:
            raise HTTPException(
                status_code=400,
                detail="Invalid API key. Key must be active and owned by you."
            )
        validated_api_key_id = api_key_id
        logger.info(f"[AUDIT] Assigning audit to API key '{api_key_obj.label}' (id={api_key_id})")

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

        # One-File-Per-Key Policy Enforcement (only for direct calls, not queue-originated)
        # Each API key can only be associated with ONE file (identified by contract_hash).
        if validated_api_key_id is not None and not _from_queue:
            early_contract_hash = compute_contract_hash(code_bytes.decode('utf-8', errors='replace'))
            existing_different_file = db.query(AuditResult).filter(
                AuditResult.api_key_id == validated_api_key_id,
                AuditResult.contract_hash != early_contract_hash,  # Different file
                AuditResult.status.in_(["queued", "processing", "completed"])  # Only count real audits
            ).first()

            if existing_different_file:
                # Clean up temp file before raising
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "one_file_per_key",
                        "message": f"This API key is already assigned to a different file: '{existing_different_file.contract_name}'. "
                                   f"Each API key can only audit one file. You can re-audit the same file, or create a new API key for different files.",
                        "existing_file": existing_different_file.contract_name,
                        "existing_audit_key": existing_different_file.audit_key,
                        "api_key_label": api_key_obj.label
                    }
                )
            logger.debug(f"[AUDIT] One-file-per-key check passed for api_key_id={validated_api_key_id}")

        # Diamond large file redirect
        if file_size > usage_tracker.size_limits.get(current_tier, 250 * 1024) and not has_diamond:
            overage_cost = usage_tracker.calculate_diamond_overage(file_size) / 100
            
            if not STRIPE_API_KEY:
                os.unlink(temp_path)
                raise HTTPException(status_code=503, detail="Payment processing unavailable: Please set STRIPE_API_KEY in environment variables.")
            
            line_items = [{"price": STRIPE_PRICE_DIAMOND, "quantity": 1}] if current_tier == "pro" else [{"price": STRIPE_PRICE_PRO, "quantity": 1}, {"price": STRIPE_PRICE_DIAMOND, "quantity": 1}]
            
            base_url = f"{request.url.scheme}://{request.url.netloc}" if request is not None else APP_BASE_URL
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
        raise HTTPException(status_code=400, detail="Failed to read uploaded file")
    
    # Tier & usage checks (DO NOT increment here - only count successful audits)
    try:
        current_tier = getattr(user, "tier", os.getenv("TIER", "free"))
        has_diamond_flag = bool(getattr(user, "has_diamond", False))
        
        # Check usage limit WITHOUT incrementing (we only increment on SUCCESS)
        tier_limits = {"free": FREE_LIMIT, "starter": STARTER_LIMIT, "beginner": STARTER_LIMIT, "pro": PRO_LIMIT, "enterprise": ENTERPRISE_LIMIT, "diamond": ENTERPRISE_LIMIT}
        current_limit = tier_limits.get(current_tier, FREE_LIMIT)
        
        # Get current audit count from user's history
        current_count = len(json.loads(user.audit_history or "[]")) if user else usage_tracker.count
        
        if current_count >= current_limit and current_tier in ["free", "starter", "beginner"]:
            logger.warning(f"Usage limit check: {effective_username} at {current_count}/{current_limit}")
            raise HTTPException(status_code=403, detail=f"Usage limit exceeded for {current_tier} tier. Limit is {current_limit}. Upgrade tier.")
        
        # Check file size limit (but don't increment count yet)
        size_limit = usage_tracker.size_limits.get(current_tier, 250 * 1024)
        if file_size > size_limit and not has_diamond_flag and current_tier not in ["enterprise", "diamond"]:
            raise HTTPException(status_code=400, detail=f"File size exceeds {current_tier} tier limit. Upgrade to continue.")
        
        logger.info(f"Audit request validated for contract {contract_address or 'uploaded'} with tier {current_tier} for user {effective_username} ({current_count}/{current_limit} audits used)")
    
    except HTTPException as e:
        # Handle size/usage limit redirects
        if file_size > usage_tracker.size_limits.get(current_tier, 250 * 1024) and not has_diamond_flag:
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
                # Use dynamic base URL from request
                size_limit_base_url = f"{request.url.scheme}://{request.url.netloc}" if request is not None else APP_BASE_URL
                session = stripe.checkout.Session.create(
                    payment_method_types=["card"],
                    line_items=[{"price": price_id, "quantity": 1}],
                    mode="subscription",
                    success_url=f"{size_limit_base_url}/complete-diamond-audit?session_id={{CHECKOUT_SESSION_ID}}&temp_id={urllib.parse.quote(os.path.basename(temp_path).split('.')[0])}&username={urllib.parse.quote(effective_username)}",
                    cancel_url=f"{size_limit_base_url}/ui",
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
                # Use dynamic base URL from request
                usage_limit_base_url = f"{request.url.scheme}://{request.url.netloc}" if request is not None else APP_BASE_URL
                session = stripe.checkout.Session.create(
                    payment_method_types=["card"],
                    line_items=[{"price": STRIPE_PRICE_STARTER, "quantity": 1}],
                    mode="subscription",
                    success_url=f"{usage_limit_base_url}/ui?session_id={{CHECKOUT_SESSION_ID}}&tier=starter",
                    cancel_url=f"{usage_limit_base_url}/ui",
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

                safe_filename = sanitize_filename(file.filename) if file.filename else "contract.sol"

                job = await audit_queue.submit(
                    username=effective_username,
                    file_content=code_bytes,
                    filename=safe_filename,
                    tier=current_tier,
                    contract_address=contract_address,
                    api_key_id=validated_api_key_id  # Persist API key assignment through queue
                )

                # Generate audit key and create AuditResult (matching /audit/submit behavior)
                # This ensures users can retrieve their audit results using the access key
                audit_key = None
                contract_hash = compute_contract_hash(code_bytes.decode('utf-8', errors='replace'))

                from sqlalchemy.exc import IntegrityError
                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        audit_key = generate_audit_key()
                        audit_result = AuditResult(
                            user_id=user.id if user else None,
                            audit_key=audit_key,
                            contract_name=safe_filename,
                            contract_hash=contract_hash,
                            contract_address=contract_address,
                            file_content=code_bytes,  # Persist for crash recovery
                            status="queued",
                            user_tier=current_tier,
                            notification_email=session_email if current_tier in ["pro", "enterprise"] else None,
                            job_id=job.job_id,
                            api_key_id=validated_api_key_id  # Assign to API key if specified
                        )
                        db.add(audit_result)
                        db.commit()
                        logger.info(f"[AUDIT_KEY] Created audit key {audit_key[:20]}... for queued job {job.job_id[:8]}...")
                        break
                    except IntegrityError:
                        db.rollback()
                        audit_key = None
                        logger.warning(f"[AUDIT_KEY] Key collision on attempt {attempt + 1}, retrying...")
                        if attempt == max_retries - 1:
                            logger.error("[AUDIT_KEY] Failed to generate unique key after max retries")
                    except Exception as e:
                        db.rollback()
                        logger.error(f"[AUDIT_KEY] Failed to save audit result: {e}")
                        audit_key = None
                        break

                status = await audit_queue.get_status(job.job_id)

                response = {
                    "queued": True,
                    "job_id": job.job_id,
                    "status": "queued",
                    "position": status["position"],
                    "queue_length": status["queue_length"],
                    "estimated_wait_seconds": status["estimated_wait_seconds"],
                    "message": f"Server busy. Audit queued at position {status['position']}."
                }

                if audit_key:
                    response["audit_key"] = audit_key
                    response["retrieve_url"] = f"/audit/retrieve/{audit_key}"
                    response["message"] += f"\n\nSave your Access Key to retrieve results: {audit_key}"

                if validated_api_key_id:
                    response["api_key_id"] = validated_api_key_id

                return response
            
            # Have capacity - increment counter and proceed
            active_audit_count += 1
            logger.info(f"[ADMISSION] Capacity available ({active_audit_count}/{MAX_CONCURRENT_AUDITS}), running immediately for {effective_username}")

            # Generate audit_key EARLY so user can save it while audit is processing
            # This allows users to leave and retrieve results later even if audit takes a long time
            safe_filename = sanitize_filename(file.filename) if file.filename else "contract.sol"
            early_contract_hash = compute_contract_hash(code_bytes.decode('utf-8', errors='replace'))

            from sqlalchemy.exc import IntegrityError
            immediate_audit_key = None
            immediate_audit_result_id = None
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    immediate_audit_key = generate_audit_key()
                    audit_result = AuditResult(
                        user_id=user.id if user else None,
                        audit_key=immediate_audit_key,
                        contract_name=safe_filename,
                        contract_hash=early_contract_hash,
                        contract_address=contract_address,
                        file_content=code_bytes,  # Persist for crash recovery
                        status="processing",  # Mark as processing, will update to completed later
                        user_tier=current_tier,
                        notification_email=session_email if current_tier in ["pro", "enterprise"] else None,
                        api_key_id=validated_api_key_id,
                    )
                    db.add(audit_result)
                    db.commit()
                    immediate_audit_result_id = audit_result.id
                    logger.info(f"[AUDIT_KEY] Created early audit key {immediate_audit_key[:20]}... for immediate audit (id={immediate_audit_result_id})")
                    break
                except IntegrityError:
                    db.rollback()
                    immediate_audit_key = None
                    logger.warning(f"[AUDIT_KEY] Key collision on attempt {attempt + 1}, retrying...")
                    if attempt == max_retries - 1:
                        logger.error("[AUDIT_KEY] Failed to generate unique key after max retries")
                except Exception as e:
                    db.rollback()
                    logger.error(f"[AUDIT_KEY] Failed to save early audit result: {e}")
                    immediate_audit_key = None
                    break

    else:
        logger.info(f"[ADMISSION] From queue - slot already acquired for {effective_username}")
        immediate_audit_key = None
        immediate_audit_result_id = None
    
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

        # Get feature flags for this tier
        tier_flags = usage_tracker.feature_flags.get(tier_for_flags, {})
        mythril_enabled = tier_flags.get("mythril", False)
        fuzzing_enabled = tier_flags.get("fuzzing", False)
        certora_enabled = tier_flags.get("certora", False)
        
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

        # Phase 2: Mythril (Pro+ only)
        mythril_results = []
        if mythril_enabled:
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

        # Phase 4: Certora Formal Verification (Enterprise only)
        # Uses caching system: check for existing results before running new verification
        certora_results = []
        certora_cache_used = False
        if certora_enabled:
            await broadcast_audit_log(effective_username, "Checking formal verification status...")
            if job_id:
                await audit_queue.update_phase(job_id, "certora", 55)
                await notify_job_subscribers(job_id, {"status": "processing", "phase": "certora", "progress": 55})
            await asyncio.sleep(0)  # Yield to event loop

            try:
                # Read contract content for hash
                with open(temp_path, 'r') as f:
                    contract_content = f.read()
                contract_hash = compute_contract_hash(contract_content)

                # Check for cached completed result first
                cached_job = get_cached_certora_result(db, user.id, contract_hash) if user else None

                # Determine if we should use cache or run fresh
                # If cache exists but was created WITHOUT Slither context, and we now have Slither findings,
                # prefer running fresh with full context for better spec generation
                use_cache = False
                if cached_job and cached_job.results_json:
                    has_slither = getattr(cached_job, 'has_slither_context', False)
                    if has_slither or not slither_findings:
                        # Cache was made with Slither context OR we don't have new Slither findings
                        use_cache = True
                    else:
                        # Cache lacks Slither context but we have findings now - run fresh
                        logger.info(f"Certora: Cached job lacks Slither context, running fresh with findings")
                        await broadcast_audit_log(effective_username, "Running enhanced Certora verification with static analysis context...")

                if use_cache and cached_job and cached_job.results_json:
                    # Use cached results!
                    await broadcast_audit_log(effective_username, "Using cached Certora verification results...")
                    certora_cache_used = True
                    try:
                        raw_results = json.loads(cached_job.results_json)
                        # Convert to expected format
                        certora_results = []
                        for rule in raw_results.get("verified_rules", []):
                            certora_results.append(rule)
                        for violation in raw_results.get("violations", []):
                            certora_results.append(violation)
                        if not certora_results:
                            # Fallback to summary
                            certora_results = [{
                                "rule": "Cached Verification",
                                "status": "verified" if cached_job.rules_violated == 0 else "issues_found",
                                "description": f"Cached result: {cached_job.rules_verified} verified, {cached_job.rules_violated} violations"
                            }]
                        logger.info(f"Certora: Using cached results from job {cached_job.job_id}")
                    except Exception as e:
                        logger.error(f"Error parsing cached Certora results: {e}")
                        certora_results = []

                else:
                    # Check for pending job
                    pending_job = get_pending_certora_job(db, user.id, contract_hash) if user else None

                    if pending_job:
                        # Job is running - poll for results
                        await broadcast_audit_log(effective_username, f"Checking pending Certora job {pending_job.job_id}...")
                        from certora import CertoraRunner
                        runner = CertoraRunner()
                        poll_result = runner._fetch_job_results(pending_job.job_url)

                        if poll_result.get("status") in ["verified", "issues_found"]:
                            # Job completed! Update DB and use results
                            pending_job.status = "completed"
                            pending_job.completed_at = datetime.now()
                            pending_job.rules_verified = poll_result.get("rules_verified", 0)
                            pending_job.rules_violated = poll_result.get("rules_violated", 0)
                            pending_job.results_json = json.dumps(poll_result)
                            db.commit()

                            certora_results = poll_result.get("verified_rules", []) + poll_result.get("violations", [])
                            await broadcast_audit_log(effective_username, f"Certora job completed: {pending_job.rules_verified} verified")
                        else:
                            # Still running - show pending status
                            certora_results = [{
                                "rule": "Formal Verification",
                                "status": "pending",
                                "description": f"Verification job {pending_job.job_id} is still running on Certora cloud. Results will be available in your next audit."
                            }]
                            await broadcast_audit_log(effective_username, "Certora job still running...")

                    else:
                        # No cache, no pending - run fresh verification
                        await broadcast_audit_log(effective_username, "Starting fresh Certora verification...")
                        certora_results, raw_results = await asyncio.to_thread(run_certora, temp_path, slither_findings)

                        # Save job to database if we got a job URL
                        job_url = raw_results.get("job_url") if raw_results else None
                        if job_url and user:
                            try:
                                # Extract job_id from URL - handle query parameters properly
                                from urllib.parse import urlparse
                                parsed_url = urlparse(job_url)
                                job_id_match = parsed_url.path.split("/")[-1] if parsed_url.path else None
                                new_job = CertoraJob(
                                    user_id=user.id,
                                    contract_hash=contract_hash,
                                    job_id=job_id_match,
                                    job_url=job_url,
                                    status="running" if raw_results.get("status") == "pending" else "completed",
                                    rules_verified=raw_results.get("rules_verified", 0),
                                    rules_violated=raw_results.get("rules_violated", 0),
                                    results_json=json.dumps(raw_results) if raw_results.get("status") != "pending" else None,
                                    has_slither_context=bool(slither_findings)  # Mark if created with Slither context
                                )
                                if raw_results.get("status") != "pending":
                                    new_job.completed_at = datetime.now()
                                db.add(new_job)
                                db.commit()
                                logger.info(f"Certora: Saved job {job_id_match} with slither_context={bool(slither_findings)}")
                            except Exception as save_err:
                                logger.error(f"Certora: Failed to save job to database: {save_err}")
                                db.rollback()

            except Exception as e:
                logger.error(f"Certora failed: {e}")
                certora_results = [{"status": "error", "reason": str(e)}]

            await asyncio.sleep(0)  # Yield to event loop
            verified_count = sum(1 for r in certora_results if r.get("status") == "verified")
            cache_msg = " (cached)" if certora_cache_used else ""
            await broadcast_audit_log(effective_username, f"Formal verification{cache_msg}: {verified_count} properties verified")

        # Results already handled above in sequential execution
        
        context = json.dumps([f if isinstance(f, dict) else getattr(f, "__dict__", str(f)) for f in slither_findings]).replace('"', '\"') if slither_findings else "No static issues found"
        context = summarize_context(context)
        
        # On-chain analysis
        details = "Uploaded Solidity code for analysis."
        onchain_analysis = None  # Store full on-chain analysis results

        if contract_address:
            if not usage_tracker.feature_flags.get(tier_for_flags, {}).get("onchain", False):
                logger.warning(f"On-chain analysis denied for {effective_username} (tier: {current_tier})")
                await broadcast_audit_log(effective_username, "On-chain analysis denied (tier restriction)")
                raise HTTPException(status_code=403, detail="On-chain analysis requires Pro tier or higher.")

            if not os.getenv("INFURA_PROJECT_ID") and not os.getenv("WEB3_PROVIDER_URL"):
                logger.error("No Web3 provider configured")
                await broadcast_audit_log(effective_username, "On-chain analysis failed: No Web3 provider")
                raise HTTPException(status_code=503, detail="On-chain analysis unavailable: Please configure Web3 provider")

            if not w3.is_address(contract_address):
                logger.error(f"Invalid Ethereum address: {contract_address}")
                await broadcast_audit_log(effective_username, "Invalid Ethereum address")
                raise HTTPException(status_code=400, detail="Invalid Ethereum address.")

            try:
                # Get on-chain analyzer instance
                analyzer = get_onchain_analyzer()
                if analyzer and analyzer.is_connected:
                    await broadcast_audit_log(effective_username, "Running on-chain analysis...")

                    # Run comprehensive on-chain analysis
                    onchain_analysis = await analyzer.analyze(
                        contract_address,
                        include_honeypot=True,
                        tier=tier_for_flags
                    )

                    # Log results
                    if onchain_analysis.get("is_contract"):
                        proxy_info = onchain_analysis.get("proxy", {})
                        backdoor_info = onchain_analysis.get("backdoors", {})
                        overall_risk = onchain_analysis.get("overall_risk", {})

                        logger.info(f"[ONCHAIN] Analysis complete: proxy={proxy_info.get('is_proxy')}, "
                                   f"backdoor_risk={backdoor_info.get('risk_level')}, "
                                   f"overall_risk={overall_risk.get('level')}")

                        await broadcast_audit_log(effective_username,
                            f"On-chain analysis complete: {overall_risk.get('level', 'N/A')} risk")

                        # Add AI context from on-chain analysis
                        if onchain_analysis.get("ai_context"):
                            details += f"\n\n{onchain_analysis['ai_context']}"
                    else:
                        logger.info(f"[ONCHAIN] Address {contract_address} is not a contract (EOA)")
                        await broadcast_audit_log(effective_username, "Address is EOA, not a contract")
                        details += f" Note: {contract_address} is an EOA, not a deployed contract."
                else:
                    # Fallback to simple bytecode fetch
                    onchain_code = w3.eth.get_code(contract_address)
                    details += f" On-chain code fetched for {contract_address} (bytecode length: {len(onchain_code)})"
                    await broadcast_audit_log(effective_username, "On-chain code fetched (basic mode)")

            except Exception as e:
                logger.error(f"On-chain analysis failed: {e}")
                await broadcast_audit_log(effective_username, f"On-chain analysis error: {str(e)[:50]}")
                # Don't fail the whole audit, just note the error
                details += f" On-chain analysis unavailable: {str(e)[:100]}"
        
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
            
            # Format Certora results with AI context (not just raw JSON)
            # This helps the AI understand what was verified vs what needs scrutiny
            formatted_certora = format_certora_for_ai(certora_results) if certora_results else "N/A (Enterprise tier only)"

            prompt = PROMPT_TEMPLATE.format(
                context=context,
                fuzzing_results=json.dumps(fuzzing_results),
                certora_results=formatted_certora,
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
                        temperature=0,  # Deterministic output for consistent results
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
                # Use pre-computed certora_results from Phase 4
                audit_json["certora_results"] = certora_results
                audit_json["formal_verification"] = certora_results

                # INTEGRATE CERTORA VIOLATIONS INTO MAIN ISSUES LIST
                # This ensures formally proven bugs appear in the main vulnerability table
                if certora_results:
                    existing_issues = audit_json.get("issues", [])
                    for cv_result in certora_results:
                        if cv_result.get("status") in ["violated", "issues_found"]:
                            # Convert Certora violation to standard issue format
                            formal_issue = {
                                "type": f"[PROVEN] {cv_result.get('rule', 'Formal Verification Issue')}",
                                "severity": cv_result.get("severity", "HIGH"),
                                "description": cv_result.get("description", "Formal verification detected a violation"),
                                "fix": cv_result.get("fix", "Review the code section related to this property"),
                                "category": cv_result.get("category", "Formal Verification"),
                                "source": "Certora Prover",
                                "proven": True,  # Flag as mathematically proven
                                "confidence": "Mathematically Proven"  # Not just high confidence
                            }
                            existing_issues.append(formal_issue)

                    audit_json["issues"] = existing_issues

                    # Update severity counts to include Certora violations
                    issues = audit_json.get("issues", [])
                    audit_json["critical_count"] = sum(1 for i in issues if i.get("severity", "").upper() == "CRITICAL")
                    audit_json["high_count"] = sum(1 for i in issues if i.get("severity", "").upper() == "HIGH")
                    audit_json["medium_count"] = sum(1 for i in issues if i.get("severity", "").upper() == "MEDIUM")
                    audit_json["low_count"] = sum(1 for i in issues if i.get("severity", "").upper() == "LOW")

            report = audit_json
            
            # Normalize predictions - handle attack_predictions or missing fields
            # normalize_audit_response creates: title, attack_vector, severity, probability
            # Frontend expects: scenario, impact
            # PDF expects: title, probability, attack_vector, financial_impact, etc.
            if "attack_predictions" in report and "predictions" not in report:
                report["predictions"] = report["attack_predictions"]
            if "predictions" not in report:
                report["predictions"] = []
            normalized_predictions = []
            for idx, p in enumerate(report.get("predictions", []), 1):
                normalized_predictions.append({
                    # Frontend compatibility
                    "scenario": p.get("title") or p.get("attack_vector") or p.get("scenario") or p.get("attack") or p.get("name") or f"Attack Scenario {idx}",
                    "impact": p.get("financial_impact") or p.get("impact") or p.get("severity") or "Significant financial risk",
                    # PDF compatibility - preserve all fields
                    "title": p.get("title") or p.get("scenario") or p.get("name") or f"Attack Scenario {idx}",
                    "severity": p.get("severity") or "High",
                    "probability": p.get("probability") or p.get("likelihood") or "Medium",
                    "attack_vector": p.get("attack_vector") or p.get("description") or "Attack vector analysis pending",
                    "financial_impact": p.get("financial_impact") or p.get("impact") or "Impact assessment pending",
                    "preconditions": p.get("preconditions") or p.get("prerequisites") or "Standard deployment conditions",
                    "time_to_exploit": p.get("time_to_exploit") or "Variable",
                    "mitigation": p.get("mitigation") or p.get("remediation") or "See recommendations",
                    "real_world_example": p.get("real_world_example") or p.get("example") or None
                })
            report["predictions"] = normalized_predictions
        
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
        pdf_path = None  # Initialize before conditional
        if usage_tracker.feature_flags.get(tier_for_flags, {}).get("reports", False):
            pdf_path = generate_compliance_pdf(
                report=report,
                username=effective_username,
                file_size=file_size,
                tier=tier_for_flags,
                compliance_data=compliance_scan if tier_for_flags in ["enterprise", "diamond"] else None
            )
        
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
        
        # INCREMENT USAGE ONLY ON SUCCESSFUL AUDIT COMPLETION
        # This ensures failed audits don't count against the user's limit
        audit_success = report.get("error") is None and report.get("risk_score") != "N/A"
        
        if audit_success:
            try:
                # Increment the usage counter now that audit succeeded
                usage_tracker.increment(file_size, effective_username, db, commit=True)
                logger.info(f"[USAGE] ✅ Audit counted for {effective_username} (successful completion)")
            except Exception as e:
                logger.error(f"[USAGE] Failed to increment usage: {e}")
        else:
            logger.info(f"[USAGE] ⚠️ Audit NOT counted for {effective_username} (failed/incomplete - not charging user)")
        
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
                except Exception as rollback_error:
                    logger.debug(f"Rollback failed (expected if no transaction): {rollback_error}")
        else:
            logger.debug("[GAMIFICATION] Guest user - stats not tracked")
        
        # Get updated audit count after successful increment
        updated_audit_count = len(json.loads(user.audit_history or "[]")) if user else usage_tracker.count
        tier_limits = {"free": FREE_LIMIT, "starter": STARTER_LIMIT, "beginner": STARTER_LIMIT, "pro": PRO_LIMIT, "enterprise": ENTERPRISE_LIMIT, "diamond": ENTERPRISE_LIMIT}
        audit_limit = tier_limits.get(current_tier, FREE_LIMIT)

        # Update or create AuditResult for immediate execution (not queued)
        # If we created an early AuditResult (with processing status), update it
        # Otherwise create a new one (fallback for edge cases)
        if not _from_queue:
            if immediate_audit_result_id:
                # UPDATE the existing AuditResult created at the start
                try:
                    audit_result = db.query(AuditResult).filter(AuditResult.id == immediate_audit_result_id).first()
                    if audit_result:
                        audit_result.status = "completed"
                        audit_result.risk_score = str(report.get("risk_score", "N/A"))
                        audit_result.issues_count = len(report.get("issues", []))
                        audit_result.pdf_path = pdf_path
                        audit_result.completed_at = datetime.now(timezone.utc)
                        audit_result.file_content = None  # Clear file_content after successful completion

                        # CRITICAL: Save full report for later retrieval via access key
                        audit_result.full_report = json.dumps(report)

                        # Count issues by severity for retrieval display
                        issues = report.get("issues", [])
                        audit_result.critical_count = sum(1 for i in issues if i.get("severity", "").lower() == "critical")
                        audit_result.high_count = sum(1 for i in issues if i.get("severity", "").lower() == "high")
                        audit_result.medium_count = sum(1 for i in issues if i.get("severity", "").lower() == "medium")
                        audit_result.low_count = sum(1 for i in issues if i.get("severity", "").lower() == "low")

                        db.commit()
                        logger.info(f"[AUDIT_KEY] Updated audit key {immediate_audit_key[:20]}... to completed (id={immediate_audit_result_id})")
                    else:
                        logger.error(f"[AUDIT_KEY] Could not find AuditResult with id={immediate_audit_result_id}")
                        immediate_audit_key = None
                except Exception as e:
                    logger.error(f"[AUDIT_KEY] Failed to update audit result: {e}")
                    try:
                        db.rollback()
                    except:
                        pass
            else:
                # Fallback: create new AuditResult if early creation failed
                safe_filename = sanitize_filename(file.filename) if file.filename else "contract.sol"
                contract_hash = compute_contract_hash(code_bytes.decode('utf-8', errors='replace'))

                # Count issues by severity
                issues = report.get("issues", [])
                critical_count = sum(1 for i in issues if i.get("severity", "").lower() == "critical")
                high_count = sum(1 for i in issues if i.get("severity", "").lower() == "high")
                medium_count = sum(1 for i in issues if i.get("severity", "").lower() == "medium")
                low_count = sum(1 for i in issues if i.get("severity", "").lower() == "low")

                from sqlalchemy.exc import IntegrityError
                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        immediate_audit_key = generate_audit_key()
                        audit_result = AuditResult(
                            user_id=user.id if user else None,
                            audit_key=immediate_audit_key,
                            contract_name=safe_filename,
                            contract_hash=contract_hash,
                            contract_address=contract_address,
                            status="completed",
                            user_tier=current_tier,
                            notification_email=session_email if current_tier in ["pro", "enterprise"] else None,
                            api_key_id=validated_api_key_id,
                            risk_score=str(report.get("risk_score", "N/A")),
                            issues_count=len(issues),
                            critical_count=critical_count,
                            high_count=high_count,
                            medium_count=medium_count,
                            low_count=low_count,
                            full_report=json.dumps(report),  # CRITICAL: Save for access key retrieval
                            pdf_path=pdf_path,
                            completed_at=datetime.now(timezone.utc)
                        )
                        db.add(audit_result)
                        db.commit()
                        logger.info(f"[AUDIT_KEY] Created audit key {immediate_audit_key[:20]}... for immediate audit (fallback)")
                        break
                    except IntegrityError:
                        db.rollback()
                        immediate_audit_key = None
                        logger.warning(f"[AUDIT_KEY] Key collision on attempt {attempt + 1}, retrying...")
                        if attempt == max_retries - 1:
                            logger.error("[AUDIT_KEY] Failed to generate unique key after max retries")
                    except Exception as e:
                        db.rollback()
                        logger.error(f"[AUDIT_KEY] Failed to save audit result: {e}")
                        immediate_audit_key = None
                        break

        response = {
            "report": report,
            "risk_score": str(report.get("risk_score", "N/A")),
            "overage_cost": overage_cost,
            "tier": current_tier,
            "audit_count": updated_audit_count,
            "audit_limit": audit_limit,
            "audits_remaining": max(0, audit_limit - updated_audit_count) if audit_limit != 9999 else "unlimited"
        }

        # Add audit key to response for immediate execution
        if immediate_audit_key:
            response["audit_key"] = immediate_audit_key
            response["retrieve_url"] = f"/audit/retrieve/{immediate_audit_key}"

        if validated_api_key_id:
            response["api_key_id"] = validated_api_key_id

        # Add on-chain analysis results if available
        if onchain_analysis and onchain_analysis.get("is_contract"):
            response["onchain_analysis"] = {
                "address": onchain_analysis.get("address"),
                "chain": onchain_analysis.get("chain"),
                "proxy": onchain_analysis.get("proxy", {}),
                "storage": {
                    "owner": onchain_analysis.get("storage", {}).get("owner"),
                    "admin": onchain_analysis.get("storage", {}).get("admin"),
                    "is_pausable": onchain_analysis.get("storage", {}).get("is_pausable"),
                    "is_paused": onchain_analysis.get("storage", {}).get("is_paused"),
                    "eth_balance": onchain_analysis.get("storage", {}).get("eth_balance"),
                    "centralization_risk": onchain_analysis.get("storage", {}).get("centralization_risk"),
                },
                "backdoors": {
                    "has_backdoors": onchain_analysis.get("backdoors", {}).get("has_backdoors"),
                    "risk_level": onchain_analysis.get("backdoors", {}).get("risk_level"),
                    "summary": onchain_analysis.get("backdoors", {}).get("summary"),
                    "dangerous_functions": onchain_analysis.get("backdoors", {}).get("dangerous_functions", [])[:5],  # Top 5
                },
                "honeypot": onchain_analysis.get("honeypot"),
                "overall_risk": onchain_analysis.get("overall_risk", {}),
            }
            # Also add to report for PDF generation
            report["onchain_analysis"] = response["onchain_analysis"]

        if pdf_path:
            # Return download URL instead of file path
            pdf_filename = os.path.basename(pdf_path)
            response["pdf_report_url"] = f"/api/reports/{pdf_filename}"
            response["pdf_filename"] = pdf_filename
        
        logger.info(f"[AUDIT_COMPLETE] {effective_username}: {updated_audit_count}/{audit_limit} audits used")
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
                db=queue_db,
                request=None,
                _from_queue=True,
                _queue_username=job.username
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
        raise HTTPException(status_code=500, detail="Failed to fetch stats")

# ============================================================================
# NEW: Admin endpoint for tier migration
# ============================================================================
@app.post("/admin/migrate-tiers")
async def migrate_tiers(request: Request, db: Session = Depends(get_db), admin_key: str = Header(None, alias="X-Admin-Key")):
    """
    Database migration endpoint to rename old tiers to new names.
    ADMIN ONLY - requires X-Admin-Key header.
    """
    expected_key = os.getenv("ADMIN_KEY", "")
    if not admin_key or not expected_key or not secrets.compare_digest(admin_key, expected_key):
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
        logger.error(f"[MIGRATION] Error: {e}")
        raise HTTPException(status_code=500, detail="Migration failed")
@app.get("/debug/echidna-env")
async def debug_echidna_env(admin_key: str = Header(None, alias="X-Admin-Key")):
    """Check Echidna installation and environment - requires X-Admin-Key header."""
    expected_key = os.getenv("ADMIN_KEY")
    if not expected_key or not admin_key or not secrets.compare_digest(admin_key, expected_key):
        raise HTTPException(status_code=403, detail="Admin access required")
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
