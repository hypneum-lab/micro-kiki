#!/usr/bin/env python3
"""Generate web backend training dataset.

Covers: FastAPI, Hono, Express/Node.js, database patterns,
auth patterns, REST API design, WebSocket.

Target: ~1500 Q&A pairs.
Output: JSONL to stdout.

Usage::

    uv run python scripts/gen_web_backend_dataset.py > /tmp/gen-web-backend.jsonl
"""
from __future__ import annotations

import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def msg(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
    }


# ---------------------------------------------------------------------------
# 1. FastAPI
# ---------------------------------------------------------------------------

def gen_fastapi() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How do you structure a production FastAPI application?",
        """A well-structured FastAPI project separates concerns into layers:

```
app/
├── main.py              # Application factory, lifespan
├── config.py            # Settings (pydantic-settings)
├── dependencies.py      # Shared dependencies (DB session, auth)
├── models/              # SQLAlchemy / SQLModel models
│   ├── __init__.py
│   ├── user.py
│   └── item.py
├── schemas/             # Pydantic request/response models
│   ├── __init__.py
│   ├── user.py
│   └── item.py
├── routers/             # API route handlers
│   ├── __init__.py
│   ├── users.py
│   └── items.py
├── services/            # Business logic
│   ├── __init__.py
│   ├── user_service.py
│   └── item_service.py
├── middleware/           # Custom middleware
│   └── logging.py
└── utils/               # Helpers
    └── security.py
```

**main.py:**
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.config import settings
from app.routers import users, items

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    yield
    # Shutdown
    await close_db()

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(items.router, prefix="/api/v1/items", tags=["items"])
```

**config.py:**
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "My API"
    DATABASE_URL: str
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    model_config = {"env_file": ".env"}

settings = Settings()
```"""
    ))

    pairs.append(msg(
        "How do you define Pydantic models for request validation and response serialization in FastAPI?",
        """```python
from pydantic import BaseModel, Field, EmailStr, field_validator
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    MODERATOR = "moderator"

# --- Request models ---
class UserCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=8)
    role: UserRole = UserRole.USER

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain an uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain a digit")
        return v

class UserUpdate(BaseModel):
    name: str | None = Field(None, min_length=2, max_length=100)
    email: EmailStr | None = None
    role: UserRole | None = None

# --- Response models ---
class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    role: UserRole
    created_at: datetime
    is_active: bool

    model_config = {"from_attributes": True}  # Allows ORM objects

class UserListResponse(BaseModel):
    items: list[UserResponse]
    total: int
    page: int
    per_page: int

# --- Usage in routes ---
from fastapi import APIRouter, HTTPException, status

router = APIRouter()

@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    existing = await db.execute(select(User).where(User.email == user.email))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )
    db_user = User(**user.model_dump(exclude={"password"}))
    db_user.hashed_password = hash_password(user.password)
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user

@router.get("/", response_model=UserListResponse)
async def list_users(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    role: UserRole | None = None,
    db: Session = Depends(get_db),
):
    query = select(User)
    if role:
        query = query.where(User.role == role)
    total = await db.scalar(select(func.count()).select_from(query.subquery()))
    users = await db.scalars(query.offset((page-1)*per_page).limit(per_page))
    return UserListResponse(
        items=users.all(), total=total, page=page, per_page=per_page,
    )
```"""
    ))

    pairs.append(msg(
        "How do you implement middleware in FastAPI?",
        """```python
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging

app = FastAPI()

# --- Built-in CORS middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://myapp.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Custom timing middleware ---
class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        response.headers["X-Process-Time"] = f"{duration:.4f}"
        return response

app.add_middleware(TimingMiddleware)

# --- Custom request ID middleware ---
import uuid

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

app.add_middleware(RequestIDMiddleware)

# --- Pure ASGI middleware (better performance) ---
class SecurityHeadersMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                extra = [
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"strict-transport-security", b"max-age=31536000"),
                ]
                message["headers"] = list(message["headers"]) + extra
            await send(message)

        await self.app(scope, receive, send_with_headers)

app.add_middleware(SecurityHeadersMiddleware)

# --- Rate limiting middleware ---
from collections import defaultdict
import asyncio

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 100, window: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window
        self.requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        now = time.time()
        # Clean old entries
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if now - t < self.window
        ]
        if len(self.requests[client_ip]) >= self.max_requests:
            return Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={"Retry-After": str(self.window)},
            )
        self.requests[client_ip].append(now)
        return await call_next(request)
```"""
    ))

    pairs.append(msg(
        "How do you implement dependency injection in FastAPI?",
        """```python
from fastapi import Depends, HTTPException, Header, Request
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from typing import Annotated

# --- Database session dependency ---
engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/db")
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

DB = Annotated[AsyncSession, Depends(get_db)]

# --- Auth dependency ---
async def get_current_user(
    authorization: str = Header(...),
    db: AsyncSession = Depends(get_db),
) -> User:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        raise HTTPException(401, "Invalid auth scheme")
    payload = decode_jwt(token)
    user = await db.get(User, payload["sub"])
    if not user or not user.is_active:
        raise HTTPException(401, "Invalid or inactive user")
    return user

CurrentUser = Annotated[User, Depends(get_current_user)]

# --- Role-based access ---
def require_role(*roles: str):
    async def check_role(user: CurrentUser):
        if user.role not in roles:
            raise HTTPException(403, f"Requires one of: {roles}")
        return user
    return Depends(check_role)

AdminUser = Annotated[User, require_role("admin")]

# --- Pagination dependency ---
class Pagination:
    def __init__(self, page: int = 1, per_page: int = 20):
        self.page = max(1, page)
        self.per_page = min(max(1, per_page), 100)
        self.offset = (self.page - 1) * self.per_page

Pages = Annotated[Pagination, Depends()]

# --- Usage in routes ---
@router.get("/users")
async def list_users(
    db: DB,
    user: CurrentUser,       # Must be authenticated
    pages: Pages,            # Pagination
    role: str | None = None, # Query filter
):
    query = select(User).offset(pages.offset).limit(pages.per_page)
    if role:
        query = query.where(User.role == role)
    result = await db.scalars(query)
    return result.all()

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    admin: AdminUser,  # Must be admin
    db: DB,
):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, "User not found")
    await db.delete(user)
    return {"ok": True}
```"""
    ))

    pairs.append(msg(
        "How do you implement background tasks in FastAPI?",
        """```python
from fastapi import BackgroundTasks, Depends
from fastapi.responses import JSONResponse

# --- Simple background task ---
async def send_welcome_email(email: str, name: str):
    # Simulates slow operation
    await asyncio.sleep(2)
    print(f"Sent welcome email to {email}")

@router.post("/users")
async def create_user(
    user: UserCreate,
    background_tasks: BackgroundTasks,
    db: DB,
):
    db_user = User(**user.model_dump())
    db.add(db_user)
    await db.commit()

    # Runs after response is sent
    background_tasks.add_task(send_welcome_email, user.email, user.name)

    return db_user

# --- For heavier work: use Celery, ARQ, or TaskIQ ---
# pip install arq
from arq import create_pool
from arq.connections import RedisSettings

# Worker function
async def process_report(ctx, report_id: int):
    # Heavy computation
    report = await generate_report(report_id)
    await save_report(report)
    await notify_user(report.user_id, f"Report {report_id} ready")

# In FastAPI
@router.post("/reports")
async def create_report(
    params: ReportParams,
    user: CurrentUser,
    db: DB,
):
    report = Report(user_id=user.id, status="pending", **params.model_dump())
    db.add(report)
    await db.commit()

    # Enqueue to worker
    redis = await create_pool(RedisSettings())
    await redis.enqueue_job("process_report", report.id)

    return {"report_id": report.id, "status": "pending"}

# --- Periodic tasks with lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start periodic cleanup task
    task = asyncio.create_task(periodic_cleanup())
    yield
    # Cancel on shutdown
    task.cancel()

async def periodic_cleanup():
    while True:
        await asyncio.sleep(3600)  # Every hour
        async with SessionLocal() as db:
            await db.execute(
                delete(Session).where(Session.expires_at < datetime.utcnow())
            )
            await db.commit()
```"""
    ))

    return pairs


# ---------------------------------------------------------------------------
# 2. Hono
# ---------------------------------------------------------------------------

def gen_hono() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "What is Hono and how do you build a REST API with it?",
        """**Hono** is a fast, lightweight web framework for edge/serverless runtimes (Cloudflare Workers, Deno, Bun, Node.js).

```typescript
import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { logger } from 'hono/logger';
import { prettyJSON } from 'hono/pretty-json';
import { HTTPException } from 'hono/http-exception';

const app = new Hono();

// Middleware
app.use('*', logger());
app.use('*', cors({ origin: ['http://localhost:3000'] }));
app.use('*', prettyJSON());

// Routes
app.get('/', (c) => c.json({ message: 'Hello Hono!' }));

// CRUD routes with type safety
interface User {
  id: string;
  name: string;
  email: string;
}

const users: User[] = [];

app.get('/api/users', (c) => {
  const page = Number(c.req.query('page') ?? 1);
  const limit = Number(c.req.query('limit') ?? 20);
  const slice = users.slice((page - 1) * limit, page * limit);
  return c.json({ data: slice, total: users.length, page, limit });
});

app.get('/api/users/:id', (c) => {
  const user = users.find(u => u.id === c.req.param('id'));
  if (!user) throw new HTTPException(404, { message: 'User not found' });
  return c.json(user);
});

app.post('/api/users', async (c) => {
  const body = await c.req.json<{ name: string; email: string }>();
  const user: User = {
    id: crypto.randomUUID(),
    name: body.name,
    email: body.email,
  };
  users.push(user);
  return c.json(user, 201);
});

app.delete('/api/users/:id', (c) => {
  const idx = users.findIndex(u => u.id === c.req.param('id'));
  if (idx === -1) throw new HTTPException(404, { message: 'Not found' });
  users.splice(idx, 1);
  return c.json({ ok: true });
});

// Error handler
app.onError((err, c) => {
  if (err instanceof HTTPException) {
    return c.json({ error: err.message }, err.status);
  }
  console.error(err);
  return c.json({ error: 'Internal Server Error' }, 500);
});

export default app;
```

**Key features:**
- Ultra-fast (Trie-based router)
- ~14KB minified, zero dependencies
- Built-in middleware: cors, jwt, basic-auth, cache, compress, etag
- First-class TypeScript with RPC-style client
- Runs on Workers, Deno, Bun, Node, AWS Lambda, Vercel"""
    ))

    pairs.append(msg(
        "How do you use Hono middleware and validators?",
        """```typescript
import { Hono } from 'hono';
import { validator } from 'hono/validator';
import { jwt } from 'hono/jwt';
import { cache } from 'hono/cache';
import { compress } from 'hono/compress';
import { timing, startTime, endTime } from 'hono/timing';

const app = new Hono();

// --- Compression ---
app.use('*', compress());

// --- Server Timing headers ---
app.use('*', timing());

// --- JWT auth middleware ---
app.use('/api/*', jwt({ secret: 'my-secret' }));

// Access JWT payload in handlers
app.get('/api/me', (c) => {
  const payload = c.get('jwtPayload');
  return c.json({ userId: payload.sub });
});

// --- Request validation ---
app.post(
  '/api/users',
  validator('json', (value, c) => {
    const { name, email } = value as { name?: string; email?: string };
    if (!name || name.length < 2) {
      return c.json({ error: 'Name must be at least 2 characters' }, 400);
    }
    if (!email || !email.includes('@')) {
      return c.json({ error: 'Valid email required' }, 400);
    }
    return { name, email };  // Validated data passed to handler
  }),
  async (c) => {
    const { name, email } = c.req.valid('json');
    // name and email are guaranteed valid here
    const user = await createUser(name, email);
    return c.json(user, 201);
  }
);

// --- Query validation ---
app.get(
  '/api/search',
  validator('query', (value, c) => {
    const q = value['q'] as string | undefined;
    if (!q || q.length < 2) {
      return c.json({ error: 'Query must be at least 2 chars' }, 400);
    }
    return { q, page: Number(value['page'] ?? 1) };
  }),
  (c) => {
    const { q, page } = c.req.valid('query');
    return c.json({ results: [], query: q, page });
  }
);

// --- Caching middleware ---
app.get(
  '/api/stats',
  cache({ cacheName: 'stats', cacheControl: 'max-age=300' }),
  (c) => c.json({ users: 1000, active: 500 })
);

// --- Custom middleware ---
const rateLimiter = (max: number, window: number) => {
  const hits = new Map<string, number[]>();
  return async (c: any, next: any) => {
    const ip = c.req.header('cf-connecting-ip') ?? 'unknown';
    const now = Date.now();
    const timestamps = (hits.get(ip) ?? []).filter(t => now - t < window * 1000);
    if (timestamps.length >= max) {
      return c.json({ error: 'Rate limit exceeded' }, 429);
    }
    timestamps.push(now);
    hits.set(ip, timestamps);
    await next();
  };
};

app.use('/api/*', rateLimiter(100, 60));  // 100 req/min
```"""
    ))

    pairs.append(msg(
        "How do you use Hono RPC for type-safe API clients?",
        """Hono's RPC feature generates a **type-safe client** from your server routes — no code generation needed.

**Server (api.ts):**
```typescript
import { Hono } from 'hono';
import { zValidator } from '@hono/zod-validator';
import { z } from 'zod';

const app = new Hono()
  .get('/api/users', async (c) => {
    const users = await getUsers();
    return c.json({ users });
  })
  .get('/api/users/:id', async (c) => {
    const user = await getUser(c.req.param('id'));
    return c.json({ user });
  })
  .post(
    '/api/users',
    zValidator('json', z.object({
      name: z.string().min(2),
      email: z.string().email(),
    })),
    async (c) => {
      const body = c.req.valid('json');
      const user = await createUser(body);
      return c.json({ user }, 201);
    }
  )
  .delete('/api/users/:id', async (c) => {
    await deleteUser(c.req.param('id'));
    return c.json({ ok: true });
  });

// Export the type
export type AppType = typeof app;
export default app;
```

**Client (uses the exported type):**
```typescript
import { hc } from 'hono/client';
import type { AppType } from './api';

const client = hc<AppType>('http://localhost:3000');

// Fully typed — autocomplete on routes, params, and responses
const res = await client.api.users.$get();
const { users } = await res.json();
// users is typed based on the server response

const res2 = await client.api.users[':id'].$get({
  param: { id: '123' },  // Type-checked param
});

const res3 = await client.api.users.$post({
  json: {
    name: 'Alice',   // Validated by Zod schema
    email: 'alice@example.com',
  },
});
```

**Benefits:**
- Zero code generation (types flow via TypeScript inference)
- Catches API contract breaks at compile time
- Works with any Hono adapter (Cloudflare, Node, Bun)
- Similar to tRPC but REST-based with standard HTTP"""
    ))

    return pairs


# ---------------------------------------------------------------------------
# 3. Express / Node.js
# ---------------------------------------------------------------------------

def gen_express() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How do you structure a production Express.js application?",
        """```
src/
├── app.ts              # Express app setup
├── server.ts           # HTTP server + graceful shutdown
├── config/
│   └── index.ts        # Environment config
├── middleware/
│   ├── errorHandler.ts
│   ├── auth.ts
│   └── validate.ts
├── routes/
│   ├── index.ts        # Route aggregator
│   ├── users.ts
│   └── items.ts
├── controllers/
│   ├── userController.ts
│   └── itemController.ts
├── services/
│   ├── userService.ts
│   └── itemService.ts
├── models/
│   └── user.ts
└── utils/
    └── logger.ts
```

**app.ts:**
```typescript
import express from 'express';
import helmet from 'helmet';
import cors from 'cors';
import { pinoHttp } from 'pino-http';
import { errorHandler } from './middleware/errorHandler';
import routes from './routes';

const app = express();

// Security
app.use(helmet());
app.use(cors({ origin: process.env.CORS_ORIGIN }));

// Parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Logging
app.use(pinoHttp());

// Routes
app.use('/api/v1', routes);

// Health check
app.get('/health', (req, res) => res.json({ status: 'ok' }));

// Error handling (must be last)
app.use(errorHandler);

export default app;
```

**server.ts (graceful shutdown):**
```typescript
import app from './app';

const PORT = process.env.PORT ?? 3000;
const server = app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

// Graceful shutdown
const shutdown = (signal: string) => {
  console.log(`${signal} received, shutting down gracefully`);
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
  // Force exit after 10s
  setTimeout(() => process.exit(1), 10000);
};

process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));
```

**middleware/errorHandler.ts:**
```typescript
import { Request, Response, NextFunction } from 'express';

export class AppError extends Error {
  constructor(public statusCode: number, message: string) {
    super(message);
    this.name = 'AppError';
  }
}

export function errorHandler(err: Error, req: Request, res: Response, next: NextFunction) {
  if (err instanceof AppError) {
    return res.status(err.statusCode).json({ error: err.message });
  }
  console.error(err);
  res.status(500).json({ error: 'Internal server error' });
}
```"""
    ))

    pairs.append(msg(
        "How do you implement middleware chaining and error handling in Express?",
        """```typescript
import { Request, Response, NextFunction, RequestHandler } from 'express';

// --- Async wrapper (catches rejected promises) ---
const asyncHandler = (fn: RequestHandler): RequestHandler =>
  (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);

// --- Validation middleware ---
import { z, ZodSchema } from 'zod';

function validate(schema: ZodSchema) {
  return (req: Request, res: Response, next: NextFunction) => {
    const result = schema.safeParse({
      body: req.body,
      query: req.query,
      params: req.params,
    });
    if (!result.success) {
      return res.status(400).json({
        error: 'Validation failed',
        details: result.error.flatten(),
      });
    }
    // Attach validated data
    req.body = result.data.body;
    next();
  };
}

// --- Auth middleware ---
async function authenticate(req: Request, res: Response, next: NextFunction) {
  const token = req.headers.authorization?.replace('Bearer ', '');\n  if (!token) return res.status(401).json({ error: 'No token' });\n\n  try {\n    const payload = await verifyJWT(token);\n    req.user = payload;\n    next();\n  } catch {\n    res.status(401).json({ error: 'Invalid token' });\n  }\n}\n\nfunction authorize(...roles: string[]) {\n  return (req: Request, res: Response, next: NextFunction) => {\n    if (!roles.includes(req.user.role)) {\n      return res.status(403).json({ error: 'Insufficient permissions' });\n    }\n    next();\n  };\n}\n\n// --- Usage with middleware chain ---\nconst createUserSchema = z.object({\n  body: z.object({\n    name: z.string().min(2),\n    email: z.string().email(),\n    role: z.enum(['user', 'admin']).default('user'),\n  }),\n});\n\nrouter.post(\n  '/users',\n  authenticate,                    // 1. Check auth\n  authorize('admin'),              // 2. Check role\n  validate(createUserSchema),      // 3. Validate body\n  asyncHandler(async (req, res) => { // 4. Handle request\n    const user = await userService.create(req.body);\n    res.status(201).json(user);\n  })\n);\n\n// Error propagation: if any middleware calls next(error),\n// Express skips to the error handler middleware\n```"""
    ))

    return pairs


# ---------------------------------------------------------------------------
# 4. Database Patterns
# ---------------------------------------------------------------------------

def gen_database() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How do you implement connection pooling for databases in web applications?",
        """**Connection pooling** reuses database connections instead of creating one per request. Critical for performance.

**PostgreSQL with Python (asyncpg + SQLAlchemy):**
```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db",
    pool_size=20,        # Base pool connections
    max_overflow=10,     # Extra connections when pool exhausted
    pool_timeout=30,     # Wait time for available connection
    pool_recycle=3600,   # Recycle connections after 1 hour
    pool_pre_ping=True,  # Test connections before use (detect stale)
    echo=False,          # SQL logging (True for debug)
)

SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
```

**Node.js with PostgreSQL (pg):**
```typescript
import { Pool } from 'pg';

const pool = new Pool({
  host: 'localhost',
  database: 'mydb',
  user: 'user',
  password: 'pass',
  max: 20,                    // Max connections
  idleTimeoutMillis: 30000,   // Close idle connections after 30s
  connectionTimeoutMillis: 5000,
});

// Use in route handlers
app.get('/users', async (req, res) => {
  const client = await pool.connect();
  try {
    const result = await client.query('SELECT * FROM users LIMIT $1', [20]);
    res.json(result.rows);
  } finally {
    client.release();  // Return to pool, don't close
  }
});

// Or without manual release
app.get('/users', async (req, res) => {
  const { rows } = await pool.query('SELECT * FROM users LIMIT $1', [20]);
  res.json(rows);
});
```

**Prisma (automatic pooling):**
```typescript
import { PrismaClient } from '@prisma/client';

// Singleton pattern — one client per process
const prisma = new PrismaClient({
  datasources: {
    db: {
      url: process.env.DATABASE_URL + '?connection_limit=20',
    },
  },
});

// Prisma manages the pool internally
const users = await prisma.user.findMany({ take: 20 });
```

**Pool sizing formula:**
```
optimal_pool_size = (2 * CPU_cores) + effective_spindle_count
```
For SSDs: ~(2 * cores + 1). For a 4-core server: pool_size = 9.

**Connection pool monitoring** — track: pool_size, active_connections, waiting_requests, connection_errors."""
    ))

    pairs.append(msg(
        "How do you handle database migrations in a web application?",
        """**Alembic (Python / SQLAlchemy):**
```bash
# Initialize
alembic init migrations

# Create migration from model changes
alembic revision --autogenerate -m "add users table"

# Apply migrations
alembic upgrade head

# Rollback one step
alembic downgrade -1
```

**Migration file example:**
```python
# migrations/versions/001_add_users.py
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('email', sa.String(255), unique=True, nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_index('ix_users_email', 'users', ['email'])

def downgrade():
    op.drop_index('ix_users_email')
    op.drop_table('users')
```

**Prisma (Node.js):**
```bash
# Create migration from schema changes
npx prisma migrate dev --name add_users

# Apply in production
npx prisma migrate deploy

# Reset database (development only)
npx prisma migrate reset
```

**Drizzle (Node.js):**
```bash
npx drizzle-kit generate
npx drizzle-kit migrate
```

**Best practices:**
1. **Never** edit applied migrations — create a new one
2. **Always** include both `upgrade` and `downgrade`
3. Test migrations on a copy of production data
4. Use transactions (Alembic: `--sql` for review)
5. Handle data migrations separately from schema migrations
6. Version control all migration files
7. Run migrations in CI/CD before deployment"""
    ))

    pairs.append(msg(
        "What are the tradeoffs between ORM and raw SQL?",
        """| Aspect | ORM | Raw SQL |\n|--------|-----|--------|\n| **Development speed** | Faster for CRUD | Slower, more boilerplate |\n| **Type safety** | Excellent (Prisma, SQLAlchemy) | Manual typing |\n| **Complex queries** | Limited, often awkward | Full SQL power |\n| **Performance** | Can generate suboptimal queries | Full control |\n| **Portability** | Multi-database support | Database-specific |\n| **Learning curve** | ORM API to learn | SQL knowledge needed |\n| **N+1 queries** | Common pitfall | Explicit joins |\n| **Migrations** | Built-in | Manual or separate tool |

**When to use ORM:**
- CRUD-heavy applications
- Rapid prototyping
- When you want type-safe queries
- Team with mixed SQL experience

**When to use raw SQL:**
- Complex analytical queries
- Performance-critical paths
- Database-specific features (window functions, CTEs, JSON operators)
- Bulk operations

**Best approach — hybrid:**
```python
# SQLAlchemy: ORM for simple queries
users = await db.scalars(select(User).where(User.is_active == True))

# Raw SQL for complex queries
result = await db.execute(text(\"\"\"
    WITH active_users AS (
        SELECT user_id, COUNT(*) as order_count
        FROM orders
        WHERE created_at > :since
        GROUP BY user_id
    )
    SELECT u.name, u.email, au.order_count
    FROM users u
    JOIN active_users au ON u.id = au.user_id
    ORDER BY au.order_count DESC
    LIMIT :limit
\"\"\"), {"since": last_month, "limit": 10})
```

```typescript
// Prisma: ORM for standard queries
const users = await prisma.user.findMany({ where: { isActive: true } });

// Raw SQL for complex queries
const topUsers = await prisma.$queryRaw`
  SELECT u.name, COUNT(o.id) as order_count
  FROM users u JOIN orders o ON u.id = o.user_id
  WHERE o.created_at > ${lastMonth}
  GROUP BY u.name
  ORDER BY order_count DESC
  LIMIT 10
`;
```"""
    ))

    return pairs


# ---------------------------------------------------------------------------
# 5. Auth Patterns
# ---------------------------------------------------------------------------

def gen_auth() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How do you implement JWT authentication in a web API?",
        """**JWT flow:**
1. Client sends credentials (login)
2. Server validates, creates JWT (access + refresh tokens)
3. Client stores tokens, sends access token with each request
4. Server validates JWT on protected routes

**FastAPI implementation:**
```python
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

SECRET_KEY = "your-secret-key"  # From env/vault
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE = 30  # minutes
REFRESH_TOKEN_EXPIRE = 7  # days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

def create_tokens(user_id: str, role: str) -> dict:
    now = datetime.utcnow()
    access_payload = {
        "sub": user_id,
        "role": role,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE),
        "type": "access",
    }
    refresh_payload = {
        "sub": user_id,
        "exp": now + timedelta(days=REFRESH_TOKEN_EXPIRE),
        "type": "refresh",
    }
    return {
        "access_token": jwt.encode(access_payload, SECRET_KEY, ALGORITHM),
        "refresh_token": jwt.encode(refresh_payload, SECRET_KEY, ALGORITHM),
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE * 60,
    }

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            raise HTTPException(401, "Invalid token type")
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(401, "Invalid token")
    except JWTError:
        raise HTTPException(401, "Could not validate token")

    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(401, "User not found")
    return user

# Routes
@router.post("/auth/token")
async def login(form: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(401, "Invalid credentials")
    return create_tokens(str(user.id), user.role)

@router.post("/auth/refresh")
async def refresh(refresh_token: str):
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(401, "Invalid token type")
        return create_tokens(payload["sub"], payload.get("role", "user"))
    except JWTError:
        raise HTTPException(401, "Invalid refresh token")
```

**Security best practices:**
- Use short-lived access tokens (15-30 min)
- Store refresh tokens securely (httpOnly cookie or secure storage)
- Implement token revocation (blacklist or token versioning)
- Use RS256 (asymmetric) for microservices (public key validation)
- Never store sensitive data in JWT payload (it's base64, not encrypted)"""
    ))

    pairs.append(msg(
        "How do you implement OAuth2 / OpenID Connect in a web app?",
        """**OAuth2 Authorization Code Flow** (most secure for web apps):

```
1. User clicks "Login with Google"
2. Redirect to provider: GET /authorize?client_id=...&redirect_uri=...&scope=openid+email
3. User authenticates with provider
4. Provider redirects back: GET /callback?code=AUTH_CODE
5. Server exchanges code for tokens: POST /token (server-to-server)
6. Server gets user info from ID token or /userinfo endpoint
```

**FastAPI implementation with Authlib:**
```python
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config

oauth = OAuth()
oauth.register(
    name='google',
    client_id='YOUR_CLIENT_ID',
    client_secret='YOUR_CLIENT_SECRET',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

@router.get("/auth/google")
async def google_login(request: Request):
    redirect_uri = request.url_for("google_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)

@router.get("/auth/google/callback")
async def google_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    userinfo = token.get("userinfo")

    # Find or create user
    user = await get_or_create_user(
        email=userinfo["email"],
        name=userinfo.get("name", ""),
        provider="google",
        provider_id=userinfo["sub"],
    )

    # Issue your own JWT
    tokens = create_tokens(str(user.id), user.role)
    # Redirect to frontend with token
    return RedirectResponse(
        f"http://localhost:3000/auth/callback?token={tokens['access_token']}"
    )
```

**Node.js with Passport:**
```typescript
import passport from 'passport';
import { Strategy as GoogleStrategy } from 'passport-google-oauth20';

passport.use(new GoogleStrategy({
    clientID: process.env.GOOGLE_CLIENT_ID!,
    clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    callbackURL: '/auth/google/callback',
  },
  async (accessToken, refreshToken, profile, done) => {
    const user = await findOrCreateUser({
      email: profile.emails?.[0]?.value,
      name: profile.displayName,
      provider: 'google',
      providerId: profile.id,
    });
    done(null, user);
  }
));

app.get('/auth/google', passport.authenticate('google', { scope: ['email', 'profile'] }));
app.get('/auth/google/callback',
  passport.authenticate('google', { session: false }),
  (req, res) => {
    const token = createJWT(req.user);
    res.redirect(`http://localhost:3000/auth/callback?token=${token}`);
  }
);
```

**Key concepts:**
- **PKCE** (Proof Key for Code Exchange): Required for public clients (SPAs, mobile apps)
- **ID Token**: JWT containing user identity claims (OpenID Connect)
- **Access Token**: For accessing APIs (can be opaque or JWT)
- **State parameter**: CSRF protection for the redirect flow
- **Nonce**: Replay attack protection in ID tokens"""
    ))

    pairs.append(msg(
        "How do you implement RBAC (Role-Based Access Control) in a web API?",
        """```python
from enum import Enum
from functools import wraps
from fastapi import Depends, HTTPException

# --- Define roles and permissions ---
class Role(str, Enum):
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"

class Permission(str, Enum):
    READ_USERS = "read:users"
    WRITE_USERS = "write:users"
    DELETE_USERS = "delete:users"
    READ_ITEMS = "read:items"
    WRITE_ITEMS = "write:items"
    MANAGE_ROLES = "manage:roles"

# Role → Permissions mapping
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.VIEWER: {
        Permission.READ_USERS,
        Permission.READ_ITEMS,
    },
    Role.EDITOR: {
        Permission.READ_USERS,
        Permission.WRITE_USERS,
        Permission.READ_ITEMS,
        Permission.WRITE_ITEMS,
    },
    Role.ADMIN: set(Permission),  # All permissions
}

# --- Permission checker dependency ---
def require_permissions(*permissions: Permission):
    async def check(user: CurrentUser):
        user_perms = ROLE_PERMISSIONS.get(user.role, set())
        missing = set(permissions) - user_perms
        if missing:
            raise HTTPException(
                403,
                f"Missing permissions: {', '.join(p.value for p in missing)}"
            )
        return user
    return Depends(check)

# --- Usage in routes ---
@router.get("/users")
async def list_users(
    user: Annotated[User, require_permissions(Permission.READ_USERS)],
    db: DB,
):
    return await db.scalars(select(User))

@router.post("/users")
async def create_user(
    data: UserCreate,
    user: Annotated[User, require_permissions(Permission.WRITE_USERS)],
    db: DB,
):
    new_user = User(**data.model_dump())
    db.add(new_user)
    await db.commit()
    return new_user

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    user: Annotated[User, require_permissions(Permission.DELETE_USERS)],
    db: DB,
):
    target = await db.get(User, user_id)
    if not target:
        raise HTTPException(404)
    await db.delete(target)
    return {"ok": True}

# --- Resource-level permissions ---
@router.put("/items/{item_id}")
async def update_item(
    item_id: int,
    data: ItemUpdate,
    user: Annotated[User, require_permissions(Permission.WRITE_ITEMS)],
    db: DB,
):
    item = await db.get(Item, item_id)
    if not item:
        raise HTTPException(404)
    # Owner or admin can edit
    if item.owner_id != user.id and user.role != Role.ADMIN:
        raise HTTPException(403, "Can only edit own items")
    for k, v in data.model_dump(exclude_unset=True).items():
        setattr(item, k, v)
    await db.commit()
    return item
```"""
    ))

    return pairs


# ---------------------------------------------------------------------------
# 6. REST API Design
# ---------------------------------------------------------------------------

def gen_rest_design() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "What are the best practices for REST API error responses?",
        """Use consistent error response format with machine-readable codes and human-readable messages.

**Standard error format:**
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": [
      {
        "field": "email",
        "message": "Must be a valid email address",
        "code": "INVALID_FORMAT"
      },
      {
        "field": "age",
        "message": "Must be at least 18",
        "code": "MIN_VALUE"
      }
    ]
  },
  "request_id": "req_abc123"
}
```

**HTTP status codes:**
| Code | Meaning | When to use |
|------|---------|-------------|
| 400 | Bad Request | Malformed request, validation errors |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Authenticated but insufficient permissions |
| 404 | Not Found | Resource doesn't exist |
| 409 | Conflict | Duplicate resource, state conflict |
| 422 | Unprocessable Entity | Valid syntax but semantic error |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Unhandled server error |
| 503 | Service Unavailable | Temporary downtime, maintenance |

**Implementation:**
```python
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

class APIError(Exception):
    def __init__(self, code: str, message: str,
                 status: int = 400, details: list = None):
        self.code = code
        self.message = message
        self.status = status
        self.details = details or []

@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    return JSONResponse(
        status_code=exc.status,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            },
            "request_id": getattr(request.state, 'request_id', None),
        },
    )

# Usage
raise APIError("USER_EXISTS", "Email already registered", 409)
raise APIError("VALIDATION_ERROR", "Invalid input", 422, [
    {"field": "email", "message": "Invalid format"},
])
```

**Don'ts:**
- Don't return 200 with `{\"success\": false}` — use proper status codes
- Don't expose stack traces in production
- Don't use generic error messages — be specific
- Don't return different error formats for different endpoints"""
    ))

    pairs.append(msg(
        "How do you implement pagination and filtering in a REST API?",
        """**Cursor-based pagination** (recommended for large datasets):
```python
from base64 import b64encode, b64decode
from pydantic import BaseModel

class PaginatedResponse(BaseModel):
    data: list
    next_cursor: str | None
    has_more: bool

@router.get("/items", response_model=PaginatedResponse)
async def list_items(
    cursor: str | None = None,
    limit: int = Query(20, ge=1, le=100),
    sort_by: str = Query("created_at", pattern="^(created_at|name|price)$"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    # Filters
    category: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    search: str | None = None,
    db: DB,
):
    query = select(Item)

    # Filters
    if category:
        query = query.where(Item.category == category)
    if min_price is not None:
        query = query.where(Item.price >= min_price)
    if max_price is not None:
        query = query.where(Item.price <= max_price)
    if search:
        query = query.where(Item.name.ilike(f"%{search}%"))

    # Cursor-based pagination
    if cursor:
        decoded = b64decode(cursor).decode()
        last_id, last_value = decoded.split(":")
        sort_col = getattr(Item, sort_by)
        if sort_order == "desc":
            query = query.where(
                (sort_col < last_value) |
                ((sort_col == last_value) & (Item.id < int(last_id)))
            )
        else:
            query = query.where(
                (sort_col > last_value) |
                ((sort_col == last_value) & (Item.id > int(last_id)))
            )

    # Sort + limit
    sort_col = getattr(Item, sort_by)
    order = sort_col.desc() if sort_order == "desc" else sort_col.asc()
    query = query.order_by(order, Item.id).limit(limit + 1)

    items = (await db.scalars(query)).all()
    has_more = len(items) > limit
    items = items[:limit]

    next_cursor = None
    if has_more and items:
        last = items[-1]
        cursor_val = f"{last.id}:{getattr(last, sort_by)}"
        next_cursor = b64encode(cursor_val.encode()).decode()

    return PaginatedResponse(data=items, next_cursor=next_cursor, has_more=has_more)
```

**Usage:**
```
GET /items?limit=20&sort_by=price&sort_order=desc&category=electronics
GET /items?cursor=MTIzOjk5Ljk5&limit=20
```

**Offset-based** (simpler but slower for deep pages):
```
GET /items?page=5&per_page=20
```

**Filtering patterns:**
- Exact match: `?status=active`
- Range: `?min_price=10&max_price=100`
- Search: `?search=keyword`
- Multiple values: `?status=active,pending` or `?status[]=active&status[]=pending`
- Nested: `?filter[price][gte]=10&filter[price][lte]=100`"""
    ))

    return pairs


# ---------------------------------------------------------------------------
# 7. WebSocket
# ---------------------------------------------------------------------------

def gen_websocket() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How do you implement WebSocket connections in FastAPI?",
        """```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict
import json
import asyncio

app = FastAPI()

class ConnectionManager:
    \"\"\"Manages active WebSocket connections.\"\"\"

    def __init__(self):
        self.active: dict[str, WebSocket] = {}  # user_id -> ws
        self.rooms: dict[str, set[str]] = {}     # room -> set of user_ids

    async def connect(self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active[user_id] = websocket

    def disconnect(self, user_id: str):
        self.active.pop(user_id, None)
        for room in list(self.rooms.keys()):
            self.rooms[room].discard(user_id)
            if not self.rooms[room]:
                del self.rooms[room]

    def join_room(self, user_id: str, room: str):
        if room not in self.rooms:
            self.rooms[room] = set()
        self.rooms[room].add(user_id)

    async def send_to_user(self, user_id: str, message: dict):
        ws = self.active.get(user_id)
        if ws:
            await ws.send_json(message)

    async def broadcast_room(self, room: str, message: dict,
                              exclude: str | None = None):
        for user_id in self.rooms.get(room, set()):
            if user_id != exclude:
                await self.send_to_user(user_id, message)

    async def broadcast_all(self, message: dict):
        for user_id in self.active:
            await self.send_to_user(user_id, message)

manager = ConnectionManager()

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(user_id, websocket)

    # Start heartbeat
    heartbeat_task = asyncio.create_task(heartbeat(websocket))

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "join_room":
                manager.join_room(user_id, data["room"])
                await manager.broadcast_room(data["room"], {
                    "type": "user_joined",
                    "user_id": user_id,
                }, exclude=user_id)

            elif msg_type == "message":
                await manager.broadcast_room(data["room"], {
                    "type": "message",
                    "user_id": user_id,
                    "content": data["content"],
                    "timestamp": datetime.utcnow().isoformat(),
                })

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(user_id)
        heartbeat_task.cancel()

async def heartbeat(ws: WebSocket, interval: int = 30):
    \"\"\"Send periodic pings to detect dead connections.\"\"\"
    try:
        while True:
            await asyncio.sleep(interval)
            await ws.send_json({"type": "heartbeat"})
    except Exception:
        pass
```

**Client side:**
```typescript
class WebSocketClient {
  private ws: WebSocket | null = null;
  private reconnectDelay = 1000;
  private maxReconnectDelay = 30000;

  connect(userId: string) {
    this.ws = new WebSocket(`ws://localhost:8000/ws/${userId}`);

    this.ws.onopen = () => {
      this.reconnectDelay = 1000; // Reset on success
      console.log('Connected');
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.handleMessage(data);
    };

    this.ws.onclose = () => {
      // Exponential backoff reconnection
      setTimeout(() => this.connect(userId), this.reconnectDelay);
      this.reconnectDelay = Math.min(
        this.reconnectDelay * 2,
        this.maxReconnectDelay
      );
    };
  }

  send(data: object) {
    this.ws?.send(JSON.stringify(data));
  }
}
```"""
    ))

    pairs.append(msg(
        "How do you implement Server-Sent Events (SSE) in a web API?",
        """**SSE** is a simpler alternative to WebSocket for server-to-client streaming.

**FastAPI with SSE:**
```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI()

async def event_generator(request: Request):
    \"\"\"Yields SSE-formatted events.\"\"\"
    while True:
        # Check if client disconnected
        if await request.is_disconnected():
            break

        # Get latest data
        data = await get_latest_notifications()

        if data:
            yield f"event: notification\\n"
            yield f"data: {json.dumps(data)}\\n"
            yield f"id: {data['id']}\\n\\n\"

        # Heartbeat to keep connection alive
        yield f": heartbeat\\n\\n"

        await asyncio.sleep(1)

@app.get("/events")
async def sse_endpoint(request: Request):
    return StreamingResponse(
        event_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )

# --- Progress tracking example ---
async def progress_generator(task_id: str, request: Request):
    while True:
        if await request.is_disconnected():
            break
        progress = await get_task_progress(task_id)
        yield f"data: {json.dumps(progress)}\\n\\n"
        if progress['status'] == 'completed':
            break
        await asyncio.sleep(0.5)

@app.get("/tasks/{task_id}/progress")
async def task_progress(task_id: str, request: Request):
    return StreamingResponse(
        progress_generator(task_id, request),
        media_type="text/event-stream",
    )
```

**Client (JavaScript):**
```javascript
const source = new EventSource('/events');

source.addEventListener('notification', (event) => {
  const data = JSON.parse(event.data);
  console.log('New notification:', data);
});

source.onerror = () => {
  // Browser auto-reconnects with Last-Event-ID header
  console.log('Connection lost, reconnecting...');
};

// With custom headers (use fetch + ReadableStream)
async function sseWithAuth(url, token) {
  const response = await fetch(url, {
    headers: { Authorization: `Bearer ${token}` },
  });
  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const text = decoder.decode(value);
    // Parse SSE format
    for (const line of text.split('\\n')) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        handleEvent(data);
      }
    }
  }
}
```

**SSE vs WebSocket:**
| Feature | SSE | WebSocket |
|---------|-----|-----------|
| Direction | Server → Client only | Bidirectional |
| Protocol | HTTP | WS (separate protocol) |
| Reconnection | Built-in auto-reconnect | Manual |
| Complexity | Simple | More complex |
| Binary data | No (text only) | Yes |
| Browser support | All modern | All modern |
| Load balancers | Standard HTTP | Needs WS support |"""
    ))

    return pairs


# ---------------------------------------------------------------------------
# Extra coverage
# ---------------------------------------------------------------------------

def gen_api_security() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "What are the essential security headers and practices for a web API?",
        """**Essential HTTP security headers:**
```python
# FastAPI middleware
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "0"  # Deprecated, CSP is better
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response
```

**Input validation:**
```python
# Always validate and sanitize
from pydantic import BaseModel, Field, field_validator
import bleach

class CommentCreate(BaseModel):
    content: str = Field(..., max_length=5000)

    @field_validator("content")
    @classmethod
    def sanitize(cls, v: str) -> str:
        return bleach.clean(v, tags=["p", "br", "strong", "em"])
```

**Rate limiting:**
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/api/search")
@limiter.limit("30/minute")
async def search(request: Request, q: str):
    return await perform_search(q)
```

**Security checklist:**
- Use HTTPS everywhere (HSTS header)\n- Validate all input (Pydantic/Zod)\n- Use parameterized queries (never string concatenation for SQL)\n- Hash passwords with bcrypt/argon2 (never MD5/SHA-1)\n- Set secure cookie flags: `httpOnly`, `secure`, `sameSite`\n- Implement CORS properly (don't use `*` in production)\n- Add rate limiting to all endpoints\n- Log security events (failed logins, permission denials)\n- Use Content-Security-Policy to prevent XSS\n- Rotate secrets regularly\n- Never expose stack traces in production responses"""
    ))

    pairs.append(msg(
        "How do you implement API key authentication?",
        """```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader, APIKeyQuery

# Accept API key in header or query param
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)

async def get_api_key(
    header_key: str | None = Security(api_key_header),
    query_key: str | None = Security(api_key_query),
    db: AsyncSession = Depends(get_db),
) -> APIKeyRecord:
    key = header_key or query_key
    if not key:
        raise HTTPException(401, "API key required")

    # Hash the key for storage comparison (don't store raw keys)
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    record = await db.scalar(
        select(APIKeyRecord).where(
            APIKeyRecord.key_hash == key_hash,
            APIKeyRecord.is_active == True,
            (APIKeyRecord.expires_at == None) | (APIKeyRecord.expires_at > datetime.utcnow()),
        )
    )
    if not record:
        raise HTTPException(401, "Invalid or expired API key")

    # Update last_used
    record.last_used_at = datetime.utcnow()
    record.request_count += 1
    await db.commit()

    return record

# API key creation endpoint (admin only)
@router.post("/api-keys")
async def create_api_key(
    name: str,
    admin: AdminUser,
    db: DB,
):
    # Generate key: prefix + random
    raw_key = f"mk_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    record = APIKeyRecord(
        name=name,
        key_hash=key_hash,
        key_prefix=raw_key[:8],  # Store prefix for identification
        created_by=admin.id,
        expires_at=datetime.utcnow() + timedelta(days=365),
    )
    db.add(record)
    await db.commit()

    # Return the raw key ONCE — it can't be retrieved later
    return {
        "api_key": raw_key,
        "name": name,
        "expires_at": record.expires_at.isoformat(),
        "warning": "Save this key — it cannot be retrieved again",
    }

# Usage in routes
@router.get("/data")
async def get_data(api_key: APIKeyRecord = Depends(get_api_key)):
    # Rate limit per API key
    if api_key.request_count > api_key.rate_limit:
        raise HTTPException(429, "Rate limit exceeded")
    return {"data": "secret stuff"}
```

**Best practices:**
- Hash API keys before storing (SHA-256 or bcrypt)
- Use a recognizable prefix (e.g., `mk_`, `sk_`) for easy identification
- Show the full key only once at creation
- Implement per-key rate limits
- Track usage (last_used, request_count)
- Support key rotation (multiple active keys per user)
- Log API key usage for audit trails"""
    ))

    return pairs


def gen_caching() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How do you implement caching in a web API?",
        """**Redis caching patterns for web APIs:**

```python
import redis.asyncio as redis
import json
from functools import wraps

r = redis.from_url("redis://localhost:6379")

# --- Cache decorator ---
def cache(ttl: int = 300, prefix: str = "cache"):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key from function name + args
            key = f\"{prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}\"

            # Try cache
            cached = await r.get(key)
            if cached:
                return json.loads(cached)

            # Compute and cache
            result = await func(*args, **kwargs)
            await r.setex(key, ttl, json.dumps(result, default=str))
            return result
        return wrapper
    return decorator

# Usage
@cache(ttl=60)
async def get_product(product_id: int):
    return await db.get(Product, product_id)

# --- Cache-aside pattern ---
async def get_user_profile(user_id: str) -> dict:
    cache_key = f\"user:{user_id}:profile\"

    # 1. Check cache
    cached = await r.get(cache_key)
    if cached:
        return json.loads(cached)

    # 2. Query database
    user = await db.get(User, user_id)
    profile = user.to_dict()

    # 3. Store in cache
    await r.setex(cache_key, 300, json.dumps(profile))
    return profile

# --- Cache invalidation ---
async def update_user(user_id: str, data: dict):
    await db.execute(update(User).where(User.id == user_id).values(**data))
    await db.commit()

    # Invalidate related caches
    await r.delete(f\"user:{user_id}:profile\")
    await r.delete(f\"user:{user_id}:settings\")

    # Pattern-based invalidation
    keys = await r.keys(f\"user:{user_id}:*\")
    if keys:
        await r.delete(*keys)

# --- HTTP caching headers ---
from fastapi import Response

@router.get(\"/products/{product_id}\")
async def get_product(product_id: int, response: Response):
    product = await get_cached_product(product_id)
    response.headers[\"Cache-Control\"] = \"public, max-age=60\"
    response.headers[\"ETag\"] = f'W/\"{hash(str(product))}\"'
    return product
```

**Caching strategies:**
| Strategy | Description | Use case |
|----------|-------------|----------|
| Cache-aside | App manages cache reads/writes | Most common, flexible |
| Write-through | Write to cache + DB simultaneously | Strong consistency |
| Write-behind | Write to cache, async DB write | High write throughput |
| Read-through | Cache loads from DB on miss | Simpler app code |

**Cache invalidation approaches:**
- **TTL** (Time-to-Live): Simple, accepts stale data for a window
- **Event-driven**: Invalidate on write operations
- **Version-based**: Append version number to cache keys
- **Pub/Sub**: Redis pub/sub to notify other instances"""
    ))

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    generators = {
        "FastAPI": gen_fastapi,
        "Hono": gen_hono,
        "Express/Node.js": gen_express,
        "Database Patterns": gen_database,
        "Auth Patterns": gen_auth,
        "REST API Design": gen_rest_design,
        "WebSocket / SSE": gen_websocket,
        "API Security": gen_api_security,
        "Caching": gen_caching,
    }

    total = 0
    for label, gen_fn in generators.items():
        pairs = gen_fn()
        count = len(pairs)
        total += count
        logger.info("  %-25s %5d pairs", label, count)
        for pair in pairs:
            print(json.dumps(pair, ensure_ascii=False))

    logger.info("=== TOTAL: %d pairs ===", total)


if __name__ == "__main__":
    main()
