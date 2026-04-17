#!/usr/bin/env python3
"""Generate YAML/JSON training dataset.

Covers: YAML syntax, JSON Schema, Docker Compose, GitHub Actions,
Kubernetes YAML, OpenAPI/Swagger, config file patterns.

Target: ~1000 Q&A pairs.
Output: JSONL to stdout.

Usage::

    uv run python scripts/gen_yaml_json_dataset.py > /tmp/gen-yaml-json.jsonl
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
# 1. YAML Syntax
# ---------------------------------------------------------------------------

def gen_yaml_syntax() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "What are YAML anchors and aliases and how do you use them?",
        "**Anchors** (`&name`) define a reusable node, and **aliases** (`*name`) reference it — like variables.\n\n```yaml\n# Define defaults with an anchor\ndefaults: &defaults\n  adapter: postgres\n  host: localhost\n  port: 5432\n\n# Reuse with alias\ndevelopment:\n  database: myapp_dev\n  <<: *defaults\n\ntest:\n  database: myapp_test\n  <<: *defaults\n\nproduction:\n  database: myapp_prod\n  <<: *defaults\n  host: db.production.com  # Override specific keys\n  port: 5433\n```\n\n**Result after resolution:**\n```yaml\ndevelopment:\n  database: myapp_dev\n  adapter: postgres\n  host: localhost\n  port: 5432\n\nproduction:\n  database: myapp_prod\n  adapter: postgres\n  host: db.production.com  # overridden\n  port: 5433               # overridden\n```\n\n**Array anchors:**\n```yaml\ncommon_volumes: &volumes\n  - ./logs:/app/logs\n  - ./config:/app/config\n\nservice_a:\n  volumes: *volumes\n\nservice_b:\n  volumes: *volumes\n```\n\n**Limitations:**\n- Anchors are file-scoped (can't reference across files)\n- `<<:` merge key is YAML 1.1 (not in YAML 1.2 spec, but widely supported)\n- Aliases create references, not copies — modifying the anchor source affects all aliases in some parsers"
    ))

    pairs.append(msg(
        "How do multiline strings work in YAML?",
        "YAML has several multiline string styles:\n\n**1. Literal block (`|`) — preserves newlines:**\n```yaml\ndescription: |\n  This is line one.\n  This is line two.\n\n  This is after a blank line.\n```\nResult: `\"This is line one.\\nThis is line two.\\n\\nThis is after a blank line.\\n\"`\n\n**2. Folded block (`>`) — joins lines with spaces:**\n```yaml\ndescription: >\n  This is a long\n  paragraph that will\n  be folded into one line.\n\n  This starts a new paragraph.\n```\nResult: `\"This is a long paragraph that will be folded into one line.\\nThis starts a new paragraph.\\n\"`\n\n**3. Chomping indicators (trailing newlines):**\n```yaml\n# Strip trailing newline\nstrip: |-\n  no trailing newline\n\n# Keep trailing newline (default)\nclip: |\n  one trailing newline\n\n# Keep all trailing newlines\nkeep: |+\n  all trailing\n  newlines kept\n\n\n```\n\n**4. Indentation indicator:**\n```yaml\n# When content starts with spaces\ncode: |2\n    indented code\n    more code\n```\nThe `2` tells YAML the content indentation is 2 (beyond the key's level).\n\n**5. Quoted strings (inline):**\n```yaml\nsingle: 'no escapes except ''single quotes'''\ndouble: \"supports \\n escapes \\t and \\\"quotes\\\"\"\nplain: just a string without quotes\n```\n\n**Common gotcha:** Strings starting with `{`, `[`, `*`, `&`, `#`, `?`, `|`, `-`, `<`, `>`, `=`, `!`, `%`, `@` must be quoted."
    ))

    pairs.append(msg(
        "What are YAML tags and how are they used?",
        "**Tags** explicitly specify the type of a YAML node.\n\n**Built-in tags:**\n```yaml\n# Explicit typing\nexplicit_string: !!str 123        # \"123\" (string, not int)\nexplicit_int: !!int \"42\"           # 42 (integer)\nexplicit_float: !!float \"3.14\"     # 3.14\nexplicit_bool: !!bool \"true\"       # true\nexplicit_null: !!null \"\"           # null\nexplicit_binary: !!binary |\n  R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==\n```\n\n**Common use case — the Norway problem:**\n```yaml\n# Without quotes, these are interpreted as booleans in YAML 1.1:\ncountry: NO     # false!\nanswer: yes     # true!\ncode: \"NO\"      # string \"NO\" (safe)\ncode: !!str NO  # string \"NO\" (also safe)\n```\n\n**YAML 1.2 fixes this:** Only `true`/`false` are booleans (not yes/no/on/off).\n\n**Custom tags (Python PyYAML):**\n```python\nimport yaml\n\n# Define custom tag\ndef env_var_constructor(loader, node):\n    value = loader.construct_scalar(node)\n    return os.environ.get(value, '')\n\nyaml.add_constructor('!env', env_var_constructor)\n\n# YAML file\n# database:\n#   password: !env DATABASE_PASSWORD\n\ndata = yaml.safe_load(open('config.yml'))\n```\n\n**Security:** Always use `yaml.safe_load()` not `yaml.load()` — `yaml.load()` can execute arbitrary Python code via tags like `!!python/object`."
    ))

    pairs.append(msg(
        "What are the common YAML gotchas that cause bugs?",
        "**1. Unquoted special values:**\n```yaml\n# Booleans in disguise (YAML 1.1)\ncountry: NO       # Parsed as false!\nversion: 1.0      # Parsed as float 1.0, not string \"1.0\"!\ntime: 12:30       # Parsed as sexagesimal number (750)!\nport: 0800        # Parsed as octal (512)!\n\n# Fix: always quote values that look like other types\ncountry: \"NO\"\nversion: \"1.0\"\ntime: \"12:30\"\nport: \"0800\"  # or just 800\n```\n\n**2. Indentation errors:**\n```yaml\n# BAD — inconsistent indentation\nserver:\n  host: localhost\n   port: 8080        # 3 spaces instead of 2 — error!\n\n# BAD — tabs (YAML forbids tabs for indentation)\nserver:\n\\thost: localhost    # Tab character — error!\n```\n\n**3. Colon in values:**\n```yaml\n# BAD — colon followed by space starts a mapping\ntitle: Chapter 1: Introduction   # Parsed as nested mapping!\n\n# Fix: quote the value\ntitle: \"Chapter 1: Introduction\"\n```\n\n**4. Anchors are not deep copies:**\n```yaml\nbase: &base\n  list:\n    - item1\n\nderived:\n  <<: *base\n  list:           # This REPLACES the list, doesn't extend it\n    - item2\n```\n\n**5. Empty values:**\n```yaml\nkey:              # null (not empty string)\nkey: \"\"           # empty string\nkey: null         # explicit null\nkey: ~            # also null\n```\n\n**6. Multi-document files:**\n```yaml\n---\nfirst: document\n---\nsecond: document\n...\n```\n`---` separates documents, `...` ends a document. Some parsers only read the first document."
    ))

    pairs.append(msg(
        "How do you validate a YAML file against a schema?",
        "Several approaches for YAML validation:\n\n**1. JSON Schema (YAML is a superset of JSON):**\n```python\nimport yaml\nimport jsonschema\n\nschema = {\n    \"type\": \"object\",\n    \"required\": [\"name\", \"version\"],\n    \"properties\": {\n        \"name\": {\"type\": \"string\", \"minLength\": 1},\n        \"version\": {\"type\": \"string\", \"pattern\": \"^\\\\d+\\\\.\\\\d+\\\\.\\\\d+$\"},\n        \"description\": {\"type\": \"string\"},\n        \"dependencies\": {\n            \"type\": \"object\",\n            \"additionalProperties\": {\"type\": \"string\"},\n        },\n    },\n    \"additionalProperties\": False,\n}\n\nwith open(\"config.yml\") as f:\n    data = yaml.safe_load(f)\n\ntry:\n    jsonschema.validate(data, schema)\n    print(\"Valid!\")\nexcept jsonschema.ValidationError as e:\n    print(f\"Validation error at {'.'.join(str(p) for p in e.absolute_path)}: {e.message}\")\n```\n\n**2. Pydantic (Python):**\n```python\nfrom pydantic import BaseModel, Field\nimport yaml\n\nclass AppConfig(BaseModel):\n    name: str = Field(min_length=1)\n    version: str = Field(pattern=r\"^\\d+\\.\\d+\\.\\d+$\")\n    port: int = Field(ge=1, le=65535)\n    debug: bool = False\n    database: DatabaseConfig\n\nclass DatabaseConfig(BaseModel):\n    host: str\n    port: int = 5432\n    name: str\n\nwith open(\"config.yml\") as f:\n    raw = yaml.safe_load(f)\n\nconfig = AppConfig(**raw)  # Validates and provides type-safe access\n```\n\n**3. yamllint (CLI):**\n```bash\npip install yamllint\nyamllint config.yml\n\n# Custom rules in .yamllint.yml\n---\nrules:\n  line-length:\n    max: 120\n  indentation:\n    spaces: 2\n  truthy:\n    check-keys: true  # Catches the NO/YES problem\n```\n\n**4. VS Code:** YAML extension with JSON Schema association:\n```json\n// .vscode/settings.json\n{\n  \"yaml.schemas\": {\n    \"https://json.schemastore.org/github-workflow.json\": \".github/workflows/*.yml\"\n  }\n}\n```"
    ))

    return pairs


# ---------------------------------------------------------------------------
# 2. JSON Schema
# ---------------------------------------------------------------------------

def gen_json_schema() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How do you write a JSON Schema for validating API payloads?",
        """```json
{
  \"$schema\": \"https://json-schema.org/draft/2020-12/schema\",
  \"$id\": \"https://api.example.com/schemas/user-create\",
  \"title\": \"Create User\",
  \"description\": \"Schema for user creation request\",
  \"type\": \"object\",
  \"required\": [\"name\", \"email\", \"password\"],
  \"properties\": {
    \"name\": {
      \"type\": \"string\",
      \"minLength\": 2,
      \"maxLength\": 100
    },
    \"email\": {
      \"type\": \"string\",
      \"format\": \"email\"
    },
    \"password\": {
      \"type\": \"string\",
      \"minLength\": 8,
      \"pattern\": \"^(?=.*[A-Z])(?=.*[0-9])\"
    },
    \"age\": {
      \"type\": \"integer\",
      \"minimum\": 18,
      \"maximum\": 150
    },
    \"role\": {
      \"type\": \"string\",
      \"enum\": [\"user\", \"editor\", \"admin\"],
      \"default\": \"user\"
    },
    \"tags\": {
      \"type\": \"array\",
      \"items\": { \"type\": \"string\" },
      \"uniqueItems\": true,
      \"maxItems\": 10
    },
    \"address\": {
      \"$ref\": \"#/$defs/Address\"
    },
    \"metadata\": {
      \"type\": \"object\",
      \"additionalProperties\": { \"type\": \"string\" }
    }
  },
  \"additionalProperties\": false,
  \"$defs\": {
    \"Address\": {
      \"type\": \"object\",
      \"required\": [\"street\", \"city\", \"country\"],
      \"properties\": {
        \"street\": { \"type\": \"string\" },
        \"city\": { \"type\": \"string\" },
        \"state\": { \"type\": \"string\" },
        \"country\": {
          \"type\": \"string\",
          \"pattern\": \"^[A-Z]{2}$\"
        },
        \"zip\": { \"type\": \"string\" }
      }
    }
  }
}
```

**Key JSON Schema keywords:**
- `type`: string, number, integer, boolean, array, object, null
- `required`: Array of required property names
- `properties`: Property definitions
- `additionalProperties`: false to disallow extra keys
- `$ref`: Reference to another schema (or definition)
- `$defs` (or `definitions`): Reusable schema fragments
- `enum`: Fixed set of allowed values
- `pattern`: Regex pattern for strings
- `format`: Semantic hint (email, uri, date-time, uuid)
- `oneOf`/`anyOf`/`allOf`: Composition operators
- `if`/`then`/`else`: Conditional schemas"""
    ))

    pairs.append(msg(
        "How do you use oneOf, anyOf, and allOf in JSON Schema?",
        """**`oneOf`** — value must match exactly ONE of the schemas:
```json
{
  \"payment\": {
    \"oneOf\": [
      {
        \"type\": \"object\",
        \"required\": [\"type\", \"card_number\"],
        \"properties\": {
          \"type\": { \"const\": \"credit_card\" },
          \"card_number\": { \"type\": \"string\", \"pattern\": \"^[0-9]{16}$\" },
          \"cvv\": { \"type\": \"string\", \"pattern\": \"^[0-9]{3,4}$\" }
        }
      },
      {
        \"type\": \"object\",
        \"required\": [\"type\", \"bank_account\"],
        \"properties\": {
          \"type\": { \"const\": \"bank_transfer\" },
          \"bank_account\": { \"type\": \"string\" },
          \"routing_number\": { \"type\": \"string\" }
        }
      },
      {
        \"type\": \"object\",
        \"required\": [\"type\", \"wallet_id\"],
        \"properties\": {
          \"type\": { \"const\": \"digital_wallet\" },
          \"wallet_id\": { \"type\": \"string\" }
        }
      }
    ]
  }
}
```

**`anyOf`** — value must match AT LEAST ONE schema:
```json
{
  \"contact\": {
    \"anyOf\": [
      { \"required\": [\"email\"], \"properties\": { \"email\": { \"type\": \"string\", \"format\": \"email\" } } },
      { \"required\": [\"phone\"], \"properties\": { \"phone\": { \"type\": \"string\" } } }
    ]
  }
}
```
Valid: `{\"email\": \"a@b.com\"}`, `{\"phone\": \"123\"}`, `{\"email\": \"a@b.com\", \"phone\": \"123\"}`

**`allOf`** — value must match ALL schemas (composition/intersection):
```json
{
  \"allOf\": [
    { \"$ref\": \"#/$defs/BaseEntity\" },
    { \"$ref\": \"#/$defs/Timestamps\" },
    {
      \"type\": \"object\",
      \"required\": [\"title\"],
      \"properties\": {
        \"title\": { \"type\": \"string\" }
      }
    }
  ]
}
```
Combines BaseEntity + Timestamps + title into one schema.

**`not`** — value must NOT match the schema:
```json
{
  \"name\": {
    \"type\": \"string\",
    \"not\": { \"enum\": [\"admin\", \"root\", \"system\"] }
  }
}
```"""
    ))

    pairs.append(msg(
        "How do you use conditional schemas (if/then/else) in JSON Schema?",
        """```json
{
  \"type\": \"object\",
  \"required\": [\"type\"],
  \"properties\": {
    \"type\": { \"enum\": [\"individual\", \"company\"] }
  },
  \"if\": {
    \"properties\": { \"type\": { \"const\": \"individual\" } }
  },
  \"then\": {
    \"required\": [\"first_name\", \"last_name\", \"date_of_birth\"],
    \"properties\": {
      \"first_name\": { \"type\": \"string\" },
      \"last_name\": { \"type\": \"string\" },
      \"date_of_birth\": { \"type\": \"string\", \"format\": \"date\" }
    }
  },
  \"else\": {
    \"required\": [\"company_name\", \"registration_number\"],
    \"properties\": {
      \"company_name\": { \"type\": \"string\" },
      \"registration_number\": { \"type\": \"string\" },
      \"vat_number\": { \"type\": \"string\" }
    }
  }
}
```

This means:\n- If `type` is `\"individual\"` → require `first_name`, `last_name`, `date_of_birth`\n- If `type` is `\"company\"` → require `company_name`, `registration_number`\n\n**Chained conditions** (multiple if/then):\n```json\n{\n  \"allOf\": [\n    {\n      \"if\": { \"properties\": { \"country\": { \"const\": \"US\" } } },\n      \"then\": { \"required\": [\"state\"], \"properties\": { \"state\": { \"type\": \"string\", \"pattern\": \"^[A-Z]{2}$\" } } }\n    },\n    {\n      \"if\": { \"properties\": { \"country\": { \"const\": \"UK\" } } },\n      \"then\": { \"required\": [\"postcode\"], \"properties\": { \"postcode\": { \"type\": \"string\", \"pattern\": \"^[A-Z]{1,2}[0-9]\" } } }\n    }\n  ]\n}\n```\n\n**Alternative: discriminated union with oneOf** (often clearer):\n```json\n{\n  \"oneOf\": [\n    {\n      \"properties\": { \"type\": { \"const\": \"individual\" }, \"first_name\": { \"type\": \"string\" } },\n      \"required\": [\"type\", \"first_name\"]\n    },\n    {\n      \"properties\": { \"type\": { \"const\": \"company\" }, \"company_name\": { \"type\": \"string\" } },\n      \"required\": [\"type\", \"company_name\"]\n    }\n  ]\n}\n```"""
    ))

    return pairs


# ---------------------------------------------------------------------------
# 3. Docker Compose
# ---------------------------------------------------------------------------

def gen_docker_compose() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "Write a production-ready Docker Compose file for a web application stack.",
        """```yaml
# docker-compose.yml
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    ports:
      - \"8000:8000\"
    environment:
      DATABASE_URL: postgresql://user:pass@db:5432/myapp
      REDIS_URL: redis://redis:6379/0
      SECRET_KEY: ${SECRET_KEY}
    env_file:
      - .env
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: \"1.0\"
        reservations:
          memory: 256M
    healthcheck:
      test: [\"CMD\", \"curl\", \"-f\", \"http://localhost:8000/health\"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: unless-stopped
    logging:
      driver: json-file
      options:
        max-size: \"10m\"
        max-file: \"3\"

  db:
    image: postgres:16-alpine
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    healthcheck:
      test: [\"CMD-SHELL\", \"pg_isready -U user -d myapp\"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD} --maxmemory 256mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    healthcheck:
      test: [\"CMD\", \"redis-cli\", \"--pass\", \"${REDIS_PASSWORD}\", \"ping\"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - \"80:80\"
      - \"443:443\"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - app
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:

networks:
  default:
    driver: bridge
```\n\n**Key patterns:**\n- Use `depends_on` with `condition: service_healthy` for proper startup order\n- Set resource limits to prevent runaway containers\n- Add healthchecks for all services\n- Use `env_file` + `${VAR}` interpolation for secrets\n- Mount named volumes for persistent data\n- Use Alpine images for smaller size"""
    ))

    pairs.append(msg(
        "How do you use Docker Compose profiles and extends?",
        """**Profiles** let you selectively start services:
```yaml
services:
  app:
    build: .
    ports: [\"8000:8000\"]
    # No profile = always starts

  db:
    image: postgres:16
    # No profile = always starts

  redis:
    image: redis:7
    profiles: [\"cache\"]  # Only starts with --profile cache

  mailhog:
    image: mailhog/mailhog
    ports: [\"8025:8025\"]
    profiles: [\"dev\"]  # Only starts with --profile dev

  prometheus:
    image: prom/prometheus
    profiles: [\"monitoring\"]

  grafana:
    image: grafana/grafana
    profiles: [\"monitoring\"]
```

**Usage:**
```bash
docker compose up                        # app + db only
docker compose --profile dev up          # app + db + mailhog
docker compose --profile cache --profile monitoring up  # all except mailhog
```

**Extends** for shared configuration:
```yaml
# docker-compose.base.yml
services:
  base-python:
    build:
      context: .
      target: base
    environment:
      PYTHONUNBUFFERED: \"1\"
    volumes:
      - ./src:/app/src

# docker-compose.yml
services:
  api:
    extends:
      file: docker-compose.base.yml
      service: base-python
    command: uvicorn app.main:app --host 0.0.0.0
    ports: [\"8000:8000\"]

  worker:
    extends:
      file: docker-compose.base.yml
      service: base-python
    command: celery -A app.worker worker

  scheduler:
    extends:
      file: docker-compose.base.yml
      service: base-python
    command: celery -A app.worker beat
```

**Override files:**
```bash
# docker-compose.yml          = base config
# docker-compose.override.yml = auto-loaded for dev
# docker-compose.prod.yml     = production overrides

docker compose up                                    # uses base + override
docker compose -f docker-compose.yml -f docker-compose.prod.yml up  # uses base + prod
```"""
    ))

    pairs.append(msg(
        "How do you handle secrets and environment variables in Docker Compose?",
        """**1. Environment variables directly:**
```yaml
services:
  app:
    environment:
      - NODE_ENV=production
      - API_KEY=${API_KEY}  # From shell environment
```

**2. Env file:**
```yaml
services:
  app:
    env_file:
      - .env           # Default
      - .env.local     # Overrides (gitignored)
```

```env
# .env
DATABASE_URL=postgresql://user:pass@db:5432/myapp
REDIS_URL=redis://redis:6379
SECRET_KEY=change-me-in-production
```

**3. Docker Secrets (Swarm mode):**
```yaml
services:
  app:
    secrets:
      - db_password
      - api_key
    environment:
      DB_PASSWORD_FILE: /run/secrets/db_password

secrets:
  db_password:
    file: ./secrets/db_password.txt
  api_key:
    external: true  # Created with: docker secret create api_key file.txt
```

In the app, read from `/run/secrets/db_password` instead of env vars.\n\n**4. Variable interpolation:**\n```yaml\nservices:\n  app:\n    image: myapp:${TAG:-latest}        # Default value\n    ports:\n      - \"${PORT:-8000}:8000\"           # Default if unset\n    environment:\n      DATABASE_URL: postgres://${DB_USER:?required}:${DB_PASS:?required}@db/app\n      # :? = error if not set\n```\n\n**5. .env file for Compose itself** (not containers):\n```env\n# .env (read by docker compose for variable substitution)\nCOMPOSE_PROJECT_NAME=myproject\nCOMPOSE_FILE=docker-compose.yml:docker-compose.prod.yml\nTAG=v1.2.3\n```\n\n**Best practices:**\n- Never commit `.env` files with real secrets\n- Use `.env.example` as a template (committed)\n- Use Docker Secrets or a vault (HashiCorp Vault, Infisical) for production\n- Use `${VAR:?error}` to fail fast on missing required vars"""
    ))

    return pairs


# ---------------------------------------------------------------------------
# 4. GitHub Actions
# ---------------------------------------------------------------------------

def gen_github_actions() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "Write a GitHub Actions CI workflow for a Python project.",
        """```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

permissions:
  contents: read

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v4
        with:
          version: \"latest\"

      - name: Install dependencies
        run: uv sync --dev

      - name: Lint with ruff
        run: uv run ruff check .

      - name: Type check
        run: uv run mypy src/

  test:
    runs-on: ubuntu-latest
    needs: lint
    strategy:
      matrix:
        python-version: [\"3.12\", \"3.13\"]
      fail-fast: false

    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_DB: test
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
        ports: [\"5432:5432\"]
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v4
        with:
          version: \"latest\"

      - name: Set up Python ${{ matrix.python-version }}
        run: uv python install ${{ matrix.python-version }}

      - name: Install dependencies
        run: uv sync --dev

      - name: Run tests
        env:
          DATABASE_URL: postgresql://test:test@localhost:5432/test
        run: uv run pytest --cov=src --cov-report=xml -v

      - name: Upload coverage
        if: matrix.python-version == '3.13'
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          token: ${{ secrets.CODECOV_TOKEN }}

  build:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t myapp:${{ github.sha }} .

      - name: Login to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Push image
        run: |
          docker tag myapp:${{ github.sha }} ghcr.io/${{ github.repository }}:latest
          docker push ghcr.io/${{ github.repository }}:latest
```"""
    ))

    pairs.append(msg(
        "How do you use GitHub Actions matrix strategy and reusable workflows?",
        """**Matrix strategy** runs a job with multiple configurations:
```yaml
jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        node-version: [18, 20, 22]
        exclude:
          - os: windows-latest
            node-version: 18
        include:
          - os: ubuntu-latest
            node-version: 22
            experimental: true
      fail-fast: false  # Don't cancel other jobs if one fails

    runs-on: ${{ matrix.os }}
    continue-on-error: ${{ matrix.experimental || false }}

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
      - run: npm ci
      - run: npm test
```

**Reusable workflow** (called by other workflows):
```yaml
# .github/workflows/deploy-reusable.yml
name: Deploy (reusable)

on:
  workflow_call:
    inputs:
      environment:
        required: true
        type: string
      image-tag:
        required: true
        type: string
    secrets:
      deploy-key:
        required: true

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: ${{ inputs.environment }}
    steps:
      - name: Deploy to ${{ inputs.environment }}
        run: |
          echo \"Deploying ${{ inputs.image-tag }} to ${{ inputs.environment }}\"
          # ssh deploy commands using ${{ secrets.deploy-key }}
```

**Calling the reusable workflow:**
```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
    steps:
      - uses: actions/checkout@v4
      - id: meta
        run: echo \"tags=${{ github.ref_name }}\" >> $GITHUB_OUTPUT
      - run: docker build -t myapp:${{ github.ref_name }} .

  deploy-staging:
    needs: build
    uses: ./.github/workflows/deploy-reusable.yml
    with:
      environment: staging
      image-tag: ${{ needs.build.outputs.image-tag }}
    secrets:
      deploy-key: ${{ secrets.STAGING_DEPLOY_KEY }}

  deploy-production:
    needs: deploy-staging
    uses: ./.github/workflows/deploy-reusable.yml
    with:
      environment: production
      image-tag: ${{ needs.build.outputs.image-tag }}
    secrets:
      deploy-key: ${{ secrets.PROD_DEPLOY_KEY }}
```

**Key concepts:**
- `workflow_call` trigger makes a workflow reusable
- `inputs` and `secrets` for parameterization
- `needs` for job dependencies
- `outputs` to pass data between jobs
- `environment` for deployment protection rules"""
    ))

    pairs.append(msg(
        "How do you cache dependencies in GitHub Actions?",
        """```yaml
# --- Node.js with npm ---
- uses: actions/setup-node@v4
  with:
    node-version: 20
    cache: 'npm'  # Built-in caching

- run: npm ci

# --- Python with uv ---
- uses: astral-sh/setup-uv@v4
  with:
    version: \"latest\"
    enable-cache: true  # Built-in uv caching

# --- Manual cache ---
- uses: actions/cache@v4
  with:
    path: |
      ~/.cache/pip
      ~/.local/share/uv
      .venv
    key: python-${{ runner.os }}-${{ hashFiles('**/pyproject.toml', '**/uv.lock') }}
    restore-keys: |
      python-${{ runner.os }}-

# --- Rust (cargo) ---
- uses: actions/cache@v4
  with:
    path: |
      ~/.cargo/bin/
      ~/.cargo/registry/index/
      ~/.cargo/registry/cache/
      target/
    key: cargo-${{ runner.os }}-${{ hashFiles('**/Cargo.lock') }}

# --- Docker layer caching ---
- uses: docker/build-push-action@v5
  with:
    context: .
    cache-from: type=gha
    cache-to: type=gha,mode=max
    push: true
    tags: myapp:latest

# --- Artifacts (persist between jobs) ---
- uses: actions/upload-artifact@v4
  with:
    name: build-output
    path: dist/
    retention-days: 1

# In another job:
- uses: actions/download-artifact@v4
  with:
    name: build-output
    path: dist/
```

**Cache key strategy:**
- Include OS, language version, and lockfile hash
- Use `restore-keys` for partial cache hits
- Cache path should include all dependency directories
- Artifacts are for passing files between jobs (different from cache)"""
    ))

    return pairs


# ---------------------------------------------------------------------------
# 5. Kubernetes YAML
# ---------------------------------------------------------------------------

def gen_kubernetes() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "Write a Kubernetes Deployment and Service for a web application.",
        """```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  namespace: default
  labels:
    app: myapp
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: myapp
        version: v1
    spec:
      containers:
        - name: myapp
          image: ghcr.io/myorg/myapp:v1.2.3
          ports:
            - containerPort: 8000
              protocol: TCP
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: myapp-secrets
                  key: database-url
            - name: LOG_LEVEL
              valueFrom:
                configMapKeyRef:
                  name: myapp-config
                  key: log-level
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              cpu: 500m
              memory: 512Mi
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
          startupProbe:
            httpGet:
              path: /health
              port: 8000
            failureThreshold: 30
            periodSeconds: 10
      terminationGracePeriodSeconds: 30

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: myapp
spec:
  selector:
    app: myapp
  ports:
    - port: 80
      targetPort: 8000
      protocol: TCP
  type: ClusterIP

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: myapp-config
data:
  log-level: \"info\"
  max-connections: \"100\"

---
# secret.yaml (use kubectl create secret or sealed-secrets in practice)
apiVersion: v1
kind: Secret
metadata:
  name: myapp-secrets
type: Opaque
stringData:
  database-url: postgresql://user:pass@db:5432/myapp
```

**Apply:**
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl get pods -l app=myapp
kubectl rollout status deployment/myapp
```"""
    ))

    pairs.append(msg(
        "How do you configure a Kubernetes Ingress with TLS?",
        """```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: myapp-ingress
  annotations:
    # For nginx ingress controller
    nginx.ingress.kubernetes.io/ssl-redirect: \"true\"
    nginx.ingress.kubernetes.io/proxy-body-size: \"10m\"
    nginx.ingress.kubernetes.io/rate-limit: \"100\"
    nginx.ingress.kubernetes.io/rate-limit-window: \"1m\"
    # For cert-manager auto-TLS
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - api.example.com
        - app.example.com
      secretName: example-com-tls  # Auto-created by cert-manager
  rules:
    - host: api.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: api-service
                port:
                  number: 80

    - host: app.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: frontend-service
                port:
                  number: 80

          - path: /api
            pathType: Prefix
            backend:
              service:
                name: api-service
                port:
                  number: 80

---
# cert-manager ClusterIssuer for Let's Encrypt
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@example.com
    privateKeySecretRef:
      name: letsencrypt-prod-key
    solvers:
      - http01:
          ingress:
            class: nginx
```

**Key concepts:**
- `ingressClassName`: Which ingress controller to use\n- `tls`: TLS termination with certificate\n- `rules`: Host-based routing\n- `pathType`: `Prefix` (matches /api/*) or `Exact` (matches /api only)\n- cert-manager automates Let's Encrypt certificate lifecycle"""
    ))

    return pairs


# ---------------------------------------------------------------------------
# 6. OpenAPI / Swagger
# ---------------------------------------------------------------------------

def gen_openapi() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How do you write an OpenAPI 3.1 specification?",
        """```yaml
openapi: 3.1.0
info:
  title: My API
  description: RESTful API for managing users and items
  version: 1.0.0
  contact:
    email: api@example.com
  license:
    name: MIT

servers:
  - url: https://api.example.com/v1
    description: Production
  - url: http://localhost:8000/v1
    description: Development

security:
  - bearerAuth: []

paths:
  /users:
    get:
      summary: List users
      operationId: listUsers
      tags: [Users]
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            minimum: 1
            default: 1
        - name: per_page
          in: query
          schema:
            type: integer
            minimum: 1
            maximum: 100
            default: 20
        - name: role
          in: query
          schema:
            $ref: '#/components/schemas/UserRole'
      responses:
        '200':
          description: Paginated list of users
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/User'
                  total:
                    type: integer
                  page:
                    type: integer

    post:
      summary: Create a user
      operationId: createUser
      tags: [Users]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UserCreate'
      responses:
        '201':
          description: User created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '409':
          $ref: '#/components/responses/Conflict'
        '422':
          $ref: '#/components/responses/ValidationError'

  /users/{userId}:
    get:
      summary: Get a user by ID
      operationId: getUser
      tags: [Users]
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: User found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '404':
          $ref: '#/components/responses/NotFound'

components:
  schemas:
    User:
      type: object
      required: [id, name, email, role, createdAt]
      properties:
        id:
          type: string
          format: uuid
        name:
          type: string
        email:
          type: string
          format: email
        role:
          $ref: '#/components/schemas/UserRole'
        createdAt:
          type: string
          format: date-time

    UserCreate:
      type: object
      required: [name, email]
      properties:
        name:
          type: string
          minLength: 2
        email:
          type: string
          format: email
        role:
          $ref: '#/components/schemas/UserRole'

    UserRole:
      type: string
      enum: [user, editor, admin]

  responses:
    NotFound:
      description: Resource not found
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
    Conflict:
      description: Resource already exists
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
    ValidationError:
      description: Validation error
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'

  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
```"""
    ))

    return pairs


# ---------------------------------------------------------------------------
# 7. Config File Patterns
# ---------------------------------------------------------------------------

def gen_config_patterns() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How do you implement the 12-factor app configuration pattern?",
        "The **12-factor app** methodology (factor III: Config) states that configuration should be stored in environment variables, NOT in code.\n\n**Why environment variables:**\n- Language/framework agnostic\n- Can't be accidentally committed to git\n- Easy to change between environments (dev/staging/prod)\n- Work with every deployment platform (Docker, K8s, Heroku, etc.)\n\n**Python pattern:**\n```python\nfrom pydantic_settings import BaseSettings, SettingsConfigDict\nfrom functools import lru_cache\n\nclass Settings(BaseSettings):\n    model_config = SettingsConfigDict(\n        env_file=\".env\",\n        env_file_encoding=\"utf-8\",\n        case_sensitive=False,\n    )\n\n    # Required (no default = must be set)\n    database_url: str\n    secret_key: str\n\n    # Optional with defaults\n    app_name: str = \"My App\"\n    debug: bool = False\n    log_level: str = \"INFO\"\n    port: int = 8000\n    cors_origins: list[str] = [\"http://localhost:3000\"]\n\n    # Nested config via prefix\n    redis_host: str = \"localhost\"\n    redis_port: int = 6379\n    redis_db: int = 0\n\n    @property\n    def redis_url(self) -> str:\n        return f\"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}\"\n\n@lru_cache\ndef get_settings() -> Settings:\n    return Settings()\n\n# Usage\nsettings = get_settings()\nprint(settings.database_url)\n```\n\n**Node.js pattern:**\n```typescript\nimport { z } from 'zod';\nimport dotenv from 'dotenv';\n\ndotenv.config();\n\nconst envSchema = z.object({\n  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),\n  PORT: z.coerce.number().default(3000),\n  DATABASE_URL: z.string().url(),\n  SECRET_KEY: z.string().min(32),\n  CORS_ORIGIN: z.string().default('http://localhost:5173'),\n});\n\nconst env = envSchema.parse(process.env);\nexport default env;\n```\n\n**File hierarchy:**\n```\n.env.example    # Template (committed) — documents required vars\n.env            # Local overrides (gitignored)\n.env.test       # Test environment\n.env.production # Never committed — use secrets manager\n```"
    ))

    pairs.append(msg(
        "Compare TOML, YAML, JSON, and INI for configuration files.",
        "| Feature | TOML | YAML | JSON | INI |\n|---------|------|------|------|-----|\n| **Comments** | `#` | `#` | None | `;` or `#` |\n| **Types** | String, int, float, bool, datetime, array, table | String, int, float, bool, null, array, object | String, number, bool, null, array, object | Strings only |\n| **Nesting** | `[section.sub]` | Indentation | `{\"a\":{\"b\":1}}` | `[section]` (1 level) |\n| **Multiline strings** | `\"\"\"..\"\"\"` or `'''..'''` | `|` and `>` | Not supported | Not supported |\n| **Date/time** | Native support | String (tag) | String | String |\n| **Trailing commas** | Yes | N/A | No (strict) | N/A |\n| **Spec strictness** | Very strict | Loose (gotchas) | Very strict | Loose, varies |\n\n**When to use each:**\n\n**TOML** — Best for human-edited config files:\n```toml\n[server]\nhost = \"0.0.0.0\"\nport = 8080\n\n[database]\nurl = \"postgresql://localhost/mydb\"\npool_size = 20\n\n[logging]\nlevel = \"info\"\nformat = \"json\"\n```\nUsed by: Rust (Cargo.toml), Python (pyproject.toml), Hugo\n\n**YAML** — Best for complex nested structures:\n```yaml\nservices:\n  api:\n    replicas: 3\n    env:\n      - name: DB_URL\n        value: postgres://...\n```\nUsed by: Kubernetes, Docker Compose, Ansible, GitHub Actions\n\n**JSON** — Best for machine-generated/consumed config:\n```json\n{\"compilerOptions\": {\"target\": \"es2022\"}}\n```\nUsed by: package.json, tsconfig.json, VS Code settings\n\n**INI** — Legacy, simple key-value:\n```ini\n[database]\nhost = localhost\nport = 5432\n```\nUsed by: Git config, systemd units, Windows apps\n\n**Recommendation:** Use TOML for new projects (strict, readable, no gotchas). Use YAML only when the ecosystem requires it (K8s, CI/CD)."
    ))

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    generators = {
        "YAML Syntax": gen_yaml_syntax,
        "JSON Schema": gen_json_schema,
        "Docker Compose": gen_docker_compose,
        "GitHub Actions": gen_github_actions,
        "Kubernetes": gen_kubernetes,
        "OpenAPI / Swagger": gen_openapi,
        "Config Patterns": gen_config_patterns,
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
