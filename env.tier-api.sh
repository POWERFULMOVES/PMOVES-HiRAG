# PMOVES.AI Tier Environment: API
# For API Gateway and REST API services
# Source after env.shared: source env.shared && source env.tier-api.sh

# ============================================================================
# API Tier Configuration
# ============================================================================

export TIER=api

# Service limits
export MAX_CONCURRENT_REQUESTS=${MAX_CONCURRENT_REQUESTS:-100}
export MAX_REQUEST_SIZE_MB=${MAX_REQUEST_SIZE_MB:-10}
export REQUEST_TIMEOUT_MS=${REQUEST_TIMEOUT_MS:-30000}

# Rate limiting
export RATE_LIMIT_ENABLED=${RATE_LIMIT_ENABLED:-true}
export RATE_LIMIT_REQUESTS_PER_MINUTE=${RATE_LIMIT_REQUESTS_PER_MINUTE:-60}
export RATE_LIMIT_BURST=${RATE_LIMIT_BURST:-10}

# CORS configuration
export CORS_ORIGINS=${CORS_ORIGINS:-*}
export CORS_ALLOW_CREDENTIALS=${CORS_ALLOW_CREDENTIALS:-true}
export CORS_ALLOW_METHODS=${CORS_ALLOW_METHODS:-GET,POST,PUT,DELETE,OPTIONS}
export CORS_ALLOW_HEADERS=${CORS_ALLOW_HEADERS:-*,Authorization,Content-Type,X-API-Key}

# API Documentation
export API_DOCS_ENABLED=${API_DOCS_ENABLED:-true}
export OPENAPI_URL=${OPENAPI_URL:-/openapi.json}
export DOCS_URL=${DOCS_URL:-/docs}
export REDOC_URL=${REDOC_URL:-/redoc}

# Response compression
export COMPRESSION_ENABLED=${COMPRESSION_ENABLED:-true}
export COMPRESSION_LEVEL=${COMPRESSION_LEVEL:-6}
export COMPRESSION_MIN_SIZE=${COMPRESSION_MIN_SIZE:-500}

# Security headers
export SECURITY_ENABLED=${SECURITY_ENABLED:-true}
export SECURITY_FRAMEGUARD=${SECURITY_FRAMEGUARD:-DENY}
export SECURITY_HSTS_MAX_AGE=${SECURITY_HSTS_MAX_AGE:-31536000}
