-- =====================================================
-- VEHICLE DAMAGE AI — DATABASE INITIALIZATION SCRIPT
-- Matches backend/app/models.py exactly.
-- DB: damage_ai  |  User: admin  |  Image: pgvector/pgvector:pg16
--
-- Tables are created with IF NOT EXISTS so re-running is safe.
-- The ORM (SQLAlchemy create_all) will also run on startup and is
-- idempotent — both sources produce the same schema.
-- =====================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- =====================================================
-- USERS
-- PK is a UUID hex string generated in Python (_uuid()).
-- Phone is the login credential (no email in the current auth flow).
-- =====================================================
CREATE TABLE IF NOT EXISTS users (
    id          VARCHAR PRIMARY KEY,
    name        VARCHAR NOT NULL,
    phone       VARCHAR NOT NULL,
    password_hash VARCHAR NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_users_phone UNIQUE (phone)
);

CREATE INDEX IF NOT EXISTS idx_users_phone ON users (phone);

-- =====================================================
-- SESSIONS
-- token is a random hex string issued on login.
-- =====================================================
CREATE TABLE IF NOT EXISTS sessions (
    token       VARCHAR PRIMARY KEY,
    user_id     VARCHAR NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions (user_id);

-- =====================================================
-- VEHICLES
-- client_id is the frontend-assigned id (e.g. "V123456").
-- data is the full VehicleRegistration JSON blob from the frontend.
-- =====================================================
CREATE TABLE IF NOT EXISTS vehicles (
    id          SERIAL PRIMARY KEY,
    client_id   VARCHAR NOT NULL,
    user_id     VARCHAR NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    data        JSONB NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_vehicle_user_client UNIQUE (user_id, client_id)
);

CREATE INDEX IF NOT EXISTS idx_vehicles_user_id ON vehicles (user_id);

-- =====================================================
-- CLAIMS
-- client_id is the frontend-assigned id (e.g. "CLM-4821").
-- data is the full Claim JSON blob from the frontend, which includes:
--   findings, detections, thumbnails, cost totals, approval decision.
-- =====================================================
CREATE TABLE IF NOT EXISTS claims (
    id          SERIAL PRIMARY KEY,
    client_id   VARCHAR NOT NULL,
    user_id     VARCHAR NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    data        JSONB NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_claim_user_client UNIQUE (user_id, client_id)
);

CREATE INDEX IF NOT EXISTS idx_claims_user_id ON claims (user_id);

-- =====================================================
-- INSURANCE CLAIMS
-- client_id is the frontend-assigned id (e.g. "INS-4821").
-- data is the full insurance claim JSON blob.
-- =====================================================
CREATE TABLE IF NOT EXISTS insurance_claims (
    id          SERIAL PRIMARY KEY,
    client_id   VARCHAR NOT NULL,
    user_id     VARCHAR NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    data        JSONB NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_insurance_user_client UNIQUE (user_id, client_id)
);

CREATE INDEX IF NOT EXISTS idx_insurance_claims_user_id ON insurance_claims (user_id);

-- =====================================================
-- USER SETTINGS
-- One row per user. data holds the full settings JSON blob.
-- =====================================================
CREATE TABLE IF NOT EXISTS user_settings (
    user_id     VARCHAR PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    data        JSONB NOT NULL,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- =====================================================
-- CLAIM IMAGES
-- Stores all images in DB as BYTEA — no filesystem dependency.
-- image_type: original | annotated | masked | merged
-- job_id links to the in-memory assessment job.
-- =====================================================
CREATE TABLE IF NOT EXISTS claim_images (
    id          SERIAL PRIMARY KEY,
    job_id      VARCHAR NOT NULL,
    image_type  VARCHAR(20) NOT NULL,
    mime_type   VARCHAR(20) NOT NULL DEFAULT 'image/jpeg',
    data        BYTEA NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_claim_images_job_id ON claim_images (job_id);
CREATE INDEX IF NOT EXISTS idx_claim_images_job_type ON claim_images (job_id, image_type);

-- =====================================================
-- VECTOR INDEX for future similarity search
-- =====================================================
-- Example (uncomment when you build the embeddings feature):
-- CREATE TABLE IF NOT EXISTS embeddings (
--     id          SERIAL PRIMARY KEY,
--     claim_id    VARCHAR NOT NULL,
--     embedding   VECTOR(768),
--     created_at  TIMESTAMPTZ DEFAULT NOW()
-- );
-- CREATE INDEX IF NOT EXISTS idx_embeddings_vector
--     ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
