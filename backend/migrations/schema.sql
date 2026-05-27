-- =====================================================
-- VEHICLE DAMAGE AI - DATABASE INITIALIZATION SCRIPT
-- =====================================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- =====================================================
-- USERS TABLE
-- =====================================================

CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    phone_number VARCHAR(15) UNIQUE NOT NULL,
    email VARCHAR(150) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    account_status VARCHAR(20)
        CHECK (
            account_status IN (
                'ACTIVE',
                'INACTIVE',
                'SUSPENDED'
            )
        )
        DEFAULT 'ACTIVE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- USER SESSIONS TABLE
-- =====================================================

CREATE TABLE user_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL,
    login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id)
        REFERENCES users(user_id)
        ON DELETE CASCADE
);

-- =====================================================
-- VEHICLE REGISTRATION TABLE
-- =====================================================

CREATE TABLE vehicle_reg (
    vehicle_id VARCHAR(20) PRIMARY KEY,
    owner_id INTEGER NOT NULL,
    vehicle_no VARCHAR(20) UNIQUE NOT NULL,
    chassis_no VARCHAR(100) UNIQUE NOT NULL,
    vehicle_company VARCHAR(100) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    vehicle_year INTEGER NOT NULL,
    insurance_policy_ref_no VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (owner_id)
        REFERENCES users(user_id)
        ON DELETE CASCADE
);

-- =====================================================
-- CLAIMS TABLE
-- =====================================================

CREATE TABLE claims (
    claim_id VARCHAR(20) PRIMARY KEY,
    user_id INTEGER NOT NULL,
    vehicle_id VARCHAR(20) NOT NULL,
    incident_type VARCHAR(100),
    incident_description TEXT,
    claim_status VARCHAR(30)
        CHECK (
            claim_status IN (
                'CREATED',
                'PROCESSING',
                'AI_RUNNING',
                'AI_COMPLETED',
                'UNDER_REVIEW',
                'APPROVED',
                'REJECTED',
                'CLOSED'
            )
        )
        DEFAULT 'CREATED',
    ai_processing_status VARCHAR(30)
        CHECK (
            ai_processing_status IN (
                'PENDING',
                'RUNNING',
                'COMPLETED',
                'FAILED'
            )
        )
        DEFAULT 'PENDING',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id)
        REFERENCES users(user_id)
        ON DELETE CASCADE,
    FOREIGN KEY (vehicle_id)
        REFERENCES vehicle_reg(vehicle_id)
        ON DELETE CASCADE
);

-- =====================================================
-- CLAIM IMAGES TABLE
-- =====================================================

CREATE TABLE claim_images (
    image_id SERIAL PRIMARY KEY,
    claim_id VARCHAR(20) NOT NULL,
    image_type VARCHAR(50) NOT NULL,
    image_url TEXT NOT NULL,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (claim_id)
        REFERENCES claims(claim_id)
        ON DELETE CASCADE
);

-- =====================================================
-- AI ANALYSIS RESULTS TABLE
-- =====================================================

CREATE TABLE ai_analysis_results (
    analysis_id VARCHAR(20) PRIMARY KEY,
    claim_id VARCHAR(20) NOT NULL,
    damage_type VARCHAR(150) NOT NULL,
    confidence_score DECIMAL(5,2),
    severity_level VARCHAR(30)
        CHECK (
            severity_level IN (
                'LOW',
                'MODERATE',
                'HIGH',
                'SEVERE'
            )
        ),
    estimated_cost NUMERIC(12,2),
    generated_report_url TEXT,
    ai_model_version VARCHAR(50),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (claim_id)
        REFERENCES claims(claim_id)
        ON DELETE CASCADE
);

-- =====================================================
-- SEGMENTED RESULTS TABLE
-- =====================================================

CREATE TABLE segmented_results (
    segment_id VARCHAR(20) PRIMARY KEY,
    analysis_id VARCHAR(20) NOT NULL,
    part_name VARCHAR(100) NOT NULL,
    segment_confidence DECIMAL(5,2),
    damage_area_percent DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (analysis_id)
        REFERENCES ai_analysis_results(analysis_id)
        ON DELETE CASCADE
);

-- =====================================================
-- MISSING PARTS TABLE
-- =====================================================

CREATE TABLE missing_parts (
    missing_id VARCHAR(20) PRIMARY KEY,
    analysis_id VARCHAR(20) NOT NULL,
    part_name VARCHAR(100) NOT NULL,
    confidence_score DECIMAL(5,2),
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (analysis_id)
        REFERENCES ai_analysis_results(analysis_id)
        ON DELETE CASCADE
);

-- =====================================================
-- REPORTS TABLE
-- =====================================================

CREATE TABLE reports (
    report_id VARCHAR(20) PRIMARY KEY,
    claim_id VARCHAR(20) NOT NULL,
    report_url TEXT NOT NULL,
    heatmap_url TEXT,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (claim_id)
        REFERENCES claims(claim_id)
        ON DELETE CASCADE
);

-- =====================================================
-- CLAIM HISTORY TABLE
-- =====================================================

CREATE TABLE claim_history (
    history_id VARCHAR(20) PRIMARY KEY,
    claim_id VARCHAR(20) NOT NULL,
    old_status VARCHAR(30),
    new_status VARCHAR(30) NOT NULL,
    changed_by VARCHAR(50),
    changed_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (claim_id)
        REFERENCES claims(claim_id)
        ON DELETE CASCADE
);

-- =====================================================
-- EMBEDDINGS TABLE (pgvector)
-- =====================================================

CREATE TABLE embeddings (
    embedding_id SERIAL PRIMARY KEY,
    analysis_id VARCHAR(20) NOT NULL,
    embedding VECTOR(768),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (analysis_id)
        REFERENCES ai_analysis_results(analysis_id)
        ON DELETE CASCADE
);

-- =====================================================
-- INDEXES
-- =====================================================

CREATE INDEX idx_claims_user_id ON claims(user_id);
CREATE INDEX idx_claims_vehicle_id ON claims(vehicle_id);
CREATE INDEX idx_claim_images_claim_id ON claim_images(claim_id);
CREATE INDEX idx_ai_analysis_claim_id ON ai_analysis_results(claim_id);
CREATE INDEX idx_segmented_results_analysis_id ON segmented_results(analysis_id);
CREATE INDEX idx_missing_parts_analysis_id ON missing_parts(analysis_id);
CREATE INDEX idx_reports_claim_id ON reports(claim_id);
CREATE INDEX idx_claim_history_claim_id ON claim_history(claim_id);

-- =====================================================
-- VECTOR INDEX (Approximate Similarity Search)
-- =====================================================

CREATE INDEX idx_embeddings_vector ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
