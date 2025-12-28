-- =============================================================================
-- ENTERPRISE RAG DATABASE SCHEMA - PRODUCTION OPTIMIZED
-- =============================================================================
-- Combines best practices from both schemas with enterprise-grade optimizations
-- Features: HNSW vector indexes, caching, analytics, RLS policies, monitoring
-- =============================================================================

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";           -- pgvector for embeddings
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements"; -- Query performance
CREATE EXTENSION IF NOT EXISTS "pg_trgm";          -- Text pattern matching
CREATE EXTENSION IF NOT EXISTS "pgcrypto";         -- Cryptography functions

-- =============================================================================
-- CORE TABLES
-- =============================================================================

-- Users table with enterprise features
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    email_verified BOOLEAN DEFAULT FALSE,
    role VARCHAR(50) DEFAULT 'user',
    plan_type VARCHAR(50) DEFAULT 'free',
    storage_limit_mb INTEGER DEFAULT 100,
    monthly_query_limit INTEGER DEFAULT 1000,
    current_month_queries INTEGER DEFAULT 0,
    last_login_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_email CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
    CONSTRAINT valid_role CHECK (role IN ('user', 'admin', 'super_admin')),
    CONSTRAINT valid_plan CHECK (plan_type IN ('free', 'pro', 'enterprise'))
);

-- Documents table with processing status
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(500) NOT NULL,
    content_type VARCHAR(100) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    file_path VARCHAR(1000) NOT NULL,
    storage_url TEXT,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Processing status
    processing_status VARCHAR(50) DEFAULT 'pending',
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    
    -- Document metadata
    page_count INTEGER,
    word_count INTEGER,
    language VARCHAR(10),
    document_hash VARCHAR(64),
    
    -- Quality metrics
    quality_score DECIMAL(3,2) DEFAULT 1.0,
    embedding_count INTEGER DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_processing_status CHECK (
        processing_status IN ('pending', 'processing', 'completed', 'failed')
    ),
    CONSTRAINT valid_quality_score CHECK (quality_score >= 0 AND quality_score <= 1)
);

-- Document embeddings with pgvector support
CREATE TABLE IF NOT EXISTS document_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Chunk information
    chunk_index INTEGER NOT NULL,
    chunk_hash VARCHAR(64) NOT NULL,
    
    -- Content
    content TEXT NOT NULL,
    content_preview VARCHAR(500),
    content_vector TSVECTOR,
    embedding VECTOR(768) NOT NULL, -- 768D for all-mpnet-base-v2
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    page_number INTEGER,
    word_count INTEGER,
    
    -- Performance tracking
    retrieval_count INTEGER DEFAULT 0,
    avg_retrieval_score DECIMAL(4,3) DEFAULT 0.0,
    last_retrieved_at TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(document_id, chunk_hash)
);

-- Chat history with comprehensive metrics
CREATE TABLE IF NOT EXISTS chat_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id UUID,
    
    -- Conversation
    user_message TEXT NOT NULL,
    bot_response TEXT NOT NULL,
    
    -- Document references
    document_ids UUID[] DEFAULT '{}',
    
    -- Performance metrics
    total_response_time_ms INTEGER NOT NULL,
    retrieval_time_ms INTEGER,
    generation_time_ms INTEGER,
    
    -- Quality metrics
    confidence_score DECIMAL(3,2) DEFAULT 0.0,
    has_documents BOOLEAN DEFAULT FALSE,
    sources_used INTEGER DEFAULT 0,
    
    -- Provider info
    provider VARCHAR(100) DEFAULT 'ollama',
    provider_model VARCHAR(100),
    template_used VARCHAR(50) DEFAULT 'default',
    model_config VARCHAR(50) DEFAULT 'conversational',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_confidence CHECK (confidence_score >= 0 AND confidence_score <= 1)
);

-- =============================================================================
-- AUTHENTICATION & AUTHORIZATION TABLES
-- =============================================================================

-- OTP table
CREATE TABLE IF NOT EXISTS otps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL,
    otp VARCHAR(10) NOT NULL,
    purpose VARCHAR(50) NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    used BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Google authentication
CREATE TABLE IF NOT EXISTS user_google_auth (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    access_token TEXT NOT NULL,
    refresh_token TEXT,
    expires_at TIMESTAMP WITH TIME ZONE,
    scopes TEXT[],
    token_type VARCHAR(50) DEFAULT 'Bearer',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id)
);

-- Meetings table
CREATE TABLE IF NOT EXISTS meetings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    scheduled_time TIMESTAMP WITH TIME ZONE NOT NULL,
    duration_minutes INTEGER DEFAULT 60,
    google_event_id VARCHAR(255),
    google_meet_link TEXT,
    calendar_link TEXT,
    status VARCHAR(50) DEFAULT 'scheduled',
    raw_request TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- CACHING TABLES FOR PERFORMANCE
-- =============================================================================

-- Query cache for frequent queries
CREATE TABLE IF NOT EXISTS query_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_hash VARCHAR(64) NOT NULL UNIQUE,
    query_text TEXT NOT NULL,
    query_embedding VECTOR(768),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- Cached response
    response_text TEXT NOT NULL,
    response_metadata JSONB DEFAULT '{}',
    
    -- Performance data
    original_response_time_ms INTEGER,
    cache_hit_count INTEGER DEFAULT 0,
    
    -- Expiry
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Embedding cache to reduce API calls
CREATE TABLE IF NOT EXISTS embedding_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text_hash VARCHAR(64) NOT NULL UNIQUE,
    text_content TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    usage_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- ANALYTICS TABLES
-- =============================================================================

-- Daily analytics
CREATE TABLE IF NOT EXISTS daily_analytics (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    total_messages INTEGER DEFAULT 0,
    avg_response_time_ms DECIMAL(8,2) DEFAULT 0,
    messages_with_documents INTEGER DEFAULT 0,
    unique_users INTEGER DEFAULT 0,
    avg_confidence_score DECIMAL(4,3) DEFAULT 0,
    cache_hit_rate DECIMAL(5,4) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User session analytics
CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_date DATE NOT NULL,
    message_count INTEGER DEFAULT 0,
    avg_response_time_ms DECIMAL(8,2) DEFAULT 0,
    documents_used INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, session_date)
);

-- Document analytics
CREATE TABLE IF NOT EXISTS document_analytics (
    id SERIAL PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    usage_date DATE NOT NULL,
    query_count INTEGER DEFAULT 0,
    avg_relevance_score DECIMAL(3,2) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, usage_date)
);

-- System health logs
CREATE TABLE IF NOT EXISTS system_health_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    component VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    response_time_ms DECIMAL(8,2),
    error_message TEXT,
    metadata JSONB DEFAULT '{}'
);

-- =============================================================================
-- HNSW INDEXES FOR OPTIMAL VECTOR SEARCH
-- =============================================================================

-- HNSW index for cosine similarity (primary for text embeddings)
CREATE INDEX IF NOT EXISTS idx_doc_embeddings_embedding_hnsw_cosine ON document_embeddings 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- HNSW for query cache
CREATE INDEX IF NOT EXISTS idx_query_cache_embedding_hnsw ON query_cache 
    USING hnsw (query_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE query_embedding IS NOT NULL;

-- =============================================================================
-- COMPREHENSIVE INDEXES
-- =============================================================================

-- Users indexes
CREATE INDEX IF NOT EXISTS idx_users_email_lower ON users(LOWER(email));
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_plan_type ON users(plan_type);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at DESC);

-- Documents indexes
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_processing_status ON documents(processing_status);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_quality_score ON documents(quality_score DESC);

-- Document embeddings indexes
CREATE INDEX IF NOT EXISTS idx_doc_embeddings_document_id ON document_embeddings(document_id);
CREATE INDEX IF NOT EXISTS idx_doc_embeddings_chunk_index ON document_embeddings(chunk_index);
CREATE INDEX IF NOT EXISTS idx_doc_embeddings_content_vector ON document_embeddings USING gin(content_vector);
CREATE INDEX IF NOT EXISTS idx_doc_embeddings_retrieval_count ON document_embeddings(retrieval_count DESC);

-- Chat history indexes
CREATE INDEX IF NOT EXISTS idx_chat_history_user_id ON chat_history(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_created_at ON chat_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_history_user_created ON chat_history(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_history_session_id ON chat_history(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_document_ids ON chat_history USING gin(document_ids);
CREATE INDEX IF NOT EXISTS idx_chat_history_has_docs ON chat_history(has_documents);

-- OTP indexes
CREATE INDEX IF NOT EXISTS idx_otps_email_purpose ON otps(email, purpose);
CREATE INDEX IF NOT EXISTS idx_otps_expires_at ON otps(expires_at);

-- Cache indexes
CREATE INDEX IF NOT EXISTS idx_query_cache_query_hash ON query_cache(query_hash);
CREATE INDEX IF NOT EXISTS idx_query_cache_expires ON query_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_embedding_cache_text_hash ON embedding_cache(text_hash);
CREATE INDEX IF NOT EXISTS idx_embedding_cache_last_used ON embedding_cache(last_used_at DESC);

-- Analytics indexes
CREATE INDEX IF NOT EXISTS idx_daily_analytics_date ON daily_analytics(date DESC);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_date ON user_sessions(user_id, session_date DESC);
CREATE INDEX IF NOT EXISTS idx_document_analytics_doc_date ON document_analytics(document_id, usage_date DESC);
CREATE INDEX IF NOT EXISTS idx_system_health_timestamp ON system_health_logs(timestamp DESC);

-- =============================================================================
-- FUNCTIONS AND TRIGGERS
-- =============================================================================

-- Update content_vector from content
CREATE OR REPLACE FUNCTION update_content_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_vector = to_tsvector('english', COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_content_vector
    BEFORE INSERT OR UPDATE ON document_embeddings
    FOR EACH ROW
    EXECUTE FUNCTION update_content_vector();

-- Update document embedding count
CREATE OR REPLACE FUNCTION update_document_embedding_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE documents 
        SET embedding_count = embedding_count + 1
        WHERE id = NEW.document_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE documents 
        SET embedding_count = embedding_count - 1
        WHERE id = OLD.document_id;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_embedding_count
    AFTER INSERT OR DELETE ON document_embeddings
    FOR EACH ROW
    EXECUTE FUNCTION update_document_embedding_count();

-- Update user query count
CREATE OR REPLACE FUNCTION update_user_query_count()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE users 
    SET current_month_queries = current_month_queries + 1,
        last_login_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.user_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_user_query_count
    AFTER INSERT ON chat_history
    FOR EACH ROW
    EXECUTE FUNCTION update_user_query_count();

-- Update daily analytics
CREATE OR REPLACE FUNCTION update_daily_analytics()
RETURNS VOID AS $$
BEGIN
    INSERT INTO daily_analytics (
        date, 
        total_messages, 
        avg_response_time_ms, 
        messages_with_documents, 
        unique_users
    )
    SELECT 
        DATE(created_at) as date,
        COUNT(*) as total_messages,
        AVG(total_response_time_ms) as avg_response_time_ms,
        COUNT(*) FILTER (WHERE has_documents = true) as messages_with_documents,
        COUNT(DISTINCT user_id) as unique_users
    FROM chat_history 
    WHERE DATE(created_at) = CURRENT_DATE
    GROUP BY DATE(created_at)
    ON CONFLICT (date) 
    DO UPDATE SET
        total_messages = EXCLUDED.total_messages,
        avg_response_time_ms = EXCLUDED.avg_response_time_ms,
        messages_with_documents = EXCLUDED.messages_with_documents,
        unique_users = EXCLUDED.unique_users,
        updated_at = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- Trigger for automatic analytics updates
CREATE OR REPLACE FUNCTION trigger_update_daily_analytics()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM update_daily_analytics();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS chat_history_analytics_trigger ON chat_history;
CREATE TRIGGER chat_history_analytics_trigger
    AFTER INSERT ON chat_history
    FOR EACH STATEMENT
    EXECUTE FUNCTION trigger_update_daily_analytics();

-- HNSW similarity search with user filtering
CREATE OR REPLACE FUNCTION hnsw_similarity_search(
    query_embedding VECTOR(768),
    user_id_param UUID,
    top_k INTEGER DEFAULT 10,
    ef_search INTEGER DEFAULT 40,
    score_threshold FLOAT DEFAULT 0.15
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    chunk_index INTEGER,
    content TEXT,
    score FLOAT,
    metadata JSONB
) AS $$
BEGIN
    -- Set HNSW search parameter
    EXECUTE format('SET LOCAL hnsw.ef_search = %s', ef_search);
    
    RETURN QUERY
    SELECT 
        de.id as chunk_id,
        de.document_id,
        de.chunk_index,
        de.content,
        1 - (de.embedding <=> query_embedding) as score,
        de.metadata
    FROM document_embeddings de
    JOIN documents d ON de.document_id = d.id
    WHERE d.user_id = user_id_param
      AND de.embedding IS NOT NULL
      AND 1 - (de.embedding <=> query_embedding) >= score_threshold
    ORDER BY de.embedding <=> query_embedding
    LIMIT top_k;
END;
$$ LANGUAGE plpgsql;

-- Cleanup expired cache
CREATE OR REPLACE FUNCTION cleanup_expired_cache()
RETURNS VOID AS $$
BEGIN
    DELETE FROM query_cache WHERE expires_at < CURRENT_TIMESTAMP;
    DELETE FROM embedding_cache WHERE last_used_at < CURRENT_TIMESTAMP - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- VIEWS FOR ANALYTICS
-- =============================================================================

CREATE OR REPLACE VIEW analytics_summary AS
SELECT 
    COUNT(*) as total_messages,
    COUNT(DISTINCT user_id) as unique_users,
    AVG(total_response_time_ms) as avg_response_time_ms,
    COUNT(*) FILTER (WHERE has_documents = true) as messages_with_docs,
    COUNT(*) FILTER (WHERE has_documents = false) as messages_without_docs,
    DATE(created_at) as date
FROM chat_history 
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY date DESC;

CREATE OR REPLACE VIEW user_activity_summary AS
SELECT 
    user_id,
    COUNT(*) as total_messages,
    AVG(total_response_time_ms) as avg_response_time_ms,
    COUNT(*) FILTER (WHERE has_documents = true) as messages_with_docs,
    MAX(created_at) as last_activity
FROM chat_history 
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY user_id
ORDER BY total_messages DESC;

-- =============================================================================
-- ROW LEVEL SECURITY (RLS)
-- =============================================================================

ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_history ENABLE ROW LEVEL SECURITY;

-- Users policies
CREATE POLICY "Users can view own profile" ON users
    FOR SELECT USING (auth.uid()::text = id::text);

CREATE POLICY "Users can update own profile" ON users
    FOR UPDATE USING (auth.uid()::text = id::text);

-- Documents policies
CREATE POLICY "Users can view own documents" ON documents
    FOR SELECT USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can insert own documents" ON documents
    FOR INSERT WITH CHECK (auth.uid()::text = user_id::text);

CREATE POLICY "Users can update own documents" ON documents
    FOR UPDATE USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can delete own documents" ON documents
    FOR DELETE USING (auth.uid()::text = user_id::text);

-- Document embeddings policies
CREATE POLICY "Users can view embeddings of own documents" ON document_embeddings
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM documents 
            WHERE documents.id = document_embeddings.document_id 
            AND documents.user_id::text = auth.uid()::text
        )
    );

CREATE POLICY "Users can insert embeddings for own documents" ON document_embeddings
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM documents 
            WHERE documents.id = document_embeddings.document_id 
            AND documents.user_id::text = auth.uid()::text
        )
    );

CREATE POLICY "Users can delete embeddings of own documents" ON document_embeddings
    FOR DELETE USING (
        EXISTS (
            SELECT 1 FROM documents 
            WHERE documents.id = document_embeddings.document_id 
            AND documents.user_id::text = auth.uid()::text
        )
    );

-- Chat history policies
CREATE POLICY "Users can view own chat history" ON chat_history
    FOR SELECT USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can insert own chat history" ON chat_history
    FOR INSERT WITH CHECK (auth.uid()::text = user_id::text);

CREATE POLICY "Users can delete own chat history" ON chat_history
    FOR DELETE USING (auth.uid()::text = user_id::text);

-- =============================================================================
-- PERFORMANCE OPTIMIZATION SETTINGS
-- =============================================================================

-- Optimize autovacuum for high-traffic tables
ALTER TABLE document_embeddings SET (
    autovacuum_vacuum_scale_factor = 0.1,
    autovacuum_analyze_scale_factor = 0.05
);

ALTER TABLE chat_history SET (
    autovacuum_vacuum_scale_factor = 0.2,
    autovacuum_analyze_scale_factor = 0.1
);

ALTER TABLE query_cache SET (
    autovacuum_vacuum_scale_factor = 0.05,
    autovacuum_analyze_scale_factor = 0.02
);

-- =============================================================================
-- COMMENTS FOR DOCUMENTATION
-- =============================================================================

COMMENT ON TABLE document_embeddings IS 'Vector embeddings with HNSW indexing for fast similarity search';
COMMENT ON INDEX idx_doc_embeddings_embedding_hnsw_cosine IS 'HNSW index for cosine similarity - optimal for text embeddings';
COMMENT ON COLUMN document_embeddings.embedding IS '768-dimensional embedding using all-mpnet-base-v2 model';
COMMENT ON TABLE query_cache IS 'LRU cache for frequent queries to improve response time';
COMMENT ON TABLE embedding_cache IS 'Cache for text embeddings to reduce API calls';
COMMENT ON FUNCTION hnsw_similarity_search IS 'Optimized HNSW search with user filtering and score threshold';
COMMENT ON VIEW analytics_summary IS 'Daily analytics summary for last 30 days';
COMMENT ON VIEW user_activity_summary IS 'Per-user activity metrics for last 30 days';