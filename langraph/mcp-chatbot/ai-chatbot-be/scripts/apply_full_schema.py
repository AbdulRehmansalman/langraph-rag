"""
Apply Full Production Schema to Supabase
========================================
This script reads the production-optimized schema.sql and applies it to
the Supabase database defined in DATABASE_URL.

Enterprise Features:
- HNSW Vector Indexes
- Analytics & Monitoring Tables
- Query & Embedding Cache
- Row Level Security (RLS)
- Automated Triggers
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def apply_schema():
    # 1. Get database URL
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        logger.error("DATABASE_URL not found in environment variables!")
        sys.exit(1)

    # 2. Fix driver prefix for SQLAlchemy if needed
    if db_url.startswith('postgresql://'):
        db_url = db_url.replace('postgresql://', 'postgresql+psycopg2://')

    # 3. Locate schema.sql
    base_dir = Path(__file__).parent.parent
    schema_path = base_dir / "app" / "database" / "schema.sql"
    
    if not schema_path.exists():
        logger.error(f"schema.sql not found at {schema_path}")
        sys.exit(1)

    logger.info(f"Reading schema from {schema_path}")
    with open(schema_path, 'r') as f:
        sql_script = f.read()

    # 4. Connect and Execute
    logger.info("Connecting to Supabase...")
    engine = create_engine(db_url, pool_pre_ping=True)
    
    try:
        with engine.connect() as conn:
            # 5. Pre-migration: Add missing columns to existing tables if needed
            logger.info("Pre-migration check: Ensuring columns exist for critical tables...")
            
            # Helper to add columns if they don't exist
            def add_columns_if_missing(table, column_defs):
                for col_name, col_type in column_defs.items():
                    try:
                        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col_name} {col_type};"))
                        conn.commit()
                    except Exception as e:
                        logger.warning(f"Could not add {col_name} to {table}: {e}")

            # Define missing columns for 'users'
            users_cols = {
                "role": "VARCHAR(50) DEFAULT 'user'",
                "plan_type": "VARCHAR(50) DEFAULT 'free'",
                "storage_limit_mb": "INTEGER DEFAULT 100",
                "monthly_query_limit": "INTEGER DEFAULT 1000",
                "current_month_queries": "INTEGER DEFAULT 0",
                "last_login_at": "TIMESTAMP WITH TIME ZONE",
                "updated_at": "TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP"
            }
            add_columns_if_missing("users", users_cols)

            # Define missing columns for 'documents'
            docs_cols = {
                "file_size_bytes": "BIGINT DEFAULT 0",
                "processing_status": "VARCHAR(50) DEFAULT 'pending'",
                "processing_started_at": "TIMESTAMP WITH TIME ZONE",
                "processing_completed_at": "TIMESTAMP WITH TIME ZONE",
                "error_message": "TEXT",
                "page_count": "INTEGER",
                "word_count": "INTEGER",
                "language": "VARCHAR(10)",
                "document_hash": "VARCHAR(64)",
                "quality_score": "DECIMAL(3,2) DEFAULT 1.0",
                "embedding_count": "INTEGER DEFAULT 0",
                "updated_at": "TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP"
            }
            add_columns_if_missing("documents", docs_cols)

            # Define missing columns for 'document_embeddings'
            embeddings_cols = {
                "content_preview": "VARCHAR(500)",
                "content_vector": "TSVECTOR",
                "retrieval_count": "INTEGER DEFAULT 0",
                "avg_retrieval_score": "DECIMAL(4,3) DEFAULT 0.0",
                "last_retrieved_at": "TIMESTAMP WITH TIME ZONE",
                "metadata": "JSONB DEFAULT '{}'",
                "page_number": "INTEGER",
                "word_count": "INTEGER"
            }
            add_columns_if_missing("document_embeddings", embeddings_cols)

            # Define missing columns for 'chat_history'
            chat_cols = {
                "session_id": "UUID",
                "total_response_time_ms": "INTEGER",
                "retrieval_time_ms": "INTEGER",
                "generation_time_ms": "INTEGER",
                "confidence_score": "DECIMAL(3,2) DEFAULT 0.0",
                "sources_used": "INTEGER DEFAULT 0",
                "provider_model": "VARCHAR(100)",
                "updated_at": "TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP"
            }
            add_columns_if_missing("chat_history", chat_cols)

            # Map old 'response_time' (NUMERIC(5,3)) to 'total_response_time_ms' if possible
            try:
                conn.execute(text("UPDATE chat_history SET total_response_time_ms = CAST(response_time * 1000 AS INTEGER) WHERE total_response_time_ms IS NULL AND response_time IS NOT NULL;"))
                conn.commit()
            except Exception:
                pass

            # Map old 'processed' boolean to 'processing_status'
            try:
                conn.execute(text("UPDATE documents SET processing_status = 'completed' WHERE processed = true;"))
                conn.execute(text("UPDATE documents SET processing_status = 'failed' WHERE processed = false AND error IS NOT NULL;"))
                conn.commit()
            except Exception:
                pass

            logger.info("Applying full schema (this may take a few moments)...")
            
            # Use raw connection to execute the multi-statement script
            raw_conn = engine.raw_connection()
            try:
                cursor = raw_conn.cursor()
                cursor.execute(sql_script)
                raw_conn.commit()
                logger.info("✓ Full schema applied successfully!")
            except Exception as e:
                raw_conn.rollback()
                logger.error(f"Failed to apply schema: {str(e)}")
                raise
            finally:
                raw_conn.close()

            # 5. Verify tables
            logger.info("Verifying tables...")
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """))
            tables = [row[0] for row in result]
            logger.info(f"Existing tables in public schema: {', '.join(tables)}")
            
            # Check for key new tables
            critical_tables = ['query_cache', 'daily_analytics', 'system_health_logs']
            missing = [t for t in critical_tables if t not in tables]
            if missing:
                logger.warning(f"Some production tables seem to be missing: {', '.join(missing)}")
            else:
                logger.info("✓ All production-optimized tables verified!")

    except Exception as e:
        logger.critical(f"Critical error during migration: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    apply_schema()
