-- knowledge schema (Person 3). __DIM__ is replaced by the MEASURED embedding dimension.
-- banking.* and system.* belong to Person 1 and are not touched here.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS knowledge;

CREATE TABLE IF NOT EXISTS knowledge.documents (
    document_id         text PRIMARY KEY,
    title               text NOT NULL,
    body                text NOT NULL,
    source_record_id    text NOT NULL,
    category            text,
    category_method     text,
    source_name         text NOT NULL,
    provenance          text NOT NULL,
    verification_status text NOT NULL CHECK (verification_status IN ('VERIFIED_PUBLIC','DEMO_LABELLED')),
    answer_label        text NOT NULL,
    version             text NOT NULL DEFAULT 'unversioned',
    effective_date      date,
    source_url          text,
    source_page         text,
    needs_fact_check    boolean NOT NULL DEFAULT false,
    fact_check_reasons  text,
    product_ambiguous   boolean NOT NULL DEFAULT false,
    language            text NOT NULL DEFAULT 'en',
    ingested_at         timestamptz NOT NULL DEFAULT now(),
    -- a record may only claim VERIFIED_PUBLIC if it names its public source
    CONSTRAINT verified_needs_url CHECK (verification_status <> 'VERIFIED_PUBLIC' OR coalesce(source_url,'') <> '')
);

CREATE TABLE IF NOT EXISTS knowledge.document_chunks (
    chunk_id     text PRIMARY KEY,
    document_id  text NOT NULL REFERENCES knowledge.documents(document_id) ON DELETE CASCADE,
    chunk_index  integer NOT NULL,
    chunk_text   text NOT NULL,
    metadata     jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding    vector(__DIM__) NOT NULL,
    fts          tsvector GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED,
    UNIQUE (document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON knowledge.document_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_fts_gin        ON knowledge.document_chunks USING gin (fts);
CREATE INDEX IF NOT EXISTS chunks_document_idx   ON knowledge.document_chunks (document_id);

-- which model produced the vectors currently stored (audit trail)
CREATE TABLE IF NOT EXISTS knowledge.embedding_runs (
    run_id       serial PRIMARY KEY,
    model_id     text NOT NULL,
    backend      text NOT NULL,
    is_standin   boolean NOT NULL,
    dimension    integer NOT NULL,
    chunk_count  integer NOT NULL,
    manifest     jsonb NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);
