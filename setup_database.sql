-- Database Setup Script for FinSolve Enhanced Backend
-- Run this in PostgreSQL after creating the database

-- Create database (run this separately first)
-- CREATE DATABASE finsolve_prod;

-- Connect to the database and run the following:

-- 1. ORGANIZATIONS TABLE
CREATE TABLE IF NOT EXISTS organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    subdomain VARCHAR(100) UNIQUE NOT NULL,
    plan_type VARCHAR(50) DEFAULT 'free',
    max_users INTEGER DEFAULT 10,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    settings JSONB DEFAULT '{}'::jsonb
);

-- 2. USERS TABLE
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    email VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL CHECK (role IN ('employee', 'engineering', 'marketing', 'finance', 'hr', 'c_level')),
    department VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    preferences JSONB DEFAULT '{}'::jsonb,
    UNIQUE(organization_id, email)
);

-- 3. DOCUMENTS TABLE
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    department VARCHAR(100) NOT NULL,
    content_type VARCHAR(100) DEFAULT 'text',
    file_size BIGINT,
    file_hash VARCHAR(64),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by UUID REFERENCES users(id),
    UNIQUE(organization_id, file_hash)
);

-- 4. DOCUMENT CHUNKS TABLE
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(64),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(document_id, chunk_index)
);

-- 5. CONVERSATION SESSIONS TABLE
CREATE TABLE IF NOT EXISTS conversation_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    session_name VARCHAR(255),
    current_topic VARCHAR(255),
    mentioned_entities TEXT[],
    topic_history TEXT[],
    preferred_language VARCHAR(20) DEFAULT 'english',
    conversation_mode VARCHAR(20) DEFAULT 'business',
    casual_interactions INTEGER DEFAULT 0,
    business_interactions INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- 6. CONVERSATION MESSAGES TABLE
CREATE TABLE IF NOT EXISTS conversation_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES conversation_sessions(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    message_type VARCHAR(20) NOT NULL CHECK (message_type IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    detected_language VARCHAR(20) DEFAULT 'english',
    conversation_mode VARCHAR(20) DEFAULT 'business',
    sources JSONB DEFAULT '[]'::jsonb,
    processing_time_ms INTEGER,
    documents_found INTEGER DEFAULT 0,
    compliance_level VARCHAR(10) DEFAULT 'LOW',
    access_denied BOOLEAN DEFAULT FALSE,
    success BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- 7. AUDIT LOGS TABLE
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    event_type VARCHAR(100) NOT NULL,
    event_details TEXT,
    sensitive BOOLEAN DEFAULT FALSE,
    ip_address INET,
    user_agent TEXT,
    session_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

-- 8. ROLE PERMISSIONS TABLE
CREATE TABLE IF NOT EXISTS role_permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL,
    department VARCHAR(100) NOT NULL,
    can_read BOOLEAN DEFAULT FALSE,
    can_write BOOLEAN DEFAULT FALSE,
    can_export BOOLEAN DEFAULT FALSE,
    can_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(organization_id, role, department)
);

-- INDEXES for better performance
CREATE INDEX IF NOT EXISTS idx_users_org_email ON users(organization_id, email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_documents_org_dept ON documents(organization_id, department);
CREATE INDEX IF NOT EXISTS idx_chunks_org ON document_chunks(organization_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON conversation_sessions(user_id, is_active);
CREATE INDEX IF NOT EXISTS idx_messages_session ON conversation_messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_user ON conversation_messages(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_org_created ON audit_logs(organization_id, created_at);
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_logs(user_id);

-- Insert default organization
INSERT INTO organizations (name, subdomain, plan_type, max_users) 
VALUES ('FinSolve Technologies', 'finsolve', 'enterprise', 1000)
ON CONFLICT (subdomain) DO NOTHING;

-- Get the organization ID for inserting users
DO $$
DECLARE
    org_id UUID;
BEGIN
    SELECT id INTO org_id FROM organizations WHERE subdomain = 'finsolve';
    
    -- Insert demo users with hashed passwords
    INSERT INTO users (organization_id, email, password_hash, name, role, department) VALUES
    (org_id, 'tony.sharma@finsolve.com', '$2b$12$' || md5('password123'), 'Tony Sharma', 'c_level', 'Executive'),
    (org_id, 'finance.lead@finsolve.com', '$2b$12$' || md5('password123'), 'Finance Lead', 'finance', 'Finance'),
    (org_id, 'peter.pandey@finsolve.com', '$2b$12$' || md5('password123'), 'Peter Pandey', 'engineering', 'Engineering'),
    (org_id, 'marketing.lead@finsolve.com', '$2b$12$' || md5('password123'), 'Marketing Lead', 'marketing', 'Marketing'),
    (org_id, 'hr.lead@finsolve.com', '$2b$12$' || md5('password123'), 'HR Lead', 'hr', 'Human Resources'),
    (org_id, 'employee@finsolve.com', '$2b$12$' || md5('password123'), 'General Employee', 'employee', 'General')
    ON CONFLICT (organization_id, email) DO NOTHING;
    
    -- Insert role permissions
    INSERT INTO role_permissions (organization_id, role, department, can_read, can_write, can_export) VALUES
    -- General access for everyone
    (org_id, 'employee', 'general', TRUE, FALSE, FALSE),
    (org_id, 'engineering', 'general', TRUE, FALSE, FALSE),
    (org_id, 'marketing', 'general', TRUE, FALSE, FALSE),
    (org_id, 'finance', 'general', TRUE, FALSE, FALSE),
    (org_id, 'hr', 'general', TRUE, FALSE, FALSE),
    (org_id, 'c_level', 'general', TRUE, TRUE, TRUE),
    
    -- Department-specific access
    (org_id, 'engineering', 'engineering', TRUE, TRUE, FALSE),
    (org_id, 'marketing', 'marketing', TRUE, TRUE, FALSE),
    (org_id, 'finance', 'finance', TRUE, TRUE, TRUE),
    (org_id, 'hr', 'hr', TRUE, TRUE, TRUE),
    
    -- C-level access to everything
    (org_id, 'c_level', 'engineering', TRUE, TRUE, TRUE),
    (org_id, 'c_level', 'marketing', TRUE, TRUE, TRUE),
    (org_id, 'c_level', 'finance', TRUE, TRUE, TRUE),
    (org_id, 'c_level', 'hr', TRUE, TRUE, TRUE),
    (org_id, 'c_level', 'executive', TRUE, TRUE, TRUE)
    ON CONFLICT (organization_id, role, department) DO NOTHING;
    
END $$;

-- Verification queries
SELECT 'Organizations created:' as info, COUNT(*) as count FROM organizations;
SELECT 'Users created:' as info, COUNT(*) as count FROM users;
SELECT 'Permissions created:' as info, COUNT(*) as count FROM role_permissions;

-- Show created users
SELECT email, name, role, department FROM users ORDER BY role;