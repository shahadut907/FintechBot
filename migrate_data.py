# scripts/migrate_csv_to_db.py - Simple migration script for beginners
"""
Simple CSV to PostgreSQL Migration Script
This script helps you move from CSV files to PostgreSQL database
"""
import asyncio
import asyncpg
import pandas as pd
import os
import hashlib
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import json

# Import your existing roles configuration
import sys
sys.path.append('..')
from config.roles import DOCUMENT_CATEGORIES, DEMO_USERS

class FinSolveMigrator:
    """Simple migrator for moving CSV data to PostgreSQL"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
        
    async def connect(self):
        """Connect to PostgreSQL database"""
        try:
            self.pool = await asyncpg.create_pool(self.database_url)
            print("✅ Connected to PostgreSQL successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to PostgreSQL: {e}")
            print("💡 Make sure PostgreSQL is running and connection details are correct")
            return False
    
    async def close(self):
        """Close database connection"""
        if self.pool:
            await self.pool.close()
            print("✅ Database connection closed")
    
    async def create_default_organization(self) -> str:
        """Create a default organization for your data"""
        org_name = "FinSolve Technologies"
        subdomain = "finsolve"
        
        async with self.pool.acquire() as conn:
            # Check if organization already exists
            existing_org = await conn.fetchval(
                "SELECT id FROM organizations WHERE subdomain = $1", 
                subdomain
            )

            
            if existing_org:
                print(f"✅ Organization '{org_name}' already exists")
                return str(existing_org)
            
            # Create new organization
            org_id = await conn.fetchval("""
                INSERT INTO organizations (name, subdomain, plan_type, max_users)
                VALUES ($1, $2, 'enterprise', 1000)
                RETURNING id
            """, org_name, subdomain)
            
            print(f"✅ Created organization: {org_name} (ID: {org_id})")
            return str(org_id)
    
    async def create_default_users(self, org_id: str):
        """Create default users from your existing DEMO_USERS"""
        print("👥 Creating default users...")
        
        async with self.pool.acquire() as conn:
            for email, user_data in DEMO_USERS.items():
                try:
                    # Hash password (in production, use proper password hashing)
                    password_hash = f"$2b$12${hashlib.md5(user_data['password'].encode()).hexdigest()}"
                    
                    user_id = await conn.fetchval("""
                        INSERT INTO users (organization_id, email, password_hash, name, role, department)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT (organization_id, email) DO UPDATE SET
                            name = EXCLUDED.name,
                            role = EXCLUDED.role,
                            department = EXCLUDED.department
                        RETURNING id
                    """, org_id, email, password_hash, user_data['name'], 
                         user_data['role'], user_data['department'])
                    
                    print(f"  ✅ Created user: {user_data['name']} ({user_data['role']})")
                    
                except Exception as e:
                    print(f"  ❌ Failed to create user {email}: {e}")
    
    async def setup_rbac_permissions(self, org_id: str):
        """Set up role-based access control permissions"""
        print("🔐 Setting up RBAC permissions...")
        
        # Define role-department access matrix
        permissions = [
            # General access for everyone
            ('employee', 'general', True, False, False),
            ('engineering', 'general', True, False, False),
            ('marketing', 'general', True, False, False),
            ('finance', 'general', True, False, False),
            ('hr', 'general', True, False, False),
            ('c_level', 'general', True, True, True),
            
            # Department-specific access
            ('engineering', 'engineering', True, True, False),
            ('marketing', 'marketing', True, True, False),
            ('finance', 'finance', True, True, True),
            ('hr', 'hr', True, True, True),
            
            # C-level access to everything
            ('c_level', 'engineering', True, True, True),
            ('c_level', 'marketing', True, True, True),
            ('c_level', 'finance', True, True, True),
            ('c_level', 'hr', True, True, True),
            ('c_level', 'executive', True, True, True),
        ]
        
        async with self.pool.acquire() as conn:
            for role, department, can_read, can_write, can_export in permissions:
                await conn.execute("""
                    INSERT INTO role_permissions 
                    (organization_id, role, department, can_read, can_write, can_export)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (organization_id, role, department) DO UPDATE SET
                        can_read = EXCLUDED.can_read,
                        can_write = EXCLUDED.can_write,
                        can_export = EXCLUDED.can_export
                """, org_id, role, department, can_read, can_write, can_export)
        
        print("✅ RBAC permissions configured")
    
    async def migrate_csv_file(self, csv_path: str, org_id: str, admin_user_id: str):
        """Migrate a single CSV file to database"""
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            filename = os.path.basename(csv_path)
            
            print(f"📊 Migrating CSV: {filename} ({len(df)} rows)")
            
            # Get file info
            file_size = os.path.getsize(csv_path)
            with open(csv_path, 'r', encoding='utf-8') as f:
                content = f.read()
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Determine department from filename
            doc_meta = DOCUMENT_CATEGORIES.get(filename, {
                "department": "general",
                "type": "data"
            })
            
            department = doc_meta.get("department", "general")
            if hasattr(department, 'value'):
                department = department.value
            
            async with self.pool.acquire() as conn:
                # Insert document record
                doc_id = await conn.fetchval("""
                    INSERT INTO documents 
                    (organization_id, filename, file_type, department, content_type, 
                     file_size, file_hash, metadata, created_by)
                    VALUES ($1, $2, 'csv', $3, 'tabular', $4, $5, $6, $7)
                    ON CONFLICT (organization_id, file_hash) DO UPDATE SET
                        filename = EXCLUDED.filename,
                        updated_at = NOW()
                    RETURNING id
                """, org_id, filename, department, file_size, file_hash, 
                     json.dumps({"rows": len(df), "columns": list(df.columns)}), 
                     admin_user_id)
                
                # Create searchable chunks from CSV data
                chunks = self.create_csv_chunks(df, filename)
                
                # Insert chunks
                for i, chunk_content in enumerate(chunks):
                    chunk_hash = hashlib.md5(chunk_content.encode()).hexdigest()
                    
                    await conn.execute("""
                        INSERT INTO document_chunks 
                        (document_id, organization_id, chunk_index, content, content_hash, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT (document_id, chunk_index) DO UPDATE SET
                            content = EXCLUDED.content,
                            content_hash = EXCLUDED.content_hash
                    """, doc_id, org_id, i, chunk_content, chunk_hash, 
                         json.dumps({"chunk_type": "csv_data", "source_file": filename}))
                
                print(f"  ✅ Created {len(chunks)} searchable chunks")
                
        except Exception as e:
            print(f"  ❌ Failed to migrate {csv_path}: {e}")
    
    async def migrate_markdown_file(self, md_path: str, org_id: str, admin_user_id: str):
        """Migrate a single Markdown file to database"""
        try:
            # Read markdown file
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            filename = os.path.basename(md_path)
            print(f"📄 Migrating Markdown: {filename}")
            
            # Get file info
            file_size = len(content.encode('utf-8'))
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Determine department from filename
            doc_meta = DOCUMENT_CATEGORIES.get(filename, {
                "department": "general",
                "type": "document"
            })
            
            department = doc_meta.get("department", "general")
            if hasattr(department, 'value'):
                department = department.value
            
            async with self.pool.acquire() as conn:
                # Insert document record
                doc_id = await conn.fetchval("""
                    INSERT INTO documents 
                    (organization_id, filename, file_type, department, content_type, 
                     file_size, file_hash, metadata, created_by)
                    VALUES ($1, $2, 'markdown', $3, 'text', $4, $5, $6, $7)
                    ON CONFLICT (organization_id, file_hash) DO UPDATE SET
                        filename = EXCLUDED.filename,
                        updated_at = NOW()
                    RETURNING id
                """, org_id, filename, department, file_size, file_hash, 
                     json.dumps({"type": "markdown"}), admin_user_id)
                
                # Create searchable chunks from markdown
                chunks = self.create_markdown_chunks(content)
                
                # Insert chunks
                for i, chunk_content in enumerate(chunks):
                    chunk_hash = hashlib.md5(chunk_content.encode()).hexdigest()
                    
                    await conn.execute("""
                        INSERT INTO document_chunks 
                        (document_id, organization_id, chunk_index, content, content_hash, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT (document_id, chunk_index) DO UPDATE SET
                            content = EXCLUDED.content,
                            content_hash = EXCLUDED.content_hash
                    """, doc_id, org_id, i, chunk_content, chunk_hash, 
                         json.dumps({"chunk_type": "markdown", "source_file": filename}))
                
                print(f"  ✅ Created {len(chunks)} searchable chunks")
                
        except Exception as e:
            print(f"  ❌ Failed to migrate {md_path}: {e}")
    
    def create_csv_chunks(self, df: pd.DataFrame, filename: str) -> List[str]:
        """Create searchable text chunks from CSV data"""
        chunks = []
        
        # Summary chunk
        summary = f"Dataset: {filename}\n"
        summary += f"Total Records: {len(df)}\n"
        summary += f"Columns: {', '.join(df.columns.tolist())}\n\n"
        
        # Add column statistics
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                summary += f"{col}: min={df[col].min()}, max={df[col].max()}, avg={df[col].mean():.2f}\n"
            elif df[col].dtype == 'object':
                unique_count = df[col].nunique()
                summary += f"{col}: {unique_count} unique values\n"
                if unique_count <= 10:
                    top_values = df[col].value_counts().head(5)
                    summary += f"  Top values: {dict(top_values)}\n"
        
        chunks.append(summary)
        
        # Create data chunks (10 rows per chunk for smaller datasets)
        chunk_size = 10
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i+chunk_size]
            chunk_content = f"Data from {filename} (rows {i} to {i + len(chunk_df) - 1}):\n\n"
            
            for idx, row in chunk_df.iterrows():
                chunk_content += f"Record {idx}:\n"
                for col, val in row.items():
                    chunk_content += f"  {col}: {val}\n"
                chunk_content += "\n"
            
            chunks.append(chunk_content)
        
        return chunks
    
    def create_markdown_chunks(self, content: str) -> List[str]:
        """Create chunks from markdown content"""
        import re
        
        # Split by headers
        sections = re.split(r'\n(?=#{1,6}\s)', content)
        
        chunks = []
        current_chunk = ""
        max_chunk_size = 1000
        
        for section in sections:
            if len(current_chunk) + len(section) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = section
            else:
                current_chunk += "\n" + section if current_chunk else section
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out very small chunks
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    async def migrate_directory(self, source_dir: str, org_id: str):
        """Migrate all CSV and Markdown files from a directory"""
        source_path = Path(source_dir)
        
        if not source_path.exists():
            print(f"❌ Source directory not found: {source_dir}")
            return
        
        # Get admin user for file ownership
        async with self.pool.acquire() as conn:
            admin_user_id = await conn.fetchval(
                "SELECT id FROM users WHERE organization_id = $1 AND role = 'c_level' LIMIT 1",
                org_id
            )
        
        if not admin_user_id:
            print("❌ No admin user found")
            return
        
        print(f"📁 Migrating files from: {source_dir}")
        
        # Migrate CSV files
        csv_files = list(source_path.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files")
        for csv_file in csv_files:
            await self.migrate_csv_file(str(csv_file), org_id, str(admin_user_id))
        
        # Migrate Markdown files
        md_files = list(source_path.glob("*.md"))
        print(f"Found {len(md_files)} Markdown files")
        for md_file in md_files:
            await self.migrate_markdown_file(str(md_file), org_id, str(admin_user_id))
        
        print(f"✅ Migration completed! Processed {len(csv_files)} CSV and {len(md_files)} MD files")
    
    async def run_full_migration(self, source_directory: str):
        """Run the complete migration process"""
        print("🚀 Starting FinSolve CSV to PostgreSQL Migration")
        print("=" * 50)
        
        try:
            # 1. Create organization
            org_id = await self.create_default_organization()
            
            # 2. Set up RBAC permissions
            await self.setup_rbac_permissions(org_id)
            
            # 3. Create users
            await self.create_default_users(org_id)
            
            # 4. Migrate documents
            await self.migrate_directory(source_directory, org_id)
            
            print("\n🎉 Migration completed successfully!")
            print(f"📊 Organization ID: {org_id}")
            print("🔐 You can now log in with your existing demo users")
            print("💡 Update your backend .env file with the database connection")
            
        except Exception as e:
            print(f"\n❌ Migration failed: {e}")
            raise


async def main():
    """Main migration function"""
    print("FinSolve Data Migration Tool")
    print("=" * 30)
    
    # Configuration
    import urllib.parse
    password = urllib.parse.quote_plus("7nt5$mV1kdv@c1$d0")
    DATABASE_URL = f"postgresql://postgres:{password}@127.0.0.1:5432/finsolve_prod"
    SOURCE_DIRECTORY = "../data/raw"  # Adjust path to your CSV/MD files
    
    # Ask user for confirmation
    print(f"📍 Database: {DATABASE_URL}")
    print(f"📁 Source Directory: {SOURCE_DIRECTORY}")
    print()
    
    confirm = input("Do you want to proceed with migration? (y/N): ").lower()
    if confirm != 'y':
        print("Migration cancelled.")
        return
    
    # Run migration
    migrator = FinSolveMigrator(DATABASE_URL)
    
    try:
        # Connect to database
        if not await migrator.connect():
            return
        
        # Run migration
        await migrator.run_full_migration(SOURCE_DIRECTORY)
        
    except Exception as e:
        print(f"❌ Migration error: {e}")
    finally:
        await migrator.close()


if __name__ == "__main__":
    # Make sure you have the required packages
    try:
        import asyncpg
        import pandas as pd
    except ImportError as e:
        print(f"❌ Missing required packages: {e}")
        print("Please install with: pip install asyncpg pandas")
        exit(1)
    
    # Run the migration
    asyncio.run(main())