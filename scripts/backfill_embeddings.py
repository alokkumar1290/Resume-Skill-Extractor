"""
One-off script to add the embedding column (if absent) and back-fill embeddings for all resumes.
Run: python scripts/backfill_embeddings.py
"""
import sqlite3
import json
import time
from pathlib import Path
import sys
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.processing.embeddings import embed_text

def get_db_path() -> Path:
    """Get the path to the SQLite database file."""
    # Try to get the database URL from environment variables
    from app.database.models import DATABASE_URL
    if DATABASE_URL and DATABASE_URL.startswith('sqlite:///'):
        db_path = Path(DATABASE_URL.replace('sqlite:///', ''))
        if db_path.exists():
            return db_path
    
    # Default path if not found in environment
    default_path = PROJECT_ROOT / 'resumes.db'
    if default_path.exists():
        return default_path
    
    # Try to find the database file in the parent directory
    parent_path = PROJECT_ROOT.parent / 'resumes.db'
    if parent_path.exists():
        return parent_path
    
    return default_path  # Will create a new database if it doesn't exist

def main():
    print("🚀 Starting resume embedding backfill...")
    
    # Get database path
    db_path = get_db_path()
    print(f"📂 Using database at: {db_path}")
    
    if not db_path.exists():
        print("❌ Database file not found. Please make sure the application has been run at least once.")
        return
    
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check if embedding column exists, add if not
        print("🔍 Checking database schema...")
        cursor.execute("PRAGMA table_info(resumes)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'embedding' not in columns:
            print("ℹ️ Adding embedding column to resumes table...")
            cursor.execute("ALTER TABLE resumes ADD COLUMN embedding TEXT")
            conn.commit()
            print("✅ Added embedding column")
        else:
            print("✅ Embedding column exists")
        
        # Get all resumes without embeddings
        print("🔍 Finding resumes that need embeddings...")
        cursor.execute("""
            SELECT id, raw_text 
            FROM resumes 
            WHERE embedding IS NULL OR embedding = ''
        """)
        
        resumes = cursor.fetchall()
        
        if not resumes:
            print("✅ All resumes already have embeddings!")
            return
            
        print(f"🔧 Found {len(resumes)} resumes that need embeddings")
        
        # Process each resume
        success_count = 0
        error_count = 0
        
        for resume_id, raw_text in tqdm(resumes, desc="Generating embeddings"):
            try:
                if not raw_text:
                    print(f"⚠️ No raw text for resume ID {resume_id}")
                    error_count += 1
                    continue
                    
                # Generate embedding
                embedding_vector = embed_text(raw_text)
                
                if not embedding_vector:
                    print(f"⚠️ Failed to generate embedding for resume ID {resume_id}")
                    error_count += 1
                    continue
                
                # Update resume with embedding
                cursor.execute(
                    "UPDATE resumes SET embedding = ? WHERE id = ?",
                    (json.dumps(embedding_vector), resume_id)
                )
                conn.commit()
                success_count += 1
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Error processing resume ID {resume_id}: {str(e)}")
                error_count += 1
                conn.rollback()
                continue
        
        # Print summary
        print("\n" + "="*50)
        print(f"✅ Successfully processed {success_count} resumes")
        if error_count > 0:
            print(f"⚠️  Failed to process {error_count} resumes")
        print("="*50)
        
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()
