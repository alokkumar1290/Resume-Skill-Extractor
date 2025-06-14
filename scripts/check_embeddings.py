"""
Simple script to check if embeddings exist in the database.
"""
import sqlite3
import json
from pathlib import Path

def main():
    db_path = Path("resumes.db")
    if not db_path.exists():
        print(f"❌ Database not found at {db_path}")
        return
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get count of resumes with and without embeddings
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN embedding IS NULL OR embedding = '' THEN 1 ELSE 0 END) as missing_embeddings,
                SUM(CASE WHEN embedding IS NOT NULL AND embedding != '' THEN 1 ELSE 0 END) as has_embeddings
            FROM resumes
        """)
        
        total, missing, has_embeddings = cursor.fetchone()
        
        print(f"📊 Database: {db_path}")
        print(f"📄 Total resumes: {total}")
        print(f"✅ Resumes with embeddings: {has_embeddings}")
        print(f"❌ Resumes missing embeddings: {missing}")
        
        # Show a sample of the embeddings
        if has_embeddings > 0:
            print("\nSample embedding (first 3 vectors):")
            cursor.execute("""
                SELECT id, name, 
                       substr(embedding, 1, 100) || '...' as embedding_sample
                FROM resumes 
                WHERE embedding IS NOT NULL AND embedding != ''
                LIMIT 1
            """)
            for row in cursor.fetchall():
                print(f"ID: {row[0]}, Name: {row[1]}")
                print(f"Embedding: {row[2]}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()
