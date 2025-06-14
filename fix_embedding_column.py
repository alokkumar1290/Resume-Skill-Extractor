import sqlite3
import os

def check_and_fix_embedding_column():
    db_path = 'resumes.db'
    
    if not os.path.exists(db_path):
        print(f"Database file {db_path} not found!")
        return
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if the column exists
        cursor.execute("PRAGMA table_info(resumes)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'embedding' not in columns:
            print("Adding 'embedding' column to 'resumes' table...")
            cursor.execute("ALTER TABLE resumes ADD COLUMN embedding TEXT")
            conn.commit()
            print("Successfully added 'embedding' column.")
        else:
            print("'embedding' column already exists in 'resumes' table.")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    check_and_fix_embedding_column()
