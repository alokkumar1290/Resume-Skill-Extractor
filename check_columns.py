import sqlite3

def check_columns():
    try:
        conn = sqlite3.connect('resumes.db')
        cursor = conn.cursor()
        
        # Get table info
        cursor.execute("PRAGMA table_info(resumes)")
        columns = cursor.fetchall()
        
        print("\nColumns in 'resumes' table:")
        print("-" * 50)
        for col in columns:
            print(f"Column {col[0]}: {col[1]} ({col[2]})")
            
        # Check if 'embedding' exists
        column_names = [col[1] for col in columns]
        if 'embedding' in column_names:
            print("\n✅ 'embedding' column exists in the database.")
        else:
            print("\n❌ 'embedding' column is MISSING from the database.")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_columns()
