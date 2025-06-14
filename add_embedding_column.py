import sys
from sqlalchemy import create_engine, text
from app.database.models import Base, DATABASE_URL

def add_embedding_column():
    # Create engine
    engine = create_engine(DATABASE_URL)
    
    # Check if the column already exists
    with engine.connect() as conn:
        # For SQLite
        result = conn.execute(
            "SELECT name FROM pragma_table_info('resumes') WHERE name='embedding'"
        )
        column_exists = result.fetchone() is not None
        
        if not column_exists:
            print("Adding 'embedding' column to 'resumes' table...")
            try:
                # Add the new column
                conn.execute(text("ALTER TABLE resumes ADD COLUMN embedding TEXT"))
                conn.commit()
                print("Successfully added 'embedding' column.")
            except Exception as e:
                print(f"Error adding column: {e}")
                conn.rollback()
        else:
            print("'embedding' column already exists in 'resumes' table.")

if __name__ == "__main__":
    add_embedding_column()
