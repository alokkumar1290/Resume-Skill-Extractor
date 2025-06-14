import os
import shutil
from pathlib import Path

def cleanup_project():
    # Files to remove
    files_to_remove = [
        'add_embedding_column.py',
        'check_columns.py',
        'fix_embedding_column.py',
        'resumes.db',
        'app/ml/train_ranker.pkl',
        '.env',  # Remove the actual .env file (keep .env.example)
        'cleanup.py'  # This script will remove itself
    ]
    
    # Directories to remove
    dirs_to_remove = [
        '__pycache__',
        'app/__pycache__',
        'app/database/__pycache__',
        'app/ml/__pycache__',
        'app/processing/__pycache__',
        'uploads'  # Remove uploads directory
    ]
    
    print("Cleaning up project...")
    
    # Remove files
    for file_path in files_to_remove:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed file: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")
    
    # Remove directories
    for dir_path in dirs_to_remove:
        try:
            if os.path.exists(dir_path):
                if os.path.isfile(dir_path):
                    os.remove(dir_path)
                else:
                    shutil.rmtree(dir_path)
                print(f"Removed directory: {dir_path}")
        except Exception as e:
            print(f"Error removing {dir_path}: {e}")
    
    print("\nCleanup complete!")
    print("\nBefore committing to GitHub, please ensure you have:")
    print("1. A .env.example file with placeholder values")
    print("2. Removed any sensitive data from the code")
    print("3. Updated the README.md with setup instructions")

if __name__ == "__main__":
    cleanup_project()
