from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, func, CheckConstraint, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import sys
import logging
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create base class for models
Base = declarative_base()

class Resume(Base):
    """Database model for storing resume information."""
    __tablename__ = 'resumes'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True)
    phone = Column(String(50), nullable=True)
    skills = Column(Text, nullable=True)  # Store as JSON string
    experience = Column(Text, nullable=True)  # Store as JSON string
    education = Column(Text, nullable=True)  # Store as JSON string
    cgpa = Column(Float, CheckConstraint('cgpa >= 0 AND cgpa <= 10', name='check_cgpa_range'), nullable=True)
    embedding = Column(Text, nullable=True)  # JSON list of floats
    hired = Column(Integer, default=0)  # 1 if candidate hired/shortlisted
    raw_text = Column(Text, nullable=True)  # Store full extracted text
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    def to_dict(self) -> dict:
        """Convert model instance to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'phone': self.phone,
            'skills': self.skills,
            'experience': self.experience,
            'education': self.education,
            'cgpa': self.cgpa,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# Database connection and session management
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./resumes.db')

try:
    # For SQLite, ensure the directory exists
    if DATABASE_URL.startswith('sqlite'):
        db_path = DATABASE_URL.split('sqlite:///')[-1]
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        logger.info(f"Using SQLite database at: {os.path.abspath(db_path)}")

    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False} if DATABASE_URL.startswith('sqlite') else {},
        echo=True  # This will log all SQL queries
    )
    
    # Enable foreign key constraints for SQLite
    if DATABASE_URL.startswith('sqlite'):
        @event.listens_for(engine, 'connect')
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
except Exception as e:
    logger.error(f"Failed to initialize database: {str(e)}")
    raise

def init_db() -> None:
    """Initialize the database."""
    try:
        logger.info("Initializing database...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

def get_db():
    """
    Dependency to get DB session.
    Yields:
        Session: Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
