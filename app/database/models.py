from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, func, CheckConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from typing import Optional

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
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith('sqlite') else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db() -> None:
    """
    Initialize the database by creating all tables.
    """
    Base.metadata.create_all(bind=engine)

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
