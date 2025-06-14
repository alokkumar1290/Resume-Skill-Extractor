from sqlalchemy.orm import Session
from sqlalchemy import or_
import json
from typing import List, Dict, Any, Optional, Tuple
import pickle, numpy as np
from pathlib import Path
from .models import Resume, SessionLocal

def set_hired(resume_id: int, hired: bool = True):
    db = SessionLocal()
    try:
        r = db.query(Resume).get(resume_id)
        if r:
            r.hired = 1 if hired else 0
            db.commit()
            return True
        return False
    finally:
        db.close()

def save_resume(resume_data: Dict[str, Any]) -> Resume:
    """
    Save resume data to the database.
    
    Args:
        resume_data: Dictionary containing resume information
        
    Returns:
        Resume: The saved Resume object
    """
    db = SessionLocal()
    try:
        # Convert lists/dicts to JSON strings
        skills = resume_data.get('skills', {})
        experience = resume_data.get('experience', [])
        education = resume_data.get('education', [])
        raw_text = resume_data.get('raw_text', '')
        
        # Ensure skills is a dictionary with technical and soft keys
        if not isinstance(skills, dict):
            skills = {'technical': [], 'soft': []}
        if 'technical' not in skills:
            skills['technical'] = []
        if 'soft' not in skills:
            skills['soft'] = []
            
        skills_json = json.dumps(skills)
        experience_json = json.dumps(experience)
        education_json = json.dumps(education)
        
        # Generate embedding for the raw text
        from app.processing.embeddings import embed_text
        try:
            embedding_vector = embed_text(raw_text)
            embedding_json = json.dumps(embedding_vector) if embedding_vector else None
        except Exception as e:
            print(f"Error generating embedding: {e}")
            embedding_json = None
        
        # Create new Resume object
        db_resume = Resume(
            name=resume_data.get('name', ''),
            email=resume_data.get('email', ''),
            phone=resume_data.get('phone', ''),
            skills=skills_json,
            experience=experience_json,
            education=education_json,
            cgpa=float(resume_data.get('cgpa')) if resume_data.get('cgpa') is not None else None,
            raw_text=raw_text,
            embedding=embedding_json
        )
        
        db.add(db_resume)
        db.commit()
        db.refresh(db_resume)
        return db_resume
    except Exception as e:
        db.rollback()
        print(f"Error saving resume: {e}")
        raise e
    finally:
        db.close()

def get_all_resumes(limit: int = 100, skip: int = 0) -> List[Resume]:
    """
    Retrieve all resumes with pagination.
    
    Args:
        limit: Maximum number of resumes to return
        skip: Number of resumes to skip
        
    Returns:
        List[Resume]: List of Resume objects
    """
    db = SessionLocal()
    try:
        return db.query(Resume).offset(skip).limit(limit).all()
    finally:
        db.close()

def get_resume_by_id(resume_id: int) -> Optional[Resume]:
    """
    Retrieve a single resume by ID.
    
    Args:
        resume_id: ID of the resume to retrieve
        
    Returns:
        Optional[Resume]: The Resume object if found, None otherwise
    """
    db = SessionLocal()
    try:
        return db.query(Resume).filter(Resume.id == resume_id).first()
    finally:
        db.close()

def search_resumes(
    skill: str = None,
    min_cgpa: float = None,
    company: str = None,
    degree: str = None,
    limit: int = 50
) -> List[Resume]:
    """
    Search resumes based on various criteria.
    
    Args:
        skill: Search for resumes with this skill
        min_cgpa: Minimum CGPA filter
        company: Search by company name in experience
        degree: Search by degree in education
        limit: Maximum number of results to return
        
    Returns:
        List[Resume]: List of matching Resume objects
    """
    db = SessionLocal()
    try:
        query = db.query(Resume)
        
        # Apply filters
        if skill:
            query = query.filter(Resume.skills.ilike(f'%{skill}%'))
            
        if company:
            query = query.filter(Resume.experience.ilike(f'%{company}%'))
            
        if degree:
            query = query.filter(Resume.education.ilike(f'%{degree}%'))
        
        # For CGPA, we need to parse the education JSON
        if min_cgpa is not None:
            # This is a simplified approach - in production, you'd want to use JSON functions
            # or a more sophisticated search engine
            all_resumes = query.all()
            filtered_resumes = []
            
            for resume in all_resumes:
                try:
                    if not resume.education:
                        continue
                        
                    educations = json.loads(resume.education)
                    if not isinstance(educations, list):
                        educations = [educations]
                        
                    for edu in educations:
                        if not isinstance(edu, dict):
                            continue
                            
                        cgpa = edu.get('cgpa')
                        if cgpa and float(cgpa) >= min_cgpa:
                            filtered_resumes.append(resume)
                            break
                except (json.JSONDecodeError, ValueError, AttributeError):
                    continue
                    
            return filtered_resumes[:limit]
            
        return query.limit(limit).all()
    finally:
        db.close()

def delete_resume(resume_id: int) -> bool:
    """
    Delete a resume by ID.
    
    Args:
        resume_id: ID of the resume to delete
        
    Returns:
        bool: True if deleted, False if not found
    """
    db = SessionLocal()
    try:
        resume = db.query(Resume).filter(Resume.id == resume_id).first()
        if resume:
            db.delete(resume)
            db.commit()
            return True
        return False
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

# ------------------ JD Matching helper ------------------

def match_job_description(jd_text: str, top_n: int = 10, min_similarity: float = 0.0) -> List[Tuple[Resume, float]]:
    """
    Find resumes that best match a job description using semantic similarity.
    
    Args:
        jd_text: The job description text
        top_n: Maximum number of matches to return
        min_similarity: Minimum similarity score (0.0 to 1.0) for a match
        
    Returns:
        List of (Resume, similarity_score) tuples, sorted by score descending
    """
    from app.processing.embeddings import embed_text, cosine_similarity
    import logging
    
    logger = logging.getLogger(__name__)
    
    if not jd_text.strip():
        logger.warning("Empty job description provided")
        return []
    
    try:
        # Generate embedding for the job description
        jd_embedding = embed_text(jd_text)
        if not jd_embedding:
            logger.error("Failed to generate embedding for job description")
            return []
            
        db = SessionLocal()
        try:
            # Get all resumes with embeddings
            resumes = db.query(Resume).filter(
                Resume.embedding.isnot(None),
                Resume.embedding != ''
            ).all()
            
            if not resumes:
                logger.info("No resumes with embeddings found")
                return []
                
            logger.info(f"Processing {len(resumes)} resumes for matching...")
            
            # Calculate similarities
            scored_resumes = []
            for resume in resumes:
                try:
                    # Parse the stored embedding
                    resume_embedding = json.loads(resume.embedding)
                    if not isinstance(resume_embedding, list):
                        logger.warning(f"Invalid embedding format for resume {resume.id}")
                        continue
                        
                    # Calculate similarity
                    similarity = cosine_similarity(jd_embedding, resume_embedding)
                    
                    # Only include if above threshold
                    if similarity >= min_similarity:
                        scored_resumes.append((resume, similarity))
                        
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Error processing resume {resume.id}: {str(e)}")
                    continue
                    
            # Sort by similarity score (descending) and take top_n
            scored_resumes.sort(key=lambda x: x[1], reverse=True)
            return scored_resumes[:top_n]
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error in match_job_description: {str(e)}", exc_info=True)
        return []

# ------------------ ML-based Ranking ------------------
MODEL_PATH = Path(__file__).resolve().parents[2]/"app"/"ml"/"train_ranker.pkl"

def ml_rank_resumes(top_n: int = 20) -> List[Tuple[Resume, float]]:
    if not MODEL_PATH.exists():
        return []
    model = pickle.load(open(MODEL_PATH, "rb"))
    db = SessionLocal()
    out = []
    try:
        for r in db.query(Resume).all():
            cgpa = r.cgpa or 0.0
            try:
                skills = json.loads(r.skills)['technical']
                num_skills = len(skills)
            except Exception:
                num_skills = 0
            try:
                exp = json.loads(r.experience)
                exp_years = len(exp)
            except Exception:
                exp_years = 0
            prob = model.predict_proba([[cgpa, num_skills, exp_years]])[0][1]
            out.append((r, prob))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:top_n]
    finally:
        db.close()

# ------------------ Ranking helper ------------------
def rank_resumes(preferred_skills: List[str] = None, top_n: int = 20) -> List[Resume]:
    """Return resumes sorted by a custom score.

    Score = (cgpa * 2) + (matched tech skills * 3) + (experience entries * 1)
    preferred_skills defaults to ["AI", "ML"] if not provided.
    """
    if preferred_skills is None or len(preferred_skills) == 0:
        preferred_skills = ["AI", "ML"]

    db = SessionLocal()
    try:
        all_resumes = db.query(Resume).all()
        scored: List[Tuple[Resume, float]] = []

        for r in all_resumes:
            # CGPA
            cgpa_val = float(r.cgpa) if r.cgpa else 0.0

            # Experience count
            try:
                exp_entries = len(json.loads(r.experience)) if r.experience else 0
            except json.JSONDecodeError:
                exp_entries = 0

            # Tech skills matching
            try:
                skills_dict = json.loads(r.skills) if r.skills else {}
                tech_skills = [s.lower() for s in skills_dict.get("technical", [])]
            except json.JSONDecodeError:
                tech_skills = []

            matches = sum(1 for skill in preferred_skills if skill.lower() in tech_skills)

            # Final score (weights can be tuned)
            score = (cgpa_val * 2) + (matches * 3) + exp_entries
            scored.append((r, score))

        # sort high to low
        scored.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in scored][:top_n]
    finally:
        db.close()
