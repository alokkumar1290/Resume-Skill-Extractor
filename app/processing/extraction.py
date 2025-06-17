import pdfplumber
import re
import logging
from typing import Dict, Any, List
from pathlib import Path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_resume_data(pdf_path: Path) -> Dict[str, Any]:
    """Extract structured data from a resume PDF."""
    text = ""
    try:
        # Extract text from PDF using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            # Extract text from each page
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        
        logger.debug(f"Extracted text from PDF: {text[:500]}...")  # Log first 500 chars
        
        # Initialize result dictionary
        result = {
            'name': '',
            'email': '',
            'phone': '',
            'cgpa': None,
            'education': [],
            'skills': {'technical': [], 'soft': []},
            'projects': []
        }
        
        if not text.strip():
            logger.error("No text was extracted from the PDF")
            return {'error': 'No text could be extracted from the PDF'}
            
        try:
            logger.debug("Starting data extraction from text")
            
            # Extract name (assuming it's at the beginning of the document)
            name_match = re.search(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text.strip())
            if name_match:
                result['name'] = name_match.group(1).strip()
                logger.debug(f"Extracted name: {result['name']}")
            
            # Extract email
            email_match = re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
            if email_match:
                result['email'] = email_match.group(0).lower()
                
            # Extract phone
            phone_match = re.search(r'\+?[\d\s-]{10,}', text)
            if phone_match:
                result['phone'] = phone_match.group(0).strip()
                
            # Extract CGPA
            cgpa_match = re.search(r'CGPA\s*[:\-]?\s*(\d+\.\d+)', text, re.IGNORECASE)
            if cgpa_match:
                try:
                    result['cgpa'] = float(cgpa_match.group(1))
                except (ValueError, TypeError):
                    pass
                    
            # Extract skills
            tech_skills = ['python', 'c++', 'java', 'sql', 'pytorch', 'matlab', 'arduino', 
                          'machine learning', 'data analysis', 'tensorflow', 'scikit-learn']
            soft_skills = ['communication', 'teamwork', 'leadership', 'problem solving', 
                          'time management', 'critical thinking']
            
            result['skills']['technical'] = [s for s in tech_skills 
                                           if re.search(r'\b' + re.escape(s) + r'\b', text, re.IGNORECASE)]
            result['skills']['soft'] = [s for s in soft_skills 
                                      if re.search(r'\b' + re.escape(s) + r'\b', text, re.IGNORECASE)]
            
            # Simple project extraction (look for bullet points)
            project_matches = re.findall(r'•\s*([^•]+?)(?=\n\s*•|$)', text, re.DOTALL)
            if project_matches:
                result['projects'] = [p.strip() for p in project_matches if len(p.strip().split()) > 3]
            
            # Simple education extraction
            edu_match = re.search(r'(?i)education\s*([^•]+?)(?=\n\s*[A-Z]|$)', text, re.DOTALL)
            if edu_match:
                result['education'] = [edu_match.group(1).strip()]
            
            logger.info("Basic extraction completed")
            return result
            
        except Exception as basic_error:
            logger.error(f"Basic extraction failed: {str(basic_error)}")
            # Return at least the raw text if everything else fails
            return {'raw_text': text}
            
    except Exception as e:
        logger.error(f"Error in extract_resume_data: {str(e)}")
        # Return minimal data structure with error information
        return {
            'error': str(e),
            'raw_text': 'Failed to extract text'
        }

def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split text into chunks for processing with LangChain.
    
    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    texts = text_splitter.split_text(text)
    return [Document(page_content=text) for text in texts]
