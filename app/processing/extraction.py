import json
import os
import pdfplumber
import re
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import LangChain components with fallback
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain_community.llms import HuggingFaceEndpoint
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain components not available. Some features may be limited.")
    LANGCHAIN_AVAILABLE = False

# Load environment variables
load_dotenv()

# Get API key and model from environment variables
HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_API_KEY")
HF_MODEL = os.getenv("HF_MODEL", 'mistralai/Mistral-7B-Instruct-v0.2')

# Initialize LLM if API key is available
llm = None
if HF_API_KEY and LANGCHAIN_AVAILABLE:
    try:
        llm = HuggingFaceEndpoint(
            repo_id=HF_MODEL,
            huggingfacehub_api_token=HF_API_KEY,
            temperature=0.3,
            max_new_tokens=1024,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            stop_sequences=["\n```"]
        )
        logger.info("Successfully initialized HuggingFace LLM")
    except Exception as e:
        logger.warning(f"Failed to initialize HuggingFace LLM: {str(e)}")
        llm = None
else:
    if not HF_API_KEY:
        logger.warning("No HuggingFace API key found. Some features may be limited.")
    if not LANGCHAIN_AVAILABLE:
        logger.warning("LangChain is not available. Some features may be limited.")

# Prompt template for resume extraction
RESUME_EXTRACTION_PROMPT = """
You are an expert resume parser. Read the provided resume text and output ONLY a valid JSON object with the following exact structure and field names:

name: string
email: string
phone: string
skills: object with two arrays -> technical (array of strings) and soft (array of strings)
experience: array of objects; each object has role, company, duration, description (all strings)
education: array of objects; each object has degree, institution, cgpa, date_range (all strings)

If a field is not present, use an empty string "" for strings or an empty array [] for arrays.
Return ONLY the JSON object and nothing else. Do not wrap it in markdown. Ensure the JSON is valid and uses double quotes for keys and string values.

Resume text:
{resume_text}
"""

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF file with improved handling of different formats.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: Extracted text with preserved formatting
    """
    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Try different extraction strategies
                strategies = [
                    lambda p: p.extract_text(x_tolerance=1, y_tolerance=1),
                    lambda p: p.extract_text(x_tolerance=3, y_tolerance=3),
                    lambda p: p.extract_text(x_tolerance=1, y_tolerance=3),
                    lambda p: p.extract_text(x_tolerance=3, y_tolerance=1)
                ]
                
                page_text = ""
                for strategy in strategies:
                    try:
                        current_text = strategy(page)
                        if current_text and len(current_text) > len(page_text):
                            page_text = current_text
                    except:
                        continue
                
                if not page_text:
                    # Fallback to default extraction
                    page_text = page.extract_text() or ""
                
                if page_text:
                    # Clean up common PDF extraction artifacts
                    page_text = page_text.replace('\x0c', '\n')  # Form feed to newline
                    page_text = ' '.join(page_text.split())  # Normalize whitespace
                    text_parts.append(page_text.strip())
        
        # Combine pages with clear section breaks
        combined_text = "\n\n".join(filter(None, text_parts))
        return combined_text.strip() or ""
        
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return ""

def clean_json_response(response: str) -> str:
    """Clean and extract JSON from LLM response."""
    json_str = response.strip()
    
    # Handle code block formatting
    if '```json' in json_str:
        json_str = json_str.split('```json')[1].split('```')[0].strip()
    elif '```' in json_str:
        json_str = json_str.split('```')[1].strip()
    
    # Find first { and last }
    json_start = json_str.find('{')
    json_end = json_str.rfind('}') + 1
    
    if json_start >= 0 and json_end > 0:
        json_str = json_str[json_start:json_end]
    
    # Fix common JSON issues
    json_str = json_str.replace('\n', ' ').replace('\t', ' ')
    json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
    json_str = re.sub(r',\s*]', ']', json_str)
    
    return json_str

def _parse_float(value: Optional[str]):
    """Extract first numeric float in a string, return None if not found."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else None

def extract_resume_data(pdf_path: Path) -> Dict[str, Any]:
    """
    Extract structured data from a resume PDF using LLM with fallback to basic extraction.
    
    Args:
        pdf_path: Path to the resume PDF file
        
    Returns:
        Dict containing extracted resume data
    """
    try:
        logger.info(f"Starting to process PDF: {pdf_path}")
        
        # Extract raw text with improved extraction
        text = extract_text_from_pdf(pdf_path)
        logger.info(f"Extracted {len(text)} characters from PDF")
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        # Basic resume data structure
        resume_data = {
            'name': '',
            'email': '',
            'phone': '',
            'skills': {'technical': [], 'soft': []},
            'experience': [],
            'education': [],
            'cgpa': None,
            'raw_text': text
        }
        
        # Try to extract basic information using regex patterns
        try:
            # Extract email
            email_match = re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
            if email_match:
                resume_data['email'] = email_match.group(0)
            
            # Extract phone number (various formats)
            phone_match = re.search(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
            if phone_match:
                resume_data['phone'] = phone_match.group(0)
            
            # Try to extract name (first line with title case that's not too long)
            first_few_lines = text.split('\n')[:5]
            for line in first_few_lines:
                line = line.strip()
                if (line.istitle() and 
                    len(line.split()) in [2, 3] and 
                    len(line) < 50 and 
                    not any(word.lower() in line.lower() for word in ['phone', 'email', 'address', 'resume'])):
                    resume_data['name'] = line
                    break
            
            # Extract skills (simple keyword matching)
            skills_keywords = ['python', 'java', 'javascript', 'sql', 'machine learning', 'data analysis',
                             'project management', 'teamwork', 'communication', 'leadership']
            
            for skill in skills_keywords:
                if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
                    if skill in ['teamwork', 'communication', 'leadership']:
                        resume_data['skills']['soft'].append(skill)
                    else:
                        resume_data['skills']['technical'].append(skill)
            
            # If LLM is available, use it for more detailed extraction
            if llm and LANGCHAIN_AVAILABLE:
                logger.info("LLM available, performing advanced extraction...")
                try:
                    # Limit text length and clean it up
                    text_sample = text[:4000].strip()
                    
                    # Create LLM chain for extraction
                    prompt = PromptTemplate(
                        input_variables=["resume_text"],
                        template=RESUME_EXTRACTION_PROMPT
                    )
                    
                    # Initialize and run LLM chain
                    chain = LLMChain(llm=llm, prompt=prompt)
                    llm_output = chain.run(resume_text=text_sample).strip()
                    
                    # Clean and parse LLM response
                    json_str = clean_json_response(llm_output)
                    llm_data = json.loads(json_str)
                    
                    # Merge LLM data with basic extraction
                    if isinstance(llm_data, dict):
                        for key in ['name', 'email', 'phone']:
                            if llm_data.get(key) and not resume_data.get(key):
                                resume_data[key] = llm_data[key]
                        
                        if 'skills' in llm_data and isinstance(llm_data['skills'], dict):
                            resume_data['skills']['technical'].extend(
                                s for s in llm_data['skills'].get('technical', []) 
                                if s not in resume_data['skills']['technical']
                            )
                            resume_data['skills']['soft'].extend(
                                s for s in llm_data['skills'].get('soft', []) 
                                if s not in resume_data['skills']['soft']
                            )
                        
                        if 'experience' in llm_data and isinstance(llm_data['experience'], list):
                            resume_data['experience'] = llm_data['experience']
                        
                        if 'education' in llm_data and isinstance(llm_data['education'], list):
                            resume_data['education'] = llm_data['education']
                            # Try to extract CGPA from education section if not already found
                            if resume_data['cgpa'] is None:
                                for edu in resume_data['education']:
                                    if isinstance(edu, dict) and 'cgpa' in edu and edu['cgpa']:
                                        try:
                                            cgpa = _parse_float(edu['cgpa'])
                                            if cgpa is not None and 0 <= cgpa <= 10:
                                                resume_data['cgpa'] = cgpa
                                                logger.info(f"Extracted CGPA from education section: {cgpa}")
                                                break
                                        except (ValueError, TypeError):
                                            continue
                        
                        if 'cgpa' in llm_data and llm_data['cgpa'] is not None and resume_data['cgpa'] is None:
                            try:
                                cgpa = _parse_float(llm_data['cgpa'])
                                if cgpa is not None and 0 <= cgpa <= 10:
                                    resume_data['cgpa'] = cgpa
                                    logger.info(f"Extracted CGPA from LLM: {cgpa}")
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Failed to parse CGPA from LLM: {str(e)}")
                    
                except Exception as llm_error:
                    logger.warning(f"LLM extraction failed, using basic extraction: {str(llm_error)}")
            
            return resume_data
            
        except Exception as basic_error:
            logger.error(f"Basic extraction failed: {str(basic_error)}")
            # Return at least the raw text if everything else fails
            return {'raw_text': text}
        
        logger.info("Basic extraction completed")
        return resume_data
        
    except Exception as e:
        logger.error(f"Error in extract_resume_data: {str(e)}")
        # Return minimal data structure with error information
        return {
            'error': str(e),
            'raw_text': text if 'text' in locals() else 'Failed to extract text'
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
