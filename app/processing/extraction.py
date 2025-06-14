import json
import os
import pdfplumber
from typing import Dict, Any, List, Optional
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceEndpoint
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Get API key and model from environment variables
HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = os.getenv("HF_MODEL", 'mistralai/Mistral-7B-Instruct-v0.2')

# Initialize LLM
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
    Extract structured data from a resume PDF using LLM.
    
    Args:
        pdf_path: Path to the resume PDF file
        
    Returns:
        Dict containing extracted resume data
    """
    try:
        print(f"\n{'='*80}\nStarting to process PDF: {pdf_path}")
        
        # Extract raw text with improved extraction
        text = extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(text)} characters from PDF")
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        # Limit text length and clean it up
        text_sample = text[:4000].strip()
        print(f"\nSample of extracted text (first 200 chars):\n{text_sample[:200]}...")
        
        # Create LLM chain for extraction
        print("\nCreating LLM prompt...")
        prompt = PromptTemplate(
            input_variables=["resume_text"],
            template=RESUME_EXTRACTION_PROMPT
        )
        
        # Use invoke instead of run as per deprecation warning
        print("Initializing LLM chain...")
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Get response from LLM
        print("\nSending request to LLM...")
        llm_output = chain.run(resume_text=text_sample).strip()
        
        print(f"\nRaw LLM response (first 500 chars):\n{llm_output[:500]}...")
        
        # Clean and parse response
        json_str = clean_json_response(llm_output)  # Uses the NEW function you added
        print(f"\nCleaned JSON string (first 200 chars):\n{json_str[:200]}...")

        try:
            resume_data = json.loads(json_str)
            
            # Validate and normalize structure
            default_resume = {
                'name': '',
                'email': '',
                'phone': '',
                'skills': {'technical': [], 'soft': []},
                'experience': [],
                'education': [],
                'cgpa': None,
                'raw_text': text
            }
            
            # Merge with parsed data
            if isinstance(resume_data, dict):
                for key in default_resume:
                    if key in resume_data:
                        if key == 'skills':
                            if isinstance(resume_data[key], dict):
                                default_resume['skills']['technical'] = resume_data[key].get('technical', [])
                                default_resume['skills']['soft'] = resume_data[key].get('soft', [])
                        elif key in ['experience', 'education']:
                            if isinstance(resume_data[key], list):
                                default_resume[key] = resume_data[key]
                        elif key == 'cgpa':
                            default_resume['cgpa'] = _parse_float(resume_data[key])
                        elif key not in ['skills', 'experience', 'education']:
                            default_resume[key] = str(resume_data[key]) if resume_data[key] is not None else ''
            
            # Derive CGPA from education entries if still None
            if default_resume['cgpa'] is None and isinstance(default_resume['education'], list):
                cgpa_values = []
                for edu in default_resume['education']:
                    if isinstance(edu, dict):
                        cg = _parse_float(edu.get('cgpa'))
                        if cg is not None:
                            cgpa_values.append(cg)
                if cgpa_values:
                    default_resume['cgpa'] = max(cgpa_values)
            
            print("\nExtraction successful!")
            return default_resume
            
        except json.JSONDecodeError as e:
            print(f"\nJSON parsing error: {str(e)}")
            # Attempt to fix common issues
            try:
                fixed_json = re.sub(r'("[^"]*")\s*(?=")', r'\1,', json_str)  # Add missing commas
                fixed_json = re.sub(r'("[^"]*")\s*(?={)', r'\1,', fixed_json)
                resume_data = json.loads(fixed_json)
                print("Fixed JSON automatically")
                return resume_data
            except:
                raise ValueError(f"Could not parse JSON. Error: {str(e)}")
                
    except Exception as e:
        print(f"\n❌ Error processing resume: {str(e)}")
        return {
            'name': 'Error',
            'email': '',
            'phone': '',
            'skills': {'technical': [], 'soft': []},
            'experience': [],
            'education': [],
            'cgpa': None,
            'raw_text': f"Error processing resume: {str(e)}"
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
