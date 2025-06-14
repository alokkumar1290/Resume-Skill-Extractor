from typing import List, Dict, Any
import json
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from config import HF_API_KEY, HF_MODEL

# Initialize the LLM
llm = HuggingFaceHub(
    repo_id=HF_MODEL,
    huggingfacehub_api_token=HF_API_KEY,
    model_kwargs={"temperature": 0.7, "max_new_tokens": 200}
)

# Prompt templates
TECHNICAL_QUESTION_PROMPT = """
Based on the following resume information, generate {num_questions} technical interview questions 
that would be relevant for a candidate with these skills and experience.

Resume Summary:
{resume_summary}

Skills: {skills}

Format the output as a JSON array of strings.
"""

BEHAVIORAL_QUESTION_PROMPT = """
Generate {num_questions} behavioral interview questions based on the candidate's work experience.
Focus on their roles, responsibilities, and achievements mentioned in their resume.

Work Experience:
{experience}

Format the output as a JSON array of strings.
"""

ROLE_SPECIFIC_PROMPT = """
Generate {num_questions} role-specific technical questions for a {target_role} position 
based on the candidate's skills and experience.

Skills: {skills}
Experience: {experience}

Format the output as a JSON array of strings.
"""

def generate_technical_questions(resume_data: Dict[str, Any], num_questions: int = 5) -> List[str]:
    """
    Generate technical interview questions based on the candidate's skills.
    
    Args:
        resume_data: Dictionary containing resume information
        num_questions: Number of questions to generate
        
    Returns:
        List of technical questions
    """
    try:
        # Prepare the resume summary
        summary = f"{resume_data.get('name', 'The candidate')} has experience in {', '.join(resume_data.get('skills', {}).get('technical', []))}."
        skills = ', '.join(resume_data.get('skills', {}).get('technical', []))
        
        # Create prompt
        prompt = PromptTemplate(
            input_variables=["resume_summary", "skills", "num_questions"],
            template=TECHNICAL_QUESTION_PROMPT
        )
        
        # Create and run chain
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run({
            'resume_summary': summary,
            'skills': skills,
            'num_questions': num_questions
        })
        
        # Parse response
        questions = json.loads(response)
        return questions if isinstance(questions, list) else []
        
    except Exception as e:
        print(f"Error generating technical questions: {str(e)}")
        return []

def generate_behavioral_questions(experience: List[Dict[str, str]], num_questions: int = 5) -> List[str]:
    """
    Generate behavioral interview questions based on work experience.
    
    Args:
        experience: List of work experience entries
        num_questions: Number of questions to generate
        
    Returns:
        List of behavioral questions
    """
    try:
        # Format experience as text
        exp_text = "\n".join([
            f"- {exp.get('role', 'Role')} at {exp.get('company', 'Company')}: {exp.get('description', '')}"
            for exp in experience
        ])
        
        # Create prompt
        prompt = PromptTemplate(
            input_variables=["experience", "num_questions"],
            template=BEHAVIORAL_QUESTION_PROMPT
        )
        
        # Create and run chain
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run({
            'experience': exp_text,
            'num_questions': num_questions
        })
        
        # Parse response
        questions = json.loads(response)
        return questions if isinstance(questions, list) else []
        
    except Exception as e:
        print(f"Error generating behavioral questions: {str(e)}")
        return []

def generate_role_specific_questions(
    resume_data: Dict[str, Any], 
    target_role: str, 
    num_questions: int = 5
) -> List[str]:
    """
    Generate role-specific technical questions.
    
    Args:
        resume_data: Dictionary containing resume information
        target_role: Target role/position
        num_questions: Number of questions to generate
        
    Returns:
        List of role-specific questions
    """
    try:
        # Prepare data
        skills = ', '.join(resume_data.get('skills', {}).get('technical', []))
        experience = "\n".join([
            f"- {exp.get('role', 'Role')} at {exp.get('company', 'Company')}: {exp.get('description', '')}"
            for exp in resume_data.get('experience', [])
        ])
        
        # Create prompt
        prompt = PromptTemplate(
            input_variables=["target_role", "skills", "experience", "num_questions"],
            template=ROLE_SPECIFIC_PROMPT
        )
        
        # Create and run chain
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run({
            'target_role': target_role,
            'skills': skills,
            'experience': experience,
            'num_questions': num_questions
        })
        
        # Parse response
        questions = json.loads(response)
        return questions if isinstance(questions, list) else []
        
    except Exception as e:
        print(f"Error generating role-specific questions: {str(e)}")
        return []

def generate_all_questions(resume_data: Dict[str, Any], target_role: str = None) -> Dict[str, List[str]]:
    """
    Generate all types of interview questions.
    
    Args:
        resume_data: Dictionary containing resume information
        target_role: Optional target role for role-specific questions
        
    Returns:
        Dictionary containing different types of questions
    """
    questions = {
        'technical': generate_technical_questions(resume_data, num_questions=5),
        'behavioral': generate_behavioral_questions(resume_data.get('experience', []), num_questions=5)
    }
    
    if target_role:
        questions['role_specific'] = generate_role_specific_questions(
            resume_data, target_role, num_questions=5
        )
    
    return questions
