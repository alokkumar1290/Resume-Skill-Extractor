import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Load environment variables
from dotenv import load_dotenv
import os
load_dotenv()

import streamlit as st
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
from app.processing.extraction import extract_resume_data
from app.database.models import Resume, init_db
from app.database.crud import save_resume, get_all_resumes, search_resumes, rank_resumes
import json
import pdfplumber

# Page configuration
st.set_page_config(
    page_title="Resume Skill Extractor",
    page_icon="📄",
    layout="wide"
)

from app.database.crud import set_hired

def display_resume(resume: Resume) -> None:
    """Display detailed view of a single resume."""
    st.markdown(f"## {resume.name or 'Unnamed Resume'}")
    
    # Basic info in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📝 Personal Information")
        st.markdown(f"**Email:** {resume.email or 'N/A'}")
        st.markdown(f"**Phone:** {resume.phone or 'N/A'}")
        if hasattr(resume, 'created_at') and resume.created_at:
            st.markdown(f"**Added on:** {resume.created_at.strftime('%Y-%m-%d %H:%M')}")
    
    with col2:
        st.markdown("### 🎓 Education")
        if resume.education:
            try:
                education_data = json.loads(resume.education) if isinstance(resume.education, str) else resume.education
                if isinstance(education_data, list) and education_data:
                    for edu in education_data[:1]:  # Show only the highest education
                        if not isinstance(edu, dict):
                            continue
                        st.markdown(f"**Degree:** {edu.get('degree', 'N/A')}")
                        st.markdown(f"**Institution:** {edu.get('institution', 'N/A')}")
                        if 'cgpa' in edu and edu['cgpa']:
                            try:
                                cgpa = float(edu['cgpa'])
                                st.markdown(f"**CGPA:** {cgpa:.2f}")
                            except (ValueError, TypeError):
                                st.markdown(f"**CGPA:** {edu['cgpa']}")
                else:
                    st.markdown("No education data")
            except (json.JSONDecodeError, TypeError):
                st.markdown("Error parsing education data")
    
    with col3:
        st.markdown("### ⚙️ Quick Stats")
        # Count technical skills
        tech_skill_count = 0
        if resume.skills:
            try:
                skills_data = json.loads(resume.skills) if isinstance(resume.skills, str) else resume.skills
                if isinstance(skills_data, dict) and 'technical' in skills_data:
                    tech_skill_count = len(skills_data['technical'])
            except (json.JSONDecodeError, AttributeError):
                pass
        
        # Count projects/experience
        project_count = 0
        if resume.experience:
            try:
                exp_data = json.loads(resume.experience) if isinstance(resume.experience, str) else resume.experience
                project_count = len(exp_data) if isinstance(exp_data, list) else 0
            except (json.JSONDecodeError, AttributeError):
                pass
        
        st.metric("Technical Skills", tech_skill_count)
        st.metric("Projects/Experience", project_count)
        st.metric("Status", "Hired ✅" if getattr(resume, 'hired', 0) == 1 else "Not Hired ❌")
    
    # Skills section
    st.markdown("---")
    st.markdown("### 🛠️ Technical Skills")
    if resume.skills:
        try:
            skills_data = json.loads(resume.skills) if isinstance(resume.skills, str) else resume.skills
            if isinstance(skills_data, dict) and 'technical' in skills_data and skills_data['technical']:
                # Display skills as chips/tags
                st.write(" ".join([f"`{skill}`" for skill in skills_data['technical'][:10]]))
                if len(skills_data['technical']) > 10:
                    st.write(f"*+ {len(skills_data['technical']) - 10} more skills*")
            else:
                st.info("No technical skills listed")
        except (json.JSONDecodeError, AttributeError):
            st.warning("Could not parse skills data")
    else:
        st.info("No skills information available")
    
    # Experience section
    st.markdown("---")
    st.markdown("### 💼 Experience")
    if resume.experience:
        try:
            exp_data = json.loads(resume.experience) if isinstance(resume.experience, str) else resume.experience
            if isinstance(exp_data, list) and exp_data:
                for exp in exp_data[:5]:  # Limit to 5 most recent
                    if not isinstance(exp, dict):
                        continue
                    with st.expander(f"{exp.get('role', 'Unknown Role')} at {exp.get('company', 'Unknown Company')}", expanded=False):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            if 'description' in exp:
                                st.markdown(exp['description'])
                        with col2:
                            if 'duration' in exp:
                                st.caption(f"⏱️ {exp['duration']}")
                            if 'location' in exp:
                                st.caption(f"📍 {exp['location']}")
            else:
                st.info("No experience data available")
        except (json.JSONDecodeError, AttributeError):
            st.warning("Could not parse experience data")
    else:
        st.info("No experience information available")

# ---------- Helper to show a clean summary ----------
def display_summary(resume_dict: Dict[str, Any]) -> None:
    """Render a concise summary of extracted resume data."""
    st.subheader("📝 Extracted Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Name:** {resume_dict.get('name','N/A')}")
        st.markdown(f"**Email:** {resume_dict.get('email','N/A')}")
        st.markdown(f"**Phone:** {resume_dict.get('phone','N/A')}")
        if resume_dict.get('cgpa') not in (None, ''):
            st.markdown(f"**CGPA:** {resume_dict['cgpa']}")

    with col2:
        skills = resume_dict.get('skills', {}) or {}
        tech_skills = skills.get('technical', [])
        soft_skills = skills.get('soft', [])
        if tech_skills or soft_skills:
            st.markdown("**Skills:**")
            if tech_skills:
                st.markdown("• Technical: " + ", ".join(tech_skills))
            if soft_skills:
                st.markdown("• Soft: " + ", ".join(soft_skills))

    # Work Experience
    experience = resume_dict.get('experience', []) or []
    if experience:
        st.markdown("\n**Work Experience:**")
        for exp in experience:
            role = exp.get('role', '')
            company = exp.get('company', '')
            duration = exp.get('duration', '')
            st.markdown(f"• **{role}** at {company} ({duration})")
    st.write("---")

def main():
    """Main application function."""
    st.title("📄 Resume Skill Extractor")
    
    # Initialize database
    init_db()
    
    # Sidebar for navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Upload Resume", "View Resumes", "Filter", "JD Match", "ML Ranking"]
    )
    
    if page == "Upload Resume":
        st.header("Upload Resume")
        uploaded_file = st.file_uploader(
            "Upload a PDF resume",
            type=["pdf"],
            accept_multiple_files=False,
            help="Upload a single PDF resume for processing (max 10 MB)"
        )
        
        if uploaded_file is not None:
            # Manual size check (10 MB)
            if uploaded_file.size > 10 * 1024 * 1024:
                st.error("File too large. Maximum size is 10 MB.")
                return
                
            with st.spinner("Processing resume..."):
                temp_path = None
                try:
                    # Log file upload
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.session_state.uploaded_file_size = f"{uploaded_file.size / 1024:.1f} KB"
                    
                    # Save uploaded file temporarily
                    temp_dir = Path("temp")
                    temp_dir.mkdir(exist_ok=True)
                    temp_path = temp_dir / f"{os.urandom(8).hex()}_{uploaded_file.name}"
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.info(f"Processing file: {uploaded_file.name} ({st.session_state.uploaded_file_size})")
                    
                    # Extract data
                    with st.spinner("Extracting data from resume..."):
                        resume_data = extract_resume_data(temp_path)
                    
                    if not resume_data:
                        raise ValueError("No data could be extracted from the resume")
                    
                    # Show clean summary upfront
                    with st.expander("📝 Extracted Information", expanded=True):
                        display_summary(resume_data)
                    
                    # Save to database
                    with st.spinner("Saving to database..."):
                        resume = save_resume(resume_data)
                    
                    st.success("✅ Resume processed and saved successfully!")
                    st.balloons()
                    
                    # Store the saved resume ID in session state
                    st.session_state.last_saved_resume_id = resume.id
                    
                    # Display the resume details directly
                    st.markdown("---")
                    display_resume(resume)
                    
                except Exception as e:
                    st.error(f"❌ Error processing resume: {str(e)}")
                    st.error("Please check the console for more details.")
                    import traceback
                    st.code(traceback.format_exc(), language='python')
                    
                finally:
                    # Clean up temporary file
                    if temp_path and temp_path.exists():
                        try:
                            temp_path.unlink()
                        except Exception as e:
                            st.warning(f"Warning: Could not delete temporary file: {e}")
    
    elif page == "View Resumes":
        st.header("Resume Database")
        
        # Add a toggle for detailed view
        show_detailed_view = st.toggle("Show Detailed View", value=False)
        
        resumes = get_all_resumes()
        
        if not resumes:
            st.info("No resumes found in the database.")
        else:
            # Prepare data for the table
            table_data = []
            for resume in resumes:
                try:
                    # Parse skills and count technical skills
                    tech_skills = []
                    if resume.skills:
                        try:
                            skills_data = json.loads(resume.skills) if isinstance(resume.skills, str) else resume.skills
                            tech_skills = skills_data.get('technical', []) if isinstance(skills_data, dict) else []
                        except (json.JSONDecodeError, AttributeError):
                            tech_skills = []
                    
                    # Count projects from experience
                    num_projects = 0
                    if resume.experience:
                        try:
                            exp_data = json.loads(resume.experience) if isinstance(resume.experience, str) else resume.experience
                            num_projects = len(exp_data) if isinstance(exp_data, list) else 0
                        except (json.JSONDecodeError, AttributeError):
                            num_projects = 0
                    
                    # Format created date
                    created_date = resume.created_at.strftime('%Y-%m-%d %H:%M') if hasattr(resume, 'created_at') and resume.created_at else 'N/A'
                    
                    # Add to table data
                    table_data.append({
                        'Name': resume.name or 'N/A',
                        'Email': resume.email or 'N/A',
                        'Phone': resume.phone or 'N/A',
                        'CGPA': f"{resume.cgpa:.2f}" if resume.cgpa is not None else 'N/A',
                        'Tech Skills': ', '.join(tech_skills[:3]) + ('...' if len(tech_skills) > 3 else ''),
                        'Projects': num_projects,
                        'Hired': bool(getattr(resume, 'hired', 0)),  # Store as boolean for checkbox
                        'Hired_Display': '✅' if getattr(resume, 'hired', 0) == 1 else '❌',  # For display only
                        'Created': created_date,
                        '_resume': resume  # Store the full resume object for detailed view
                    })
                except Exception as e:
                    import logging
                    logging.error(f"Error processing resume {getattr(resume, 'id', 'unknown')}: {str(e)}")
            
            # Display the table
            if table_data:
                # Convert to DataFrame for better display
                import pandas as pd
                df = pd.DataFrame(table_data)
                
                # Display the table with editable Hired column
                display_cols = [col for col in df.columns if not col.startswith('_') and col != 'Hired_Display']
                
                # Create a copy of the dataframe for editing
                edited_df = st.data_editor(
                    df[display_cols],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Tech Skills': st.column_config.TextColumn(
                            'Tech Skills',
                            help='Top 3 technical skills',
                            max_chars=50
                        ),
                        'CGPA': st.column_config.NumberColumn(
                            'CGPA',
                            format='%.2f',
                            help='Cumulative Grade Point Average'
                        ),
                        'Hired': st.column_config.CheckboxColumn(
                            'Hired',
                            help='Mark as hired',
                            default=False
                        )
                    },
                    disabled=([col for col in display_cols if col != 'Hired']),  # Only allow editing the Hired column
                    key='resume_editor'
                )
                
                # Update database if any hiring status changed
                if not df[display_cols].equals(edited_df):
                    for idx, row in edited_df.iterrows():
                        if idx in df.index and df.at[idx, 'Hired'] != row['Hired']:
                            resume_id = df.at[idx, '_resume'].id
                            set_hired(resume_id, row['Hired'])
                    st.success("Hiring status updated successfully!")
                    st.rerun()
                
                # Show detailed view if toggled
                if show_detailed_view and not df.empty:
                    st.subheader("Detailed View")
                    selected_index = st.selectbox(
                        "Select a resume to view details:",
                        range(len(df)),
                        format_func=lambda i: f"{df.iloc[i]['Name']} - {df.iloc[i]['Email']}"
                    )
                    
                    if 0 <= selected_index < len(df):
                        selected_resume = df.iloc[selected_index]['_resume']
                        display_resume(selected_resume)
                        st.markdown("---")
                        st.header("Upload New Resume")
    
    elif page == "Filter":
        st.header("Filter Resumes")
        
        # Gather all resumes and build unique technical skills list
        all_resumes = get_all_resumes()
        unique_skills = set()
        for r in all_resumes:
            try:
                skills_dict = json.loads(r.skills) if r.skills else {}
                unique_skills.update(skills_dict.get("technical", []))
            except json.JSONDecodeError:
                continue
        skill_options = sorted(unique_skills)
        
        selected_skill = st.selectbox("Select technical skill", ["All"] + skill_options)
        min_cgpa = st.number_input("Minimum CGPA", min_value=0.0, max_value=10.0, step=0.1, value=0.0)
        
        if st.button("Apply Filters"):
            filtered = []
            for r in all_resumes:
                # CGPA filter
                if r.cgpa is None:
                    cg_ok = (min_cgpa == 0.0)
                else:
                    cg_ok = r.cgpa >= min_cgpa
                # Skill filter
                if selected_skill == "All":
                    skill_ok = True
                else:
                    try:
                        skills_dict = json.loads(r.skills) if r.skills else {}
                        skill_ok = selected_skill in skills_dict.get("technical", [])
                    except json.JSONDecodeError:
                        skill_ok = False
                if cg_ok and skill_ok:
                    filtered.append(r)
            
            if not filtered:
                st.info("No resumes match the selected filters.")
            else:
                st.success(f"Found {len(filtered)} matching resumes")
                for res in filtered:
                    display_resume(res)
    
    elif page == "JD Match":
        st.header("🔍 Match Resumes to Job Description")
        
        # Job description input
        col1, col2 = st.columns([3, 1])
        with col1:
            jd_text = st.text_area(
                "Paste the job description here:",
                height=200,
                help="Enter the full job description to find matching resumes"
            )
        
        with col2:
            st.markdown("### Or upload a PDF")
            jd_file = st.file_uploader(
                "Upload JD PDF",
                type=["pdf"],
                accept_multiple_files=False,
                key="jd_pdf",
                help="Upload a PDF job description"
            )
            
            # If a PDF is uploaded, extract its text and override textarea
            if jd_file is not None:
                try:
                    with pdfplumber.open(jd_file) as pdf:
                        jd_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                    st.success("Successfully extracted text from PDF")
                except Exception as e:
                    st.error(f"Failed to read JD PDF: {e}")
        
        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                top_n = st.slider(
                    "Number of matches to show:",
                    min_value=1,
                    max_value=50,
                    value=10,
                    help="Maximum number of matching resumes to display"
                )
            with col2:
                min_similarity = st.slider(
                    "Minimum similarity score:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.05,
                    help="Higher values return only the closest matches"
                )
        
        if st.button("Find Matches", type="primary"):
            if not jd_text.strip():
                st.warning("Please enter a job description or upload a PDF")
            else:
                with st.spinner("Finding best matches..."):
                    from app.database.crud import match_job_description
                    matches = match_job_description(
                        jd_text,
                        top_n=top_n,
                        min_similarity=min_similarity
                    )
                    
                    if not matches:
                        st.info("No matching resumes found. Try adjusting the minimum similarity score.")
                    else:
                        st.success(f"Found {len(matches)} matching resumes")
                        
                        # Display matches in tabs
                        tabs = st.tabs([f"#{i+1}" for i in range(len(matches))])
                        
                        for idx, (resume, score) in enumerate(matches):
                            with tabs[idx]:
                                # Calculate match percentage (0-100%)
                                match_percent = min(100, int(score * 100))
                                
                                # Create a progress bar for the match score
                                st.progress(match_percent / 100)
                                st.caption(f"Match Score: {match_percent}%")
                                
                                # Display resume details in columns
                                col1, col2 = st.columns([1, 3])
                                
                                with col1:
                                    # Basic info card
                                    st.markdown(f"### {resume.name}")
                                    
                                    if resume.email:
                                        st.markdown(f"📧 {resume.email}")
                                    if resume.phone:
                                        st.markdown(f"📞 {resume.phone}")
                                    if resume.cgpa is not None:
                                        st.metric("CGPA", f"{resume.cgpa:.2f}")
                                    
                                    # Skills
                                    try:
                                        skills = json.loads(resume.skills)
                                        if skills and skills.get('technical'):
                                            st.markdown("**Technical Skills:**")
                                            for skill in skills['technical'][:5]:  # Show top 5 skills
                                                st.markdown(f"- {skill}")
                                            if len(skills['technical']) > 5:
                                                st.caption(f"+ {len(skills['technical']) - 5} more")
                                    except (json.JSONDecodeError, AttributeError):
                                        pass
                                
                                with col2:
                                    # Experience
                                    try:
                                        experience = json.loads(resume.experience)
                                        if experience:
                                            st.markdown("### Experience")
                                            for exp in experience[:2]:  # Show top 2 experiences
                                                st.write(f"**{exp.get('role', 'Role')}**")
                                                st.write(f"*{exp.get('company', 'Company')}*")
                                                if exp.get('duration'):
                                                    st.caption(f"⏱️ {exp['duration']}")
                                                if exp.get('description'):
                                                    st.write(exp['description'])
                                                st.write("---")
                                    except (json.JSONDecodeError, AttributeError):
                                        pass
                                    
                                    # Education
                                    try:
                                        education = json.loads(resume.education)
                                        if education:
                                            st.markdown("### Education")
                                            for edu in education[:1]:  # Show top education
                                                st.write(f"**{edu.get('degree', 'Degree')}**")
                                                st.write(f"*{edu.get('institution', 'Institution')}*")
                                                if edu.get('date_range'):
                                                    st.caption(f"📅 {edu['date_range']}")
                                    except (json.JSONDecodeError, AttributeError):
                                        pass
                                
                                # View full resume button
                                if st.button("View Full Resume", key=f"view_{resume.id}"):
                                    st.session_state['view_resume_id'] = resume.id
                                    st.experimental_rerun()
                                
                                # Hired toggle
                                is_hired = st.checkbox(
                                    "Mark as Hired",
                                    value=resume.hired or False,
                                    key=f"hired_{resume.id}",
                                    on_change=lambda r=resume: set_hired(r.id, not r.hired)
                                )

    elif page == "ML Ranking":
        st.header("ML-based Resume Ranking")
        top_n = st.slider("Top N", 1, 50, 10, key="ml_top_n")
        from app.database.crud import ml_rank_resumes
        results = ml_rank_resumes(top_n=top_n)
        if not results:
            st.info("Model not trained or no resumes available.")
        else:
            st.success(f"Top {len(results)} ranked resumes")
            for idx, (res, prob) in enumerate(results, 1):
                st.markdown(f"### #{idx} — {res.name} (Hire Prob: {prob:.2f})")
                display_resume(res)


if __name__ == "__main__":
    main()
