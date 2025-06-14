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
    """Display a single resume in an expandable section."""
    with st.expander(f"📄 {resume.name}", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            st.write(f"**Name:** {resume.name}")
            st.write(f"**Email:** {resume.email}")
            st.markdown(f"**Phone:** {resume.phone}")

            
            if resume.education:
                st.subheader("Education")
                for edu in json.loads(resume.education):
                    st.write(f"**Degree:** {edu.get('degree', 'N/A')}")
                    st.write(f"**Institution:** {edu.get('institution', 'N/A')}")
                    st.write(f"**CGPA:** {edu.get('cgpa', 'N/A')}")
                    st.write("---")
        
        with col2:
            if resume.skills:
                st.subheader("Skills")
                skills = json.loads(resume.skills)
                if skills.get('technical'):
                    st.write("**Technical:** " + ", ".join(skills['technical']))
                if skills.get('soft'):
                    st.write("**Soft Skills:** " + ", ".join(skills['soft']))
            
            if resume.experience:
                st.subheader("Work Experience")
                for exp in json.loads(resume.experience):
                    st.write(f"**{exp.get('role', 'N/A')}**")
                    st.write(f"{exp.get('company', 'N/A')} - {exp.get('duration', 'N/A')}")
                    st.write(exp.get('description', ''))
                    st.write("---")

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
                try:
                    # Save uploaded file temporarily
                    temp_path = Path("temp_resume.pdf")
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Extract data
                    resume_data = extract_resume_data(temp_path)
                    
                    # Show clean summary upfront
                    display_summary(resume_data)
                    
                    # Save to database
                    resume = save_resume(resume_data)
                    
                    # Clean up
                    temp_path.unlink()
                    
                    st.success("Resume processed successfully!")
                    st.balloons()
                    
                    # Display the extracted data
                    display_resume(resume)
                    
                except Exception as e:
                    st.error(f"Error processing resume: {str(e)}")
    
    elif page == "View Resumes":
        st.header("All Resumes")
        resumes = get_all_resumes()
        
        if not resumes:
            st.info("No resumes found in the database.")
        else:
            # ---------- Compact table summary ----------
            summary_rows = []
            for r in resumes:
                try:
                    skills = json.loads(r.skills) if r.skills else {}
                except json.JSONDecodeError:
                    skills = {}
                summary_rows.append({
                    "ID": r.id,
                    "Name": r.name,
                    "Email": r.email,
                    "Phone": r.phone,
                    "CGPA": r.cgpa if r.cgpa is not None else "",
                    "Tech Skills": ", ".join(skills.get("technical", [])) if skills.get("technical") else "",
                    "No. of Projects": (lambda exp: (len(exp) if exp else None))(json.loads(r.experience) if r.experience else []),
                    "Hired": bool(r.hired),
                    "Created": r.created_at.strftime("%Y-%m-%d") if r.created_at else ""
                })
            import copy
            df_original = pd.DataFrame(summary_rows)
            edited_df = st.data_editor(
                df_original,
                column_config={"Hired": st.column_config.CheckboxColumn()},
                hide_index=True,
                key="resume_editor"
            )
            # Update DB if any hired flag changed
            if not edited_df["Hired"].equals(df_original["Hired"]):
                for _, row in edited_df.iterrows():
                    orig_val = df_original.loc[df_original["ID"]==row["ID"], "Hired"].values[0]
                    if row["Hired"] != orig_val:
                        set_hired(int(row["ID"]), bool(row["Hired"]))
                st.success("Updated hired flags.")
                st.experimental_rerun()
            
            for resume in resumes:
                display_resume(resume)
    
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
