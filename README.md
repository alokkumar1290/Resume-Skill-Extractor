# Resume Skill Extractor

A Streamlit-based web application for extracting and analyzing skills from resumes. The application processes PDF resumes, extracts key information, and provides advanced search and AI-powered question generation capabilities.

## Features

- **Resume Parsing**: Extract personal information, skills, work experience, and education from PDF resumes
- **Database Storage**: Store and manage resume data in SQLite
- **Advanced Search**: Filter resumes by skills, education, experience, and more
- **AI Question Generation**: Generate interview questions based on resume content
- **Responsive UI**: Clean, user-friendly interface built with Streamlit

## Prerequisites

- Python 3.9+
- pip
- Docker (optional, for containerized deployment)
- OpenAI API key (for AI question generation)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd resume-skill-extractor
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### Running the Application

1. Start the Streamlit app:
   ```bash
   streamlit run app/main.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

### Using Docker

1. Build the Docker image:
   ```bash
   docker build -t resume-skill-extractor .
   ```

2. Run the container:
   ```bash
   docker run -p 8501:8501 -e OPENAI_API_KEY=your_openai_api_key_here resume-skill-extractor
   ```

3. Access the application at `http://localhost:8501`

## Project Structure

```
resume-skill-extractor/
├── app/
│   ├── main.py                 # Main Streamlit application
│   ├── processing/             # Resume processing modules
│   │   ├── extraction.py       # PDF and text extraction
│   │   └── questions.py        # AI question generation
│   ├── database/               # Database models and operations
│   │   ├── models.py
│   │   └── crud.py
│   └── utils/                  # Utility functions
│       └── config.py           # Configuration settings
├── uploads/                    # Directory for uploaded resumes
├── .env.example               # Example environment variables
├── requirements.txt           # Python dependencies
└── Dockerfile                 # Docker configuration
```

## Configuration

Configure the application by setting environment variables in the `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key (required for AI features)
- `DATABASE_URL`: Database connection URL (default: SQLite)
- `UPLOAD_FOLDER`: Directory to store uploaded files (default: ./uploads)


## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- Streamlit for the amazing web framework
- LangChain for LLM integration
- PyPDF2 and pdfplumber for PDF processing
- SQLAlchemy for database ORM
