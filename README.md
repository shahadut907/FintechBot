FinSolve AI
FinSolve AI is an intelligent role-based assistant for business operations. It provides secure, multilingual chat capabilities with document processing, RAG-powered search, and real-time analytics. Built with React frontend and FastAPI backend for efficient performance.
Table of Contents
Installation
Usage
Features
Contributing
License
Contact
Installation
Prerequisites
Python 3.8+
Node.js 16+
PostgreSQL (for database)
Ollama (for local LLM)

Steps
Clone the repository:

git clone https://github.com/your-repo/finsolve-ai.git
cd finsolve-ai

Backend setup:
cd src/backend
pip install -r requirements.txt
# Set environment variables (e.g: DATABASE_URL)
uvicorn endpoints:app --reload

Frontend setup:
cd ../frontend
npm install
npm start

Usage
Start backend server (as above).
Open http://localhost:3000 in browser.
Login with role-based credentials (e.g., demo users in LoginForm.js).
Chat: Ask questions, upload files, view responses. Example: Upload financial CSV, query "Q4 revenue".


Features
Role-based access control for 6 user types.
Multilingual query detection and translation.
GPU-optimized RAG for document search.
Real-time admin dashboard with CRUD and analytics.
Persistent conversation memory with DB/Redis.

Contributing
Fork the repository.
Create a branch: git checkout -b feature/your-feature.
Commit changes: git commit -m "Add your-feature".
Push: git push origin feature/your-feature.
Submit pull request.

Contact
Email: shahaduthossen172@gmail.com (Author) 

