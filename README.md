# 🔍 Data Privacy Assistant  
**Understanding what data GPT Actions really collect — beyond what privacy policies say.**

### 🧠 Project Summary  
**Data Privacy Assistant** is an interactive web application that explores **what types of personal data are actually collected by GPT Actions**, even when not clearly disclosed in their privacy policies.  
It combines an **LLM-based chatbot** with **interactive data visualizations** to reveal and explain hidden data collection patterns.

### ✨ Core Idea  
While companies often publish privacy policies, users rarely see **what types of data are actually gathered and processed** in real use.  
This project bridges that gap by:
- Analyzing GPT Action metadata and datasets,  
- Revealing which **data types and sensitive information** appear most frequently,  
- And allowing users to **ask questions interactively** through a conversational interface.

### 💬 Features
- 💬 **Conversational LLM Chatbot** — Answers user questions about data privacy and collection.  
- 📊 **Dynamic Visualizations** — Displays data type distributions and sensitive data occurrences.  
- 🕹️ **Play the Game** — A small privacy-awareness game that makes the learning interactive and fun.

### 🎯 Goal  
To **increase user awareness** about personal data handling in LLM ecosystems,  
and to **visualize the gap** between what privacy policies state and what data is actually being collected.
  

---

## 📥 Installation & Setup

Follow these steps to clone and run the project on your local machine:

```bash
 1) Clone the project from GitHub
git clone https://github.com/elifdikmn/DataPrivacy.git

 2) Navigate into the project directory
cd DataPrivacy

 3-Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

 4) Navigate into the backend folder
cd backend

 5) Upgrade pip and install dependencies
pip install -r requirements.txt

 6) Update your .env file
The .env file already exists in the backend folder.
Open it in any text editor (VS Code, nano, Notepad, etc.)
and replace "YOUR API KEY" with your actual Together API key.

 7) Run the FastAPI backend server
uvicorn llm_handler:app --reload
 

  ✅ Backend will start at:
  http://127.0.0.1:8000
```
### 💻 Frontend Setup (React)
```bash
Follow these steps to run the React frontend.

 1) Open a new terminal window or tab  
Keep the backend running in the first terminal.

 2) Navigate to the frontend folder
cd DataPrivacy/frontend

 3) Install dependencies
npm install

 4) Run the React development server
npm start
```
