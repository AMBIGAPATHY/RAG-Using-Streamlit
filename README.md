# 📚 AMBIGAPATHY's Chatbot (Simple PDF Question-Answering App)

Welcome to **AMBIGAPATHY's Chatbot**!  
This is an easy-to-use web app that lets you:

- Upload **PDF files**
- Ask **questions** based on those PDFs
- Get **smart answers** powered by an AI model!

You don't need any machine learning background to use this app.

---

## ✨ What This App Does

- You **upload PDF documents** (like books, reports, notes).
- The app **reads** all the content from the PDFs.
- It **splits** the text into small pieces (chunks).
- It **stores** these pieces in a searchable database (called FAISS).
- When you **ask a question**, it **searches** the relevant information.
- It uses an **AI model** to **generate a nice answer** for you.

✅ All happens automatically after you upload and ask!

---

## 🛠️ What You Need To Run This App

- Python installed (version 3.10 or higher)
- Internet connection (only for the first time, to download AI models)

---

## 🏗️ Project Files

| File | Purpose |
|:-----|:--------|
| `rag.py` | Main Python file that runs the app |
| `requirements.txt` | List of Python packages needed |
| `README.md` | This instruction guide |
| `.env` (you create) | To safely store your HuggingFace token |
| `faiss_index/` folder | Where the searchable database is saved (after processing PDFs)

---

## ⚙️ How To Set It Up (Step-by-Step)

### 1. Download or Clone the Project

You can either download the ZIP file or run:

```bash
git clone https://github.com/yourgithub/rag-chatbot.git
cd rag-chatbot
```

---

### 2. Create a Virtual Environment

This keeps your Python packages clean.

```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# or
source .venv/bin/activate  # Mac/Linux
```

---

### 3. Install Required Packages

Install everything you need:

```bash
pip install -r requirements.txt
```

---

### 4. Set Your HuggingFace API Token

Create a file named `.env` in the project folder.  
Inside `.env`, add:

```
hugging_face_token=your_actual_huggingface_token_here
```

You can get a free HuggingFace token by signing up at [huggingface.co](https://huggingface.co/).

---

### 5. Run the App

Start the chatbot!

```bash
streamlit run rag.py
```

It will automatically open in your web browser at `http://localhost:8501`.

---

## 💬 How To Use The App

1. **Upload PDFs** from the sidebar.
2. Click **"Process PDFs"**.
3. In the text box, **ask any question** related to the PDFs you uploaded.
4. **Get an instant answer!**

Example:
> Upload a "Physics Book PDF" ➔  
> Ask: "What is Newton's second law?" ➔  
> Get a smart answer!

---

## 🔥 Important Tips

- First time you run, it will **download AI models** — wait patiently.
- If you upload new PDFs, **click "Process PDFs" again** to update the database.
- If you see any error about `.env` or token, check if your HuggingFace API key is correct.
- Works **completely offline after the first run** (except for model download).

---

## 🛠️ What Technologies Are Used?

- **Streamlit** → For building the web app
- **FAISS** → For fast document search
- **HuggingFace Transformers** → For text generation
- **LangChain** → For managing the flow between search and answer
- **PyPDF2** → For reading PDFs

---

## 🐛 Troubleshooting

| Problem | What to Do |
|:--------|:-----------|
| App not starting | Check if you activated `.venv` |
| Can't install packages | Update pip: `pip install --upgrade pip` |
| Model download fails | Check internet connection |
| Token error | Make sure `.env` is properly set with the right token |

---

## 📩 Contact

- **Developer**: AMBIGAPATHY
- **Email**: ambigapathy.s2002@gmail.com
- **GitHub**: [AMBIGAPATHY's Rag](https://github.com/AMBIGAPATHY/RAG-Using-Streamlit)

---

# 🎉 Enjoy Uploading PDFs and Asking Anything!
Super simple, super smart 🚀

---

# 📦 Example of Project Structure After Setup:

```
rag-chatbot/
│
├── rag.py
├── requirements.txt
├── README.md
├── .env
├── faiss_index/ (auto-created after processing PDFs)
│   ├── index.faiss
│   └── index.pkl
```

✅ Everything clean and ready!

---

# 📌 Extra: Content of `requirements.txt`

```plaintext
streamlit>=1.31.0
langchain>=0.1.8
langchain-community>=0.0.18
langchain-core>=0.1.28
transformers>=4.38.2
PyPDF2>=3.0.1
faiss-cpu>=1.7.4
torch>=2.1.2
python-dotenv>=1.0.1
```

---

# 🎯 This README is now super easy to understand even for beginners!

---

Would you also like me to prepare a `.zip` file for you with:
- `rag.py`
- `README.md`
- `requirements.txt`
- a sample `.env`
- and folder structure?



