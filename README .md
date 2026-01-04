# 🧠 EquiLens: LLM-Powered Financial News Research Assistant

EquiLens is an **AI-powered financial news research assistant** that allows users to analyze real-time financial news articles and ask intelligent questions.  
It leverages **Retrieval-Augmented Generation (RAG)** using **FAISS** and **HuggingFace LLMs** to provide **accurate answers with source citations**.

---

## 🚀 Key Features

- 🔗 Accepts multiple financial news article URLs  
- 🧠 Semantic search using LLM embeddings  
- 📌 Generates grounded answers from real news content  
- 🔍 Displays original source links  
- 💻 Interactive UI built with Streamlit  
- 💸 Completely free (no OpenAI API key required)  
- ⚡ Token-safe and optimized for small LLMs  

---

## 🏗️ Architecture

User Query  
→ FAISS Vector Search  
→ Relevant News Chunks  
→ LLM (Flan-T5)  
→ Answer + Source URLs

---

## 🧠 Technologies Used

| Category | Tools |
|--------|-------|
| Frontend | Streamlit |
| LLM | HuggingFace (google/flan-t5-base) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| Framework | LangChain |
| Language | Python |

---

## 📁 Project Structure

```
EquiLens/
│
├── app.py
├── requirements.txt
├── faiss_index/
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
```
git clone https://github.com/your-username/EquiLens.git
cd EquiLens
```

### 2️⃣ Install Dependencies
```
pip install -r requirements.txt
```

### 3️⃣ Run Application
```
streamlit run app.py
```

---

## 🖥️ How to Use

1. Paste financial news URLs in the sidebar  
2. Click **Process URLs**  
3. Ask a question related to the articles  
4. View AI-generated answer with sources  

---

## 🧪 Sample Query

**What would Elon Musk’s total fortune be if SpaceX IPO happens?**

---

## 🧠 Why RAG?

- Prevents hallucinations  
- Ensures answers are source-grounded  
- Improves trust and transparency  

---

## 🌐 Deployment

- Local system  
- Streamlit Cloud  
- Any Python-supported cloud VM  

---

## 🎓 Academic Value

- End-to-end RAG implementation  
- Real-world financial NLP use case  
- Ideal for final-year projects & AI portfolios  

---

## 📌 Future Enhancements

- Chat history  
- PDF & report ingestion  
- Advanced financial analytics  
- Authentication system  

---

## 👨‍💻 Author

**Rakesh**  
AI & Machine Learning Enthusiast  
Domain: Generative AI | NLP | RAG Systems
