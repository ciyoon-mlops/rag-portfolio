# 📚 RAG System with Qwen LLM

A complete **Retrieval-Augmented Generation (RAG)** system that enables question-answering over PDF documents using Qwen language models via Ollama and FAISS vector search.

Python
LangChain
FAISS
Ollama

## 🎯 Overview

This project implements an end-to-end RAG pipeline that:

- Processes PDF documents and converts them into searchable vector embeddings
- Stores embeddings in a FAISS vector database for efficient similarity search
- Retrieves relevant document chunks based on user queries
- Generates accurate answers using Qwen LLM models (7B or 14B variants)

## 🏗️ Architecture

```
┌─────────────────┐
│   PDF Document  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Chunking  │ (1000 chars, 200 overlap)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embeddings    │ (jhgan/ko-sroberta-multitask)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FAISS Vector   │ (Persistent storage)
│     Database    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│   Query Input   │─────▶│   Retriever  │ (Top-K: 5)
└─────────────────┘      └──────┬───────┘
                                 │
                                 ▼
                          ┌──────────────┐
                          │  Qwen LLM    │ (14B)
                          │  (Ollama)    │
                          └──────┬───────┘
                                 │
                                 ▼
                          ┌──────────────┐
                          │   Answer +   │
                          │   Sources    │
                          └──────────────┘
```

## 📁 Project Structure

```
.
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── build_faiss_index.py           # Vector DB creation script
├── query_rag_ollama.py            # RAG query system (Ollama + FAISS)
├── example_usage.py               # Interactive usage examples
├── eval_rag_reference.py          # Optional: evaluate vs reference.json
├── reference.json                 # Optional: gold Q&A for eval
├── faiss_index/                   # Generated vector database
│   ├── index.faiss
│   └── index.pkl
└── sayno_230405.pdf               # Source document (repo root)
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Ollama** installed and running
3. **Qwen models** pulled in Ollama:
  ```bash
   ollama pull qwen2:7b
   # or
   ollama pull qwen2.5:14b
  ```

### Installation

1. Clone the repository:
  ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
  ```
2. Set up Python prerequisite:
  ```bash
   pyenv install 3.12.13
   pyenv local 3.12.13
   python -m venv .venv
   source .venv/bin/activate
  ```
3. Install dependencies:
  ```bash
   pip install -r requirements.txt
  ```
4. Build the vector database (run from the repo root so **`sayno_230405.pdf`** resolves as a path relative to that directory):
  ```bash
   python build_faiss_index.py
  ```
   This will create the `faiss_index/` folder with the vector embeddings.
5. Run the RAG query system:
  ```bash
   python query_rag_ollama.py
  ```
   Or use the interactive example:

```bash
python example_usage.py
```

## 📝 Scripts Description

### 1. `build_faiss_index.py` - Vector Database Builder

**Purpose**: Creates and persists the FAISS vector database from PDF documents.

**Features**:

- Loads PDF documents using LangChain's PyPDFLoader (default: **`sayno_230405.pdf`** at the repository root, relative path)
- Splits documents into chunks (1000 characters, 200 overlap)
- Generates embeddings using HuggingFace **`jhgan/ko-sroberta-multitask`**
- Creates FAISS vector store and saves to `faiss_index/` directory

**Usage**:

```bash
python build_faiss_index.py
```

### 2. `query_rag_ollama.py` - Complete RAG System (14B)

**Purpose**: Full-featured RAG query system using the larger 14B Qwen model (configurable).

**Features**:

- Loads pre-built FAISS vector database
- Uses `qwen2.5:14b` model for higher accuracy (7B optional in code)
- Retrieves top 5 relevant document chunks
- Provides answers with source document citations
- Exposes `ask_question()` for programmatic or interactive use

**Usage**:

```bash
python query_rag_ollama.py
```

## 🔧 Configuration

### Model Selection

In **`query_rag_ollama.py`**, you can switch between models:

```python
# For faster inference (7B model)
llm = OllamaLLM(model="qwen2:7b", temperature=0.3)

# For better accuracy (14B model)
llm = OllamaLLM(model="qwen2.5:14b", temperature=0.3)
```

### Retrieval Parameters

Adjust the number of retrieved documents:

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Change k value
```

### Chunking and PDF path

Modify chunk size, overlap, and PDF path in **`build_faiss_index.py`**:

```python
loader = PyPDFLoader("sayno_230405.pdf")  # relative to cwd when you run the script

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Adjust chunk size
    chunk_overlap=200     # Adjust overlap
)
```

### Embeddings

Both **`build_faiss_index.py`** and **`query_rag_ollama.py`** must use the same model:

```python
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
```

## 💡 Example Usage

### Programmatic Usage

```python
from query_rag_ollama import ask_question

# Ask questions about the document
ask_question("월급쟁이의 재테크 수단은?")
ask_question("영어 스피킹 잘하는 방법은?")
ask_question("일을 잘하는 방법은?")
```

### Interactive Mode

Run the example script for an interactive session:

```bash
python example_usage.py
```

**Output Example** (from `query_rag_ollama.py` / `example_usage.py`):

```
Question: 월급쟁이의 재테크 수단은?

=== Retrieved Documents and Similarity Scores (lower score = more similar) ===
1. Score: ... | Content: ...

Answer: [Generated answer based on retrieved documents]

=== Referenced Documents (partial) ===
1. [Relevant document chunk 1]...
```

## 🛠️ Technologies Used

- **LangChain**: Document processing, text splitting, and chain orchestration
- **HuggingFace Embeddings**: jhgan/ko-sroberta-multitask for vector embeddings
- **FAISS**: Facebook AI Similarity Search for efficient vector storage and retrieval
- **Ollama**: Local LLM inference with Qwen models
- **PyPDFLoader**: PDF document loading and parsing

## 📊 Key Features

✅ **Persistent Vector Storage**: FAISS index saved locally for reuse  
✅ **Multiple Model Support**: Choose between 7B (fast) or 14B (accurate) models  
✅ **Source Citation**: Answers include references to source document chunks  
✅ **Configurable Retrieval**: Adjustable top-K retrieval for optimal results  
✅ **Modular Design**: Separate scripts for building and querying  

## 🔍 How It Works

1. **Document Processing**: PDF is loaded and split into semantic chunks
2. **Embedding Generation**: Each chunk is converted to a vector using **`jhgan/ko-sroberta-multitask`**
3. **Vector Storage**: Embeddings are stored in FAISS for fast similarity search
4. **Query Processing**: User query is embedded and matched against stored vectors
5. **Context Retrieval**: Top-K most similar chunks are retrieved
6. **Answer Generation**: LLM generates answer using retrieved context
7. **Source Attribution**: Original document chunks are returned for verification

## 📈 Performance Considerations

- **14B Model**: Higher accuracy (~5-8 seconds per query), better for production
- **FAISS**: Sub-second retrieval even with large document collections
- **Chunk Size**: 1000 chars with 200 overlap balances context and granularity

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Qwen team for the excellent language models
- LangChain for the comprehensive RAG framework
- Facebook AI Research for FAISS
- HuggingFace for embedding models

---

**Built with ❤️ using Python, LangChain, FAISS, and Qwen LLM**
