# rag_sayno2_faiss.py — full script (copy and replace file contents as needed)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings   # current recommended API
from langchain_community.vectorstores import FAISS       # updated import

EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"

# 1. Load PDF (set the path to your own file!)
loader = PyPDFLoader("sayno_230405.pdf")  # repo root; run from project directory
documents = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)   # defines chunks here

# 3. Embedding model (better accuracy for Korean retrieval)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# 4. Build FAISS vector store and persist
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")   # save locally for later reload

print("FAISS vector store created. Total chunks:", len(chunks))