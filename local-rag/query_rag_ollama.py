# rag_sayno_full.py (Modified full code with English comments)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS   # Import FAISS
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA

EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"

# 1. Embedding model (must use the SAME model as when creating the vector DB!)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# 2. Load the saved FAISS vector DB ← This part was missing and caused the error!
vectorstore = FAISS.load_local(
    folder_path="faiss_index",                  # folder name used with save_local
    embeddings=embeddings,
    allow_dangerous_deserialization=True     # Required option (ignore security warning)
)

# 3. Retriever setup (from here it's the original code)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Increased k=5 slightly for better accuracy ↑

# 4. Ollama LLM (using 14B for better quality; switch to 7B for faster speed)
#llm = OllamaLLM(model="qwen2:7b", temperature=0.3)
llm = OllamaLLM(model="qwen2.5:14b", temperature=0.3)
# For more accurate answers: model="qwen2.5:14b"

# 5. RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 6. Question answering function (same as before)
def ask_question(query):
    print(f"\nQuestion: {query}")

    # Search directly with retriever to check similarity scores
    docs_with_score = vectorstore.similarity_search_with_score(query, k=5)
    print("\n=== Retrieved Documents and Similarity Scores (lower score = more similar) ===")
    for i, (doc, score) in enumerate(docs_with_score, 1):
        print(f"{i}. Score: {score:.4f} | Content: {doc.page_content[:200]}...")

    result = qa_chain.invoke({"query": query})
    print(f"Answer: {result['result']}")
    
    print("\n=== Referenced Documents (partial) ===")
    for i, doc in enumerate(result["source_documents"][:2], 1):
        print(f"{i}. {doc.page_content[:300]}...")

# Test questions (translated to English for consistency)
if __name__ == "__main__":
    ask_question("월급쟁이의 재테크 수단은?")
    #ask_question("영어 스피킹 잘하는 방법은?")
    #ask_question("일을 잘는 방법은?")
    # 원하는 질문 자유롭게 추가!