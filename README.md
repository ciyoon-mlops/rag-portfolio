# RAG Portfolio – Retrieval-Augmented Generation Projects

This is my portfolio showing different ways to build Retrieval-Augmented Generation (RAG) systems.  
I am using my Azure AI-102 certification knowledge and building step by step from basic to advanced.

The goal is to show how RAG can evolve:  
Local → Developer Tool Integration → Advanced Graph RAG → Cloud Deployment.

## Current Projects

### Local RAG (FAISS + LangChain Agentic) ✅ Completed
- Loads PDF document ("세이노의 가르침" – official free PDF)
- Splits text into chunks and builds FAISS vector database
- Uses HyDE and Multi-Query techniques
- Agentic RAG that can decide when to search or answer directly
- Works with local LLM (Ollama) or OpenAI
- Demo screenshots/GIFs are in the folder

→ Check the folder: [/local-rag](./local-rag)

## In Progress / Planned

### MCP RAG (Developer Tool Integration) 🔄 In Progress
- Turns the Local RAG into an MCP server (using FastAPI)
- Lets VS Code extension **Continue.dev** call my private documents in real time
- Great for coding with my own knowledge base

### Graph RAG (Microsoft GraphRAG) ⏳ Planned
- Builds a knowledge graph from the documents
- Very strong for big-picture questions ("What are the main ideas in the whole book?")
- Usually 2–3 times better than basic RAG for complex datasets

### Azure RAG ⏳ Planned
- Cloud version using Azure AI Search + Azure OpenAI
- Ready for production with security and scaling

## Why This Portfolio?
I want to show the full journey of RAG technology through real code.  
I add one part at a time and keep improving.  
This helps me learn deeply and shows employers that I can grow with new AI tools.

Feel free to explore the code and try it yourself!

## Contact
- LinkedIn: www.linkedin.com/in/ciyoon-mlops
- Please contact me via LinkedIn message (GitHub email is private to avoid spam)

Thank you for visiting! 🚀