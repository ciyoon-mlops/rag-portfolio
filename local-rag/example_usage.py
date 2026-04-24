"""
Example usage of the RAG system.
This script demonstrates how to use the RAG query system interactively.
"""

from query_rag_ollama import ask_question

def main():
    """Interactive RAG query examples."""
    
    print("=" * 60)
    print("RAG System - Question Answering Demo")
    print("=" * 60)
    
    # Example queries
    queries = [
        "세이노는 누구인가",
        "카페에 글을 올리기 시작한 이유",
        "월급쟁이의 재테크 수단은?",
        "영어 스피킹 잘하는 방법은?",
        "일을 잘하는 방법은?",
    ]
    
    for query in queries:
        ask_question(query)
        print("\n" + "-" * 60 + "\n")
    
    # Interactive mode
    print("Entering interactive mode. Type 'quit' to exit.")
    while True:
        user_query = input("\n질문을 입력하세요: ").strip()
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if user_query:
            ask_question(user_query)

if __name__ == "__main__":
    main()

