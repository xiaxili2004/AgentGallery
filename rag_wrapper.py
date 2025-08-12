from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from vector_store import VectorStoreManager
from langchain.schema import Document
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class rag_instance:
    def __init__(self, vector_store_path: str = "vector_store", index_name: str = "test_index", max_splits_per_batch: int = 100):
        """
        Initialize the RAG system for LegCo documents.
        
        Args:
            vector_store_path: Path to the vector store directory
            index_name: Name of the index collection
            max_splits_per_batch: Maximum number of splits to insert at one time
        """
        # Check if vector store directory exists
        index_path = Path(vector_store_path)
        if not index_path.exists():
            raise FileNotFoundError(
                f"Vector store directory not found at {vector_store_path}"
            )
        
        # Initialize vector store (no documents argument needed)
        self.vector_store = VectorStoreManager(
            index_path=vector_store_path,
            index_name=index_name,
            max_splits_per_batch=max_splits_per_batch
        )
        
        # Load vector store
        self.vector_store.load_vector_store()
        
        # Initialize LLM
        self.llm = ChatTongyi(
            model_name="qwen-max",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        
        # Create prompt template
        self.prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Keep the answer concise and relevant to the question.

        Context:
        {context}

        Question: {question}

        Answer:"""
        
        self.qa_chain = self._create_qa_chain()

    def _create_qa_chain(self):
        """Create the QA chain with custom prompt."""
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.vector_store.as_retriever(
                search_kwargs={"k": 4}  # Number of relevant chunks to retrieve
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def query(self, question: str):
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to ask
            
        Returns:
            dict: Response containing answer and source documents
        """
        result = self.qa_chain({"query": question})
        
        # Format the response
        response = {
            "answer": result["result"],
            "sources": [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown")
                }
                for doc in result["source_documents"]
            ]
        }
        
        return response

    def print_response(self, response: dict):
        """Print the response in a formatted way."""
        print("\nAnswer:")
        print(response["answer"])
        
        print("\nSources:")
        for i, source in enumerate(response["sources"], 1):
            print(f"\nSource {i}:")
            print(f"From: {source['source']}")
            print(f"Content: {source['content'][:200]}...")  # Show first 200 chars

# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = rag_instance()
    
    # Example questions
    questions = [
        "What are the key points about national key labs?",
        "What is the government's policy on research funding?",
        "How does the government support innovation?"
    ]
    
    # Process each question
    for question in questions:
        print(f"\nQuestion: {question}")
        response = rag.query(question)
        rag.print_response(response) 