from langchain_community.embeddings import DashScopeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VectorStoreManager:
    def __init__(self, index_path: str = "vector_store", index_name: str = "test_index", max_splits_per_batch: int = 100):
        """
        Initialize the vector store manager.
        Args:
            index_path: Path to store the vector store files
            index_name: Name of the index collection
            max_splits_per_batch: Maximum number of splits to insert at one time
        """
        self.index_path = Path(index_path)
        self.index_name = index_name
        self.max_splits_per_batch = max_splits_per_batch
        self.embeddings = DashScopeEmbeddings(
            model="text-embedding-v3",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vector_store = None
        self._ensure_chromadb_instance()

    def _ensure_chromadb_instance(self):
        """Ensure a persistent ChromaDB instance exists, create if not."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        # Try to load existing, or create empty collection if not exists
        self.vector_store = Chroma(
            persist_directory=str(self.index_path),
            embedding_function=self.embeddings,
            collection_name=self.index_name
        )
        print(f"ChromaDB instance ready at {self.index_path} (collection: {self.index_name})")
        print(f"Current document count: {self.vector_store._collection.count()}")

    def add_documents(self, new_documents: list[Document]):
        """
        Add new documents to the existing vector store, batching if needed.
        Args:
            new_documents: List of new documents to add
        """
        # Split new documents
        new_splits = self.text_splitter.split_documents(new_documents)
        print(f"Split new documents into {len(new_splits)} chunks")
        # Batch insert
        for i in range(0, len(new_splits), self.max_splits_per_batch):
            batch = new_splits[i:i+self.max_splits_per_batch]
            self.vector_store.add_documents(batch)
            print(f"Inserted batch {i//self.max_splits_per_batch+1}: {len(batch)} splits")
        print(f"Total document count after insertion: {self.vector_store._collection.count()}")

    def load_vector_store(self):
        """Reload the vector store from disk (not usually needed with ChromaDB)."""
        print(f"Loading vector store from {self.index_path} with name {self.index_name}...")
        self.vector_store = Chroma(
            persist_directory=str(self.index_path),
            embedding_function=self.embeddings,
            collection_name=self.index_name
        )
        print("Vector store loaded successfully")
        print(f"The current document count is {self.vector_store._collection.count()}")

    def save_vector_store(self):
        """Save the vector store to disk."""
        if self.vector_store:
            print(f"Vector store saved to {self.index_path}")
            print(f"The current document count is {self.vector_store._collection.count()}")

    def similarity_search(self, query: str, k: int = 4):
        """
        Perform similarity search on the vector store.
        Args:
            query: Search query
            k: Number of results to return
        Returns:
            List of relevant documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not created or loaded yet")
        return self.vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 4):
        """
        Perform similarity search with scores.
        Args:
            query: Search query
            k: Number of results to return
        Returns:
            List of tuples containing (document, score)
        """
        if self.vector_store is None:
            raise ValueError("Vector store not created or loaded yet")
        return self.vector_store.similarity_search_with_score(query, k=k)

    def filtered_similarity_search(self, query: str, filter_dict: dict = None, k: int = 4):
        """
        Perform similarity search with metadata filtering.
        Args:
            query: Search query
            filter_dict: Dictionary of metadata filters. Example:
                {
                    "source": "test1.txt",  # exact match
                    "date": {"$gte": "2023-01-01"},  # greater than or equal
                    "category": {"$in": ["AI", "ML"]}  # in list
                }
            k: Number of results to return
        Returns:
            List of relevant documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not created or loaded yet")
        
        return self.vector_store.similarity_search(
            query,
            k=k,
            filter=filter_dict
        )

    def inspect_collection(self):
        """
        Inspect the current collection's metadata and statistics.
        Returns:
            dict: Collection information including count, metadata fields, and sample documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not created or loaded yet")
        
        collection = self.vector_store._collection
        count = collection.count()
        
        # Get unique metadata fields
        metadata_fields = set()
        if count > 0:
            # Get a sample of documents to inspect metadata
            sample_docs = collection.get()["metadatas"]
            for doc in sample_docs:
                metadata_fields.update(doc.keys())
        
        # Get a sample of documents
        sample_docs = []
        if count > 0:
            results = collection.get(limit=3)  # Get first 3 documents as sample
            for i in range(len(results["ids"])):
                sample_docs.append({
                    "id": results["ids"][i],
                    "metadata": results["metadatas"][i],
                    "content_preview": results["documents"][i][:100] + "..." if results["documents"][i] else ""
                })
        
        return {
            "collection_name": self.index_name,
            "total_documents": count,
            "metadata_fields": list(metadata_fields),
            "sample_documents": sample_docs
        }

# Example usage
if __name__ == "__main__":
    # Example documents
    docs = [
        Document(
            page_content="This is a test document about AI.",
            metadata={"source": "test1.txt", "category": "AI", "date": "2024-01-01"}
        ),
        Document(
            page_content="This is another test document about machine learning.",
            metadata={"source": "test2.txt", "category": "ML", "date": "2024-01-02"}
        )
    ]
    
    # Initialize vector store manager
    vs_manager = VectorStoreManager(
        index_path="vector_store",
        index_name="test_chromadb",
        max_splits_per_batch=100
    )
    
    # Add documents
    vs_manager.add_documents(docs)
    
    # Inspect the collection
    collection_info = vs_manager.inspect_collection()
    print("\nCollection Information:")
    print(f"Collection Name: {collection_info['collection_name']}")
    print(f"Total Documents: {collection_info['total_documents']}")
    print(f"Metadata Fields: {collection_info['metadata_fields']}")
    print("\nSample Documents:")
    for doc in collection_info['sample_documents']:
        print(f"\nID: {doc['id']}")
        print(f"Metadata: {doc['metadata']}")
        print(f"Content Preview: {doc['content_preview']}")
    
    # Example search
    results = vs_manager.similarity_search("test")
    for doc in results:
        print(f"Content: {doc.page_content}")
        print(f"Source: {doc.metadata['source']}")
        print("---") 