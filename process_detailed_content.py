from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import os
import re


def extract_detailed_content(file_path):
    """
    Extract only the 'Detailed content:' sections from the input file.
    Returns a list of Document objects containing only the detailed content.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split the content by statement sections
    statement_sections = re.split(r'={50,} Statement \d+ ={50,}', content)
    
    # Create a list to store all detailed content
    documents = []
    
    # Extract detailed content sections using regex
    for i, section in enumerate(statement_sections):
        if not section.strip():
            continue
            
        # Find all "Detailed content:" sections in this statement
        detailed_content_matches = re.finditer(r'Detailed content:(.*?)(?=SOURCE \d+:|$)', 
                                              section, re.DOTALL)
        
        for j, match in enumerate(detailed_content_matches):
            detailed_text = match.group(1).strip()
            if detailed_text:
                # Create a document with metadata
                statement_match = re.search(r'Input Statement: (.*?)(?=\n|$)', section)
                statement_text = statement_match.group(1) if statement_match else f"Statement {i}"
                
                documents.append(Document(
                    page_content=detailed_text,
                    metadata={
                        "source": f"statement_{i}_source_{j+1}",
                        "statement": statement_text
                    }
                ))
    
    return documents


def process_detailed_content(input_file="extracted_info.txt", output_dir="embeddings"):
    """
    Process only the detailed content from the input file:
    1. Extract detailed content sections
    2. Chunk the text
    3. Generate embeddings
    4. Store in FAISS index
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract detailed content as document objects
    documents = extract_detailed_content(input_file)
    
    print(f"Found {len(documents)} detailed content sections to process")
    
    # Create text chunks using RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    
    # Clean chunks (remove any that are just whitespace or too short)
    cleaned_chunks = [doc for doc in chunks 
                     if doc.page_content and len(doc.page_content.strip()) >= 10]
    
    print(f"Created {len(cleaned_chunks)} chunks after cleaning")
    
    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create and save the vector store
    vector_store = FAISS.from_documents(cleaned_chunks, embedding_model)
    index_path = os.path.join(output_dir, 'faiss_index')
    vector_store.save_local(index_path)
    
    print(f"Successfully created and saved embeddings to {index_path}")
    
    # Return the processed chunks for inspection if needed
    return cleaned_chunks


if __name__ == "__main__":
    try:
        chunks = process_detailed_content()
        print(f"Processing completed successfully. Generated {len(chunks)} chunks.")
        
        # Print a sample of the first chunk to verify content
        if chunks:
            print("\nSample of first chunk content:")
            print("-" * 50)
            print(chunks[0].page_content[:150] + "...")
            print("-" * 50)
    except Exception as e:
        print(f"Error during processing: {e}")