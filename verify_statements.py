import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

def load_embeddings(embeddings_dir="embeddings/faiss_index"):
    """Load the FAISS index with embeddings"""
    # Initialize the embedding model (same as used during creation)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load the FAISS index
    vector_store = FAISS.load_local(embeddings_dir, embedding_model, allow_dangerous_deserialization=True)
    
    print("Successfully loaded FAISS index")
    return vector_store


def semantic_search(vector_store, query, top_k=4):
    """Search for semantically similar content in the vector store"""
    results = vector_store.similarity_search(query, k=top_k)
    return results


def verify_statement_with_llm(statement, context_documents, groq_api_key):
    """Use Groq LLM to verify if the statement is factually correct"""
    
    # Prepare context from the retrieved documents
    context = "\n\n".join([doc.page_content for doc in context_documents])
    
    # Initialize the Groq LLM
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024
    )
    
    # Create a prompt template for fact verification
    prompt_template = """
    You are a factual verification assistant. Your task is to determine if the given statement is factually correct based solely on the provided context.
    
    Statement to verify: "{statement}"
    
    Context information:
    {context}
    
    Based ONLY on the context provided above, determine if the statement is factually correct. If the statement is correct, explain why. If it's incorrect, explain what aspects are incorrect and provide the correct information. If the context doesn't contain enough information to verify the statement, state that you cannot verify it.
    
    Your verification format should be:
    
    Verification Result: [CORRECT / PARTIALLY CORRECT / INCORRECT / CANNOT VERIFY]
    Explanation: [Your detailed explanation with specific references to the context]
    
    Only use information from the provided context. Do not use any external knowledge.
    """
    
    prompt = PromptTemplate(
        input_variables=["statement", "context"],
        template=prompt_template
    )
    
    # Create a chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run the chain
    result = chain.run(statement=statement, context=context)
    
    return result


def verify_statements(statements, embeddings_dir="embeddings/faiss_index"):
    """Verify a list of statements using semantic search and Groq LLM"""
    # Load environment variables
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please add it to your .env file.")
    
    # Load the embeddings
    vector_store = load_embeddings(embeddings_dir)
    
    # Create a file to store the results
    with open("verification_results.txt", "w", encoding="utf-8") as results_file:
        # Process each statement
        for i, statement in enumerate(statements, 1):
            print(f"\n{'='*80}\nVerifying Statement {i}: {statement}\n{'='*80}")
            results_file.write(f"\n{'='*80}\nVerifying Statement {i}: {statement}\n{'='*80}\n\n")
            
            # Perform semantic search
            context_docs = semantic_search(vector_store, statement)
            
            # Write retrieved context to file
            results_file.write("Retrieved Context:\n")
            for j, doc in enumerate(context_docs, 1):
                results_file.write(f"\nContext Source {j}:\n{doc.page_content}\n")
            
            # Verify with LLM
            verification_result = verify_statement_with_llm(statement, context_docs, groq_api_key)
            
            # Print and save result
            print(f"\nVerification Result:\n{verification_result}")
            results_file.write(f"\nVerification Result:\n{verification_result}\n")
    
    print(f"\nAll verification results have been saved to verification_results.txt")


if __name__ == "__main__":
    # List of statements to verify
    statements = [
        "In Finland, teachers are required to have a master's degree.",
        "The first university was founded in Bologna, Italy, in 1088.",
        "The No Child Left Behind Act, passed in the United States in 2002, significantly impacted education policy.",
        "With approximately 773 million illiterate adults worldwide, access to basic education remains a critical global challenge."
    ]
    
    # Verify the statements
    verify_statements(statements)