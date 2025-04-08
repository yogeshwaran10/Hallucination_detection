import os
import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from bs4 import BeautifulSoup
import json
import re
import fitz
import time
from rich.markup import escape
from langchain.document_loaders import PyPDFLoader
from urllib.parse import urlparse
import tempfile

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SCRAPINGBEE_API_KEY = os.getenv("SCRAPINGBEE_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")  # You'll need to add this to your .env file

console = Console()

# List of statements for extraction and later verification
statements = [
    "In Finland, teachers are required to have a master's degree.",
    "The first university was founded in Bologna, Italy, in 1088.",
    "The No Child Left Behind Act, passed in the United States in 2002, significantly impacted education policy.",
    "With approximately 773 million illiterate adults worldwide, access to basic education remains a critical global challenge."
]

def get_search_query(statement: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert at creating concise, effective search queries that a human would type into a search engine. "
                    "Your task is to create a short, specific search query (under 10 words) that will help find accurate information about a statement. "
                    "Focus on the key facts, entities, and relationships in the statement."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Create a concise search query (maximum 10 words) that a human would type into Google to verify this statement:\n\n"
                    f"\"{statement}\"\n\n"
                    "Include only the search query, without quotation marks or explanations."
                )
            }
        ],
        "temperature": 0.2
    }

    res = requests.post(url, headers=headers, json=payload)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"].strip()

def search_web(query: str) -> list:
    """Search the web using Serper API (Google Search API alternative)"""
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query,
        "num": 5
    })
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    response.raise_for_status()
    search_results = response.json()
    
    return search_results.get('organic', [])

def scrape_url(url: str) -> str:
    """Scrape content from a URL or extract text from a PDF."""
    if url.lower().endswith(".pdf"):
        try:
            response = requests.get(url)
            response.raise_for_status()

            # Use PyMuPDF for better text extraction
            pdf_bytes = response.content
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pdf_text = []
            
            for page in doc[:3]:  # First 3 pages
                text = page.get_text().strip()
                if text:
                    # Remove common PDF artifacts
                    text = re.sub(r'\n\s*\n', '\n\n', text)
                    text = re.sub(r'\s{3,}', '  ', text)
                    pdf_text.append(text)
            
            doc.close()
            
            if not pdf_text:
                return "PDF contained no extractable text"
                
            cleaned_text = "\n\n".join(pdf_text)
            if len(cleaned_text) < 100 or not re.search(r'\b[a-zA-Z]{3,}\b', cleaned_text):
                return "PDF text extraction quality too low"
            
            return cleaned_text

        except Exception as e:
            return f"Error processing PDF: {str(e)}"
    else:
        try:
            params = {
                "api_key": SCRAPINGBEE_API_KEY,
                "url": url,
                "render_js": "false"
            }

            response = requests.get("https://app.scrapingbee.com/api/v1", params=params)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            content_elements = soup.select('article, .article, .content, main, #content, .main-content')
            
            paragraphs = []
            if content_elements:
                for element in content_elements:
                    for p in element.find_all('p'):
                        text = p.get_text(strip=True)
                        if len(text) > 50:
                            paragraphs.append(text)
            else:
                for p in soup.find_all('p'):
                    text = p.get_text(strip=True)
                    if len(text) > 50:
                        paragraphs.append(text)
            
            if paragraphs:
                return "\n\n".join(paragraphs[:3])
            return soup.get_text()[:1000]
        
        except Exception as e:
            return f"Error scraping URL: {str(e)}"

def extract_information(statement: str) -> str:
    """Process a statement to extract relevant information from multiple web sources"""
    query = get_search_query(statement)
    console.print(f"[bold cyan]Search Query:[/bold cyan] {query}")
    
    search_results = search_web(query)
    if not search_results:
        return "No search results found."
    
    all_content = []
    for i, result in enumerate(search_results[:3]):
        title = result.get('title', 'No title')
        link = result.get('link', '')
        snippet = result.get('snippet', 'No snippet available')
        
        console.print(f"[bold]Source {i+1}:[/bold] {title}")
        console.print(f"[dim]{link}[/dim]")
        
        source_info = f"SOURCE {i+1}: {title}\n{snippet}\n"
        
        if link:
            try:
                full_content = scrape_url(link)
                full_content = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', full_content)
                full_content = full_content.replace('[', '\\[').replace(']', '\\]')
                
                if 150 < len(full_content) < 10000:
                    source_info += f"\nDetailed content:\n{full_content[:800]}...\n"
                    console.print("[green]✓ Successfully extracted content[/green]")
                else:
                    console.print("[yellow]⚠ Limited content quality, using snippet[/yellow]")
            except Exception as e:
                console.print(f"[red]Error processing content: {e}[/red]")
        
        all_content.append(source_info)
        time.sleep(1)
    
    return "\n\n" + "-"*50 + "\n\n".join(all_content)

def execute_main_extraction():
    """Runs the main extraction routine and saves information to a file."""
    with open("extracted_info.txt", "w", encoding="utf-8") as f:
        for i, statement in enumerate(statements, 1):
            console.rule(f"[bold yellow]Statement {i}")
            console.print(f"[bold]Input Statement:[/bold] {statement}")
            f.write(f"\n{'='*50} Statement {i} {'='*50}\n")
            f.write(f"Input Statement: {statement}\n\n")
            console.print("[bold]Searching and extracting information...[/bold]")
            query = get_search_query(statement)
            f.write(f"Search Query: {query}\n\n")
            info = extract_information(statement)
            f.write(f"Extracted Information:\n{info}\n\n")
            try:
                safe_query = escape(query)
                safe_info = escape(info)
                panel_content = Text.from_markup(f"[bold]Search Query:[/bold] {safe_query}\n\n{safe_info}")
                console.print(Panel.fit(
                    panel_content,
                    title="Extracted Information",
                    title_align="left",
                    border_style="green"
                ))
            except Exception as e:
                console.print(f"[red]Error displaying panel: {e}[/red]")
                console.print("Information has been saved to file successfully.")
        f.write("\nInformation extracted from multiple sources successfully.\n")
    console.print("\n[bold green]Information saved to extracted_info.txt[/bold green]")

def orchestrate_complete_process():
    """Orchestrate the complete process:
       1. Extract information and save it.
       2. Process detailed content and generate embeddings.
       3. Verify statements using the created embeddings.
    """
    # Step 1: Extraction
    execute_main_extraction()

    # Step 2: Process detailed content to generate FAISS embeddings
    console.rule("[bold magenta]Processing Detailed Content & Generating Embeddings[/bold magenta]")
    try:
        import process_detailed_content
        chunks = process_detailed_content.process_detailed_content()
    except Exception as e:
        console.print(f"[red]Error during detailed content processing: {e}[/red]")
        return

    # Step 3: Verify statements using the FAISS index
    console.rule("[bold blue]Verifying Statements Using Embeddings & LLM[/bold blue]")
    try:
        import verify_statements
        # Using the same statements list defined above.
        verify_statements.verify_statements(statements)
    except Exception as e:
        console.print(f"[red]Error during statement verification: {e}[/red]")

def execute_main_extraction_custom(statements_input):
    """Runs the main extraction routine with a custom list of statements and saves information to a file."""
    global statements
    statements = statements_input  # Overwrite the global variable with custom input

    with open("extracted_info.txt", "w", encoding="utf-8") as f:
        for i, statement in enumerate(statements, 1):
            console.rule(f"[bold yellow]Statement {i}")
            console.print(f"[bold]Input Statement:[/bold] {statement}")
            f.write(f"\n{'='*50} Statement {i} {'='*50}\n")
            f.write(f"Input Statement: {statement}\n\n")
            console.print("[bold]Searching and extracting information...[/bold]")
            query = get_search_query(statement)
            f.write(f"Search Query: {query}\n\n")
            info = extract_information(statement)
            f.write(f"Extracted Information:\n{info}\n\n")
            try:
                safe_query = escape(query)
                safe_info = escape(info)
                panel_content = Text.from_markup(f"[bold]Search Query:[/bold] {safe_query}\n\n{safe_info}")
                console.print(Panel.fit(
                    panel_content,
                    title="Extracted Information",
                    title_align="left",
                    border_style="green"
                ))
            except Exception as e:
                console.print(f"[red]Error displaying panel: {e}[/red]")
                console.print("Information has been saved to file successfully.")
        f.write("\nInformation extracted from multiple sources successfully.\n")

    console.print("\n[bold green]Information saved to extracted_info.txt[/bold green]")


def orchestrate_custom_process_with_results(statements_input):
    """
    Orchestrate the complete process with custom statements:
      1. Extract information and save it.
      2. Process detailed content and generate embeddings.
      3. Verify statements using the created embeddings.
    Returns a list of dicts: {
       "statement": str,
       "status": "INCORRECT"|"PARTIALLY CORRECT"|"CORRECT"|"OTHER",
       "verification_result": str (the LLM's full response)
    }
    """
    # Step 1: Extraction using custom statements
    execute_main_extraction_custom(statements_input)

    # Step 2: Process detailed content to generate FAISS embeddings
    console.rule("[bold magenta]Processing Detailed Content & Generating Embeddings[/bold magenta]")
    try:
        import process_detailed_content
        process_detailed_content.process_detailed_content()
    except Exception as e:
        console.print(f"[red]Error during detailed content processing: {e}[/red]")
    
    # Step 3: Verify statements using the FAISS index
    results = []
    try:
        import verify_statements
        # Load the FAISS embeddings
        groq_api_key = os.getenv("GROQ_API_KEY")
        vector_store = verify_statements.load_embeddings()  # from verify_statements.py

        for statement in statements_input:
            # Perform semantic search to obtain context documents
            context_docs = verify_statements.semantic_search(vector_store, statement)
            # Use Groq LLM to verify the statement
            verification_result = verify_statements.verify_statement_with_llm(statement, context_docs, groq_api_key)

            # Determine status by scanning the LLM response
            verification_result_upper = verification_result.upper()
            if "INCORRECT" in verification_result_upper:
                status = "INCORRECT"
            elif "PARTIALLY CORRECT" in verification_result_upper:
                status = "PARTIALLY CORRECT"
            elif "CORRECT" in verification_result_upper:
                status = "CORRECT"
            else:
                status = "OTHER"

            results.append({
                "statement": statement,
                "status": status,
                "verification_result": verification_result
            })

    except Exception as e:
        console.print(f"[red]Error during statement verification: {e}[/red]")

    return results



if __name__ == "__main__":
    orchestrate_complete_process()
