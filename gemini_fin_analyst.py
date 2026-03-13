import os
from dotenv import load_dotenv
from google import genai  # Modern 2026 Google Gen AI SDK
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

def run_gemini_financial_analyst(pdf_path, query):
    """
    Performs RAG (Retrieval-Augmented Generation) to analyze financial PDFs
    using local embeddings and Gemini 2.5 Flash.
    """
    
    # 1. Document Ingestion & Preprocessing
    print("📂 Loading and splitting PDF document...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Splitting text into manageable chunks for semantic search
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 2. Local Semantic Indexing (RAG)
    # Using HuggingFace local models to ensure privacy and reduce API costs
    print("🧠 Generating local vector embeddings (HuggingFace + FAISS)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    print(f"🔍 Retrieving relevant context for query: '{query}'")
    relevant_docs = vectorstore.similarity_search(query, k=5)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # 3. Configure Google Gen AI Client (2026 SDK)
    # Forcing 'v1' api_version to bypass 404 errors common in v1beta
    print("🚀 Connecting to Gemini 2.5 Flash (v1 Stable)...")
    client = genai.Client(
        api_key=os.getenv("GOOGLE_API_KEY"),
        http_options={'api_version': 'v1'}
    )
    
    # Constructing a professional persona-based prompt
    prompt = f"""
    You are a Senior Financial Analyst. 
    Use the provided context extracted from official documents to answer the user query.
    Be technical, precise, and objective. 
    If the answer is not in the context, state that you do not have enough information.
    
    CONTEXT:
    {context}
    
    QUERY:
    {query}
    
    PROFESSIONAL ANALYSIS:"""

    # 4. Generate AI Response
    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=prompt
    )
    
    return response.text

if __name__ == "__main__":
    # Ensure 'report.pdf' exists in your directory
    file_name = "report.pdf" 
    
    if os.path.exists(file_name):
        try:
            # Running the analysis
            analysis_result = run_gemini_financial_analyst(
                file_name, 
                "Summarize key financial metrics, revenue trends, and strategic outlook."
            )
            
            print("\n" + "="*60)
            print("📊 FINANCIAL ANALYSIS REPORT (GEMINI 2.5 FLASH)")
            print("="*60)
            print(analysis_result)
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ Execution Error: {e}")
    else:
        print(f"❌ Error: File '{file_name}' not found in the current directory.")