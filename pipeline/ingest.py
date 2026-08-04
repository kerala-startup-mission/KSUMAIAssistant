import os
import sys
import re
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import ollama
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.config import DATA_DIRS, EMBEDDING_MODEL, COLLECTION_NAME, CHUNK_SIZE, LLM_MODEL, CHROMA_HOST, CHROMA_PORT
# DYNAMIC CHUNK OVERLAP: Automatically calculates ~15% overlap to preserve boundary context
CHUNK_OVERLAP = int(CHUNK_SIZE * 0.15)

def _generate_chunk_summary(chunk_text: str) -> str:
    """Uses Gemma to generate a dense, noise-free abstract of the text chunk for superior vector matching."""
    system_prompt = (
        "You are a data indexing assistant. Read the provided text block and generate a highly concise 1 to 2 sentence abstract. "
        "Focus strictly on the scheme names, eligibility criteria, monetary amounts, and core topics. "
        "DO NOT use conversational filler (e.g., 'This text discusses...'). Output ONLY the pure summary."
    )
    
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': chunk_text}
            ],
            options={'temperature': 0.0, 'num_predict': 100, 'num_ctx': 2048} 
        )
        summary = response['message']['content'].strip()
        return summary if summary else chunk_text
    except Exception as e:
        # ADDED FLUSH
        print(f"      [!] Summary generation failed, falling back to raw text: {e}", flush=True)
        return chunk_text

def ingest_data():
    # ADDED FLUSH
    print("Starting Advanced KSUM Data Ingestion Pipeline (Smart Append Mode)...", flush=True)
    print(f"Initializing local embedding model ({EMBEDDING_MODEL})...", flush=True)
    
    chrome_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    
    # 1. DO NOT DELETE THE DB. Get the existing collection or create it if it's the first run.
    collection = chrome_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )
    
    # 2. ASK THE DB WHAT IT ALREADY KNOWS
    existing_data = collection.get(include=["metadatas"])
    existing_files = set()
    
    if existing_data and existing_data.get("metadatas"):
        for meta in existing_data["metadatas"]:
            if meta and "file" in meta:
                existing_files.add(meta["file"])
                
    # ADDED FLUSH
    print(f"Database currently contains {len(existing_files)} indexed document(s).", flush=True)
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False 
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]
    )

    total_new_chunks = 0
    
    # --- NEW: PRE-SCAN FILES TO PROVIDE ACCURATE PROGRESS COUNTS ---
    files_to_process = []
    
    for category_name, folder_path in DATA_DIRS.items():
        if not os.path.exists(folder_path):
            print(f"Warning: Folder '{folder_path}' not found. Skipping.", flush=True)
            continue
            
        for filename in os.listdir(folder_path):
            if not filename.endswith(".md"):
                continue
                
            if filename in existing_files:
                print(f"  -> Skipping {filename} (Already indexed in ChromaDB)", flush=True)
                continue
                
            files_to_process.append((category_name, folder_path, filename))

    if not files_to_process:
        print("\nNo new files found. Database is up to date!", flush=True)
        return
        
    print(f"\nFound {len(files_to_process)} new file(s) to ingest.", flush=True)

    # --- NEW: ITERATE OVER THE PRE-COMPILED LIST WITH AN ENUMERATE COUNTER ---
    for index, (category_name, folder_path, filename) in enumerate(files_to_process, 1):
        
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()
        
        source_url = filename
        url_match = re.search(r'\*\s*\*\*URL:\*\*\s*(https?://[^\s]+)', file_content)
        if url_match:
            source_url = url_match.group(1)
        
        md_header_splits = markdown_splitter.split_text(file_content)
        final_splits = text_splitter.split_documents(md_header_splits)
        
        documents = []
        summaries = []
        metadatas = []
        ids = []
        
        # ADDED FLUSH & PROGRESS COUNTER
        print(f"[{index}/{len(files_to_process)}] Indexing NEW FILE: {filename} ({len(final_splits)} chunks)", flush=True)
        
        for idx, split in enumerate(final_splits):
            original_text = split.page_content
            documents.append(original_text)
            
            meta = split.metadata.copy() 
            meta.update({
                "source": source_url, 
                "file": filename, 
                "chunk_index": idx,
                "category": category_name,
                "has_funding_info": any(w in original_text.lower() for w in ["lakh", "amount", "grant", "fund", "₹", "rs"]),
                "has_eligibility_info": any(w in original_text.lower() for w in ["eligible", "criteria", "apply", "who can"])
            })
            metadatas.append(meta)
            
            # 4. UNIQUE IDs: Bind the chunk ID to the filename so they never overlap with older files
            ids.append(f"{filename}_chunk_{idx}")
            
            # ADDED FLUSH
            print(f"      -> Summarizing chunk {idx + 1}/{len(final_splits)}...", flush=True)
            summaries.append(_generate_chunk_summary(original_text))
        
        if documents:
            # ADDED FLUSH
            print(f"      -> Computing vectors and injecting into ChromaDB...", flush=True)
            summary_embeddings = embedding_func(summaries)
            
            collection.add(
                documents=documents, 
                embeddings=summary_embeddings, 
                metadatas=metadatas, 
                ids=ids
            )
            total_new_chunks += len(documents)
            
    if total_new_chunks > 0:
        # ADDED FLUSH
        print(f"\nSuccess! Embedded {total_new_chunks} new chunks into ChromaDB.", flush=True)
    else:
        # ADDED FLUSH
        print("\nNo new files found. Database is up to date!", flush=True)

if __name__ == "__main__":
    ingest_data()