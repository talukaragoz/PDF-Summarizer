from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
import os
import shutil
from pypdf import PdfReader
import asyncio
from contextlib import asynccontextmanager

from utils import *

GEMINI_KEY = open("environment_variables/GEMINI_KEY.txt", "r").read()

# Current storage of PDFs
pdf_metadata = {}

MAX_FILE_SIZE = 10 * 1024 * 1024    # 10 MB

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup initialization of database
    init_db()
    yield
    # Shutdown
    clear_db()

app = FastAPI(lifespan=lifespan)


async def pdf_text_extraction(pdf_id: str, path: str):
    try:
        pdf_text = await extract_text_from_pdf(path)
        insert_extracted_text(pdf_id, pdf_text)
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_id}: {str(e)}")


@app.post("/v1/pdf")
async def pdf_ingestion(file: UploadFile, background_tasks: BackgroundTasks):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Please upload a PDF file!!")
    
    file.file.seek(0,2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds the maximum limit of 10 MB")

    pdf_id = generate_pdf_id()
    safe_filename = "".join([c for c in file.filename if c.isalnum() or c in ('-', '_', '.')])
    file_path = f"data/input/{pdf_id}_{safe_filename}"

    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Extract metadata
        with open(file_path, "rb") as f:
            pdf = PdfReader(f)
            page_count = len(pdf.pages)
        
        try:
            PdfReader(file_path)
        except:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Invalid PDF file")
        
        insert_pdf_metadata(pdf_id, file.filename, file_path, file_size, page_count)
        
        background_tasks.add_task(pdf_text_extraction, pdf_id, file_path)
        
        return {
            "pdf_id": pdf_id,
            "original_filename": file.filename,
            "file_size": file_size,
            "page_count" : page_count
        }
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.get("/v1/chat/{pdf_id}")
async def pdf_chat(pdf_id):
    return {"message": "Hello World"}