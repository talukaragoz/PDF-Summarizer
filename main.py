from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
import os
import shutil
from pypdf import PdfReader
import asyncio
from contextlib import asynccontextmanager
import google.generativeai as genai

from utils import *

GEMINI_KEY = open("environment_variables/GEMINI_KEY.txt", "r").read()
genai.configure(api_key=GEMINI_KEY)

# Current storage of PDFs
pdf_metadata = {}

MAX_FILE_SIZE = 10 * 1024 * 1024    # 10 MB

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup initialization of database
    await init_db()
    yield
    # Shutdown
    await clear_db()

app = FastAPI(lifespan=lifespan)


async def pdf_text_extraction(pdf_id: str, path: str):
    try:
        pdf_text = await extract_text_from_pdf(path)
        await insert_extracted_text(pdf_id, pdf_text)
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
        
        await insert_pdf_metadata(pdf_id, file.filename, file_path, file_size, page_count)
        
        background_tasks.add_task(pdf_text_extraction, pdf_id, file_path)
        
        return {
            "pdf_id": pdf_id
        }
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.post("/v1/chat/{pdf_id}")
async def pdf_chat(pdf_id: str, request: ChatRequest):
    pdf_metadata = await get_pdf_metadata(pdf_id)
    if not pdf_metadata:
        raise HTTPException(status_code=404, detail="PDF not found")
    
    extracted_text = await get_extracted_text(pdf_id)
    if not extracted_text:
        raise HTTPException(status_code=404, detail="Extracted text not found")
    
    # Check if the response is cached
    cahced_response = await get_cached_response(pdf_id, request.prompt)
    if cahced_response:
        return {"response": cahced_response}
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        full_prompt = f"Based on the following text from a PDF, please answer this question: {request.prompt}\n\nPDF Content: {extracted_text[:30000]}\n\nPlease provide your answer in plaintext only. Do not provide any markdown features in the text- not even new lines!"
        
        response = model.generate_content(full_prompt)
        
        await cache_query_response(pdf_id, request.prompt, response.text)
        
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while processing your request: {str(e)}") 