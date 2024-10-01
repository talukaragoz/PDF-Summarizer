from fastapi import FastAPI, UploadFile, HTTPException
import os
import shutil
from pypdf import PdfReader

from utils import generate_pdf_id

app = FastAPI()

GEMINI_KEY = open("environment_variables/GEMINI_KEY.txt", "r").read()

# Current storage of PDFs
pdf_metadata = {}

MAX_FILE_SIZE = 10 * 1024 * 1024    # 10 MB

@app.post("/v1/pdf")
async def pdf_ingestion(file: UploadFile):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Please upload a PDF file!!")
    
    
    file.file.seek(0,2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds the maximum limit of 10 MB")

    pdf_id = generate_pdf_id()
    print(pdf_id)
    safe_filename = "".join([c for c in file.filename if c.isalnum() or c in ('-', '_', '.')])
    file_location = f"data/input/{pdf_id}_{safe_filename}"

    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Extract metadata
        with open(file_location, "rb") as f:
            pdf = PdfReader(f)
            page_count = len(pdf.pages)
        
        try:
            PdfReader(file_location)
        except:
            os.remove(file_location)
            raise HTTPException(status_code=400, detail="Invalid PDF file")
        
        # Store metadata
        pdf_metadata[pdf_id] = {
            "original_filename": file.filename,
            "file_location": file_location,
            "file_size": file_size,
            "page_count" : page_count
        }
        
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