import uuid
import asyncio
from pypdf import PdfReader

def generate_pdf_id() -> str:
    return str(uuid.uuid4())

async def extract_text_from_pdf(path: str) -> str:
    def extract():
        with open(path, "rb") as file:
            pdf = PdfReader(file)
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text

    return await asyncio.to_thread(extract)