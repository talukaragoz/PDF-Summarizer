import uuid
import asyncio
from pypdf import PdfReader
import sqlite3
from contextlib import contextmanager

DATABASE_NAME = "data/pdf_chat.db"

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

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Create pdfs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pdfs (
            id TEXT PRIMARY KEY,
            original_filename TEXT NOT NULL,
            file_location TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            page_count INTEGER NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create extracted_text table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS extracted_text (
            pdf_id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (pdf_id) REFERENCES pdfs (id)
        )
        ''')
        
        # Create faq_cache table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS faq_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_id TEXT NOT NULL,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (pdf_id) REFERENCES pdfs (id)
        )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_faq_cache_pdf_id ON faq_cache (pdf_id)')
        
        conn.commit()

def insert_pdf_metadata(pdf_id: str, filename: str, file_path: str, file_size: int, page_count: int):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO pdfs (id, original_filename, file_location, file_size, page_count)
        VALUES (?, ?, ?, ?, ?)
        ''', (pdf_id, filename, file_path, file_size, page_count))
        conn.commit()

def insert_extracted_text(pdf_id: str, text: str):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO extracted_text (pdf_id, text)
        VALUES (?, ?)
        ''', (pdf_id, text))
        conn.commit()

def get_pdf_metadata(pdf_id: str):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM pdfs WHERE id = ?', (pdf_id,))
        return cursor.fetchone()

def get_extracted_text(pdf_id: str):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT text FROM extracted_text WHERE pdf_id = ?', (pdf_id,))
        result = cursor.fetchone()
        return result[0] if result else None