import uuid
import asyncio
from pypdf import PdfReader
import aiosqlite
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine

DATABASE_NAME = "data/pdf_chat.db"
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

class ChatRequest(BaseModel):
    prompt: str

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

@asynccontextmanager
async def get_async_db_connection():
    conn = await aiosqlite.connect(DATABASE_NAME)
    try:
        yield conn
    finally:
        await conn.close()

async def init_db():
    async with get_async_db_connection() as conn:
        await conn.executescript('''
        CREATE TABLE IF NOT EXISTS pdfs (
            id TEXT PRIMARY KEY,
            original_filename TEXT NOT NULL,
            file_location TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            page_count INTEGER NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS extracted_text (
            pdf_id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (pdf_id) REFERENCES pdfs (id)
        );
        
        CREATE TABLE IF NOT EXISTS faq_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_id TEXT NOT NULL,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (pdf_id) REFERENCES pdfs (id)
        );
        
        CREATE INDEX IF NOT EXISTS idx_faq_cache_pdf_id ON faq_cache (pdf_id);
        ''')

async def insert_pdf_metadata(pdf_id: str, filename: str, file_path: str, file_size: int, page_count: int):
    async with get_async_db_connection() as conn:
        await conn.execute('''
        INSERT INTO pdfs (id, original_filename, file_location, file_size, page_count)
        VALUES (?, ?, ?, ?, ?)
        ''', (pdf_id, filename, file_path, file_size, page_count))
        await conn.commit()

async def insert_extracted_text(pdf_id: str, text: str):
    async with get_async_db_connection() as conn:
        await conn.execute('''
        INSERT INTO extracted_text (pdf_id, text)
        VALUES (?, ?)
        ''', (pdf_id, text))
        await conn.commit()

async def get_pdf_metadata(pdf_id: str):
    async with get_async_db_connection() as conn:
        async with conn.execute('SELECT * FROM pdfs WHERE id = ?', (pdf_id,)) as cursor:
            return await cursor.fetchone()

async def get_extracted_text(pdf_id: str):
    async with get_async_db_connection() as conn:
        async with conn.execute('SELECT text FROM extracted_text WHERE pdf_id = ?', (pdf_id,)) as cursor:
            result = await cursor.fetchone()
            return result[0] if result else None

async def clear_db():
    async with get_async_db_connection() as conn:
        await conn.executescript('''
        DELETE FROM faq_cache;
        DELETE FROM extracted_text;
        DELETE FROM pdfs;
        ''')

async def cache_query_response(pdf_id: str, query: str, response: str):
    embedding = model.encode(query)
    embedding_bytes = embedding.tobytes()
    
    async with get_async_db_connection() as conn:
        await conn.execute('''
        INSERT INTO faq_cache (pdf_id, query, response, embedding)
        VALUES (?, ?, ?, ?)
        ''', (pdf_id, query, response, embedding_bytes))
        await conn.commit()

async def get_cached(pdf_id: str):
    async with get_async_db_connection() as conn:
        async with conn.execute('''
        SELECT query, response, embedding
        FROM faq_cache
        WHERE pdf_id = ?
        ''', (pdf_id,)) as cursor:
            results = await cursor.fetchall()
    
    return [(query, response, np.frombuffer(embedding, dtype=np.float32)) 
            for query, response, embedding in results]

async def find_similar_query(pdf_id: str, new_query: str, similarity_threshold: float = 0.82):
    new_embedding = model.encode(new_query)
    cached_queries = await get_cached(pdf_id)
    
    if not cached_queries:
        return None
    
    for query, response, embedding in cached_queries:
        similarity = 1 - cosine(new_embedding, embedding)
        if similarity > similarity_threshold:
            return response
    return None

async def get_cached_response(pdf_id: str, query: str):
    cached_response = await find_similar_query(pdf_id, query)
    if cached_response:
        return cached_response  # True indicates a cache hit
    else:
        return None