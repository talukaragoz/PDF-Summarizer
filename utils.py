import uuid
import aiosqlite
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import hashlib
from dotenv import load_dotenv
import os

from logging_config import logger

# Load environment variables from .env file
load_dotenv()

DATABASE_NAME = os.getenv("DATABASE_NAME")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

class ChatRequest(BaseModel):
    prompt: str

def generate_pdf_id() -> str:
    """
    Generate a unique identifier for a PDF.

    Returns:
        str: A unique UUID as a string.
    """
    pdf_id = str(uuid.uuid4())
    logger.debug(f"Generated new PDF ID: {pdf_id}")
    return pdf_id

async def pdf_text_extraction(pdf_id: str, path: str):
    """
    Extract text from a PDF file, split it into chunks, and store it in a vector database.

    Args:
        pdf_id (str): The unique identifier of the PDF.
        path (str): The file path to the PDF.

    Raises:
        Exception: If there's an error during text extraction or processing.
    """
    logger.info(f"Starting text extraction for PDF {pdf_id}")
    try:
        loader = PyPDFLoader(path)
        pages = loader.load_and_split()
        logger.debug(f"Loaded {len(pages)} pages from PDF {pdf_id}")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(pages)
        logger.debug(f"Split PDF {pdf_id} into {len(docs)} chunks")
        
        embeddings = HuggingFaceEmbeddings()
        
        persist_directory = f"data/chroma/{pdf_id}"
        vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
        vectorstore.persist()
        logger.info(f"Created and persisted vectorstore for PDF {pdf_id}")
        
        full_text = "\n".join([page.page_content for page in pages])
        
        await insert_extracted_text(pdf_id, full_text, persist_directory)
        logger.info(f"Inserted extracted text for PDF {pdf_id}")
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_id}: {str(e)}")

def calculate_pdf_hash(file_path: str) -> str:
    """
    Calculate the SHA-256 hash of a PDF file.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The hexadecimal representation of the SHA-256 hash.
    """
    logger.debug(f"Calculating hash for file: {file_path}")
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    hash_value = sha256_hash.hexdigest()
    logger.debug(f"Calculated hash: {hash_value}")
    return hash_value

@contextmanager
def get_db_connection():
    """
    Context manager for synchronous database connections.

    Yields:
        sqlite3.Connection: A connection to the SQLite database.
    """
    logger.debug("Opening database connection")
    conn = sqlite3.connect(DATABASE_NAME)
    try:
        yield conn
    finally:
        conn.close()
        logger.debug("Closed database connection")

@asynccontextmanager
async def get_async_db_connection():
    """
    Asynchronous context manager for database connections.

    Yields:
        aiosqlite.Connection: An asynchronous connection to the SQLite database.
    """
    logger.debug("Opening async database connection")
    conn = await aiosqlite.connect(DATABASE_NAME)
    try:
        yield conn
    finally:
        await conn.close()
        logger.debug("Closed async database connection")

async def init_db():
    """
    Initialize the database by creating necessary tables if they don't exist.
    """
    logger.info("Initializing database")
    async with get_async_db_connection() as conn:
        await conn.executescript('''
        CREATE TABLE IF NOT EXISTS pdfs (
            id TEXT PRIMARY KEY,
            original_filename TEXT NOT NULL,
            content_hash TEXT UNIQUE,
            file_location TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            page_count INTEGER NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS extracted_text (
            pdf_id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            vectorstore_dir TEXT NOT NULL,
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
    logger.info("Database initialized")

async def insert_pdf_metadata(pdf_id: str, filename: str, content_hash: str, file_path: str, file_size: int, page_count: int):
    """
    Insert metadata for a PDF into the database.

    Args:
        pdf_id (str): The unique identifier of the PDF.
        filename (str): The original filename of the PDF.
        content_hash (str): The hash of the PDF content.
        file_path (str): The path where the PDF is stored.
        file_size (int): The size of the PDF in bytes.
        page_count (int): The number of pages in the PDF.
    """
    logger.info(f"Inserting PDF metadata for {pdf_id}")
    async with get_async_db_connection() as conn:
        await conn.execute('''
        INSERT INTO pdfs (id, original_filename, content_hash, file_location, file_size, page_count)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (pdf_id, filename, content_hash, file_path, file_size, page_count))
        await conn.commit()
    logger.debug(f"PDF metadata inserted for {pdf_id}")

async def insert_extracted_text(pdf_id: str, text: str, vectorstore_dir):
    """
    Insert extracted text and vectorstore directory for a PDF into the database.

    Args:
        pdf_id (str): The unique identifier of the PDF.
        text (str): The extracted text from the PDF.
        vectorstore_dir (str): The directory where the vectorstore is saved.
    """
    logger.info(f"Inserting extracted text for PDF {pdf_id}")
    async with get_async_db_connection() as conn:
        await conn.execute('''
        INSERT INTO extracted_text (pdf_id, text, vectorstore_dir)
        VALUES (?, ?, ?)
        ''', (pdf_id, text, vectorstore_dir))
        await conn.commit()
    logger.debug(f"Extracted text inserted for PDF {pdf_id}")

async def get_pdf_metadata(pdf_id: str):
    """
    Retrieve metadata for a specific PDF from the database.

    Args:
        pdf_id (str): The unique identifier of the PDF.

    Returns:
        tuple: A tuple containing the PDF metadata, or None if not found.
    """
    logger.debug(f"Fetching PDF metadata for {pdf_id}")
    async with get_async_db_connection() as conn:
        async with conn.execute('SELECT * FROM pdfs WHERE id = ?', (pdf_id,)) as cursor:
            return await cursor.fetchone()

async def get_extracted_text(pdf_id: str):
    """
    Retrieve the extracted text for a specific PDF from the database.

    Args:
        pdf_id (str): The unique identifier of the PDF.

    Returns:
        str: The extracted text, or None if not found.
    """
    logger.debug(f"Fetching extracted text for PDF {pdf_id}")
    async with get_async_db_connection() as conn:
        async with conn.execute('SELECT text FROM extracted_text WHERE pdf_id = ?', (pdf_id,)) as cursor:
            result = await cursor.fetchone()
            return result[0] if result else None

async def get_vectorstore_dir(pdf_id: str):
    """
    Retrieve the vectorstore directory for a specific PDF from the database.

    Args:
        pdf_id (str): The unique identifier of the PDF.

    Returns:
        str: The vectorstore directory, or None if not found.
    """
    logger.debug(f"Fetching vectorstore for PDF {pdf_id}")
    async with get_async_db_connection() as conn:
        async with conn.execute('SELECT vectorstore_dir FROM extracted_text WHERE pdf_id = ?', (pdf_id,)) as cursor:
            result = await cursor.fetchone()
            return result[0] if result else None

async def clear_db():
    """
    Clear all data from the database and remove Chroma directories.
    """
    logger.debug(f"Clearing database")
    async with get_async_db_connection() as conn:
        await conn.executescript('''
        DELETE FROM faq_cache;
        DELETE FROM extracted_text;
        DELETE FROM pdfs;
        ''')
    # Remove Chroma persist directories
    shutil.rmtree('data/chroma', ignore_errors=True)
    logger.info("Database cleared and Chroma directories removed")

async def cache_query_response(pdf_id: str, query: str, response: str):
    """
    Cache a query and its response for a specific PDF.

    Args:
        pdf_id (str): The unique identifier of the PDF.
        query (str): The user's query.
        response (str): The generated response.
    """
    logger.info(f"Caching query response for PDF {pdf_id}")
    embedding = model.encode(query)
    embedding_bytes = embedding.tobytes()
    
    async with get_async_db_connection() as conn:
        await conn.execute('''
        INSERT INTO faq_cache (pdf_id, query, response, embedding)
        VALUES (?, ?, ?, ?)
        ''', (pdf_id, query, response, embedding_bytes))
        await conn.commit()
    logger.debug(f"Query response cached for PDF {pdf_id}")

async def get_cached(pdf_id: str):
    """
    Retrieve all cached queries and responses for a specific PDF.

    Args:
        pdf_id (str): The unique identifier of the PDF.

    Returns:
        list: A list of tuples containing (query, response, embedding) for the PDF.
    """
    logger.debug(f"Fetching cached responses for PDF {pdf_id}")
    async with get_async_db_connection() as conn:
        async with conn.execute('''
        SELECT query, response, embedding
        FROM faq_cache
        WHERE pdf_id = ?
        ''', (pdf_id,)) as cursor:
            results = await cursor.fetchall()
    
    return [(query, response, np.frombuffer(embedding, dtype=np.float32)) 
            for query, response, embedding in results]

async def get_pdf_by_hash(content_hash: str):
    """
    Retrieve PDF metadata by its content hash.

    Args:
        content_hash (str): The hash of the PDF content.

    Returns:
        tuple: A tuple containing the PDF metadata, or None if not found.
    """
    logger.debug(f"Fetching PDF by hash: {content_hash}")
    async with get_async_db_connection() as conn:
        async with conn.execute('''
        SELECT * 
        FROM pdfs 
        WHERE content_hash = ?
        ''', (content_hash,)) as cursor:
            return await cursor.fetchone()

async def find_similar_query(pdf_id: str, new_query: str, similarity_threshold: float = 0.82):
    """
    Find a similar cached query for a given PDF and new query.

    Args:
        pdf_id (str): The unique identifier of the PDF.
        new_query (str): The new query to compare against cached queries.
        similarity_threshold (float): The minimum cosine similarity to consider a match.

    Returns:
        str: The response of the most similar cached query, or None if no similar query is found.
    """
    logger.info(f"Finding similar query for PDF {pdf_id}")
    new_embedding = model.encode(new_query)
    cached_queries = await get_cached(pdf_id)
    
    if not cached_queries:
        logger.debug(f"No cached queries found for PDF {pdf_id}")
        return None
    
    for query, response, embedding in cached_queries:
        similarity = 1 - cosine(new_embedding, embedding)
        if similarity > similarity_threshold:
            logger.info(f"Similar query found for PDF {pdf_id} with similarity {similarity}")
            return response
    logger.debug(f"No similar query found for PDF {pdf_id}")
    return None

async def get_cached_response(pdf_id: str, query: str):
    """
    Get a cached response for a given PDF and query.

    Args:
        pdf_id (str): The unique identifier of the PDF.
        query (str): The user's query.

    Returns:
        str: The cached response if found, or None if not found.
    """
    logger.info(f"Attempting to get cached response for PDF {pdf_id}")
    cached_response = await find_similar_query(pdf_id, query)
    if cached_response:
        logger.info(f"Cache hit for PDF {pdf_id}")
        return cached_response
    else:
        logger.info(f"Cache miss for PDF {pdf_id}")
        return None