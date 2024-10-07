import pytest
from fastapi.testclient import TestClient
from main import app
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
import os
import shutil
from utils import init_db, clear_db
from fastapi import HTTPException

# Setup test client
client = TestClient(app)

# Use a separate test database
TEST_DATABASE_NAME = "test_pdf_chat.db"

@pytest.fixture(scope="module")
def test_app():
    # Setup
    original_db = os.environ.get("DATABASE_NAME")
    os.environ["DATABASE_NAME"] = TEST_DATABASE_NAME
    asyncio.run(init_db())
    yield app
    # Teardown
    asyncio.run(clear_db())
    if original_db:
        os.environ["DATABASE_NAME"] = original_db
    else:
        del os.environ["DATABASE_NAME"]
    if os.path.exists(TEST_DATABASE_NAME):
        os.remove(TEST_DATABASE_NAME)

@pytest.fixture(autouse=True)
def run_around_tests():
    # Before each test
    yield
    # After each test
    if os.path.exists("data/input"):
        shutil.rmtree("data/input")
    if os.path.exists("data/chroma"):
        shutil.rmtree("data/chroma")

def test_pdf_chat(test_app):
    with patch('main.get_pdf_metadata', new_callable=AsyncMock) as mock_get_metadata, \
         patch('main.get_extracted_text', new_callable=AsyncMock) as mock_get_text, \
         patch('main.get_vectorstore_dir', new_callable=AsyncMock) as mock_get_vectorstore, \
         patch('main.get_cached_response', new_callable=AsyncMock) as mock_get_cached, \
         patch('main.Chroma') as mock_chroma, \
         patch('main.genai.GenerativeModel') as mock_genai:
        
        mock_get_metadata.return_value = ("test_id", "test.pdf", "hash123", "/path/to/file", 1000, 5)
        mock_get_text.return_value = "Extracted text"
        mock_get_vectorstore.return_value = "/path/to/vectorstore"
        mock_get_cached.return_value = None  # Simulate no cached response
        mock_chroma_instance = MagicMock()
        mock_chroma_instance.similarity_search.return_value = ["Relevant chunk"]
        mock_chroma.return_value = mock_chroma_instance
        mock_genai_instance = MagicMock()
        mock_genai_instance.generate_content.return_value = MagicMock(text="Generated response")
        mock_genai.return_value = mock_genai_instance
        
        response = client.post("/v1/chat/test_id", json={"prompt": "Test question"})
        
        assert response.status_code == 200
        assert response.json() == {"response": "Generated response"}
        mock_get_metadata.assert_called_once_with("test_id")
        mock_get_text.assert_called_once_with("test_id")
        mock_get_vectorstore.assert_called_once_with("test_id")
        mock_get_cached.assert_called_once()
        mock_chroma_instance.similarity_search.assert_called_once()
        mock_genai_instance.generate_content.assert_called_once()