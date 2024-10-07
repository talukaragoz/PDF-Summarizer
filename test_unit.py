import pytest
from unittest.mock import AsyncMock, patch, MagicMock, Mock
import uuid
from utils import *
from main import pdf_ingestion, pdf_chat
import logging
from contextlib import asynccontextmanager

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Utility function tests

def test_generate_pdf_id():
    pdf_id = generate_pdf_id()
    assert isinstance(pdf_id, str)
    assert uuid.UUID(pdf_id)  # Verify it's a valid UUID

@pytest.mark.asyncio
async def test_pdf_text_extraction(tmp_path):
    # Create a mock PDF file
    pdf_content = b"%PDF-1.5\n%Test PDF content"
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(pdf_content)
    
    pdf_id = str(uuid.uuid4())
    
    with patch('utils.PyPDFLoader') as mock_loader, \
         patch('utils.RecursiveCharacterTextSplitter') as mock_splitter, \
         patch('utils.HuggingFaceEmbeddings') as mock_embeddings, \
         patch('utils.Chroma') as mock_chroma, \
         patch('utils.insert_extracted_text') as mock_insert:
        
        mock_loader.return_value.load_and_split.return_value = [Mock(page_content="Test content")]
        mock_splitter.return_value.split_documents.return_value = [Mock(page_content="Split content")]
        mock_chroma.from_documents.return_value = Mock()
        mock_insert.return_value = AsyncMock()
        
        await pdf_text_extraction(pdf_id, str(pdf_path))
        
        mock_loader.assert_called_once_with(str(pdf_path))
        mock_splitter.assert_called_once()
        mock_embeddings.assert_called_once()
        mock_chroma.from_documents.assert_called_once()
        mock_insert.assert_called_once()

def test_calculate_pdf_hash(tmp_path):
    pdf_content = b"%PDF-1.5\n%Test PDF content"
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(pdf_content)
    
    hash_value = calculate_pdf_hash(str(pdf_path))
    assert isinstance(hash_value, str)
    assert len(hash_value) == 64  # SHA-256 hash length

# Database operation tests

@pytest.mark.asyncio
async def test_init_db():
    mock_conn = AsyncMock()
    mock_conn.executescript = AsyncMock()

    # Create a mock for the async context manager
    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = mock_conn
    mock_cm.__aexit__.return_value = None

    # Mock the get_async_db_connection function
    with patch('utils.get_async_db_connection', return_value=mock_cm):
        await init_db()
        
        mock_cm.__aenter__.assert_awaited_once()
        mock_conn.executescript.assert_awaited_once()
        mock_cm.__aexit__.assert_awaited_once()

@pytest.mark.asyncio
async def test_insert_pdf_metadata():
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.commit = AsyncMock()

    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = mock_conn
    mock_cm.__aexit__.return_value = None

    with patch('utils.get_async_db_connection', return_value=mock_cm):
        pdf_id = str(uuid.uuid4())
        await insert_pdf_metadata(pdf_id, "test.pdf", "hash123", "/path/to/file", 1000, 5)
        
        mock_cm.__aenter__.assert_awaited_once()
        mock_conn.execute.assert_awaited_once()
        mock_conn.commit.assert_awaited_once()
        mock_cm.__aexit__.assert_awaited_once()

@pytest.mark.asyncio
async def test_get_extracted_text():
    # Mock cursor
    mock_cursor = AsyncMock()
    mock_cursor.fetchone.return_value = ("Extracted text",)

    execute_args = None

    # Mock execute function that returns an async context manager
    @asynccontextmanager
    async def mock_execute(*args, **kwargs):
        nonlocal execute_args
        execute_args = args  # Capture the arguments
        yield mock_cursor

    # Mock connection
    mock_conn = AsyncMock()
    mock_conn.execute = mock_execute

    # Mock get_async_db_connection
    @asynccontextmanager
    async def mock_get_async_db_connection():
        yield mock_conn

    with patch('utils.get_async_db_connection', mock_get_async_db_connection):
        pdf_id = str(uuid.uuid4())
        logger.debug(f"Testing get_extracted_text with pdf_id: {pdf_id}")
        
        try:
            result = await get_extracted_text(pdf_id)
            logger.debug(f"Result from get_extracted_text: {result}")
            
            assert result == "Extracted text"
            logger.debug("Assert passed: result matches expected text")
            
            mock_cursor.fetchone.assert_awaited_once()
            logger.debug("Assert passed: fetchone called")
            
            # Check the SQL query
            assert execute_args is not None, "execute was not called"
            assert execute_args[0] == 'SELECT text FROM extracted_text WHERE pdf_id = ?'
            assert execute_args[1] == (pdf_id,)
            logger.debug("Assert passed: correct SQL query and parameters used")
            
        except Exception as e:
            logger.error(f"Error occurred: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error traceback:", exc_info=True)
            raise

        logger.debug("Test completed successfully")

@pytest.mark.asyncio
async def test_cache_query_response():
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.commit = AsyncMock()

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])

    @asynccontextmanager
    async def mock_db_connection():
        yield mock_conn

    with patch('utils.get_async_db_connection', mock_db_connection), \
         patch('utils.model', mock_model):
        
        pdf_id = str(uuid.uuid4())
        query = "Test query"
        response = "Test response"
        
        await cache_query_response(pdf_id, query, response)
        
        mock_model.encode.assert_called_once_with(query)
        mock_conn.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

@pytest.mark.asyncio
async def test_find_similar_query():
    with patch('utils.get_cached') as mock_get_cached, \
         patch('utils.model.encode') as mock_encode, \
         patch('utils.cosine') as mock_cosine:
        mock_get_cached.return_value = [
            ("Similar query", "Cached response", b"encoded_similar_query")
        ]
        mock_encode.return_value = b"encoded_new_query"
        mock_cosine.return_value = 0.1  # High similarity
        
        pdf_id = str(uuid.uuid4())
        new_query = "New test query"
        
        result = await find_similar_query(pdf_id, new_query)
        
        assert result == "Cached response"
        mock_get_cached.assert_called_once_with(pdf_id)
        mock_encode.assert_called_once_with(new_query)

@pytest.mark.asyncio
async def test_get_pdf_metadata():
    mock_cursor = AsyncMock()
    mock_cursor.fetchone.return_value = ("test_id", "test.pdf", "hash123", "/path/to/file", 1000, 5)

    execute_args = None

    # Mock execute function that returns an async context manager
    @asynccontextmanager
    async def mock_execute(*args, **kwargs):
        nonlocal execute_args
        execute_args = args  # Capture the arguments
        yield mock_cursor
    
    # Mock connection
    mock_conn = AsyncMock()
    mock_conn.execute = mock_execute

    db_connection_called = False
    
    # Mock get_async_db_connection
    @asynccontextmanager
    async def mock_get_async_db_connection():
        nonlocal db_connection_called
        db_connection_called = True
        yield mock_conn

    with patch('utils.get_async_db_connection', mock_get_async_db_connection):
        pdf_id = "test_id"
        try:
            result = await get_pdf_metadata(pdf_id)
            
            assert result == ("test_id", "test.pdf", "hash123", "/path/to/file", 1000, 5)
            # Assert that fetchone was called once
            mock_cursor.fetchone.assert_called_once()
            
            # Assert that execute was called with the correct SQL query and parameters
            assert execute_args is not None, "execute was not called"
            assert execute_args[0] == 'SELECT * FROM pdfs WHERE id = ?'
            assert execute_args[1] == (pdf_id,)
            
            # Assert that the result has the correct number of elements
            assert len(result) == 6, "Result should have 6 elements"
            
            # Assert the types of the returned values
            assert isinstance(result[0], str), "PDF ID should be a string"
            assert isinstance(result[1], str), "Filename should be a string"
            assert isinstance(result[2], str), "Hash should be a string"
            assert isinstance(result[3], str), "File path should be a string"
            assert isinstance(result[4], int), "File size should be an integer"
            assert isinstance(result[5], int), "Page count should be an integer"
            
            # Assert that get_async_db_connection was called
            assert db_connection_called, "get_async_db_connection was not called"
            
            print("All assertions passed successfully!")
        except Exception as e:
            logger.error(f"Error occurred: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error traceback:", exc_info=True)
            raise

@pytest.mark.asyncio
async def test_get_vectorstore_dir():
    mock_cursor = AsyncMock()
    mock_cursor.fetchone.return_value = ("/path/to/vectorstore",)

    execute_args = None

    # Mock execute function that returns an async context manager
    @asynccontextmanager
    async def mock_execute(*args, **kwargs):
        nonlocal execute_args
        execute_args = args  # Capture the arguments
        yield mock_cursor
    
    # Mock connection
    mock_conn = AsyncMock()
    mock_conn.execute = mock_execute

    db_connection_called = False
    
    # Mock get_async_db_connection
    @asynccontextmanager
    async def mock_get_async_db_connection():
        nonlocal db_connection_called
        db_connection_called = True
        yield mock_conn

    with patch('utils.get_async_db_connection', mock_get_async_db_connection):
        pdf_id = "test_id"
        result = await get_vectorstore_dir(pdf_id)
        
        mock_cursor.fetchone.assert_called_once()
        
        assert result == "/path/to/vectorstore"
        assert execute_args is not None, "execute was not called"
        
        # Assert that get_async_db_connection was called
        assert db_connection_called, "get_async_db_connection was not called"
@pytest.mark.asyncio
async def test_pdf_text_extraction():
    with patch('utils.PyPDFLoader') as mock_loader, \
         patch('utils.RecursiveCharacterTextSplitter') as mock_splitter, \
         patch('utils.HuggingFaceEmbeddings') as mock_embeddings, \
         patch('utils.Chroma') as mock_chroma, \
         patch('utils.insert_extracted_text') as mock_insert:
        
        mock_loader.return_value.load_and_split.return_value = [MagicMock(page_content="Test content")]
        mock_splitter.return_value.split_documents.return_value = [MagicMock(page_content="Split content")]
        mock_chroma.from_documents.return_value = MagicMock()
        
        pdf_id = str(uuid.uuid4())
        file_path = "/path/to/test.pdf"
        
        await pdf_text_extraction(pdf_id, file_path)
        
        mock_loader.assert_called_once_with(file_path)
        mock_splitter.assert_called_once()
        mock_embeddings.assert_called_once()
        mock_chroma.from_documents.assert_called_once()
        mock_insert.assert_called_once()

@pytest.mark.asyncio
async def test_pdf_ingestion():
    mock_file = MagicMock()
    mock_file.filename = "test.pdf"
    mock_file.content_type = "application/pdf"
    mock_file.file.read.return_value = b"%PDF-1.5\nTest PDF content"
    mock_file.file.seek = MagicMock()
    mock_file.file.tell.return_value = 1000  # Simulate file size

    mock_pdf_reader = MagicMock()
    mock_pdf_reader.pages = [MagicMock(), MagicMock()]  # Simulate 2 pages

    with patch('main.generate_pdf_id', return_value="test_id"), \
         patch('main.calculate_pdf_hash', return_value="test_hash"), \
         patch('main.get_pdf_by_hash', return_value=None), \
         patch('main.insert_pdf_metadata', AsyncMock()), \
         patch('main.pdf_text_extraction', AsyncMock()), \
         patch('main.shutil.copyfileobj', MagicMock()), \
         patch('builtins.open', MagicMock()), \
         patch('os.makedirs', MagicMock()), \
         patch('main.PdfReader', return_value=mock_pdf_reader):

        result = await pdf_ingestion(mock_file, AsyncMock())
        
        assert result == {"pdf_id": "test_id"}


@pytest.mark.asyncio
async def test_pdf_chat():
    with patch('main.get_pdf_metadata') as mock_get_metadata, \
         patch('main.get_extracted_text') as mock_get_text, \
         patch('main.get_vectorstore_dir') as mock_get_vectorstore, \
         patch('main.get_cached_response') as mock_get_cached, \
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
        
        result = await pdf_chat("test_id", MagicMock(prompt="Test question"))
        
        assert result == {"response": "Generated response"}
        mock_get_metadata.assert_called_once_with("test_id")
        mock_get_text.assert_called_once_with("test_id")
        mock_get_vectorstore.assert_called_once_with("test_id")
        mock_get_cached.assert_called_once()
        mock_chroma_instance.similarity_search.assert_called_once()
        mock_genai_instance.generate_content.assert_called_once()