# PDF Chat API

## Overview

The PDF Chat API is a sophisticated FastAPI application that enables users to upload PDF documents, extract their content, and engage in an interactive chat based on the PDF's contents. This project leverages advanced natural language processing techniques, including the Google Gemini API, to provide intelligent responses to user queries about the uploaded PDFs.

Key features of the application include:
- PDF upload and processing
- Text extraction and vectorization
- Integration with Google's Gemini API for advanced language understanding
- Caching mechanism for faster responses to similar queries
- Comprehensive error handling and logging
- Rate limiting and timeout protection

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <project-directory>
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Environment Configuration

It's recommended to use a Conda environment with Python 3.10 for this project. Here's how to set it up:

1. Create a new Conda environment:
   ```
   conda create -n pdf-chat-api python=3.10
   ```

2. Activate the environment:
   ```
   conda activate pdf-chat-api
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root directory and add the following environment variables:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   DATABASE_NAME=pdf_chat.db
   ```

   Replace `your_gemini_api_key_here` with your actual Google Gemini API key.

## API Endpoints

### 1. PDF Upload

- **Endpoint**: `/v1/pdf`
- **Method**: POST
- **Description**: Upload a PDF file for processing and chatting.
- **Request**:
  - Content-Type: `multipart/form-data`
  - Body: PDF file (max size: 10MB)
- **Response**:
  ```json
  {
    "pdf_id": "unique_pdf_identifier"
  }
  ```

### 2. Chat with PDF

- **Endpoint**: `/v1/chat/{pdf_id}`
- **Method**: POST
- **Description**: Send a query about a specific PDF and receive an AI-generated response.
- **Request**:
  ```json
  {
    "prompt": "What is the main topic of this PDF?"
  }
  ```
- **Response**:
  ```json
  {
    "response": "The main topic of this PDF is..."
  }
  ```

## Data Storage

The application uses SQLite for data storage, with the following structure:

1. **PDFs Table**: Stores metadata about uploaded PDFs.
   - Columns: id, original_filename, content_hash, file_location, file_size, page_count, upload_date

2. **Extracted Text Table**: Stores the extracted text content and vectorstore directory for each PDF.
   - Columns: pdf_id, text, vectorstore_dir, extraction_date

3. **FAQ Cache Table**: Stores cached responses for faster retrieval of similar queries.
   - Columns: id, pdf_id, query, response, embedding, created_at

The application also uses Chroma for vector storage, which allows for efficient similarity search when processing user queries.

## Testing

The project includes both unit tests and integration tests. To run the tests, follow these steps:

1. Ensure you're in the project root directory and your Conda environment is activated.

2. Run the unit tests:
   ```
   pytest test_unit.py
   ```

3. Run the integration tests:
   ```
   pytest test_integration.py
   ```

Note: The integration tests use a separate test database to avoid interfering with the production database.

When contributing to the project, please ensure that all existing tests pass and add new tests for any new functionality you implement.

## Error Handling and Logging

The application implements comprehensive error handling and logging:

- Custom error handlers for various scenarios (e.g., rate limiting, timeouts, Google API errors)
- Detailed logging of operations, including PDF processing, API calls, and database interactions
- Use of structured logging for easy parsing and analysis

## Performance and Scalability

The application includes several features to enhance performance and scalability:

- Rate limiting to prevent abuse
- Timeout middleware to handle long-running requests
- Caching of responses for similar queries
- Use of background tasks for time-consuming operations like PDF text extraction

## Security Considerations

- API keys and sensitive information are stored in environment variables
- Input validation and sanitization to prevent security vulnerabilities
- Rate limiting to mitigate potential DoS attacks

For any questions or issues, please open an issue in the project repository or contact the maintainers directly.