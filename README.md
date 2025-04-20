# MCP-RAG: Model Context Protocol with RAG 🚀

A powerful and efficient RAG (Retrieval-Augmented Generation) implementation using GroundX and OpenAI, built with Modern Context Processing (MCP).

## 🌟 Features

- **Advanced RAG Implementation**: Utilizes GroundX for high-accuracy document retrieval
- **Model Context Protocol**: Seamless integration with MCP for enhanced context handling
- **Type-Safe**: Built with Pydantic for robust type checking and validation
- **Flexible Configuration**: Easy-to-customize settings through environment variables
- **Document Ingestion**: Support for PDF document ingestion and processing
- **Intelligent Search**: Semantic search capabilities with scoring

## 🛠️ Prerequisites

- Python 3.12 or higher
- OpenAI API key
- GroundX API key
- MCP CLI tools

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mcp-rag
```

2. Create and activate a virtual environment:
```bash
uv sync
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```


## ⚙️ Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Configure your environment variables in `.env`:
```env
GROUNDX_API_KEY="your-groundx-api-key"
OPENAI_API_KEY="your-openai-api-key"
BUCKET_ID="your-bucket-id"
```

## 🚀 Usage

### Starting the Server

Run the inspect server using:
```bash
mcp dev server.py
```

### Document Ingestion

To ingest new documents:
```python
from server import ingest_documents

result = ingest_documents("path/to/your/document.pdf")
print(result)
```

### Performing Searches

Basic search query:
```python
from server import process_search_query

response = process_search_query("your search query here")
print(f"Query: {response.query}")
print(f"Score: {response.score}")
print(f"Result: {response.result}")
```

With custom configuration:
```python
from server import process_search_query, SearchConfig

config = SearchConfig(
    completion_model="gpt-4",
    bucket_id="custom-bucket-id"
)
response = process_search_query("your query", config)
```

## 📚 Dependencies

- `groundx` (≥2.3.0): Core RAG functionality
- `openai` (≥1.75.0): OpenAI API integration
- `mcp[cli]` (≥1.6.0): Modern Context Processing tools
- `ipykernel` (≥6.29.5): Jupyter notebook support

## 🔒 Security

- Never commit your `.env` file containing API keys
- Use environment variables for all sensitive information
- Regularly rotate your API keys
- Monitor API usage for any unauthorized access

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request