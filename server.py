import os
from dotenv import load_dotenv
from groundx import GroundX, Document
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
import os
import openai
from groundx import GroundX
from typing import Optional
from dotenv import load_dotenv
import asyncio



load_dotenv()

mcp = FastMCP("mcp-rag")
client = GroundX(api_key=os.getenv("GROUNDX_API_KEY"))


class SearchResponse(BaseModel):
    query: str
    score: float
    result: str

class SearchConfig(BaseModel):
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    groundx_api_key: str = Field(default_factory=lambda: os.getenv("GROUNDX_API_KEY"))
    completion_model: str = "gpt-4o"
    bucket_id: int = Field(default_factory=lambda: os.getenv("BUCKET_ID"))


@mcp.tool()
def process_search_query(query: str, config: Optional[SearchConfig] = None) -> SearchResponse:
    """
    Process a search query using GroundX and OpenAI.
    
    Args:
        query: The search query string
        config: Optional SearchConfig object for customization
    
    Returns:
        SearchResponse object containing the query, score, and result
    """
    if config is None:
        config = SearchConfig()

    # Initialize clients
    client = GroundX(api_key=config.groundx_api_key)
    openai.api_key = config.openai_api_key

    # System instruction for the AI
    instruction = """`You are a highly knowledgeable assistant. Your primary role is to assist developers by answering questions related to documents they have uploaded and that have been processed by the GroundX proprietary ingestion pipeline. This pipeline creates semantic objects and is known for delivering the highest accuracy in RAG retrievals on the market.

Key Responsibilities:
    1.	Document Verification and Summary:
	- When the developer asks about specific documents, your task is to verify that the documents have been successfully uploaded and processed. You may be asked to summarize the contents of a document, describe the types of data extracted, or confirm the presence of specific information.

	2.	Handling Filenames:
	- Developers might refer to documents by filename. They may make minor spelling or case errors when mentioning filenames. Your task is to interpret these filenames as accurately as possible based on context and provide relevant responses, using the correct filenames from the processed content.

	3.	Demonstrating RAG Capabilities:
	- Developers may test the accuracy and efficacy of the GroundX RAG ingestion by asking general or specific questions. Your answers should demonstrate the high accuracy and reliability of the retrievals, showcasing how well the system processes and retrieves information from their documents.
    - If asked to summarize, extract, or perform any other operation on the ingested documents, provide detailed and precise answers based on the semantic objects created during ingestion.

Your Responses Should Be:
	1.	Accurate and Detailed: Base your responses on the processed documents available in the system. Provide accurate and detailed information, ensuring that the developers see the full capabilities of the GroundX RAG system.

	2.	Clear and Technical: Tailor your responses to a developer audience. Be precise in your explanations, using technical language where appropriate to demonstrate your understanding of the ingestion and retrieval process.

	3.	Supportive of Testing: Understand that developers may be testing the system's capabilities. If a developer asks a general or test question, your response should help validate that the ingestion and retrieval processes have been successful.

	4.	Context-Aware: Take into account any context provided by the developer, especially if they mention specific documents or filenames. Ensure that your answers are relevant to the specific queries asked.

	5.	Informative about Errors: If a document cannot be found or if there is an issue with retrieval, inform the developer clearly and suggest checking the document upload process. Do not assume an error in the system unless explicitly noted.


Handling Specific Scenarios:
	1.	Document Not Found: If a document referenced by the developer is not found in the system, respond with a message that indicates this and suggest they verify the upload process.

	2.	Incorrect or Ambiguous Filenames: If the developer refers to a document with a filename that is slightly incorrect or ambiguous, attempt to match it with the closest available document and confirm with the developer.

	3.	General Questions: When asked general questions, please rely on your general knowledge of the world.`"""

    # Perform content search
    content_response = client.search.content(
        id=config.bucket_id,
        query=query,
    )
    results = content_response.search

    # Generate completion using OpenAI
    completion = openai.chat.completions.create(
        model=config.completion_model,
        messages=[
            {
                "role": "system",
                "content": f"{instruction}\n===\n{results.text}\n===\n"
            },
            {"role": "user", "content": query},
        ],
    )

    return SearchResponse(
        query=query,
        score=results.score,
        result=completion.choices[0].message.content)




@mcp.tool()
def search_doc_for_rag_context(query: str) -> str:
    """
    Searches and retrieves relevant context from a knowledge base,
    based on the user's query.
    Args:
        query: The search query supplied by the user.
    Returns:
        str: Relevant text content that can be used by the LLM to answer the query.
    """
    response = client.search.content(
        id=os.getenv("BUCKET_ID"),
        query=query,
        n=5,
    )

    return response.search.text

@mcp.tool()
def ingest_documents(local_file_path: str) -> str:
    """
    Ingest documents from a local file into the knowledge base.
    Args:
        local_file_path: The path to the local file containing the documents to ingest.
    Returns:
        str: A message indicating the documents have been ingested.
    """
    file_name = os.path.basename(local_file_path)
    client.ingest(
        documents=[
            Document(
            bucket_id=os.getenv("BUCKET_ID"),
            file_name=file_name,
            file_path=local_file_path,
            file_type="pdf",
            search_data=dict(
                key = "value",
            ),
            )
        ]
    )
    return f"""Ingested {file_name} into the knowledge base. 
               It should be available in a few minutes"""

async def main():
    await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())