import os
import json
import logging
from typing import List, Dict

from langchain_community.document_loaders import CSVLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.messages import HumanMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Setting up OpenAI API key")
os.environ["OPENAI_API_KEY"] = ""

logger.info("Initializing OpenAI embeddings")
embeddings = OpenAIEmbeddings()

logger.info("Creating in-memory vector store")
vector_store = InMemoryVectorStore(embeddings)

logger.info("Initializing ChatOpenAI model with gpt-3.5-turbo")
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def load_csvs_to_db(file_paths: List[str]) -> None:
    """Load multiple CSV files into the PostgreSQL database with their embeddings."""
    logger.info(f"Starting to load CSV files: {file_paths}")
    for file_path in file_paths:
        logger.info(f"Loading file: {file_path}")
        loader = CSVLoader(file_path=file_path)
        documents = loader.load()
        logger.info(f"Adding {len(documents)} documents to vector store")
        vector_store.add_documents(documents)
    logger.info("Finished loading all CSV files")

def query_similar_documents(user_query: str, top_n: int = 5) -> List[Dict[str, str]]:
    """Retrieve documents similar to the user's query and return structured JSON."""
    logger.info(f"Performing similarity search for query: '{user_query}'")
    results = vector_store.similarity_search(user_query, k=top_n)
    
    logger.debug(f"Raw similarity search results: {results}")

    logger.info("Preparing context from search results")
    context = "\n\n".join([f"Document {i+1}:\n{result.page_content}" for i, result in enumerate(results)])
    
    logger.info("Setting up response schemas and output parser")
    response_schemas = [
        ResponseSchema(name="document_number", description="The number of the document."),
        ResponseSchema(name="content_summary", description="A brief summary of the document content.")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    
    logger.info("Creating prompt template")
    prompt_template = PromptTemplate(
        template="Based on the following documents:\n{context}\n\nPlease provide a summary of each document with its corresponding number in JSON format.\n{format_instructions}",
        input_variables=["context"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )
    
    logger.info("Formatting prompt with context")
    prompt = prompt_template.format(context=context)
    
    logger.info("Sending prompt to chat model")
    response = chat_model([HumanMessage(content=prompt)])
    
    logger.info("Parsing model response")
    structured_response = output_parser.parse(response.content)
    
    logger.info("Returning structured response")
    return structured_response

# Example usage:
logger.info("Starting example usage")
load_csvs_to_db(['./csv/customers-100.csv', './csv/people-100.csv'])
logger.info("Querying for Tracey's information")
response = query_similar_documents("Bring me Tracey's information.")
logger.info("Displaying final response")
print(json.dumps(response, indent=2))
logger.info("Example completed")
