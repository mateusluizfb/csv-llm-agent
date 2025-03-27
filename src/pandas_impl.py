import os
import json
import logging
from typing import List, Dict
import pandas as pd

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Setting up OpenAI API key")
os.environ["OPENAI_API_KEY"] = ""

logger.info("Initializing ChatOpenAI model with gpt-3.5-turbo")
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def get_csv_columns(file_paths: List[str]) -> Dict[str, Dict[str, str]]:
    """Read CSV files and return their column names and data types."""
    logger.info(f"Reading columns from CSV files: {file_paths}")
    columns_info = {}
    
    for file_path in file_paths:
        logger.info(f"Reading file: {file_path}")
        df = pd.read_csv(file_path)
        columns_info[file_path] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    return columns_info

def generate_pandas_query(columns_info: Dict[str, Dict[str, str]], user_query: str) -> str:
    """Generate pandas query code based on user question and dataframe columns."""
    logger.info("Generating pandas query from user question")
    
    prompt_template = """
    You are working with a pandas dataframe in Python. The name of the dataframe is df.

    The columns and data types of a dataframe are given below as a Python dictionary with keys showing column names and values showing the data types.
    {columns_info}

    I will ask question, and you will output the Python code using pandas dataframe to answer my question. Do not provide any explanations. Do not respond with anything except the output of the code.

    Question: {user_query}
    Output Code: 
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["columns_info", "user_query"]
    )
    
    formatted_prompt = prompt.format(
        columns_info=json.dumps(columns_info, indent=2),
        user_query=user_query
    )
    
    logger.info("Sending prompt to chat model")
    response = chat_model([HumanMessage(content=formatted_prompt)])
    
    return response.content.strip()

def query_csv_data(file_paths: List[str], user_query: str) -> List[Dict]:
    """Query CSV data based on user question and return results as JSON."""
    logger.info(f"Querying CSV data for: '{user_query}'")
    
    # Get columns info from all CSVs
    columns_info = get_csv_columns(file_paths)
    
    results = []
    
    for file_path, cols_info in columns_info.items():
        logger.info(f"Processing file: {file_path}")
        
        # Generate pandas query
        pandas_query = generate_pandas_query(cols_info, user_query)
        logger.info(f"Generated pandas query: {pandas_query}")
        
        try:
            # Read CSV and execute query
            df = pd.read_csv(file_path)
            result_df = eval(pandas_query, {'df': df})
            
            # Convert results to JSON
            if not result_df.empty:
                results.extend(result_df.to_dict('records'))
        except Exception as e:
            logger.error(f"Error executing query on {file_path}: {str(e)}")
            continue
    
    return results

# Example usage:
logger.info("Starting example usage")
file_paths = ['./csv/customers-100.csv', './csv/people-100.csv']
logger.info("Querying for Tracey's information")
response = query_csv_data(file_paths, "bring me Tracey's data")
logger.info("Displaying final response")
print(json.dumps(response, indent=2))
logger.info("Example completed")
