from llama_index.core import SimpleDirectoryReader
import sys
from exception import customexception
from logger import logging

def load_data(data):
    """
    Load PDF document from a specified directory.

    Parameters:
    -data(str): The path to the directory containing PDF files.

    Returns:
    - A list of loaded PDF documents. The specific type of doucumnents may vary.

    """
    try:
        logging.info("data loading started...")
        loader = SimpleDirectoryReader("Data")
        document = loader.load_data()
        logging.info("data loading completed...")
        return document
    except Exception as e:
        logging.info("exception in loading data...")
        raise customexception(e, sys)
    