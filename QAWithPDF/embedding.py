from llama_index.core import VectorStoreIndex
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter


import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

sys.path.append(parent_dir)
sys.path.append(current_dir)

# from QAWithPDF.data_ingestion import load_data
from data_ingestion import load_data
# from QAWithPDF.model_api import load_model
from model_api import load_model
from exception import customexception
from logger import logging

class customexception(Exception):

    def __init__(self, error_message, error_details: sys):
        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info()
        print(exc_tb)

        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
            self.file_name, self.lineno, str(self.error_message)
        )
 
 
def download_gemini_embedding(model, document):
    """
    Downloads and initializes a Gemini Embedding model for vector embeddings.

    Returns:
    - VectorStoreIndex: An index of vector embeddings for efficient similarity queries.
    """
    try:
        logging.info("")
        gemini_embed_model = GeminiEmbedding(model_name='models/embedding-001')
        index = VectorStoreIndex.from_documents(
            document, embed_model=gemini_embed_model, llm=model
        )
        Settings.llm = model
        Settings.embed_model = gemini_embed_model
        Settings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=80)
        index.storage_context.persist()

        logging.info("")
        # query_engine = index.as_query_engine()
        query_engine = index.as_chat_engine()
        return query_engine
    except Exception as e:
        raise customexception(e, sys)