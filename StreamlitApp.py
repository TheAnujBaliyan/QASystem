import streamlit as st
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
child_dir = os.path.abspath(os.path.join("QAWithPDF", current_dir))

sys.path.append(child_dir)


from QAWithPDF.data_ingestion import load_data
from QAWithPDF.embedding import download_gemini_embedding
from QAWithPDF.model_api import load_model

def save_uploaded_file(uploaded_file):
    temp_dir = "Data"

    temp_file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return temp_file_path

def remove_file(file_path):
    # Delete the file if it exists
    if os.path.exists(file_path):
        os.remove(file_path)

def main():
    st.set_page_config("QA with Documents")

    doc = st.file_uploader("Upload Your Documents")

    st.header("QA with Documents(Information Retrival)")

    user_question = st.text_input('Ask your Question')

    if st.button("submit & process"):
        with st.spinner("Processing..."):
            # print(doc)
            temp_file_path = save_uploaded_file(doc)
            document = load_data("")
            remove_file(temp_file_path)
            
            model = load_model()
            query_engine = download_gemini_embedding(model, document)

            response = query_engine.query(user_question)

            st.write(response.response)

    
if __name__=="__main__":
    main()
