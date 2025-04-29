import os
os.environ["STREAMLIT_WATCHDOG_MODE"] = "none"

import torch
if hasattr(torch, 'classes'):
    torch.classes.__path__ = []

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, set_seed
import os
import warnings
import dotenv
 
# Suppress TensorFlow and PyTorch warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["STREAMLIT_WATCHDOG_MODE"] = "none"  # Disable Streamlit file watcher
 
# Apply your Hugging Face API key directly
hugging_face_token = os.getenv("hugging_face_token", "your_default_token_here")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hugging_face_token

 
# Set random seed for reproducibility
set_seed(42)
 
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text
 
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return text_splitter.split_text(text)
 
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    os.makedirs("faiss_index", exist_ok=True)
    vector_store.save_local("faiss_index")
    return vector_store
 
def create_chain():
    # Load a stable and widely supported model
    pipe = pipeline(
        "text-generation",
        model="gpt2",  # Using GPT-2 as a reliable alternative
        tokenizer="gpt2",
        device=-1,  # Use CPU
        pad_token_id=50256,  # Set padding token ID for GPT-2
        max_new_tokens=256  # Set max_new_tokens globally
    )
 
    prompt_template = """
    You are an AI assistant that provides helpful answers based on provided documents.
 
    Context: {context}
    Question: {question}
    Answer:
    """
 
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
 
    llm = HuggingFacePipeline(pipeline=pipe)
 
    def format_docs(docs):
        return "\n".join(doc.page_content for doc in docs)
 
    chain = (
        {"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
 
def user_input(user_question):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question, k=2)
 
        chain = create_chain()
        response = chain.invoke({"docs": docs, "question": user_question})
 
        # Display the response directly
        st.write("Reply:", response.strip())
 
    except RuntimeError as e:
        if "faiss_index" in str(e):
            st.error("Error: Please upload and process PDFs first before asking questions.")
        else:
            st.error(f"Runtime error: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
 
def main():
    st.set_page_config(page_title="AMBIGAPATHY's CHATBOT", page_icon="🤖", layout="wide")
    st.header("AMBIGAPATHY's CHATBOT", divider='rainbow')
 
    with st.sidebar:
        st.title("Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("PDFs processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing PDFs: {str(e)}")
            else:
                st.error("Please upload PDF files first.")
 
    user_question = st.text_input("Ask a Question")
    if user_question:
        user_input(user_question)
 
if __name__ == "__main__":
    main()