import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_text_chunks(text):
    """
    Splits the extracted text into smaller chunks for better processing.

    Args:
        text (str): The full extracted text.

    Returns:
        list: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Generates embeddings from text chunks and stores them in a FAISS index.

    Args:
        text_chunks (list): List of text chunks to be embedded.

    Returns:
        None
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def process_query(user_question):
    """
    Processes a user query by searching for relevant context in a FAISS vector store
    and generating a response using the Gemini AI model.

    Args:
        user_question (str): The user's input question.

    Returns:
        str: AI-generated response based on the retrieved context.
    """
    try:
        # Initialize embeddings and load vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings)

        # Retrieve relevant documents
        docs = vector_store.similarity_search(user_question, k=4)

        # Combine content from retrieved documents
        context = "\n".join([doc.page_content for doc in docs])

        # Create the prompt
        prompt = f"""
        Answer the question as detailed as possible from the provided context. 
        If the answer is not in the provided context, just say "Answer is not available in the context".
        Do not provide incorrect information.

        Context: {context}

        Question: {user_question}

        Answer:
        """

        # Initialize Gemini model and get response
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        response = model.invoke(prompt)
        extracted_text = response
        print(extracted_text.content)
        return extracted_text.content

    except Exception as e:
        return f"An error occurred: {str(e)}"

def fetch_data_from_web(url):
    """
    Fetches and extracts text content from a given webpage URL.

    Args:
        url (str): The URL of the webpage to scrape.

    Returns:
        str: Extracted text content from the webpage.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text and clean it
        text = soup.get_text(separator='\n')
        lines = (line.strip() for line in text.splitlines())
        text = '\n'.join(line for line in lines if line)

        return text
    except Exception as e:
        st.error(f"Error fetching webpage: {str(e)}")
        return None

def main():
    """
    Main function for Streamlit application to chat with documents using Gemini AI.

    Provides options to  fetch text from a webpage, process user queries,
    and generate AI-based responses.
    """
    st.set_page_config("Chat with Documents")
    st.header("Chat with WebContent using Gemini💁")

    # Sidebar for document upload
    with st.sidebar:
        st.title("Menu:")

        # Web URL input
        web_url = st.text_input("Insert web link:")

        if st.button("Process"):
            with st.spinner("Processing..."):
                try:
                    # Process web content if URL is provided
                    if web_url:
                        raw_text = fetch_data_from_web(web_url)
                        if raw_text:
                            text_chunks = get_text_chunks(raw_text)
                            get_vector_store(text_chunks)

                    st.success("Done")
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")

    # Main chat interface
    user_question = st.text_input("Ask a Question:")
    if user_question:
        with st.spinner("Searching for answer..."):
            response = process_query(user_question)
            st.write("Reply:", response)

if __name__ == "__main__":
    main()
