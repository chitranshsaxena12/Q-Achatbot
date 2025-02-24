## **Demonstrating Content Ingestion and Retrieval for Answer Generation**

This code allows users to **ingest** content from web pages and then **retrieve relevant information** using **vector embeddings** and **semantic search** to generate answers using the **Gemini AI model**. Below is a breakdown of how content ingestion and retrieval work in this implementation.

---

### **📌 Step 1: Content Ingestion**
Content ingestion involves extracting and storing text data for later retrieval.


#### **🔹 1.1 Extracting Text from Web Pages**
- If a user provides a web link, the function `fetch_data_from_web()` scrapes text from the webpage using `BeautifulSoup`.
- It removes unnecessary script and style elements before returning the cleaned-up text.

📌 **Implementation:**
```python
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
```

---

#### **🔹 1.2 Splitting Text into Chunks**
- Large text documents are **split into smaller chunks** to improve search efficiency.
- The `RecursiveCharacterTextSplitter` from **LangChain** ensures that text chunks maintain semantic meaning.

📌 **Implementation:**
```python
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
```

---

#### **🔹 1.3 Storing Text Embeddings in FAISS**
- The `GoogleGenerativeAIEmbeddings` model converts text chunks into **vector embeddings**.
- The embeddings are stored in **FAISS**, a vector search engine optimized for **fast similarity search**.

📌 **Implementation:**
```python
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
```

---

### **📌 Step 2: Information Retrieval**
Once the content is stored, the system can **search and retrieve** relevant information based on a user's query.

#### **🔹 2.1 Searching for Relevant Context**
- The `process_query()` function searches for relevant **text chunks** in FAISS.
- It retrieves **the top 4 most similar** chunks based on the user's query.

📌 **Implementation:**
```python
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
        # Load stored embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings)

        # Retrieve relevant documents
        docs = vector_store.similarity_search(user_question, k=4)

        # Combine retrieved content
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

        # Generate response using Gemini AI
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        response = model.invoke(prompt)
        extracted_text = response
        print(extracted_text.content)
        return extracted_text.content

    except Exception as e:
        return f"An error occurred: {str(e)}"
```

---

### **📌 Step 3: User Interaction via Streamlit**
The **Streamlit UI** allows users to:
- Enter a webpage URL 🌐
- Ask questions ❓

📌 **Implementation:**
```python
def main():
    """
    Main function for Streamlit application to chat with documents using Gemini AI.
    
    Provides options to fetch text from a webpage, process user queries,
    and generate AI-based responses.
    """
    st.set_page_config("Chat with Documents")
    st.header("Chat with WebContent using Gemini💁")

    with st.sidebar:
        st.title("Menu:")
        web_url = st.text_input("Insert web link:")

        if st.button("Process"):
            with st.spinner("Processing..."):
                try:
                    if web_url:
                        raw_text = fetch_data_from_web(web_url)
                        if raw_text:
                            text_chunks = get_text_chunks(raw_text)
                            get_vector_store(text_chunks)
                    st.success("Done")
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")

    user_question = st.text_input("Ask a Question:")
    if user_question:
        with st.spinner("Searching for answer..."):
            response = process_query(user_question)
            st.write("Reply:", response)

if __name__ == "__main__":
    main()
```

---

### **📌 Full Pipeline:**
1. **User enters a URL.**
2. **Extracted text** is split into **chunks**.
3. **Text embeddings** are generated and stored in **FAISS**.
4. **User submits a query**.
5. **FAISS retrieves relevant text chunks**.
6. **Gemini AI generates a response** based on retrieved information.
7. **Response is displayed in Streamlit**.

---

### **Summary**
✅ **Content ingestion** is done via web scraping.  
✅ **Text is split into chunks** and **stored using FAISS embeddings**.  
✅ **Retrieval** is performed via **semantic search** on FAISS.  
✅ **Answers are generated** using **Google's Gemini AI model**.  
✅ **User interaction** is facilitated via a **Streamlit web app**.  
