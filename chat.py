import os

import langchain_together
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader, \
    UnstructuredPowerPointLoader
from langchain.agents import create_react_agent, AgentExecutor
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain import hub
import wikipediaapi
from langchain.tools import Tool
from langchain_together import Together
from transformers import pipeline
import faiss
import pickle
# Load environment variables
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))


def load_document(file_path):
    # Lấy phần mở rộng của tệp
    file_extension = os.path.splitext(file_path)[1].lower()

    # Kiểm tra loại tệp và sử dụng loader phù hợp
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)

    elif file_extension == ".doc" or file_extension == ".docx":
        loader = Docx2txtLoader(file_path)

    elif file_extension == ".xls" or file_extension == ".xlsx":
        loader = UnstructuredExcelLoader(file_path)

    elif file_extension == ".ppt" or file_extension == ".pptx":
        loader = UnstructuredPowerPointLoader(file_path)

    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    # Tải và trả về tài liệu đã tải
    documents = loader.load()
    return documents


def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(doc_chunks, embeddings)

    # Save the vectorstore metadata along with the index
    vectorstore_path = "vectorstore.index"
    faiss.write_index(vectorstore.index, vectorstore_path)

    metadata_path = "vectorstore.metadata"  # Path to save metadata
    with open(metadata_path, "wb") as f:
        pickle.dump((vectorstore.docstore, vectorstore.index_to_docstore_id), f)  # Save metadata

    return vectorstore


def get_wikipedia_summary(topic: str) -> str:
    """
    Retrieve a summary for a given topic from Wikipedia.

    Args:
    - topic (str): The topic to search on Wikipedia.

    Returns:
    - str: A summary of the topic if found, otherwise an error message.
    """
    user_agent = "MyApp/1.0 (contact@example.com)"  # Customize with your app's info
    wiki_wiki = wikipediaapi.Wikipedia(language='en', user_agent=user_agent)  # Set the language and user agent
    page = wiki_wiki.page(topic)

    if page.exists():
        return f"Summary for '{topic}':\n\n{page.summary}"
    else:
        return f"'{topic}' page not found on Wikipedia."


def get_today_date(input: str) -> str:
    import datetime
    today = datetime.date.today()
    return f"\n {today} \n"


def get_summarized_text(query=None):  # Add a default argument
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    if "retrieved_text" not in st.session_state:
        return "No content to summarize. Please upload and search documents first."
    article = st.session_state.retrieved_text
    try:
        summary = summarizer(article, max_length=1000, min_length=30, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        print(f"Error in summarization: {e}")
        return "Error generating summary. Please try again or adjust the text length."


def get_relevant_document(query: str) -> str:
    if "vectorstore" not in st.session_state:
        return "Vectorstore not setup. Please upload documents first."
    vectorstore = st.session_state.vectorstore
    results = vectorstore.similarity_search(query, k=4)
    total_content = "\n\nBelow are the related document's content: \n\n"
    for result in results:
        total_content += result.page_content + "\n"
    st.session_state.retrieved_text = total_content
    return total_content


def create_agent(vectorstore):
    tools = [
        Tool(name="Get Relevant Document", func=get_relevant_document, description="Get relevant document content."),
        Tool(name="Get Today's Date", func=get_today_date, description="Get today's date."),
        Tool(name="Wikipedia Summary", func=get_wikipedia_summary, description="Get a summary from Wikipedia.")
    ]
    tools.append(
        Tool(name="Get Summarized Text", func=get_summarized_text, description="Get summarized text.")
    )
    prompt_react = hub.pull("hwchase17/react")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    # Use a more powerful model if available, adjust based on your needs and quota
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    #llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)

    # together_client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    # llm = langchain_together.ChatTogether(client=together_client, model="llama-3.1-70b-versatile", temperature=0)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    react_agent = create_react_agent(llm, tools=tools, prompt=prompt_react)
    return AgentExecutor(agent=react_agent, tools=tools, memory=memory, handle_parsing_errors=True, verbose=True)


# Streamlit app
st.title("Chat with Documents")  # Updated title

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
    embeddings = HuggingFaceEmbeddings()
    vectorstore_path = "vectorstore.index"
    metadata_path = "vectorstore.metadata"

    if os.path.exists(vectorstore_path) and os.path.exists(metadata_path):
        index = faiss.read_index(vectorstore_path)
        with open(metadata_path, "rb") as f:
            docstore, index_to_docstore_id = pickle.load(f)

        st.session_state.vectorstore = FAISS(embeddings.embed_query, index, docstore,
                                             index_to_docstore_id)  # Corrected!

    else:
        st.session_state.vectorstore = None

# File uploader
uploaded_files = st.file_uploader("Upload your files", type=["pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx"],
                                  accept_multiple_files=True)

if uploaded_files:
    # Create upload directory if it doesn't exist
    upload_dir = os.path.join(working_dir, "upload")
    os.makedirs(upload_dir, exist_ok=True)

    documents = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        documents.extend(load_document(file_path))

    st.session_state.vectorstore = setup_vectorstore(documents)
    st.session_state.agent = create_agent(st.session_state.vectorstore)

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask Llama...")

if user_input and st.session_state.agent:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.agent({"input": user_input})

        # Get the answer from the correct key depending on Langchain version
        assistant_response = response.get("output", response.get("answer"))
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
elif user_input:  # Handle the case when the agent is not yet initialized.
    st.write("Please upload documents first to initialize the agent.")
