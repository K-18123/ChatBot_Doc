import os
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters.character import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredPDFLoader, TextLoader, WebBaseLoader, UnstructuredExcelLoader, UnstructuredPowerPointLoader, Docx2txtLoader
from langchain.agents import load_tools, initialize_agent, AgentType
import langchain_together
from langchain_together import Together

#Load bến môi trường
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

def load_foulder(file_path):
    loader = DirectoryLoader(
        file_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    #loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()
    return documents

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

def setup_vectorstore(documents):# làm sao để lưu vectorstore trên bộ nhớ máy
    embeddings = HuggingFaceEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(
        #separator="/n",
        chunk_size=1000,
        chunk_overlap=200
    )
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore

def load_vectorstore(index_path="faiss_index.index"):
    # Đọc FAISS index từ file
    index = FAISS.read_index(index_path)
    vectorstore = FAISS(embeddings=HuggingFaceEmbeddings(), index=index)
    return vectorstore

def initialize_agents():
    #together_client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    #llm = langchain_together.ChatTogether(client=together_client, model="llama-3.1-70b-versatile", temperature=0)
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
    tools = load_tools(["llm-math", "wikipedia", "serpapi"], llm=llm)
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True
    )
    return agent

def create_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0
    )
    #together_client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    #llm = langchain_together.ChatTogether(client=together_client, model="llama-3.1-70b-versatile", temperature=0)


    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        memory=memory,
        verbose=True
    )
    return chain


def router(user_input, agent):
    # Check if the input is related to calculation or math
    if "calculate" in user_input.lower() or "math" in user_input.lower():
        return agent.run(user_input)

    # Check if the input is a Wikipedia search
    elif "wikipedia" in user_input.lower() or "define" in user_input.lower():
        return agent.run(user_input)

    elif "search" in user_input.lower() or "google" in user_input.lower() or "what about" in user_input.lower():
        return agent.run(user_input)  # SerpAPI will handle this query

        # Fallback to conversation chain if no specific tool matched
    else:
        return st.session_state.conversation_chain({"question": user_input})["answer"]


st.set_page_config(
    page_title="ChatWithDocument",
    layout="centered",
    page_icon="🦾",
)

st.markdown("<h1 style='text-align: center; '>Chat With Document - LLAMA 3.1</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: grey;'>🤖</h1>", unsafe_allow_html=True)



#xay dung chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []




uploaded_files = st.file_uploader(
    label="Upload your pdf files",
    type=["pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx"],
    accept_multiple_files=True
)
# Xử lý nếu có tệp được tải lên
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        file_path = f"{working_dir}/upload/{uploaded_file.name}"

        if os.path.exists(file_path):#xóa tệp nếu nó tồn tại
            os.remove(file_path)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Đọc nội dung của từng tệp PDF và thêm vào danh sách documents
        documents.extend(load_document(file_path))

    # Thiết lập vectorstore từ tất cả tài liệu đã tải lên nếu chưa thiết lập
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = setup_vectorstore(documents)

    # Thiết lập conversation chain nếu chưa thiết lập
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)
    if "agent" not in st.session_state:
        st.session_state.agent = initialize_agents()


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Ask Llama...")


if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)


    with st.chat_message("assistant"):
        # response = st.session_state.conversation_chain({"question": user_input})
        # assistant_response = response["answer"]
        assistant_response = router(user_input, st.session_state.agent)
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})