import os
import gradio as gr
from dotenv import load_dotenv

from langchain_text_splitters.character import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
import warnings
warnings.filterwarnings("ignore")
#Load bến môi trường
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

def load_document(file_path):
    loader = DirectoryLoader(
        file_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    #loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()
    return documents

# def load_document_test(file_path):
#     loader = UnstructuredPowerPointLoader("ppt/Mds2_b4_Matplotlib_Seaborn.pptx")
#     documents = loader.load()
#     return documents
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

def create_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0
    )
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

# Đường dẫn cố định tới dữ liệu
fixed_file_path = "data"

vectorstore, conversation_chain = None, None
if os.path.exists(fixed_file_path):
    documents = load_document(fixed_file_path)
    vectorstore = setup_vectorstore(documents)
    conversation_chain = create_chain(vectorstore)

# Hàm xử lý trò chuyện
def chat_with_llama(user_input):
    global chat_history, conversation_chain
    if conversation_chain:
        chat_history.append({"role": "user", "content": user_input})
        response = conversation_chain({"question": user_input})
        assistant_response = response["answer"]
        chat_history.append({"role": "assistant", "content": assistant_response})
        return [(message["role"], message["content"]) for message in chat_history]
    return [("assistant", "Sorry, the system is not ready.")]

# Tạo giao diện Gradio
iface = gr.Interface(
    fn=chat_with_llama,
    inputs="text",
    outputs="chatbot",
    title="Chat With Document - LLAMA 3.1",
    description="Hệ thống hội thoại với tài liệu sử dụng mô hình LLAMA 3.1."
)

iface.launch()
