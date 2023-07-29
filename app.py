import os
import tempfile
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain

import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


st.title("Personal Assistant Chatbot")
st.subheader("Upload your pdf and ask questions")
st.write("---")


# TODO: load to langchain directly without saving temp file
docs = []
uploaded_file = st.file_uploader("Choose a file", type=['pdf'])
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())
    # st.write("Full path of the uploaded file:", temp_file_path)

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

# split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150)
docs_split = text_splitter.split_documents(docs)

# define embedding
embeddings = OpenAIEmbeddings()
# create vector database from data
vector_db = Chroma.from_documents(docs_split, embeddings)

# define retriever
retriever = vector_db.as_retriever(
    search_type="similarity", search_kwargs={"k": 2})
# create a chatbot chain. Memory is managed externally.
qa = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

st.write("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Enter your question here"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    response = qa(
        {"question": prompt, "chat_history": st.session_state.chat_history})
    # print(response)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response['answer'])

    # Add user message to messages
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Add assistant response to messages
    st.session_state.messages.append(
        {"role": "assistant", "content": response['answer']})
    # Add chat history to chat_history
    st.session_state.chat_history.append((prompt, response['answer']))
