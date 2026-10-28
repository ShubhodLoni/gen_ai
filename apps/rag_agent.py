from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st

# data in st session state
if "docs" not in st.session_state:
    st.session_state.docs = False
if "agent" not in st.session_state:
    st.session_state.agent = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def process_document(path):
    # Load the document
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    #split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents=docs)

    #embeddings and vectorDB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = InMemoryVectorStore.from_documents(
        documents=docs, 
        embedding=embeddings
    )


    #create agent - tool, llm, prompt
    llm=ChatGroq(model="openai/gpt-oss-20b")


    @tool
    def retrieve_context(query:str) -> str:
        """Retrieve relevant documents from the vector store based on the query."""
        context=""
        docs = vectorstore.similarity_search(query, k=4)
        for doc in docs:
            context += doc.page_content + "\n\n"
        return context

    System_prompt = """You are a helpful assistant that answers questions based on the context provided. If the answer is not contained within the context, politely respond that you don't know. Always use the 'retrieve_context' tool for questions requiring external knowledge. Do not make up answers. If the context is empty, respond that you don't know."""


    memory = InMemorySaver()


    agent = create_agent(
        model = llm,
        tools = [retrieve_context],
        system_prompt = System_prompt,
        checkpointer = memory
    )

    st.session_state.agent = agent
    st.session_state.docs = True

    config = {"configurable": {"thread_id": "1"}}

#upload ui
if not st.session_state.docs:
    uploaded_file = st.file_uploader(label="Upload a PDF file", type=["pdf"], accept_multiple_files=True)

    if uploaded_file:
        with st.spinner("Processing the uploaded PDF file..."):
            path = "./doc_files/"
            for file in uploaded_file:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())

            process_document(path)
            st.rerun()

#chat UI
if st.session_state.docs and st.session_state.agent:
    for message in st.session_state.messages:
        role = message.get("role")
        content =  message.get("content")
        st.chat_message(role).markdown(content)
    query = st.chat_input("Ask anything related to document uploaded")
    if query:
        st.session_state.messages.append({"role":"user","content":query})
        st.chat_message("user").markdown(query)
        response = st.session_state.agent.invoke(
            {"messages":[{"role":"user", "content":query}]},
            {"configurable":{"thread_id":"1"}}
        )

        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role":"ai","content":query})