import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

##Creating Vectorstore
def get_vectorstore(url):
    
    ##Getting the text in Document Form
    loader = WebBaseLoader(url)
    document = loader.load()
    
    ##Creating Chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    ##Creating Embeddings and Vectorstore
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)    
    vectorstore = Chroma.from_documents(document_chunks, embeddings)
    
    return vectorstore


##Creating Content(Context) Retriever Chain
def get_retrieval_chain(vectorstore):
    
    ##Setting up the Google LLM
    #llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0.2)
    llm = ChatOpenAI()

    ##Setting up the Retriever
    retriever = vectorstore.as_retriever()
    
    ##Creating the Prompt for the Chain
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retrieval_chain = create_history_aware_retriever(llm=llm, retriever=retriever, prompt=prompt)
    
    return retrieval_chain


##Creating the Conversational RAG Chain
def create_conversational_RAG_chain(retriever_chain):
    
    ##Setting up the LLM
    #llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0.2, convert_system_message_to_human=True)
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    ##Creating the stuff documents chain
    stuff_documents_chain = create_stuff_documents_chain(llm = llm, prompt=prompt)
    
    ##Creating the Conversational RAG Chain (Combining both the Chains)
    conversational_rag_chain = create_retrieval_chain(retriever_chain, stuff_documents_chain)
    
    return conversational_rag_chain
    
    
##To get the user response
def get_user_response(user_input, retriever_chain):
    conversational_chain = create_conversational_RAG_chain(retriever_chain)
    
    response = conversational_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response["answer"]


##Creating App's Interface
st.set_page_config(page_title="Chat with HTML", page_icon="🌎")
st.header("Chat with Website")


##Sidebar
with st.sidebar:
    st.header("Page Settings")
    st.subheader("Enter the URL below:")
    webpage_url = st.text_input("Enter your Webpage URL here...")
    

if webpage_url is None or webpage_url == "":
    st.info("Please enter a Valid URL.")
    
    
else:
    ##Creating the Chat History and making it persistant so it does not initialized everytime any changes are made in the Application
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, How can I help you?")
        ]

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = get_vectorstore(webpage_url)
    
    ##Getting the Retriever Chain
    retriever_chain = get_retrieval_chain(st.session_state.vectorstore)
    
    ##User Input
    user_input = st.chat_input("Enter your response here...")

    if user_input is not None and user_input != "":
        response = get_user_response(user_input, retriever_chain)
        
        ##Appending to Chat History
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=response))
        
        
    ##Displaying Chat History in Application
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
            
        elif isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)