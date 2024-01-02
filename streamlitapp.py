import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from HTMLTemplate import css, user_template, bot_template
import os
from dotenv import load_dotenv

embedding_model_name = os.environ.get('EMBEDDING_MODEL_NAME')

urls = ['https://www.mosaicml.com/blog/mpt-7b',
        'https://stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models',
        'https://lmsys.org/blog/2023-03-30-vicuna/']


def load_urls(urls):
    loaders = UnstructuredURLLoader(urls=urls)
    data = loaders.load()
    return data


def get_text_chunks(data):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    text_chunks = text_splitter.split_documents(data)
    return text_chunks


def get_vector_store(text_chunks):
    # embeddings=OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
    # llm=ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(),
                                                               memory=memory)
    return conversation_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chatbot for our Own Website", page_icon=":chatbot:")
    st.write(css, unsafe_allow_html=True)
    st.header("Chatbot For Your Own Website 💬")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    user_question = st.text_input("Please ask any question")
    if user_question:
        handle_user_input(user_question)
    with st.sidebar:
        st.title("LLM Chatapp using LangChain")
        st.markdown('''
        This app is an LLM powered Chatbot built using:
        - [Streamlit](https://streamlit.io/) 
        - [OpenAI](https://platform.openai.com/docs/models) LLM
        - [Lang Chain](https://python.langchain.com/)
        ''')

        if st.button("Start"):
            with st.spinner("Processing"):
                # Load the Data
                data = load_urls(urls)
                # Split the Text into Chunks
                text_chunks = get_text_chunks(data)
                print(len(text_chunks))
                # Create a Vector Store
                vectorstore = get_vector_store(text_chunks)
                # Create a Conversation Chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

                st.success("Completed")


if __name__ == '__main__':
    main()
