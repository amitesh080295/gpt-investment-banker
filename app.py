import os
from apikey import api_key

import streamlit as st
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

os.environ['OPENAI_API_KEY'] = api_key

llm = OpenAI(temperature=0.1, verbose=True)
loader = PyPDFLoader('annual_report.pdf')
pages = loader.load_and_split()
store = Chroma.from_documents(pages, collection_name='annualreport')

vectorstore_info = VectorStoreInfo(
    name='annual_report',
    description='A banking annual report as PDF',
    vectorstore=store
)

toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

st.title('🦜🔗 GPT Investment Banker')
prompt = st.text_input('Type your prompt here!')

if prompt:
    response = agent_executor.run(prompt)
    st.write(response)

    with st.expander('Document Similarity Search'):
        search = store.similarity_search_with_score(prompt)
        st.write(search[0][0].page_content)