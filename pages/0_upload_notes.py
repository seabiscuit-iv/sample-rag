# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import numpy as np

import streamlit as st
from streamlit.hello.utils import show_code



import os
import streamlit as st
from streamlit.logger import get_logger
import PyPDF2
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import hashlib
from openai import OpenAI

LOGGER = get_logger(__name__)

PINECONE_API_KEY=os.environ['PINECONE_API_KEY']
PINECONE_API_ENV=os.environ['PINECONE_API_ENV']
PINECONE_INDEX_NAME=os.environ['PINECONE_INDEX_NAME']

client=OpenAI(api_key=os.environ['OPENAI_API_KEY'])

def embed(text,filename):
    # pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    pinecone = Pinecone(api_key = PINECONE_API_KEY)
    index = pinecone.Index(PINECONE_INDEX_NAME)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap  = 200,length_function = len,is_separator_regex = False)
    docs=text_splitter.create_documents([text])
    for idx,d in enumerate(docs):
        hash=hashlib.md5(d.page_content.encode('utf-8')).hexdigest()
        embedding=client.embeddings.create(model="text-embedding-ada-002", input=d.page_content).data[0].embedding
        metadata={"hash":hash,"text":d.page_content,"index":idx,"model":"text-embedding-ada-002","docname":filename}
        index.upsert([(hash,embedding,metadata)])
    return

def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

#
# Direcly access Text Input    
#
st.markdown("Upload Notes directly")
filename = st.text_input("Note Name")
uploaded_text = st.text_area("Enter Note","")
if st.button('Process and Upload Notes'):
    embedding = embed(uploaded_text, filename)
