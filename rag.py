#!/usr/bin/env python
# coding: utf-8

# In[9]:


from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
import streamlit as st


# In[ ]:

st.title("ASIMKKA MED BOT")



# In[10]:


from pinecone import Pinecone,ServerlessSpec
load_dotenv()
pc=Pinecone(api_key="68234ea3-63f7-4ebf-99db-9f5fdecca789")


# In[11]:


index_name = "sadat"
if index_name not in pc.list_indexes().names():
  pc.create_index(
    name="sadat",
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
      cloud="aws",
      region="us-east-1"
    ),
    
  )


# In[12]:


def load_pdf(data):
    loader=DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)

    documents=loader.load()
    return documents



# In[13]:


##extracting data
data_ext=load_pdf("data/")


# In[14]:


##splitting all docs
def text_split(data_ext):
    splited_text=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    text_chunks=splited_text.split_documents(data_ext)
    return text_chunks


# In[15]:


text_chunks=text_split(data_ext)


# In[16]:


## download model for Embedding
def hf_embeddings():
    embedding=HuggingFaceEmbeddings(model_name = ("sentence-transformers/all-MiniLM-L6-v2"))
    return embedding


# In[17]:


embeddings=hf_embeddings()


# In[18]:


import os

if index_name not in pc.list_indexes().names():
    os.environ['PINECONE_API_KEY'] = '68234ea3-63f7-4ebf-99db-9f5fdecca789'
    index_name="sadat"

    docsearch=PineconeVectorStore.from_documents(text_chunks,embeddings,index_name=index_name)
os.environ['PINECONE_API_KEY'] = '68234ea3-63f7-4ebf-99db-9f5fdecca789'
docsearch=PineconeVectorStore.from_existing_index(index_name,embeddings)


# In[19]:


docsearch=PineconeVectorStore.from_existing_index(index_name,embeddings)
question="cancer"
docs=docsearch.similarity_search(question,k=3)
print('result: ',docs)


# In[20]:


load_dotenv()
import os
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")


# In[22]:



# Print the response from the model



# In[23]:


from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


# In[24]:


from langchain.chains import RetrievalQA

qa=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={'k':2}),
        return_source_documents=True,
        )


# In[ ]:



user_input=st.text_input("What is your Question")
if user_input:
    result=qa.invoke({"query": user_input})
    st.write("Response:",result["result"])
    print("Response: ", result["result"])
# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




