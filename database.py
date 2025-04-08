from langchain_core.documents import Document
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import chromadb
from typing import List
import tempfile
import streamlit as st
import shutil

CHROMA_PATH = "./demo-rag-chroma"
COLLECTION_NAME = "rag_app"
def add_to_vector_collection(all_splits:list[Document], filename: str, original_filename:str):
    collection = get_vector_collection()
    documents, metadatas, ids = [],[],[]

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        ids.append(f"{filename}_{idx}")
        metadata = split.metadata
        metadata["document_name"] = original_filename
        metadatas.append(metadata)

    collection.upsert(
        documents= documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success("Data added to the vector store!")


def get_vector_collection() -> chromadb.Collection:
    #Able to use ollama as api embedding function
    ollama_ef = OllamaEmbeddingFunction(
        url = "http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma_client.get_or_create_collection(
        name= COLLECTION_NAME,
        embedding_function= ollama_ef,
        metadata= {"hnsw:space": "cosine"}, #Calculo de similaridade
    )

def process_document(uploaded_file: UploadedFile) -> List[Document]: 
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())

    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    os.unlink(temp_file.name) #Delete temp file

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 400,
        chunk_overlap = 100,
        separators= ["\n\n", "\n", ".", "?", "!", " ", ""],
    )

    return text_splitter.split_documents(docs)

def query_collection(prompt:str, n_results: int = 99999 , exclude_docs:list[str] = None, max_embeddings_per_doc: int = 200):
    collection = get_vector_collection()
    CONTROL_NUMBER = 350

    query_params ={
        "query_texts" :[prompt],
        "n_results": n_results * CONTROL_NUMBER,
    }

    if exclude_docs:
        query_params["where"] = {"document_name": {"$nin": exclude_docs}}

    results = collection.query(**query_params)
    print("--------")
    filtered_documents = []
    filtered_ids = []
    doc_count = {}

    print(f"Results: {results}")

    for doc, metadata,doc_id in zip(results["documents"][0], results["metadatas"][0], results["ids"][0]):
        doc_name = metadata["document_name"]
        if doc_name not in doc_count:
            doc_count[doc_name] = 0

        if doc_count[doc_name] < max_embeddings_per_doc:
            filtered_documents.append(doc)
            filtered_ids.append(doc_id)
            #print("++++++++++++++++++")
            #print(f"filtered_documents: {filtered_documents}/ filtered_ids {filtered_ids}  ")
            #print("++++++++++++++++++")
            doc_count[doc_name] += 1

        if len(filtered_documents) >= n_results:
            break

    results["documents"] = [filtered_documents]
    results["ids"] = [filtered_ids]

    return results

# Ele ta bugando, quando eu apago a base de dados e tento inserir um novo aquivo ele simplesmente n adiciona:
#ValueError: Could not connect to tenant default_tenant. Are you sure it exists?
def reset_database():
    try:
        # chroma_client = get_vector_collection()
        # chroma_client.delete_collection(name= COLLECTION_NAME)

        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
            st.success("DB deleted")
    #Ele apaga a collection, porém os embeddings se mantem na bd. Porém não é utilizada para conteúdo.
    #chroma_client.reset()
    #ValueError: Resetting is not allowed by this configuration (to enable it, set `allow_reset` to `True` in your Settings() or include `ALLOW_RESET=TRUE` in your environment variables)
    except Exception as e:
        st.error(e)
    
def get_document_names() -> list[str]:
    try:
        collection = get_vector_collection()

        results = collection.get(include=["metadatas"])
        metadatas = results["metadatas"]

        #Extract unique doc inside bd
        document_names = set()
        for num, metadata in enumerate(metadatas):
            if metadata and "document_name" in metadata:
                document_names.add(metadata["document_name"])

        return list(document_names)
    except:
        st.write("")

def remove_document_from_db(filename:str):
    try:   
        collection= get_vector_collection()

        collection.delete(where={"document_name": filename})
        return True
    
    except Exception as e:
        st.write(e)
        return False

    