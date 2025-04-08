import streamlit as st

import database


def delete_db_button():
        db_get = st.button("Reset Database")

        if db_get:
            database.reset_database()

def display_list_of_documents():
    st.subheader("Documents available")
    documents_names = database.get_document_names()
    if documents_names:
        for doc_name in documents_names:
            col_doc_name, col_doc_include, col_del_button = st.columns([4,3,1]) #Collums layout
            
            with col_doc_name:
                st.write(doc_name)

            with col_doc_include:
                if f"toggle_{doc_name}" not in st.session_state:
                    st.session_state[f"toggle_{doc_name}"] = True #padrão é 1 -> incluso na busca
                st.checkbox("Included in search", key=f"toggle_{doc_name}")


                # print(f"session: toogle{doc_name}")
                # print(st.session_state[f"toggle_{doc_name}"])
                # print(f"st session: {st.session_state}")

            with col_del_button:
                delete_doc_button = st.button("X", key=f"delete_{doc_name}" )
                if delete_doc_button:
                    with st.spinner(f"Deleting {doc_name}"):
                        is_success = database.remove_document_from_db(doc_name)
                        if is_success:
                            st.toast(f"Document '{doc_name}' deleted successfully!")



def sidebar():


    with st.sidebar:

        st.set_page_config(page_title="RAG Question Answer")
        st.header("Rag Question Answer")
        uploaded_file= st.file_uploader("Upload PDF File for QnA", type=["pdf"], accept_multiple_files=True)

        process = st.button(
            "Process"
        )

        if uploaded_file and process:
            with st.spinner("Inserting the documents in the database...", show_time = True):
                for doc in uploaded_file:
                    normalize_uploaded_file_name = doc.name.translate(
                        str.maketrans({"-":"_", ".": "_", " ":"_"})
                    )
                    all_splits = database.process_document(doc)
                    database.add_to_vector_collection(all_splits, normalize_uploaded_file_name, doc.name)

        display_list_of_documents()

