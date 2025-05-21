import streamlit as st

import ollama
import database
from sidebar import sidebar


SYSTEM_PROMPT = """
You are a PDH Professor focused on Material Science papers.
Your task is to write an outline of a review paper on the subject given within CONTEXT.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs and sections.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""
SYSTEM_PROMPT_AFTER = """

You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
The next line will be offered you the prompt again:
""" 

def call_llm(context: str, prompt:str):
    messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}" ,
            },
            {
                "role": "system",
                "content": SYSTEM_PROMPT_AFTER,
            },
                        {
                "role": "user",
                "content": f"Prompt : {prompt}" ,
            },
        ]
    response = ollama.chat(
        model= st.session_state.llm_model,
        stream = False,
        messages = messages
    )

    return response['message']['content']



def combine_drafts(draft1: str, draft2:str, prompt:str):

    
    combine_prompt = f"""
        You are an expert in information synthesis. Your task is to combine two versions of a scientific article structure on the same topic into a single refined version.

        Topic: {prompt}

        Version 1:
        {draft1}

        Version 2:
        {draft2}

        Instructions:

            Carefully analyze both versions

            Identify the strengths of each

            Combine the best parts from each version

            Maintain a logical and cohesive structure

            Produce a single refined version that is better than both individual versions

        Return only the final refined structure, without additional comments.
    """
    messages = [
            {
                "role": "system",
                "content": "You are an assistant specialized in combining and refining scientific article structures.",
            },
            {
                "role": "user",
                "content": combine_prompt ,
            }
        ]
    response = ollama.chat(
        model= st.session_state.llm_model,
        stream = True,
        messages = messages
    )

    #Como está no modo stream, a resposa virá por chunks
    #O último chunk virá com a mensagem "done"
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break


def make_one_draft(context: str, prompt:str):
    messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}" ,
            },
            {
                "role": "system",
                "content": SYSTEM_PROMPT_AFTER,
            },
                        {
                "role": "user",
                "content": f"Prompt : {prompt}" ,
            },
        ]

    response = ollama.chat(
        model= st.session_state.llm_model,
        stream = True,
        messages = messages
    )


    #Como está no modo stream, a resposa virá por chunks
    #O último chunk virá com a mensagem "done"
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break


def call_two_drafts(context: str, prompt:str):
    
    draft1= call_llm(context, prompt)

    draft2= call_llm(context, prompt)

    
    combined = combine_drafts(draft1, draft2, prompt)
    
    return combined


def generate_chat(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
    
        with st.spinner("Looking for answers...", show_time= True):
            
            excluded_docs= [
                doc_name for doc_name in database.get_document_names()
                if f"toggle_{doc_name}" in st.session_state and not st.session_state[f"toggle_{doc_name}"]
            ]
            most_similar_docs = database.query_collection(prompt, exclude_docs=excluded_docs)


            stream = make_one_draft(most_similar_docs["documents"], prompt)
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})


        
            with st.expander("See retrivied documents"):
                st.write(most_similar_docs)
            with st.expander("See Prompt sent to LLM"):
                messages = [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": f"Context: {most_similar_docs['documents']}, Question: {prompt}" ,
                        },
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT_AFTER,
                        },
                        {
                            "role": "user",
                            "content": f"{prompt}" ,
                        },
                ]
                st.write(messages)


def show_chat_interface():
    st.header("RAG Question Answer")


    if "messages" not in st.session_state:
        st.session_state.messages=[]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.first_interaction == True:
        st.session_state.first_interaction = False

        initial_prompt = f"""
Generate an outline of a review paper on the subject {st.session_state.research_topic}.
Use the understanding provided in the PDFs and the chunks presented in the prompt as Context.
I want the review to be comprehensive and also provide details about the methods.
I will later ask you to expand the context of the sections in the outline.
"""

        generate_chat(initial_prompt)


    if prompt := st.chat_input("Ask a question related to your document"):
        generate_chat(prompt)

   

def show_welcome_screen():
    st.header("Welcome to RAG Question Answer")
    st.markdown(""" 
        #### This is an application where you can generate a draft for your scientific paper.
        
        ##### How to use
        1. Upload your PDF documents using the sidebar on the left
        2. Click "Add to Database" button to add the documents into database
        3. Wait for the documents to be processed. You will see a confirmation message
        4. Specify the research topic for the LLM to create a scientific draft
        5. Select your LLM. Choose Llama3.2 if you have a low spec machine. Otherwise, select deepseek for better results.
        6. Click "Generate Draft" to generate your first draft
                
        You can Toggle documents on/off to include/exclude them from searches
""")

    document_names = database.get_document_names()

    research_topic = st.text_input(
        "Enter your research topic:",
        placeholder = "e.g Advanced Materials for Solar Cells",
        help = "This will be used to customize your scientific draft"

    )
    
    llm_model = st.radio(
        "Choose one Large Language Model to generate the Cientific Draft.",
        ["llama3.2:3b","deepseek-r1"],
        index= 0,
    )

    generate_button = st.button("Generate Draft", disabled = not bool(document_names), help ="Upload documents to generate draft" if not document_names else "Click to generate", key = "generate_button")

    if generate_button and research_topic != "": 
        st.session_state.research_topic = research_topic
        st.session_state.llm_model = llm_model
        st.session_state.show_chat = True
        st.rerun()

def main():
    st.set_page_config(page_title="RAG Question Answer", initial_sidebar_state="expanded")

    ##Inicializa as variáveis de sessões
    if "show_chat" not in st.session_state:
        st.session_state.show_chat = False
    if "first_interaction" not in st.session_state:
        st.session_state.first_interaction = True
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "files_processed" not in st.session_state:
        st.session_state.files_processed = False
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = "llama3.2:3b"
    if "research_topic" not in st.session_state:
        st.session_state.research_topic = ""
    sidebar()

    if st.session_state.show_chat == False:
        show_welcome_screen()
    else:
        show_chat_interface()


if __name__ == "__main__":
    main()