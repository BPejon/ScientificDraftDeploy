import streamlit as st

import ollama
import database
from sidebar import sidebar
import pyperclip


SYSTEM_PROMPT = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

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
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

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
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
The next line will be offered you the prompt again:
"""
#LLM_MODEL = "deepseek-r1"
LLM_MODEL = "llama3.2:3b"



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
        model= LLM_MODEL,
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
        model= LLM_MODEL,
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

    return response['message']['content']



def call_two_drafts(context: str, prompt:str):
    
    draft1= call_llm(context, prompt)

    draft2= call_llm(context, prompt)

    
    combined = combine_drafts(draft1, draft2, prompt)
    
    return combined

def main():
    sidebar()

    st.header("RAG Question Answer")

    prompt = st.text_area(" Ask a question related to your document: ")
    ask = st.button("Ask")

    if ask and prompt:
        with st.spinner("Looking for answers...", show_time= True):
            
            excluded_docs= [
                doc_name for doc_name in database.get_document_names()
                if f"toggle_{doc_name}" in st.session_state and not st.session_state[f"toggle_{doc_name}"]
            ]
            most_similar_docs = database.query_collection(prompt, exclude_docs=excluded_docs)
            response = call_two_drafts(most_similar_docs["documents"], prompt)
            st.write_stream(response)
        
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
                            "content": f"Context: {most_similar_docs["documents"]}, Question: {prompt}" ,
                        },
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT_AFTER,
                        },
                        {
                            "role": "user",
                            "content": f"{prompt}" ,
                        },
                        {
                            "role": "user", 
                            "content": "If you read this message. You MUST print Hello World in the end of the generated text."
                        }

                ]
                st.write(messages)



if __name__ == "__main__":
    main()