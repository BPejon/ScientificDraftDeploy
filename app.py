import streamlit as st

import ollama
import database
from sidebar import sidebar


SYSTEM_PROMPT :str= """
You are a Professor on Materials Science.
Your task is to write the outline of a review paper on the topic given within CONTEXT.

text embeddings will be passed as "Context:"
user intent will be passed as "Goal:"

You must:
1. Thoroughly analyze the content provided by the text embeddings, identifying key information relevant to the user's intent and goal.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the user intent, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the content provided.
5. If there is not sufficient information to fully satisfy the user goal, state this clearly in your response.


Important: Base your entire response solely on the information in the documents provided. Do not include any external knowledge or assumptions not present in the given texts.
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

def make_one_draft(context: str, prompt:str):
    messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Goal: {prompt}" ,
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

def refine_full_article(all_sections: list[str], user_prompt: str,context_chunks: list[str]):
    full_text = "\n\n".join(all_sections)

    # Monte os C1..Cn (ex.: strings vindas de most_similar_docs["documents"][0])
    numbered_ctx = []
    for i, ch in enumerate(context_chunks, start=1):
        numbered_ctx.append(f"C{i}: {ch}")
    context_block = "\n".join(numbered_ctx)

    refine_prompt = f"""
        You are a senior scientific editor in Materials Science. You will receive:
        (1) a DRAFT review article, and
        (2) a CONTEXT = list of retrieved passages (C1...Cn) from PDF sources.

        Your job is to produce a SINGLE, coherent, fact-checked review article that uses ONLY the information supported by the CONTEXT.

        STRICT RULES
        - Rely strictly on CONTEXT. If a claim is not clearly supported by any Ci, either remove it or rewrite it as uncertainty and tag it [UNSUPPORTED].
        - Do not invent numbers, mechanisms, colors, or applications. Do not rely on  outside knowledge.
        - Enforce consistent terminology and spelling exactly as they appear MOST FREQUENTLY in CONTEXT (resolve variants; choose one canonical form and use it throughout).
        - Keep units, symbols, and acronyms consistent; define them once when first used.
        - Prefer precise, non-generic statements. Avoid marketing language.
        - Maintain logical section flow (Introduction → thematic sections → Conclusion). Merge overlaps and eliminate redundancy.
        - Do NOT include your reasoning or chain of thought.

        OUTPUT FORMAT (exactly):
        1) Revised Article
        <<final, polished article; add in-text source tags like [C3] where claims are supported>>

        2) Consistency & Terminology Fixes
        - Canonical terms enforced: ...
        - Units/style rules applied: ...

        3) Unsupported or Removed Claims
        - <short claim> — reason; no support in Ci → [UNSUPPORTED]

        INPUT
        THEME: {user_prompt}
        DRAFT:
        {full_text}

        CONTEXT (C1...Cn):
        {context_block}
    """

    response = ollama.chat(
        model=st.session_state.llm_model,
        stream=False,
        messages=[
            {"role": "system", "content": "You are a scientific editor specialized in materials science."},
            {"role": "user", "content": refine_prompt},
        ],
    )
    return response["message"]["content"]

def generate_text_llm_no_stream(context: str, prompt:str, system_prompt:str = SYSTEM_PROMPT):
    messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Goal: {prompt}" ,
            },
        ]

    response = ollama.chat(
        model= st.session_state.llm_model,
        stream = False,
        messages = messages
    )
    full_response = response['message']['content']
    # Extrai o conteúdo <think> se existir
    # Extrai TODO o conteúdo <think> (mesmo que mal formado ou múltiplos)
    think_contents = []
    while "<think>" in full_response:
        start = full_response.find("<think>") + len("<think>")
        end = full_response.find("</think>", start) if "</think>" in full_response[start:] else len(full_response)
        
        think_content = full_response[start:end].strip()
        think_contents.append(think_content)
        
        # Remove o bloco atual (com ou sem fechamento)
        full_response = full_response[:start-len("<think>")] + full_response[end+(len("</think>") if "</think>" in full_response[start:] else 0):]
    
    # Armazena todos os conteúdos <think> encontrados
    st.session_state.think_content = "\n\n---\n\n".join(think_contents) if think_contents else ""
    
    # Remove quaisquer fragmentos remanescentes
    full_response = full_response.replace("<think>", "").replace("</think>", "").strip()
   
    return full_response


def get_most_similar_docs(prompt:str, n_chunks: int= 10, max_chuks_per_docs:int = 15):
    excluded_docs= [
        doc_name for doc_name in database.get_document_names()
        if f"toggle_{doc_name}" in st.session_state and not st.session_state[f"toggle_{doc_name}"]
    ]

    most_similar_docs = database.query_collection(prompt,n_chunks,excluded_docs, max_chuks_per_docs)
    
    return most_similar_docs

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


        
            with st.expander("See retrieved documents"):
                st.write(most_similar_docs)
            with st.expander("See Prompt sent to LLM"):
                messages = [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": f"Context: {most_similar_docs['documents']}, Goal: {prompt}" ,
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

def polish_article(unified_text: str):
    polish_prompt = f"""
        You are a senior scientific editor in Materials Science. 
        You will receive a draft of a review article that may still contain:
        - incorrect definitions,
        - imprecise terminology,
        - factual errors,
        - awkward phrasing or grammar.

        Your task:
        1. Correct scientific inaccuracies in definitions and descriptions, as well as factual errors. 
        - Example: Langmuir–Blodgett (LB) films are molecular monolayers/multilayers transferred from a Langmuir trough, not generic thin films of atoms.
        2. Standardize terminology and ensure internal consistency.
        3. Fix grammar, style, and readability while maintaining an academic tone.
        4. Do not add new content beyond clarification and correction.
        5. If you must remove speculative or unsupported claims, do so silently, leaving only scientifically accurate content.

        DRAFT TO CORRECT:
        {unified_text}
    """

    response = ollama.chat(
        model="llama3.2:3b",
        stream=False,
        messages=[
            {"role": "system", "content": "You are a scientific editor specialized in materials science."},
            {"role": "user", "content": polish_prompt},
        ],
    )
    return response["message"]["content"]


def generate_sections(user_prompt:str):
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # garantir lista limpa
    st.session_state.sections_drafts = []

    section_prompt= f"""
        You are a materials science researcher preparing the **outline of a review article** on a user selected theme.

        Topic {user_prompt}

        Instructions:
        - Create up to 12  logical and progressive sections for a literature review on this topic.
        - Use standard scientific review article structure: Introduction, thematic sections (3–6), and Conclusion.
        - Section titles must be short (up to  10 words), precise, and non-overlapping.
        - Do not include explanations or extra text. Return only the list of sections in the format:

        1 - Introduction
        2 - Section title
        ...
        8 - Conclusion
    """
    most_similar_docs = get_most_similar_docs(section_prompt)

    sections_response = generate_text_llm_no_stream(most_similar_docs["documents"], section_prompt, "Do not put the <think> on the result text ") #Vazio pro system prompt, pq quero que o prompt acima se sobressaia
    
    print(f"Sections antes de filtrar {sections_response}")
    # Extrai as seções da resposta
    sections = [line.strip() for line in sections_response.split('\n') if line.strip()]
    
    print(f"Sections {sections}")
    
    with st.chat_message("user"):
            st.markdown(user_prompt)
    print(f"Sections antes de entrar : {sections}")

   
    # Gerar os drafts de todas as seções e salvar
    for section_theme in sections:
        draft_prompt = f"""
            You are writing a **single section** of a scientific review article.

        	Article theme: {user_prompt}  
        	Section: {section_theme}

        	Requirements:
        	- Write a well-structured, coherent section with proper scientific tone.
        	- Base the text strictly on the provided context. If the context lacks information, clearly state the limitation instead of inventing content.
        	- Maintain logical consistency with typical scientific review structure.
        	- Use precise terminology (avoid vague or generic claims).
        	- Maximum length: 250 words.
        	- Do NOT repeat information from Introduction or Conclusion unless necessary for clarity.
"""
        most_similar_docs_section_theme = get_most_similar_docs(draft_prompt, 10, 5)

        draft_response = generate_text_llm_no_stream(
            most_similar_docs_section_theme["documents"], draft_prompt
        )

        # salvar draft da seção
        st.session_state.sections_drafts.append(f"{section_theme}\n{draft_response}")

    final_article = refine_full_article(st.session_state.sections_drafts, user_prompt, most_similar_docs ) #O que estou passando em most_similar docs?

    polish_text = polish_article(final_article)

    with st.chat_message("assistant"):
        st.markdown(polish_text)
    st.session_state.messages.append({"role": "assistant", "content": polish_text})
            
def show_chat_interface():
    st.header("RAG Question Answer")

    if "messages" not in st.session_state:
        st.session_state.messages=[]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.first_interaction == True:
        st.session_state.first_interaction = False


    with st.spinner("Generating sections and first draft..."):
        generate_sections(st.session_state.research_topic)

        #generate_chat(initial_prompt)


    if prompt := st.chat_input("Ask a question related to your document"):
        generate_chat(prompt)

   

def show_welcome_screen():
    st.header("Welcome to RAG Question Answer")
    st.markdown(""" 
        #### Explore the literature to identify the important topics and relevant contents for a review paper on your subject.
        
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
        "Enter your research theme:",
        placeholder = "e.g Advanced Materials for Solar Cells",
        help = "This will be used to customize your scientific draft"

    )
    
    llm_model = st.radio(
        "Choose one Large Language Model to generate the Scientific Draft.",
        ["llama3.2:3b","deepseek-r1"],
        index= 1,
    )

    generate_button = st.button("Generate", disabled = not bool(document_names), help ="Upload documents to generate draft" if not document_names else "Click to generate", key = "generate_button")

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