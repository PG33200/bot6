import os
import streamlit as st
from io import StringIO
import re
import sys
from modules.history import ChatHistory
from modules.layout import Layout
from modules.utils import Utilities
from modules.sidebar import Sidebar

# Recueil de la clé API OpenAI
user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your openAI API key, sk-",
    type="password")

# Téléchargement du fichier texte
uploaded_file = st.sidebar.file_uploader("Upload", type="txt")

if user_api_key and uploaded_file:
    # Lire le contenu du fichier texte
    text_data = uploaded_file.getvalue().decode("utf-8")

    # Ici, vous pourriez diviser `text_data` en segments si nécessaire
    # Par exemple, en utilisant `text_data.split(".")` pour diviser par phrases
    # Pour cet exemple, nous supposerons que vous traitez `text_data` comme un seul document

    # Initialisation des embeddings OpenAI avec la clé API
    embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)

    # Utilisation d'un exemple simplifié pour créer un seul embedding pour tout le texte
    # Dans une application réelle, vous pourriez vouloir créer des embeddings pour des segments individuels
    document_embedding = embeddings.encode([text_data])

    # Initialiser FAISS avec les embeddings (ici simplifié pour un seul document)
    vectors = FAISS()
    vectors.add([document_embedding])

    # Initialisation de la chaîne de récupération conversationnelle
    chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', openai_api_key=user_api_key),
                                                  retriever=vectors.as_retriever())

    # Suite du script pour la gestion de la conversation...


    def conversational_chat(query):
        
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        
        return result["answer"]
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
        
    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
                
#streamlit run tuto_chatbot_csv.py