import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
import json
import random
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: black;
        }
        .stTextInput input {
            background-color:black;
            border-radius: 20px;
            padding: 12px;
        }
        .stButton>button {
            background-color: black;
            color: white;
            border-radius: 20px;
            padding: 10px 24px;
        }
        .chat-bubble {
            padding: 15px 20px;
            border-radius: 25px;
            margin: 10px 0;
            max-width: 80%;
        }
        .user-bubble {
            background-color: #E3F2FD;
            margin-left: auto;
            color: black;
        }
        .bot-bubble {
            background-color: #7d7979;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize core components in session state
if 'core_loaded' not in st.session_state:
    st.session_state.core_loaded = False
    st.session_state.tokenizer = None
    st.session_state.model = None
    st.session_state.bert_model = None
    st.session_state.intents_dict = None
    st.session_state.classes = None

# Loading Pre-trained Models and Assets
@st.cache_resource
def load_core_components():
    try:
        # Load BERT components
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        
        # Load chatbot model
        model = load_model('./model/conversationalchatbotmodel.h5')
        
        # Load and validate intents
        with open("./Data/intents.json", "r") as f:
            intents_dict = json.load(f)
            if 'intents' not in intents_dict:
                raise ValueError("JSON file missing 'intents' key")
            for intent in intents_dict['intents']:
                if 'tag' not in intent:
                    if 'tags' in intent and isinstance(intent['tags'], list) and intent['tags']:
                    
                        intent['tag'] = intent['tags'][0]
                else:
                    raise KeyError(f"Intent missing both 'tag' and valid 'tags' key: {intent}")

        
        # Load classes
        with open('./Data/classes.pkl', 'rb') as f:
            classes = pickle.load(f)

        return tokenizer, bert_model, model, intents_dict, classes
    
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        st.stop()

# Load components once at startup if not already loaded
if not st.session_state.core_loaded:
    try:
        (st.session_state.tokenizer,
         st.session_state.bert_model,
         st.session_state.model,
         st.session_state.intents_dict,
         st.session_state.classes) = load_core_components()
        
        st.session_state.core_loaded = True
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        st.stop()

# Helper Functions for Inference
def get_bert_embedding(sentence):
    try:
        inputs = st.session_state.tokenizer(
            sentence, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=512
        )
        with torch.no_grad():
            outputs = st.session_state.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()
    except Exception as e:
        st.error(f"Embedding error: {str(e)}")
        return np.zeros((1, 768))

def predict_class(sentence):
    try:
        embedding = get_bert_embedding(sentence)
        res = st.session_state.model.predict(embedding, verbose=0)[0]
        ERROR_THRESHOLD = 0.35
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return [{'intent': st.session_state.classes[r[0]], 
                 'probability': float(r[1])} for r in results] if results else []
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return []

def get_response(intents_list):
    try:
        if not intents_list:
            return "I'm here to listen. Could you tell me more about how you're feeling?"
        
        tag = intents_list[0]['intent']
        for intent in st.session_state.intents_dict['intents']:
            if tag == intent['tag']:
                return random.choice(intent['responses'])
        return "I want to understand better. Could you rephrase that?"
    
    except Exception as e:
        st.error(f"Response error: {str(e)}")
        return "Something went wrong. Please try again."

# Streamlit Interface Components
st.title("🫂  Mindful Companion")
st.markdown("""
    *Your safe space for mental health support*  
    I'm here to listen, not judge. Feel free to share what's on your mind.
""")

# Sidebar with Resources
with st.sidebar:
    st.header("💡 Mental Health Resources")
    st.markdown("""
        - **Crisis Hotline**: 988 Suicide & Crisis Lifeline at 988
        - [Find a Therapist](https://www.psychologytoday.com)
        - [Mental Health Exercises](https://www.mindful.org)
        - [Self-Care Tips](https://www.helpguide.org)
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/1057/1057231.png", width=50)

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f'<div class="chat-bubble {message["role"]}-bubble">{message["content"]}</div>', 
                   unsafe_allow_html=True)

# User input
if prompt := st.chat_input("Type your message here to chat..."):
    # Add user message to chat history 
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat
    with st.chat_message("user"):
        st.markdown(f'<div class="chat-bubble user-bubble">{prompt}</div>', 
                   unsafe_allow_html=True)
    
    # Get response from model
    predicted_intents = predict_class(prompt)
    response = get_response(predicted_intents)
    
    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Display bot response in chat
    with st.chat_message("assistant"):
        st.markdown(f'<div class="chat-bubble bot-bubble">{response}</div>', 
                   unsafe_allow_html=True)

## Update the Quick Action Buttons section with this code:

# Quick Action Buttons
def handle_quick_action(prompt):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate and add bot response
    predicted_intents = predict_class(prompt)
    response = get_response(predicted_intents)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Force a rerun to show updates
    st.experimental_rerun()

st.markdown("### 💬 Quick Start Topics")
col1, col2, col3 = st.columns(3)
with col1:
    st.button("Feeling Anxious 😰", 
              on_click=handle_quick_action, 
              args=("I'm feeling anxious",))
with col2:
    st.button("Sad Mood 😢", 
              on_click=handle_quick_action, 
              args=("I've been feeling sad",))
with col3:
    st.button("Stress Relief 🧘", 
              on_click=handle_quick_action, 
              args=("How can I manage stress?",))