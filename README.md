# Mindful Companion - Mental Health Conversational Chatbot

Here is a hosted link to the Streamlit app: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mental-health-conversational-chatbot.streamlit.app/) 

A therapeutic chatbot designed to provide emotional support and mental health resources using BERT-based NLP.

![Chatbot Interface Demo](./assets/demo.gif) 

## Features

- 🤖 BERT-powered intent recognition
- 🧠 Mental health FAQ knowledge base
- 🚨 Crisis resource integration
- 💬 Conversational interface with Streamlit
- 🎯 Quick-access topics for common issues
- 📚 Mental health education content

## Technologies Used

- **NLP:** Hugging Face Transformers (BERT)
- **ML Framework:** TensorFlow/Keras
- **Web Interface:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Backend:** Python 3.10+

## Installation

### Prerequisites
- Python 3.10
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/charite-uwatwembi/Mental-Health-Conversational-ChatBot
cd Mental-Health-Conversational-ChatBot
```

2. Install dependencies:

```bash

pip install -r requirements.txt
```

3. Download and install BERT model components:

```bash
python -c "from transformers import BertTokenizer, BertModel; BertTokenizer.from_pretrained('bert-base-uncased'); BertModel.from_pretrained('bert-base-uncased')"
```

## Usage
Start the chatbot:

```bash
streamlit run app.py
The application will launch in your default browser at http://localhost:8501
```