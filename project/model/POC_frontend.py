import plotly.graph_objects
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import transformers
import requests
import time
import torch
from huggingface_hub import InferenceClient
from scipy.special import softmax

def get_dataframe():
    # load dataset.
    full_df = pd.read_csv("archive/news-article-categories.csv")
    return full_df

def update_answer():
    client = InferenceClient(
         "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct",
         token="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",
    )
    response = client.text_generation(prompt=st.session_state["text"] + "When was this text generated? (answer only in MM-YYYY format, no dates, only month and year)")
    print(response)
    st.session_state.date = response
    return

def update_textarea():
    if st.session_state["file selector"] is None:
        st.session_state.text = "Select an example text."
        return
    st.session_state["text"] = st.session_state.file_content_list[st.session_state["file selector"]]

def update_summary():
    client = InferenceClient(
        "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
        token="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",
    )
    response = client.summarization(st.session_state["text"])
    print(response["summary_text"])
    st.session_state["summary"] = response["summary_text"]
    return

def update_predictions():
    logits = st.session_state.model(**st.session_state.tokenizer(st.session_state["text"], return_tensors="tf",
                                                                 truncation=True, max_length=300)).logits
    st.session_state.probabilities = softmax(logits)
    st.session_state.prediction = st.session_state.id_label[np.argmax(logits, axis=1)[0]]
    print(st.session_state.probabilities[0])
    return

def update_both():
    update_textarea()
    update_summary()
    update_answer()
    update_predictions()

# main
st.set_page_config(page_title="Disposition Tool", page_icon="wf.png", layout="wide")
if "files" not in st.session_state:
    st.session_state.files = []
if "df" not in st.session_state:
    st.session_state.df = get_dataframe()
if "random_rows" not in st.session_state:
    st.session_state.random_rows = pd.DataFrame()
if "file_content_list" not in st.session_state:
    st.session_state.file_content_list = dict()
if "file selector" not in st.session_state:
    st.session_state["file selector"] = None
if "date" not in st.session_state:
    st.session_state["date"] = "Date Generated Here."
if "model" not in st.session_state:
    st.session_state.model = transformers.TFAutoModelForSequenceClassification.from_pretrained(
        "C:\\Users\\Stubh\\PycharmProjects\\pythonProject1\\model", local_files_only=True
    )
    print(st.session_state.model.summary())
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = transformers.RobertaTokenizerFast.from_pretrained(
        "C:\\Users\\Stubh\\PycharmProjects\\pythonProject1\\tokenizer"
    )
    print(st.session_state.tokenizer)
if "label_id" not in st.session_state:
    st.session_state.label_id = {'ARTS & CULTURE': 0,
     'BUSINESS': 1,
     'COMEDY': 2,
     'CRIME': 3,
     'EDUCATION': 4,
     'ENTERTAINMENT': 5,
     'ENVIRONMENT': 6,
     'MEDIA': 7,
     'POLITICS': 8,
     'RELIGION': 9,
     'SCIENCE': 10,
     'SPORTS': 11,
     'TECH': 12,
     'WOMEN': 13}
if "id_label" not in st.session_state:
    st.session_state.id_label = {v: k for k, v in st.session_state.label_id.items()}
if "probabilites" not in st.session_state:
    st.session_state.probabilities = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
if "labels" not in st.session_state:
    st.session_state.labels = ["ARTS & CULTURE", "BUSINESS", "COMEDY", "CRIME", "EDUCATION", "ENTERTAINMENT", "ENVIRONMENT", "MEDIA", "POLITICS", "RELIGION", "SCIENCE", "SPORTS", "TECH", "WOMEN"]
if "prediction" not in st.session_state:
    st.session_state.prediction = "Prediction Here."
col1, col2 = st.columns([0.2,0.7], gap="medium", border=True)
st.sidebar.image("wf.png", width=150)
st.sidebar.title("**Disposition Tool** - _POC_")
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
if st.sidebar.button("Load a new example batch of 10 files."):
    st.session_state.random_rows = st.session_state.df.sample(n=10)
    st.session_state["files"] = st.session_state.random_rows['title'].to_list() # Replace with your list of files
    for index, row in st.session_state.random_rows.iterrows():
        st.session_state.file_content_list[row['title']] = row['body']
    st.session_state["file selector"] = st.session_state.files[0]
    update_both()
with col1:
    st.write("### *Metrics*")
    st.write("##### **Month Generated:**")
    st.text_area("(As generated with text as input, *MM-YYYY* format)", key="date", value="Date Generated Here.", height=70)
    st.write("##### Confidence Scores:")
    fig = plotly.graph_objects.Figure(plotly.graph_objects.Bar(
    x=st.session_state.probabilities[0],
    y=st.session_state.labels,
    orientation='h',
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig, height=400)
    st.text_area("Predicted Class:", key="prediction", value="Prediction Here.", height=70)
with col2:
    selected_file = st.selectbox("Select an example text.", st.session_state["files"], key="file selector",
                                 on_change=update_both)
    t_area = st.text_area("File Content", key="text", value="File Contents Here.", height=500)
    t_area_summary = st.text_area("Summary", key="summary", value="Summary Here.", height=100)