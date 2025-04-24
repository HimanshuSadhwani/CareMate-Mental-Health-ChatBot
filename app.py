from transformers import BertTokenizer, BertModel
import torch
from flask import Flask, render_template, request
import os
import json
import requests
import pickle
import joblib
import numpy as np
import pandas as pd
import nltk 
import string 
import re
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from textblob.sentiments import *
from nltk.stem.wordnet import WordNetLemmatizer

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Initialize Flask app
app = Flask(__name__)
app.static_folder = 'static'

# Load pre-trained sentence embeddings
sent_bertphrase_embeddings = joblib.load('model/questionembedding.dump')
sent_bertphrase_ans_embeddings = joblib.load('model/ansembedding.dump')

# Load dataset
df = pd.read_csv("model/20200325_counsel_chat.csv", encoding="utf-8")

# Stopwords and lemmatizer
stop_w = stopwords.words('english')
lmtzr = WordNetLemmatizer()

# Clean text function
def clean(column, df, stopwords=False):
    df[column] = df[column].apply(str)
    df[column] = df[column].str.lower().str.split()
    # Remove stop words
    if stopwords:
        df[column] = df[column].apply(lambda x: [item for item in x if item not in stop_w])
    # Remove punctuation
    df[column] = df[column].apply(lambda x: [item for item in x if item not in string.punctuation])
    df[column] = df[column].apply(lambda x: " ".join(x))

# New function to get BERT embeddings
def get_embeddings(texts):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)
    embeddings = output.last_hidden_state.mean(dim=1)  # Mean pooling to get sentence embeddings
    return embeddings

# Retrieve and print FAQ answers
def retrieveAndPrintFAQAnswer(question_embedding, sentence_embeddings, FAQdf):
    max_sim = -1
    index_sim = -1
    valid_ans = []
    
    for index, faq_embedding in enumerate(sentence_embeddings):
        sim = cosine_similarity(faq_embedding, question_embedding)[0][0]
        if sim >= max_sim:
            max_sim = sim
            index_sim = index
            valid_ans.append(index_sim)
    
    max_a_sim = -1
    answer = ""
    for ans in valid_ans:
        answer_text = FAQdf.iloc[ans, 8]
        answer_em = sent_bertphrase_ans_embeddings[ans]
        similarity = cosine_similarity(answer_em, question_embedding)[0][0]
        if similarity > max_a_sim:
            max_a_sim = similarity
            answer = answer_text
    
    if max_a_sim < 0.50:
        return "Could you please elaborate on your situation more? I don't really understand."
    return answer

    if not answer or max_a_sim < 0.50:
        return "I'm here to help! Could you tell me more about your situation?"
    return answer


# Retrieve function
def retrieve(sent_bertphrase_embeddings, example_query):
    max_ = -1
    max_i = -1
    for index, emb in enumerate(sent_bertphrase_embeddings):
        sim_score = cosine_similarity(emb, example_query)[0][0]
        if sim_score > max_:
            max_ = sim_score
            max_i = index
    return str(df.iloc[max_i, 8])

# Clean and lemmatize text
def clean_text(greetings):
    greetings = greetings.lower()
    greetings = ' '.join(word.strip(string.punctuation) for word in greetings.split())
    re.sub(r'\W+', '', greetings)
    greetings = lmtzr.lemmatize(greetings)
    return greetings

# Predictor function
def predictor(userText):
    data = [userText]
    x_try = pd.DataFrame(data, columns=['text'])
    clean('text', x_try, stopwords=True)
    
    for index, row in x_try.iterrows():
        question = row['text']
        question_embedding = get_embeddings([question])
        answer = retrieveAndPrintFAQAnswer(question_embedding, sent_bertphrase_embeddings, df)
        
        # Check if fallback message repeats, return alternative
        if "Could you please elaborate" in answer:
            fallback_responses = [
                "I'm here to help. Can you tell me a bit more?",
                "That sounds tough. Would you like to share more?",
                "I understand. Please give me some more details."
            ]
            return random.choice(fallback_responses)
        
        return answer

# Greetings and goodbye messages
greetings = ['hi', 'hey', 'hello', 'heyy', 'good evening', 'good morning', 'good afternoon', 'fine', 'okay', 'great']
happy_emotions = ['i feel good', 'life is good', "i'm doing good", "i've had a wonderful day"]
goodbyes = ['thank you', 'bye', 'thanks and bye', 'goodbye', 'see ya later', 'adios', 'talk to you later']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    cleanText = clean_text(str(userText))
    
    # Check sentiment
    blob = TextBlob(userText, analyzer=PatternAnalyzer())
    polarity = blob.sentiment.polarity

    if cleanText in greetings:
        return "Hello! How may I help you today?"
    elif polarity > 0.7:
        return "That's great! Do you still have any questions for me?"
    elif cleanText in happy_emotions:
        return "That's great! Do you still have any questions for me?"  
    elif cleanText in goodbyes:
        return "Hope I was able to help you today! Take care, bye!"
    
    topic = predictor(userText)
    return topic

if __name__ == "__main__":
    app.run()
