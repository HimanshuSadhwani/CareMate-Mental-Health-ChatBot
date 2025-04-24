# 🤖 CareMate - Mental Health ChatBot

CareMate is a conversational chatbot designed to support mental wellness by providing empathetic and relevant responses. It utilizes BERT embeddings and a preprocessed counseling dataset to help users express themselves in a safe, supportive environment.

---

## 🌟 Features

- 🗨️ **Conversational chatbot** with a clean, responsive web UI  
- 🔍 **BERT (`bert-base-uncased`)** used for sentence embeddings  
- 📐 **Cosine similarity**-based response retrieval  
- 😊 **Sentiment-aware responses** using TextBlob  
- 🧠 **Preprocessed mental health counseling dataset**  
- 🪶 **Lightweight Flask backend**

---

## 📁 Project Structure

```yaml
project_structure:
  root: CareMate-Mental-Health-ChatBot/
  contents:
    - app.py: Main Flask application
    - requirements.txt: Required Python packages
    - README.md: Project documentation
    - model/:
        - questionembedding.dump: BERT embeddings of questions
        - ansembedding.dump: BERT embeddings of answers
        - 20200325_counsel_chat.csv: Cleaned counseling dataset
    - static/:
        - icon.jpg: Bot image
        - styles/:
            - style.css: Frontend styling
    - templates/:
        - index.html: Chat interface (HTML)

```

---

## ⚙️ Installation & Usage Steps

```yaml
Steps:
  - Step 1: Clone the repository
    command: |
      git clone https://github.com/HimanshuSadhwani/CareMate-Mental-Health-ChatBot.git
      cd CareMate-Mental-Health-ChatBot

  - Step 2: Create a virtual environment
    windows: |
      python -m venv env
      env\Scripts\activate
    linux_mac: |
      python3 -m venv env
      source env/bin/activate

  - Step 3: Install dependencies
    command: |
      pip install -r requirements.txt

  - Step 4: Run the application
    command: |
      python app.py
    navigate_to: http://127.0.0.1:5000
```

---

## 📊 Dataset
- **Source**: Counseling conversations dataset

- **File**: 20200325_counsel_chat.csv

- **Preprocessing**: Stopword removal, punctuation stripping, lemmatization

---


## 🤖 How It Works
**1.** User Input is cleaned and embedded using BERT.

**2.** Cosine Similarity is calculated with pre-embedded counseling dataset questions.

**3.** Relevant Answer is retrieved, optionally verified with answer embeddings.

**4.** Sentiment Analysis is done using TextBlob to adapt response tone.

---


## 🛠️ Technologies Used
- Python 3.10

- Flask

- BERT (via HuggingFace transformers)

- TextBlob (Sentiment Analysis)

- scikit-learn

- NLTK (Text preprocessing)

- HTML/CSS/JavaScript (Frontend)

---


## 📃 License
This project is licensed under the MIT License. See the LICENSE file for details.

---


## 🙌 Acknowledgements
- BERT by Google AI

- TextBlob for sentiment scoring

- Dataset adapted from counseling community platforms

---


## 🧠 *CareMate is not a replacement for professional therapy. It’s meant to provide general emotional support and should be used accordingly.*
