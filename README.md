# 🎓 CampusBuddy AI

A beginner-friendly **NLP retrieval chatbot** that answers student questions by finding the closest matching FAQ using **TF-IDF** and **cosine similarity**.

## ✨ Features
- Ask student-life, study, programming, AI/ML, GitHub, and career questions
- TF-IDF text vectorization (questions plus answers, with the question weighted higher)
- Small alias table so abbreviations like AI, ML and NLP are understood
- "You might also ask" suggestions from the next-best matches
- Says "no close match" instead of guessing when similarity is low
- Cosine-similarity matching
- Similarity score with a clear explanation of what it means
- Explore the editable FAQ knowledge base
- Modern Streamlit interface
- Easy-to-edit CSV dataset

## 🧠 AI concepts used
1. **Natural Language Processing (NLP):** works with user-written questions.
2. **Text preprocessing:** lowercasing and English stop-word removal through the vectorizer.
3. **TF-IDF:** converts text into numerical feature vectors.
4. **Cosine similarity:** compares the user's question with saved FAQ questions.
5. **Information retrieval:** returns the answer connected to the closest question.

> This is a retrieval-based chatbot, not a generative AI model. It selects an answer from the FAQ dataset and does not create new facts. Its similarity score is not a probability of correctness.

## ✅ Testing
```bash
python -m unittest discover -s tests -v
```
The tests check CSV validation, retrieval on a 16-query evaluation set (paraphrases, abbreviations and off-topic questions) and that the Streamlit app runs. Note: I tuned the alias table while building that set, so its score is optimistic. Known weak spots are indirect phrasing with no shared words (e.g. "I keep putting things off") and very short queries. Measuring on questions written by someone else is a good next step.

## 🛠️ Requirements
- Python 3.10 or newer
- pip

## ▶️ Run on Windows
1. Download or clone this repository.
2. Open the project folder in VS Code.
3. Open the VS Code terminal (**Terminal → New Terminal**).
4. Install the packages:

   ```bash
   py -m pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   py -m streamlit run app.py
   ```

6. Your browser should open the app. If it does not, use the local URL printed in the terminal.

## ▶️ Run on macOS / Linux
```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

## 📁 Project structure
```text
CampusBuddy_AI/
│
├── app.py            # Streamlit interface
├── retrieval.py      # TF-IDF search logic (testable, no UI)
├── tests/
│   └── test_retrieval.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── faqs.csv
│
└── .streamlit/
    └── config.toml
```

## ➕ Add your own FAQs
Open `data/faqs.csv` and add a new row with:
- `category`
- `question`
- `answer`

Keep each answer inside one CSV cell. Restart the app after editing the dataset.

## 🚀 Future improvements
- Add more campus-specific FAQs
- Add a feedback button for helpful / not helpful answers
- Compare TF-IDF with other text representations
- Grow the evaluation set with independently written questions
- Try sentence embeddings and compare them with TF-IDF

## 👩‍💻 Portfolio description
**CampusBuddy AI — NLP FAQ Chatbot:** Built a student helpdesk chatbot using Python, Streamlit, TF-IDF vectorization, and cosine similarity to retrieve relevant answers from a custom FAQ knowledge base.
