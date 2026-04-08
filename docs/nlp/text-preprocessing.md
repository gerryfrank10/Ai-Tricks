# Text Processing

Text processing is a crucial step in Natural Language Processing (NLP) workflows. It involves converting raw, unstructured text into structured and usable formats. This section covers practical techniques from basic text cleaning to advanced processing for downstream NLP tasks.

---

## 📄 **Sections in Text Processing**

### 1. **Text Preprocessing**
Preprocessing is the initial and most critical step in text processing. This step ensures that text data is clean and structured.

Key steps:
- **Lowercasing**:
  ```python
  text = "Hello World!"
  text = text.lower()  # Output: "hello world!"
  ```
- **Removing punctuation and special characters**:
  ```python
  import re
  text = "Hello, World! @2023"
  clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
  # Output: "Hello World 2023"
  ```
- **Stopword removal**:
  ```python
  from nltk.corpus import stopwords
  stop_words = set(stopwords.words('english'))
  words = ["this", "is", "an", "example"]
  filtered_words = [word for word in words if word not in stop_words]
  # Output: ["example"]
  ```
- **Stemming and Lemmatization**:
  ```python
  from nltk.stem import PorterStemmer, WordNetLemmatizer
  ps = PorterStemmer()
  wl = WordNetLemmatizer()
  print(ps.stem("running"))  # Output: "run"
  print(wl.lemmatize("running", pos="v"))  # Output: "run"
  ```

Packages:
- `NLTK`
- `SpaCy`
- `re` (Regular Expressions module for cleaning)

Reference: [Text Preprocessing Guide](Text-Preprocessing.md)

---

### 2. **Tokenization**
Tokenization breaks down text into smaller units.

**a. Word Tokenization**:
```python
from nltk.tokenize import word_tokenize
text = "Hello, how are you?"
tokens = word_tokenize(text)
# Output: ['Hello', ',', 'how', 'are', 'you', '?']
```

**b. Sentence Tokenization**:
```python
from nltk.tokenize import sent_tokenize
text = "AI is amazing. It automates tasks effectively."
sentences = sent_tokenize(text)
# Output: ["AI is amazing.", "It automates tasks effectively."]
```

**c. Subword Tokenization (e.g., Byte Pair Encoding)**:
Subword tokenization is necessary for handling out-of-vocabulary words.
```python
# Example using Hugging Face's Tokenizers
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("unbelievable")
# Output: ['un', '##believable']
```

Packages:
- `NLTK`
- `transformers` (for subword tokenization)

Reference: [Tokenization in NLP](https://placeholder_link.com)

---

### 3. **Text Cleaning**
Cleaning involves removing irrelevant parts of text that don’t contribute to analysis.

Steps:
- Removing HTML tags:
  ```python
  from bs4 import BeautifulSoup
  html = "<p>This is a sentence.</p>"
  clean_text = BeautifulSoup(html, "html.parser").get_text()
  # Output: "This is a sentence."
  ```

- Expanding contractions:
  ```python
  from contractions import fix
  text = "don't"
  expanded_text = fix(text)
  # Output: "do not"
  ```

Packages:
- `BeautifulSoup` (for cleaning HTML)
- `contractions`

---

### 4. **Text Normalization**
Text normalization ensures consistency.

Steps:
- **Lowercasing**:
  ```python
  text = "Normalization IS Important!"
  text = text.lower()  # Output: "normalization is important!"
  ```
- **Spelling correction**:
  ```python
  from autocorrect import Speller
  spell = Speller()
  text = "thiss is misspeled"
  corrected_text = spell(text)
  # Output: "this is misspelled"
  ```

Packages:
- `autocorrect`

---

### 5. **Vectorization**
Vectorization converts text into numerical formats.

**a. Bag of Words (BoW)**:
```python
from sklearn.feature_extraction.text import CountVectorizer
texts = ["This is an example.", "This is another example."]
vectorizer = CountVectorizer()
vectorized = vectorizer.fit_transform(texts)
print(vectorized.toarray())
# Output: [[1 1 1 1], [1 0 1 1]]
```

**b. TF-IDF**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(texts)
print(vectors.toarray())
```

**c. Word Embeddings** (Word2Vec, GloVe):
```python
from gensim.models import Word2Vec
sentences = [["hello", "world"], ["text", "processing", "is", "fun"]]
model = Word2Vec(sentences, vector_size=100, window=2, min_count=1)
print(model.wv["hello"])  # Output: Word embedding for "hello"
```

Packages:
- `scikit-learn`
- `gensim`

Reference: [Understanding Word Embeddings](https://placeholder_link.com)

---

### 6. **Sentiment Analysis**
Sentiment analysis uses text polarity to find whether the sentiment is positive, negative, or neutral.

Simple Example:
```python
from textblob import TextBlob
text = "I love programming!"
analysis = TextBlob(text)
print(analysis.sentiment.polarity)  # Output: Positive polarity (1)
```

For advanced deep learning examples:
```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I love AI Tricks!")[0]
print(result)
# Output: {'label': 'POSITIVE', 'score': 0.9997}
```

Packages:
- `TextBlob`
- `transformers` (Hugging Face)

Reference: [A Guide to Sentiment Analysis](https://placeholder_link.com)

---

### 7. **Language Detection and Translation**
- **Detecting Language**:
  ```python
  from langdetect import detect
  text = "Bonjour le monde"
  language = detect(text)
  # Output: "fr"
  ```

- **Translation**:
  ```python
  from googletrans import Translator
  translator = Translator()
  translated = translator.translate("Bonjour", src="fr", dest="en")
  print(translated.text)  # Output: "Hello"
  ```

Packages:
- `langdetect`
- `googletrans`

---

### 8. **Named Entity Recognition (NER)**
Identify entities like names, dates, and locations.

```python
import spacy
nlp = spacy.load("en_core_web_sm")
text = "Microsoft was founded by Bill Gates in 1975."
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)
# Output: "Microsoft ORG", "Bill Gates PERSON", "1975 DATE"
```

Packages:
- `spaCy`
- Hugging Face transformers for advanced NER.

---

## 🔗 **References and Resources**

1. [Text Preprocessing Guide](Text-Preprocessing.md)  
2. [Tokenization in NLP](https://placeholder_link.com)  
3. [Understanding Word Embeddings](https://placeholder_link.com)  
4. [A Guide to Sentiment Analysis](https://placeholder_link.com)  
5. [Processing Large Text Datasets](https://placeholder_link.com)

---

Efficient text processing is the backbone of NLP applications. Use these techniques to streamline workflows and achieve meaningful results in your projects!