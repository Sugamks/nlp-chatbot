import string
import random
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Load data and preprocess
f = open('/content/data.txt', 'r', errors='ignore')
raw_doc = f.read()
raw_doc = raw_doc.lower()

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

sentence_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

def StemTokens(tokens):
    return [stemmer.stem(token) for token in tokens]

remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)

def StemNormalize(text):
    wh_words = {'what', 'why', 'where', 'when', 'who', 'how'}
    return StemTokens([token for token in nltk.word_tokenize(text.lower().translate(remove_punc_dict)) if token not in stop_words and token not in wh_words])

greet_inputs = ('hello', 'hi', 'whassup', 'how are you?')
greet_responses = ('hi', 'Hey', 'Hey There!', 'There there!!')

def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)

# Prepare the vectorizer and LSA model
TfidfVec = TfidfVectorizer(tokenizer=StemNormalize)
tfidf = TfidfVec.fit_transform(sentence_tokens)

lsa_model = TruncatedSVD(n_components=100, n_iter=100)
lsa_model.fit(tfidf)

# Define a function to generate responses
def kbprocessing(user_response):
    robo1_response = ""
    # Check if user input contains a greeting
    greet_response = greet(user_response)
    if greet_response:
      robo1_response = robo1_response + greet_response
      return robo1_response
    else:
      sentence_tokens.append(user_response)
      tfidf = TfidfVec.transform(sentence_tokens)
      concept_vectors = lsa_model.transform(tfidf)

      vals = cosine_similarity(concept_vectors[-1].reshape(1,-1), concept_vectors)
      idx = vals.argsort()[0][-2]
      flat = vals.flatten()
      flat.sort()
      req_tfidf = flat[-2]

      if (req_tfidf == 0):
          robo1_response = robo1_response + "I am sorry. Unable to understand you!"
      else:
          robo1_response = robo1_response + sentence_tokens[idx]

      sentence_tokens.remove(user_response)
      return robo1_response
