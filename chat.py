import nltk
import io
import random
import numpy as np
import string
import warnings
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

f = open('E:\chatBot\globalwarming.txt','r',errors = 'ignore')
raw = f.read()
raw = raw.lower()
nltk.download('punkt')
nltk.download('wordnet')
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

# print(sent_tokens[:2])
# print(word_tokens[:2])

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct),None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello","hi","hey","greetings","what's up","sup")
GREETING_RESPONSES = ["Bol BC","AA gya Laudu","are kese ho","bol bsdk" ]
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def response(user_response):
    chatbot_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1],tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf == 0):
        chatbot_response = chatbot_response + "I am sorry! I don't understand"
        return chatbot_response
    else:
        chatbot_response = chatbot_response + sent_tokens[idx]
        return chatbot_response

flag = True
print("ChatBot: My name is Dubey. I will answer all your queries about Global Warming. Type Bye to leave")
while(flag == True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response!="bye"):
        if(user_response=='thanks' or user_response == 'thank you'):
            flag = False
            print("Dubey:  You are Welcome...")
        else:
            if(greeting(user_response)!=None):
                print("Dubey: " + greeting(user_response))
            else:
                print("Dubey :", end = "")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("Dubey : Bhak Bsdk")

