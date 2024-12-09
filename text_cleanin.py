#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, log_loss

import nltk
from nltk.util import ngrams
nltk.download('stopwords')
from nltk.corpus import stopwords
stop=set(stopwords.words('english'))
import string  # Add this import at the top of your script

from nltk.stem import WordNetLemmatizer 




# In[2]:


def clean_text1(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# In[3]:


#Removing Emojis
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# In[4]:


lemma=WordNetLemmatizer()
nltk.download('wordnet')
def clean_text(text):
    
    """
    It takes text as an input and clean it by applying several methods
    
    """
     #simplifying text
    text=re.sub(r"i'm","i am",text)
    text=re.sub(r"he's","he is",text)
    text=re.sub(r"she's","she is",text)
    text=re.sub(r"that's","that is",text)
    text=re.sub(r"what's","what is",text)
    text=re.sub(r"where's","where is",text)
    text=re.sub(r"\'ll"," will",text)
    text=re.sub(r"\'ve"," have",text)
    text=re.sub(r"\'re"," are",text)
    text=re.sub(r"\'d"," would",text)
    text=re.sub(r"won't","will not",text)
    text=re.sub(r"can't","cannot",text)
    
    
    string = ""
    for word in text.split():
        if word not in stop:
            string+=lemma.lemmatize(word)+" "
    
    return string


# In[5]:


def expand_abbreviations(text):
    abbreviations = {
        "u": "you",
        "r": "are",
        "4": "for",
        "&": "and"
    }
    words = text.split()
    expanded_words = [abbreviations[word] if word in abbreviations else word for word in words]
    return " ".join(expanded_words)
from spellchecker import SpellChecker

spell = SpellChecker()
spell.word_frequency.load_words(["wildfire", "evacuation", "shelter"])

def correct_spellings(text):
    # Ensure the input is a string
    if not isinstance(text, str) or text.strip() == "":
        return text  # Return as-is for invalid input
    
    corrected_text = []
    words = text.split()  # Split text into words
    
    # Identify misspelled words
    misspelled_words = spell.unknown(words)
    
    for word in words:
        # Check if the word is misspelled
        if word in misspelled_words:
            correction = spell.correction(word)
            # Keep original if no correction is found
            corrected_text.append(correction if correction else word)
        else:
            corrected_text.append(word)  # Keep the word as-is if not misspelled
    
    return " ".join(corrected_text)  # Rejoin the corrected words


# In[6]:


def clean_text(text):
    '''Clean the text using various methods.'''
    text = clean_text1(text)  # Clean the text with basic rules
    text = remove_emoji(text)  # Remove emojis
    text = expand_abbreviations(text)  # Expand abbreviations
    text = correct_spellings(text)  # Correct spellings
    
    # Lemmatization: Reduce words to their root form
    cleaned_text = " ".join([lemma.lemmatize(word) for word in text.split() if word not in stop])
    
    return cleaned_text

