# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 18:43:01 2023

@author: rohit
"""

import regex as re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
    
stopwords_list = stopwords.words('english')
exclude = string.punctuation

def remove_html_tags(text):
    reg_rule = re.compile('<.*?>')
    return re.sub(reg_rule, '', text)

def remove_urls(text):
    reg_rule = re.compile(r'http\S+|www.\S+')
    return re.sub(reg_rule, '', text)

def remove_stopwords(text):
    clean_text = []  
    for word in text.split():
        if word in stopwords_list: # stp = stopwords.words('english')
            clean_text.append('')
        else:
            clean_text.append(word)
    
    x = clean_text[:]
    return " ".join(x)


# Remove punctuation 
def remove_punc(text):
    return text.translate(str.maketrans('','',exclude))

#Add stemming and lemmatization     
def text_data_cleaning(df):
    df['review'] = df['review'].str.lower()
    df['review'] = df['review'].apply(remove_html_tags)
    df['review'] = df['review'].apply(remove_urls)
    df['review'] = df['review'].apply(remove_punc)  
    df['review'] = df['review'].apply(remove_stopwords)
    return df
    
def tfidf_features_fit(df):
    tfidf = TfidfVectorizer(min_df=0.01,max_df=0.1)
    tfidf_matrix = tfidf.fit_transform(df['review'])
    return tfidf,tfidf_matrix.toarray()

def tfidf_features_transform(tfidf,df):
    tfidf_matrix = tfidf.transform(df['review']) 
    return tfidf_matrix.toarray()
   