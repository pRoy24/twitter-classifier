
# coding: utf-8

# In[150]:


from jupyter_core.paths import jupyter_data_dir
import constants

from cassandra.cluster import Cluster
import spacy
import pandas as pd
from sklearn_pandas import DataFrameMapper, cross_val_score
# update table save features
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import math

nlp = spacy.load('en_core_web_lg')
cluster = Cluster(['127.0.0.1'])
session = cluster.connect('sharelock')


# In[151]:


topic_list_query = "SELECT * from sharelock.topic_list"
topic_rows = session.execute(topic_list_query)
topic_row_list = list(topic_rows)

topic_frames = pd.DataFrame(topic_row_list)  


# In[152]:


categories = set()
for idx, topic_item in topic_frames.iterrows():
    categories.add(topic_item['category'])

CATEGORY_LIST = list(categories) 
def get_sentence_score(tweet_doc):
    # number of entitites * number of sentences
    num_sents = len(list(tweet_doc.sents))
    num_ents = len(list(tweet_doc.ents))
    return num_sents * num_ents
        
    
# number of hash-tags or mentions
def get_tag_score(doc):
    tag_score = 0
    for token in doc:
        if "#" in token.text or "@" in token.text:
            tag_score = tag_score + 1 
    return tag_score  


upper_caps_stops = ['I']

spam_word_stops = constants.SPAM_WORDS
def get_structure_score(doc):
    token_penalty = 0
    for token in doc:
        token_text = token.text
        prev_token_caps = False
        if token_text.isupper():
            if prev_token_caps:
                token_penalty = token_penalty + 10
            prev_token_caps = True
        else:
            prev_token_caps = False
        if token_text in upper_caps_stops:
            token_penalty = token_penalty + 10
        if token_text.lower() in spam_word_stops:
            token_penalty = token_penalty + 10 
    return token_penalty    


import numpy as np
from langdetect import detect

# get corpus of news text on topic
def get_topic_web_corpus(category):
    sql_str = "SELECT * from sharelock.topic_link_data where link_category='"+category+"' allow filtering;"
    corpus_data = session.execute(sql_str)
    topic_corpus = pd.DataFrame(list(corpus_data))
    #print(len(topic_corpus))
    topic_corpus['body_doc'] = topic_corpus['link_text'].apply(nlp)
    topic_corpus['head_doc'] = topic_corpus['link_title'].apply(nlp)
    return topic_corpus

web_corpus_list = {}

doc_title_list = []
doc_body_list = []

for category in CATEGORY_LIST:
    web_corpus = get_topic_web_corpus(category)
    web_corpus_list[category] = web_corpus

    
def get_body_relevance_score(sentence_doc, frame_category):
    category_text_corpus = web_corpus_list[frame_category]

    body_relevance_score = []
    counter = 0
    for corpus_body in category_text_corpus['body_doc']:
        body_sents = corpus_body.sents
        counter = counter + len(list(body_sents)) 
        for corpus_sentence in body_sents:
            body_relevance_score.append(corpus_sentence.as_doc().similarity(sentence_doc))        
    return body_relevance_score    


def get_head_relevance_score(sentence_doc, frame_category):
    category_text_corpus = web_corpus_list[frame_category]

    head_relevance_score = []
    counter = 0
    for corpus_head in category_text_corpus['head_doc']:
        head_sents = corpus_head.sents
        counter = counter + len(list(head_sents))   
        for corpus_sentence in head_sents:
            head_relevance_score.append(corpus_sentence.as_doc().similarity(sentence_doc))       
    return head_relevance_score  
    
def get_raw_score(frame):
    user_score = 1
    if frame['user_score'] > 1:
        user_score = math.log(frame['user_score'])
    base_score = frame['likes'] + frame['retweets'] * 2
    if base_score < 1:
        base_score = 1
    return base_score * user_score 


# In[153]:


from datetime import datetime, timedelta
import json
pd.set_option('display.max_colwidth', -1)

time_now = datetime.now()

time_frame = time_now - timedelta(minutes = 30)
time_frame = '{0:%Y-%m-%d %H:%M:%S}'.format(time_frame)

query = "SELECT * from sharelock.streaming_tweets where inserted_at>'" + time_frame + "' allow filtering"

rows = session.execute(query)
row_list = list(rows)
result = []

if len(row_list) > 0:
    frame_result = pd.DataFrame(row_list)        
def start_classify_batch():
    for idx, topic_item in topic_frames.iterrows():
        category = topic_item['category']
        topic = topic_item['topic']
        result = frame_result.query("topic=='"+topic+"'")  
        result.set_index('tweet_id')
        result = result.drop_duplicates(subset='tweet_id', keep='last')
        print(category)
        time_frame = time_now - timedelta(minutes = 22)
        result = result[result['topic'] == topic]
        input_frame = result.sort_values(by=['tweet_time'], ascending=False)
        input_frame['tweet_time'] = pd.to_datetime(input_frame['tweet_time'])
        input_frame = input_frame[(input_frame['inserted_at'] > time_frame)]

        if not input_frame.empty:    
            input_frame['text_tokens'] = input_frame['tweet_text'].apply(nlp)
            input_frame['sentence_score'] = input_frame['text_tokens'].apply(get_sentence_score)
            input_frame['tag_score'] = input_frame['text_tokens'].apply(get_tag_score)
            input_frame['structure_score'] = input_frame['text_tokens'].apply(get_structure_score)
            input_frame['raw_score'] = input_frame.apply(get_raw_score, axis=1)
            input_frame['body_relevance_score'] = input_frame['text_tokens'].apply(get_body_relevance_score, args = (category,))
            input_frame['head_relevance_score'] = input_frame['text_tokens'].apply(get_head_relevance_score, args = (category,))


            mapper_model = joblib.load("%s-mapper.pkl" % category)
            feature_vector = mapper_model.transform(input_frame)   
            head_relevance_matrix = input_frame['body_relevance_score'].values
            body_relevance_matrix = input_frame['head_relevance_score'].values
            final_features = []

            for idx, f in enumerate(feature_vector):
                final_features.append(list(feature_vector[idx]) + list(head_relevance_matrix[idx]) + list(body_relevance_matrix[idx]))


            soft_relevant_classifier = joblib.load("%s-relevance-classifier.pkl" % category)
            soft_spam_classifier = joblib.load("%s-spam-classifier.pkl" % category)
            input_frame["relevant_prediction"] = soft_relevant_classifier.predict(final_features)
            input_frame["spam_prediction"] = soft_spam_classifier.predict(final_features)

            input_frame = input_frame.query('relevant_prediction==1').query('spam_prediction==0')
            input_frame['is_graded'] = input_frame['is_graded'].astype('int')
            input_frame['inserted_at'] = input_frame['inserted_at']
            input_frame['raw_score'] = input_frame['raw_score'].astype('int')

            df_json = input_frame[['tweet_text', 'tweet_id', 'raw_score', 'user_score', 'likes', 'retweets']].to_json(orient = "records")
            query = "INSERT INTO sharelock.active_tweets (topic, inserted_at, category, tweet_batch) values (?, ?, ?, ?)";
            nvals = [topic, time_now.timestamp(), category, df_json]

            prepared = session.prepare(query)
            session.execute(prepared, (nvals))




