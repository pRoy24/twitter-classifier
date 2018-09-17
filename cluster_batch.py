
# coding: utf-8

# In[65]:


from cassandra.cluster import Cluster
import spacy
import pandas as pd
import operator
from datetime import datetime
import textacy
nlp = spacy.load('en_core_web_lg')
cluster = Cluster(['127.0.0.1'])
session = cluster.connect('sharelock')
en = textacy.load_spacy('en_core_web_lg')


# Clean text rows, lemmatize, remove stop words. 
# 

# In[66]:


from emoji import UNICODE_EMOJI

def is_emoji(s):
    count = 0
    for emoji in UNICODE_EMOJI:
        count += s.count(emoji)
        if count > 1:
            return False
    return bool(count)

def get_tweet_tokens_list(tweet):
    t_doc = nlp(tweet)
    token_list = []
    for token in t_doc:
        if token.is_alpha and not token.is_stop and "-PRON-" not in token.lemma_ and not is_emoji(token.lemma_):
            token_list.append(token.lemma_.strip()) 
    return token_list       


# In[67]:


from sklearn.feature_extraction.text import TfidfVectorizer
def dummy_fun(doc):
    return doc

def get_tf_idf_vector(text_token_list):
    tfidf = TfidfVectorizer(
        analyzer='word',
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None)  
    pos_vectorized = tfidf.fit_transform(text_token_list)
    return pos_vectorized
    


# In[68]:


def get_similarity_vector(tweet_text_list):
    similarity_list = []
    for tweet1 in tweet_text_list:
        sim_row = []
        for tweet2 in tweet_text_list:
            sim_row.append(tweet1.similarity(tweet2))
        similarity_list.append(sim_row)
    return  similarity_list  


# In[69]:


import math
def get_raw_score(frame):
    user_score = 1
    if frame['user_score'] > 1:
        user_score = math.log(frame['user_score'])
    base_score = frame['likes'] + frame['retweets'] * 2
    if base_score < 1:
        base_score = 1
    return base_score * user_score 


# In[70]:


import re
stop_ents = ["@"]
def is_valid_ent(ent, topic):
    unspaced_ent = ent.replace(" ", "")
    phone_no_reg = '/^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$/im'                        
    stop_found = False
    num_word_in_ent = len(ent.split())
    for stop in stop_ents:
        if stop in ent:
            stop_found = True
    return ent != topic and not stop_found and len(ent) > 3               and not "http" in ent and not ent.replace(" ", "").isdigit() and "amp" not in ent and ent != '-PRON-'               and num_word_in_ent <= 3 

def get_ent_freq_score(vec, topic):
    ent_list = []
    for row in vec:
        for vec_ent in row:
            if (is_valid_ent(vec_ent.lemma_, topic)):
                ent_list.append(vec_ent.lemma_)
    return ent_list
        


# In[71]:


def get_entity_match_vector(frame, entity_list, topic):
    entity_matrix = []
    for row in frame:
        row_entity_list = []
        for row_ent in row.ents:
            if is_valid_ent(row_ent.lemma_, topic):
                row_entity_list.append(row_ent.lemma_.strip())
        match_ent_vector = []                       
        for ent in entity_list:
            if (ent in row_entity_list):
                match_ent_vector.append(1)
            else:
                match_ent_vector.append(0)
                
        entity_matrix.append(match_ent_vector)
    return entity_matrix


# In[72]:


def get_lemmatized_ents(doc, topic):
    ent_set = set()
    for row_ent in doc.ents:
        if is_valid_ent(row_ent.lemma_, topic):
            ent_set.add(row_ent.lemma_.strip())
    
    for token in doc:
        if token.pos_ in ["PROPN", "NOUN"]:
            ent_set.add(token.lemma_)
            
    for chunk in doc.noun_chunks:
        ent_set.add(chunk.lemma_.strip())
        ent_set.add(chunk.root.lemma_.strip())
        #ent_set.add(chunk.root.head.lemma_.strip())
    if "-PRON-" in ent_set:
        ent_set.remove("-PRON-")
    return list(ent_set)        


# In[73]:


def get_ranked_ent_list(frame, entity_list, topic):
    r_ents = {}
    for row in frame:
        if is_valid_ent(row_ent.lemma_, topic):
            curr = row_ent.lemma_.strip()
            if r_ents[curr]:
                r_ents[curr] = r_ents[curr] + 1
            else:
                r_ents[curr] = 1         
    return r_ents            
    


# In[74]:


def remove_duplicate_posts(sorted_result):
    for idx1, frame1 in sorted_result.iterrows():
        doc1 = frame1['tweet_clean_tokens']
        for idx2, frame2 in  sorted_result.iterrows():
            doc2 = frame2['tweet_clean_tokens']
            if idx1 != idx2 and doc1.similarity(doc2) > 0.99:
                sorted_result.drop(idx2, inplace=True)
    return sorted_result           


# In[75]:


import pytextrank
import sys
import json
from gensim.summarization.summarizer import summarize

def process_text_rank_for_corpus(topic_dict):
    tweet_text_corpus = ""
    for t in topic_dict:
        topic_dict[t].sort(key=lambda x: x['prob'], reverse=True)
        for tweet_row in topic_dict[t][2:6]:
            tweet_text_corpus = tweet_text_corpus + tweet_row['tweet_clean_text']

    


# In[76]:


import re
from langdetect import detect
RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)

def strip_emoji(text):
    return RE_EMOJI.sub(r'', text)
    return tweet_text

def get_cleaned_text(tweet_text):
    tweet_text = re.sub(r"(?:\@|https?\://)\S+", "", tweet_text)
    tweet_text = strip_emoji(tweet_text)
    return tweet_text   


# In[77]:


upper_caps_stops = ['I']
import constants

spam_word_stops = constants.SPAM_WORDS

def get_structure_score(doc):
    token_penalty = 0
    for token in doc:
        token_text = token.text
        prev_token_caps = False
        if "#" in token_text or "@" in token_text or "..." in token_text:
            token_penalty = token_penalty + 10
        if token_text.isupper() and token_text not in upper_caps_stops:
            if prev_token_caps:
                token_penalty = token_penalty + 10
            prev_token_caps = True
        else:
            prev_token_caps = False
    
    return token_penalty    


# In[78]:


def get_sent_weights(tweet, tweet_terms):
    tweet_text = tweet['tweet_text']
    tt_doc = nlp(tweet_text)
    sentence_scores = []
    for sent in tt_doc.sents:
        ent_score = 0
        sent_text = sent.text
        for term in tweet_terms:
            if term['term'] in sent.lemma_:
                ent_score = ent_score + ((term['weight'] + 1)* 10)
        sent_doc = sent.as_doc()   
        num_words = 0
        for token in sent_doc:
            if token.is_alpha and not token.is_stop and not token.text.isupper() and not "#" in token.text:
                num_words = num_words + 1

        word_score = 0
        if num_words >= 6 and num_words <= 30:
            word_score = 100

        str_score = get_structure_score(sent_doc)
        grammar_score = 0
        
        if list(textacy.extract.pos_regex_matches(sent, r"<DET>? (<NOUN>+ <ADP|CONJ>)* <NOUN>+")):
            grammar_score = grammar_score + 100
        elif list(textacy.extract.pos_regex_matches(sent, r"<VERB>?<ADV>*<VERB>+")):
            grammar_score = grammar_score + 50
        elif list(textacy.extract.pos_regex_matches(sent, r"<PREP> <DET>? (<NOUN>+<ADP>)* <NOUN>+")):
            grammar_score = grammar_score + 20
        
        nc = textacy.extract.noun_chunks(tt_doc)
        grammar_score = grammar_score + (len(list(nc)))
        
        nv_triples = textacy.extract.subject_verb_object_triples(tt_doc)
        nv_grams =  textacy.extract.ngrams(tt_doc, 3, filter_stops=True, filter_punct=True, filter_nums=False, include_pos=None, exclude_pos=None, min_freq=1)

        grammar_score = grammar_score + (len(list(nv_triples)))
        grammar_score = grammar_score + (len(list(nv_grams)))
        
        user_normalized_score = 1
        user_score = int(tweet['user_score'])
        if user_score > 1:
            user_normalized_score = int(math.log(user_score))
        
        final_score_str = str(word_score) + str(user_normalized_score) + str(int(ent_score)) + str(grammar_score)

        
        sentence_scores.append({"text": strip_emoji(re.sub(r"(?:\https?\://)\S+", "", sent.text)) , "ent_score": ent_score, "grammar_score": grammar_score,
                                "word_score": word_score, "structure_penalty": str_score, "final_score": final_score_str})
    return sentence_scores
        


# In[79]:


import json
import pandas as pd
import numpy as np
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import textacy
from sklearn.feature_extraction.text import TfidfVectorizer

def start_cluster_batch():
    topic_list_query = "SELECT * from sharelock.topic_list"
    topic_rows = session.execute(topic_list_query)
    topic_row_list = list(topic_rows)
    topic_frames = pd.DataFrame(topic_row_list)
    for idx, frame in topic_frames.iterrows():
        topic = frame['topic']
        category = frame['category']
        query = "SELECT * from sharelock.active_tweets where topic='"+topic+"'order by inserted_at desc limit 30"
        rows = session.execute(query)
        ent_dict = {}    
        sorted_json = {}

        row_list = []
        for row in rows:
            xd = json.loads(row.tweet_batch)
            row_list = row_list + xd

        sorted_result = df = pd.DataFrame(data=row_list) 
        sorted_result.set_index('tweet_id')
        sorted_result = sorted_result.drop_duplicates(subset='tweet_id', keep='first')

        # Clean results by dropping items with similarity score o.98 or higher

        sorted_result['tweet_tokens'] = sorted_result['tweet_text'].apply(nlp)
        sorted_result['tweet_clean_text'] = sorted_result['tweet_text'].apply(get_cleaned_text)
        sorted_result['tweet_clean_tokens'] = sorted_result['tweet_clean_text'].apply(nlp)
        sorted_result = remove_duplicate_posts(sorted_result)

        corpus = textacy.Corpus(lang="en_core_web_lg", texts = list(sorted_result['tweet_text']), metadatas=list(sorted_result['tweet_id']))


        terms_list = (doc.to_terms_list(ngrams=(1, 2, 3), named_entities=True, normalize=u'lemma', lemmatize=True, lowercase=True, as_strings=True,                                        filter_stops=True, filter_punct=True, min_freq=1, exclude_pos=("PRON", "X", "PUNCT", "SYM"))
                      for doc in corpus)
  
        vectorizer = textacy.Vectorizer(tf_type='linear', apply_idf=True, idf_type='smooth')

        textacy.text_utils.clean_terms(terms_list)
    
        doc_term_matrix = vectorizer.fit_transform(terms_list)
        
        num_topics = int(len(sorted_result)/10)

        model = textacy.tm.TopicModel('nmf', n_topics=num_topics)
        model.fit(doc_term_matrix)    

        doc_topic_matrix = model.transform(doc_term_matrix)


        topic_cluster = {}
        for topic_idx, top_terms in model.top_topic_terms(vectorizer.id_to_term, topics=-1, top_n=8, weights=True):
            dct = dict(top_terms)
            tt_list = []
            for j in dct.keys():
                    tt_list.append({"term":j, "weight": dct[j]})
            topic_cluster["topic-"+str(topic_idx)] = {"terms": tt_list}

        for topic_idx, top_docs in model.top_topic_docs(doc_topic_matrix, topics=-1, top_n=6, weights=True):
            dct = dict(top_docs)
            tweet_in_topic_list = []
            for j in dct.keys():
                query_str = "tweet_id="+ corpus[j].metadata
                curr = sorted_result[sorted_result['tweet_id']==corpus[j].metadata]
                curr_frame_row = curr.iloc[0]
                is_attached_to_topic = False
                for prev_topic in topic_cluster:
                    if 'tweets' in topic_cluster[prev_topic]:
                        tweet_list = topic_cluster[prev_topic]['tweets']
                        for tweet in tweet_list:
                            if tweet['tweet_id'] == curr.iloc[0]['tweet_id']:
                                is_attached_to_topic = True
                                break

                if not is_attached_to_topic:
                    tweet_in_topic_list.append({"tweet_id": curr.iloc[0]['tweet_id'], "tweet_text": curr.iloc[0]['tweet_text'],
                                            "user_score": str(curr.iloc[0]['user_score']), "raw_score": str(curr.iloc[0]['raw_score'])})
            if tweet_in_topic_list:
                topic_cluster["topic-"+str(topic_idx)]['tweets'] = tweet_in_topic_list

        for curr_topic in topic_cluster:
            if 'tweets' in topic_cluster[curr_topic]:
                sent_weights = []
                for tweet in topic_cluster[curr_topic]['tweets']:
                    sent_weights = sent_weights + get_sent_weights(tweet, topic_cluster[curr_topic]['terms'])
                sent_weights = sorted(sent_weights, key=lambda x: x['final_score'], reverse=True)
                top_sents =  sent_weights[0:2]
                sorted_top_sents = sorted(sent_weights, key=lambda x: x['ent_score'], reverse=True)
                topic_title = ""
                topic_title_list = []
                for sent in sorted_top_sents:
                    if sent['structure_penalty'] < 50 and sent['word_score'] > 0:
                        topic_title_list.append(sent['text'].strip('\n'))           
                topic_cluster[curr_topic]['title'] =  topic_title_list
        
        result_dict = {}
        for k in topic_cluster.keys():
            if 'tweets' in topic_cluster[k]:
                 result_dict[k] = topic_cluster[k]
            

        insert_at = datetime.datetime.now().timestamp()

        insert_values = [topic, category, insert_at, json.dumps(result_dict)]

        sql_query = "INSERT into sharelock.topic_clusters (topic, category, inserted_at, tweet_cluster) values (?, ?, ?, ?)"
        try:
            prepared = session.prepare(sql_query)
            session.execute(prepared, (insert_values))
        except Exception as e:
            print(e)

