from cassandra.cluster import Cluster
import spacy
import pandas as pd
import operator
from datetime import datetime
import warnings
import json
warnings.filterwarnings("always")
nlp = spacy.load('en_core_web_lg')
cluster = Cluster(['172.31.66.231'])
session = cluster.connect('sharelock')

import math
def get_row_score(row):
    user_score = 1
    if (row['user_score'] > 1):
        user_score = math.log(row['user_score'])
    return (row['likes'] + row['retweets'] * 2 ) * user_score

stop_ents = ['#', "@"]
def is_valid_ent(ent, topic):
    stop_found = False
    for stop in stop_ents:
        if stop in ent:
            stop_found = True
    return ent != topic and not stop_found and len(ent) > 3\
               and not "http" in ent and not ent.replace(" ", "").isdigit()

def get_ent_dictionary(sorted_result, topic):
    ent_dict = {}

    for idx, row in sorted_result.iterrows():
        text_item = row['tweet_text']
        tweet_item = row
        text_doc = nlp(row['tweet_text'])
        for ent in text_doc.ents:
            stripped_ent = ent.lemma_.strip()
            if (is_valid_ent(stripped_ent, topic)):
                if stripped_ent in ent_dict.keys():
                    for t in ent_dict[stripped_ent]:
                        tFound = False
                        if t['tweet_id'] == tweet_item['tweet_id'] or nlp(t['tweet_text']).similarity(text_doc) > 0.95:
                            tFound = True
                            break   
                        if not tFound:
                            ent_dict[stripped_ent].append(tweet_item)
                            
                else:
                    ent_dict[stripped_ent] = [tweet_item]               
    return ent_dict            

def get_post_tag_map(row, ent_dict):
    post_tag_dict = []
    stop_ents = ['#', "@"]

    text_item = row['tweet_text']
    tweet_item = row
    text_doc = nlp(row['tweet_text'])
    tweet_id = row['tweet_id']
    entity_list = []
    for key in ent_dict:
        frame_list = ent_dict[key]
        key_exists = False
        for frame in frame_list:
            if frame['tweet_id'] == tweet_id:
                key_exists = True
                break
                if not key_exists:
                    data = {}
                    data["name"] = key
                    data["rank"] = len(frame_list)
                    post_tag_dict.append(data)
    return post_tag_dict               
               
def start_tag_buckets():
    topic_list_query = "SELECT * from sharelock.topic_list"
    topic_rows = session.execute(topic_list_query)
    topic_row_list = list(topic_rows)
    topic_frames = pd.DataFrame(topic_row_list)  
    
    for idx, frame in topic_frames.iterrows():
        topic = frame['topic']
        category = frame['category']
        query = "SELECT * from sharelock.active_tweets where topic='"+topic+"'order by inserted_at desc limit 2"
        rows = session.execute(query)
        ent_dict = {}    
        sorted_json = {}
        xd = json.loads(rows[0].tweet_batch)
        sorted_result = df = pd.DataFrame(data=xd) 
        if not sorted_result.empty:
            ent_dict = get_ent_dictionary(sorted_result, topic)
            for idx, row in sorted_result.iterrows():
                post_ent_list = get_post_tag_map(row, ent_dict) 
                enity_score = 0
                for ent in post_ent_list:
                    enity_score = enity_score + ent['rank']
    
                SQL_QUERY = "INSERT INTO sharelock.tweet_tags_with_rank (tweet_id , topic, category, entity_list, entity_score) values (?,?,?,?, ?)"
                params = [row['tweet_id'], topic, category, json.dumps(post_ent_list), enity_score]
                prepared = session.prepare(SQL_QUERY)
                session.execute(prepared, params)
                # Create api endpoint for posts with entity rankings  
    
            if ent_dict:
                frame_score_list = {}
                for key in ent_dict:
                    frame_score = 0
                    frame_list = ent_dict[key]
                    frame_tag = key
                    frame_tweet_list = []
                    if (len(frame_list) > 0):
                        for frame in frame_list:
                            frame_score = frame_score + frame['raw_score']
                            frame_obj = {}
                            raw_score = frame['raw_score']
                            tweet_id = frame['tweet_id']
                            tweet_text = frame['tweet_text']
    
                            frame_tweet_list.append({"id": tweet_id, "text": tweet_text, "score": raw_score})
    
                            insert_time = int(datetime.utcnow().timestamp())   
                            insert_values = [topic, category, frame_tag, json.dumps(frame_tweet_list), frame_score, insert_time]
                            sql_query = "INSERT INTO sharelock.filtered_tag_frame (topic, category, tag, tweet_map, frame_raw_score, inserted_at) values (?, ?, ?, ?, ?, ?)"
                            try:
                                prepared = session.prepare(sql_query)
                                session.execute(prepared, (insert_values))
                            except Exception as e:
                                pass
