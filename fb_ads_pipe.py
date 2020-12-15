# # -*- coding: utf-8 -*-
# """
# Created on Mon Oct 14 16:40:43 2019

# @author: sgupta27
# """

from __future__ import absolute_import, division, print_function

import pip

# def import_or_install(package):
#     try:
#         __import__(package)
#     except ImportError:
#         pip.main(['install', package])
#         if package=="spacy":
#             import spacy
#         elif package=="nltk":
#             import nltk
      
           
import csv
import os
import sys
import logging
import regex as re
logger = logging.getLogger()


import numpy as np

import nltk
import spacy
nlp = spacy.load("en_core_web_sm")
from spacy.matcher import PhraseMatcher 
matcher = PhraseMatcher(nlp.vocab)

csv.field_size_limit(2147483647) #Increase CSV reader's field limit incase we have long text.


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class BinaryClassificationProcessor(DataProcessor):
    """Processor for binary classification dataset."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:41:37 2019

@author: sgupta27
"""

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_example_to_feature(example_row):
    # return example_row
    example, label_map, max_seq_length, tokenizer, output_mode = example_row

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if output_mode == "classification":
        label_id = label_map[example.label]
    elif output_mode == "regression":
        label_id = float(example.label)
    else:
        raise KeyError(output_mode)

    return InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_id=label_id)
from datetime import timedelta,datetime
# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG
# Operators; we need this to operate!
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago
from datetime import datetime
import requests
import json
import pandas as pd
import pymongo

# import nltk
# from nltk.corpus import stopwords
# import re

# import spacy 
# from spacy.matcher import PhraseMatcher 
# nlp = spacy.load('en_core_web_sm')
# matcher = PhraseMatcher(nlp.vocab)



import sys
import torch
import pickle
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss

# from tools import *
from multiprocessing import Pool, cpu_count
# import convert_examples_to_features

from tqdm import tqdm_notebook, trange
import os
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


from datetime import timedelta,datetime
# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG
# Operators; we need this to operate!
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago
from datetime import datetime
import requests
import json
import pandas as pd
import pymongo
import time
from datetime import date
from airflow.models import Variable
# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
at=Variable.get('ads_token').split()
token=at[0]
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2020,3,12,2,0,0),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=1),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    'execution_timeout': timedelta(days=5),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}
dag = DAG(
    'fb_ads_pipe',
    default_args=default_args,
    description='A dag to dump and mark our database with new ads',
   # schedule_interval='0 */4 * * *',
    schedule_interval='@daily'
)

dbase='POTUS_2020_DEV'
url1 = 'mongodb://%s:%s@denver.ischool.syr.edu'
mongoClient= pymongo.MongoClient(url1 % ('bitslab', '0rang3!'))
db = mongoClient[dbase]
col=db["fb_ads_dump"]
data=[]




import datetime
# import json
# import pymongo
# import requests

## -----------------------------
## get a handle to the mongo database
## -----------------------------


# def get_database():
#     server_url = "mongodb://%s:%s@denver.ischool.syr.edu" % ("bitslab", "0rang3!")
#     database_name = "POTUS_2020_DEV"
#     mongo_client = pymongo.MongoClient(server_url)
#     return mongo_client[database_name]


## -----------------------------

## -----------------------------
## init some legacy variables
## -----------------------------


# db = get_database() # legacy support
# col = db["fb_ads_extended_test"] # legacy support


## -----------------------------

## -----------------------------
## "at" comes from AIRFLOW
## -----------------------------
##at = Variable.get('ads_token').split()


# at = [
#     "EAAIgVseTUHgBAMgOdVoJnHllnzTAoZBxzZANXWFxJqpnuHykQfsEHlXJDvPaDZAyn0odkyegu2TN2XfdAdwIZC519vHMinyXfMDJAGjm0yz1TVZBaquVVbCPTmCv4cOZAE32XLiu7ffluziFa9R0Ljzg32sI93stJ3CUVoSdo4hfX0hk2ZA7izh",
#     "EAAsZBFEmusggBAGTjqNgJx5CQpa5fvbt9sxRZBqZCwNFVZBSb7SpKIlUYX3jt4nMogCysxZBafsmWZBX6VeFzYvMh2DyhnZCzuC1FBtUTNHhZCtp9xRCnu6Wc2MLf86HzrkeQ5PijON83vvnz0wiO1CAaERAA5QU2AEtziZAzsUTi0YVCizQYB9NB"
# ]
# token = at[1]


## -----------------------------

## -----------------------------
## Should be a drop in replacement if I have done things right
## note that the real work this method should do has been mostly commented out.
## -----------------------------


def get_data(ds, **kwargs):
    global data
    global page_ids
    global token
    global at


    ## -----------------------------
    ## get some globals
    ## We are probably fine to switch to the versions that are commented out rather than using the globals
    ## -----------------------------
    #mongo_database = get_database()
    mongo_database = db

    #target_collection = mongo_database["fb_ads_extended_dump"]
    target_collection = col

    #fb_token = "EAAIgVseTUHgBAMgOdVoJnHllnzTAoZBxzZANXWFxJqpnuHykQfsEHlXJDvPaDZAyn0odkyegu2TN2XfdAdwIZC519vHMinyXfMDJAGjm0yz1TVZBaquVVbCPTmCv4cOZAE32XLiu7ffluziFa9R0Ljzg32sI93stJ3CUVoSdo4hfX0hk2ZA7izh"
#     fb_token = token

    fb_url = "https://graph.facebook.com/v5.0/ads_archive"
    ## -----------------------------

    ## -----------------------------
    ## how far back should we look for activity?
    ## once this is working correctly, we can probably set this to something lower
    ## -----------------------------
#     dt=(date.today()-timedelta(1))
#     dt5=(date.today()-timedelta(5))
    
    days_back =  datetime.timedelta(5)
    date_since = datetime.date.today() - days_back
    ## -----------------------------

    ## -----------------------------
    ## We want these fields back from facebook
    ## -----------------------------
    facebook_fields = [
        "ad_creation_time",
        "ad_creative_body",
        "ad_creative_link_caption",
        "ad_creative_link_description",
        "ad_creative_link_title",
        "ad_delivery_start_time",
        "ad_delivery_stop_time",
        "ad_snapshot_url",
        "currency",
        "demographic_distribution",
        "funding_entity",
        "impressions",
        "page_id",
        "page_name",
        "publisher_platforms",
        "region_distribution",
        "spend"
    ]
    ## -----------------------------

    ## -----------------------------
    ## default parameters for a facebook API request
    ## once paired with a page id we should get back
    ## all ad_buys with any activity since "date_since"
    ## -----------------------------
    facebook_params = {
        "access_token": token,
        "ad_active_status": "ALL",
        "ad_type": "POLITICAL_AND_ISSUE_ADS",
        "ad_reached_countries": "[\'US\']",
        "ad_delivery_date_min": str(date_since),
        "fields": ",".join(facebook_fields)
    }
    ## -----------------------------

    ## -----------------------------
    ## clear out rows/documents from the target collections
    ## not executed at the moment
    ## -----------------------------
    target_collection.remove()
    ## -----------------------------

    ## -----------------------------
    ## method used to process a page of facebook results
    ## this way we don't have to repeate this code
    ## -----------------------------
    def process_ad_buys(ad_buys, target_collection):
        ## -----------------------------
        ## If there are no ad_buys we can skip the rest
        ## -----------------------------
        if not ad_buys:
            return 0
        ## -----------------------------

        ## -----------------------------
        ## set some pre-scoring state
        ## -----------------------------
        for ad_buy in ad_buys:
            ad_buy["marked"] = 0
            ad_buy["marked_topic"] = 0
        ## -----------------------------

        ## -----------------------------
        ## load this "page" of data to mongo
        ## not executed at the moment
        ## -----------------------------
        target_collection.insert_many(ad_buys)
        ## -----------------------------

        return len(ad_buys)
    ## -----------------------------

    ## -----------------------------
    ## find active candidates
    ## at the moment restricted to Harris
    ## -----------------------------
    valid_candidates = [candidate for candidate in mongo_database["cand_info"].find({"drop_date": float("NaN")})]
#     valid_candidates = [candidate for candidate in mongo_database["cand_info"].find({"drop_date": float("NaN")}) if candidate["candidate_name"] == "Joe Biden" or candidate["candidate_name"] == "Kamala Harris"]
    ## -----------------------------

    for candidate in valid_candidates:
        ad_buys_processed = 0
        print(candidate["candidate_name"])

        ## -----------------------------
        ## Find ads for this candidate.
        ## -----------------------------
        these_facebook_params = facebook_params.copy()
        these_facebook_params["search_page_ids"] = candidate["page_id"]
        ## -----------------------------

        ## -----------------------------
        ## fetch a "page" of ads (upto 25) from facebook
        ## -----------------------------
        resp = requests.get(fb_url, params=these_facebook_params.items())
        resp_json = json.loads(resp.text)
        next_page_request = resp_json.get("paging", {}).get("next")
        ad_buys = resp_json.get("data", [])
        ## -----------------------------

        ad_buys_processed += process_ad_buys(ad_buys, target_collection)
        print("\r\tad buys loaded: %s" % (ad_buys_processed), flush=True, end="")

        ## -----------------------------
        ## process the next page of data
        ## -----------------------------
        while next_page_request is not None:
            ## -----------------------------
            ## fetch a "page" of ads from facebook
            ## -----------------------------
            resp = requests.get(next_page_request)
            resp_json = json.loads(resp.text)
            next_page_request = resp_json.get("paging", {}).get("next")
            ad_buys = resp_json.get("data", [])
            ## -----------------------------

            ad_buys_processed += process_ad_buys(ad_buys, target_collection)
            print("\r\tad buys loaded: %s" % (ad_buys_processed), flush=True, end="")
        ## -----------------------------

        print("\n")
## -----------------------------

# get_data(None)



# def get_data(ds, **kwargs):
#     global data
#     global page_ids
#     global token
#     global at
    
#     col.remove()
#     dt=(date.today()-timedelta(1))
#     dt5=(date.today()-timedelta(5))
#     resp={}
#     print("hellooo")
#     t=time.time()
#     for page in db['cand_info'].find(no_cursor_timeout=True):
#         print(page['candidate_name'])
#         drp_dt=page['drop_date']
#         y=0
#         if type(drp_dt)!=float:
#             drp_dt=datetime.strptime(drp_dt, '%m/%d/%Y').date()
#             if drp_dt>=dt:
#                 y=1
#         else:
#             y=1
#         if y==1: 
#             params = (('fields', 'ad_creation_time,ad_creative_body,ad_creative_link_caption,ad_creative_link_description,ad_creative_link_title,ad_delivery_start_time,ad_delivery_stop_time,ad_snapshot_url,currency,demographic_distribution,funding_entity,impressions,page_id,page_name,publisher_platforms,region_distribution,spend'),('ad_active_status','ALL'),('search_page_ids', page['page_id']),('ad_type', 'POLITICAL_AND_ISSUE_ADS'),('ad_reached_countries', '[\'US\']'),('access_token', token))
#             resp = requests.get('https://graph.facebook.com/v5.0/ads_archive', params=params)
#             resp_json = json.loads(resp.text)
#             x={}
#             i=0
#             ct=0
#             while True:
#                 try:
#                     ct=ct+1
#                     print("page: ",ct)
#                     for item in resp_json["data"]:
#                         item['marked']=0
#                         item['marked_topic']=0
#                         col.insert_one(item)
#                     try:
#                         try:
#                             if datetime.strptime(resp_json["data"][-1]["ad_delivery_stop_time"][0:10], '%Y-%m-%d').date()<dt:    
#                                 break
#                         except KeyError:
#                             print(resp_json["data"][-1])
#                             if datetime.strptime(resp_json["data"][-1]["ad_delivery_start_time"][0:10], '%Y-%m-%d').date()<dt5:
#                                 break
#                     except IndexError:
#                         try:
#                             x["data"][-1]["ad_delivery_stop_time"][0:10]
#                         except KeyError:
#                             pass
#                         pass
#                     try:
#                         x=resp_json
#                         resp = requests.get(resp_json['paging']['next'])
#                         resp_json=json.loads(resp.text)
#                     except KeyError:
#                         break
#                 except KeyError:
#                     if time.time()-t<10:
#                         break
#                     t=time.time()
#                     i=(i+1)%2
#                     token=at[i]
#                     print("token changed")
#         else:
#             print("Candidate Not Collected!")
#     print(len(data))



def predict(ds, **kwargs):
    global db
    print(os.getcwd())
    data=[]
    ls=[]
    for doc in col.distinct('ad_creative_body',{'marked':0}):
        if len(data)<2000:
            data.append({'ad_creative_body':doc})
        else:
            break           

    if len(data)>0:
        print(len(data))
        df=pd.DataFrame(data)
        df['label']=0
        dev_df_bert = pd.DataFrame({
            'id':range(len(df)),
            'label':df['label'],
            'alpha':['a']*df.shape[0],
            'text': df['ad_creative_body'].replace(r'\n', ' ', regex=True)
        })

        dev_df_bert.to_csv('./home/jay/airflow/dags/data/dev.tsv', sep='\t', index=False, header=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # This is where BERT will look for pre-trained models to load parameters from.
        CACHE_DIR = './home/jay/airflow/dags/cache/'

        # The maximum total input sequence length after WordPiece tokenization.
        # Sequences longer than this will be truncated, and sequences shorter than this will be padded.
        MAX_SEQ_LENGTH = 128

        TRAIN_BATCH_SIZE = 24
        EVAL_BATCH_SIZE = 8
        LEARNING_RATE = 2e-5
        RANDOM_SEED = 42
        GRADIENT_ACCUMULATION_STEPS = 1
        WARMUP_PROPORTION = 0.1
        OUTPUT_MODE = 'classification'
        NUM_TRAIN_EPOCHS = 1
        CONFIG_NAME = "config.json"
        WEIGHTS_NAME = "pytorch_model.bin"
        Data = 'FB16'
        DATA_DIR = "./home/jay/airflow/dags/data/"
        categories = ["Attack", "Advocacy", "CTA", "Issue","Image"]
#         categories = ["Attack", "Advocacy", "Ceremonial", "CTA", "CI", "Image", "Issue"]
        for Category in categories:
            print(Category)
            TASK_NAME = Data+Category
            BERT_MODEL =  TASK_NAME+'.tar.gz'

            # The output directory where the fine-tuned model and checkpoints will be written.
            OUTPUT_DIR = './home/jay/airflow/dags/outputs/'+TASK_NAME+'/'
            tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIR + 'vocab.txt', do_lower_case=False)
            processor = BinaryClassificationProcessor()
            eval_examples = processor.get_dev_examples(DATA_DIR)
            label_list = processor.get_labels() # [0, 1] for binary classification
            num_labels = len(label_list)
            eval_examples_len = len(eval_examples)


            label_map = {label: i for i, label in enumerate(label_list)}
            eval_examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in eval_examples]

            process_count = cpu_count() - 1
#             if __name__ ==  '__main__':
        #         print('Preparing to convert' {eval_examples_len} examples..')
        #         print(f'Spawning {process_count} processes..')
            with Pool(process_count) as p:
                eval_features = list(p.imap(convert_example_to_feature, eval_examples_for_processing))

            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)


            # Load pre-trained model (weights)
            model = BertForSequenceClassification.from_pretrained(CACHE_DIR + BERT_MODEL, cache_dir=CACHE_DIR, num_labels=len(label_list))



            model.to(device)

            model.eval()
            eval_loss = 0
            nb_eval_steps = 0
            preds = []

            for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, labels=None)

                # create eval loss and other metric required by the task

                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds[0] = np.append(
                        preds[0], logits.detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps
            preds = preds[0]
            preds = np.argmax(preds, axis=1)
            df[Category]=preds

        del df['label']

        dc=df.to_dict('records')

        for doc in dc:
            doc['class']=[]
            for c in categories:
                if doc[c]==1:
                    doc['class'].append(c)
                del doc[c]
        print(len(dc))
        print(dc[0])
        print("Pushing into DB")
        for doc in dc:
            for x in col.find({"ad_creative_body":doc['ad_creative_body'],'marked':0}):
                x['marked']=1
                x['class']=doc['class']
                col.update_one({'_id': x['_id']},{"$set":x},True)    
        return "Done"

def predict_civility(ds, **kwargs):
    global db
    print(os.getcwd())
    ls=[]
    while True:
        data=[]
        for doc in col.distinct('ad_creative_body',{'civility_class':None}):
            if len(data)<10000:
                data.append({'ad_creative_body':doc})
            else:
                break           

        if len(data)>0:
            print(len(data))
            df=pd.DataFrame(data)
            df['label']=0
            dev_df_bert = pd.DataFrame({
                'id':range(len(df)),
                'label':df['label'],
                'alpha':['a']*df.shape[0],
                'text': df['ad_creative_body'].replace(r'\n', ' ', regex=True)
            })

            dev_df_bert.to_csv('./home/jay/airflow/dags/data/dev.tsv', sep='\t', index=False, header=False)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # This is where BERT will look for pre-trained models to load parameters from.
            CACHE_DIR = './home/jay/airflow/dags/cache/'

            # The maximum total input sequence length after WordPiece tokenization.
            # Sequences longer than this will be truncated, and sequences shorter than this will be padded.
            MAX_SEQ_LENGTH = 128

            TRAIN_BATCH_SIZE = 24
            EVAL_BATCH_SIZE = 8
            LEARNING_RATE = 2e-5
            RANDOM_SEED = 42
            GRADIENT_ACCUMULATION_STEPS = 1
            WARMUP_PROPORTION = 0.1
            OUTPUT_MODE = 'classification'
            NUM_TRAIN_EPOCHS = 1
            CONFIG_NAME = "config.json"
            WEIGHTS_NAME = "pytorch_model.bin"
            Data = 'FBAds'
            DATA_DIR = "./home/jay/airflow/dags/data/"
            categories = ["Uncivil"]
    #         categories = ["Attack", "Advocacy", "Ceremonial", "CTA", "CI", "Image", "Issue"]
            for Category in categories:
                print(Category)
                TASK_NAME = Data+Category
                BERT_MODEL =  TASK_NAME+'.tar.gz'

                # The output directory where the fine-tuned model and checkpoints will be written.
                OUTPUT_DIR = './home/jay/airflow/dags/outputs/'+TASK_NAME+'/'
                tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIR + 'vocab.txt', do_lower_case=False)
                processor = BinaryClassificationProcessor()
                eval_examples = processor.get_dev_examples(DATA_DIR)
                label_list = processor.get_labels() # [0, 1] for binary classification
                num_labels = len(label_list)
                eval_examples_len = len(eval_examples)


                label_map = {label: i for i, label in enumerate(label_list)}
                eval_examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in eval_examples]

                process_count = cpu_count() - 1
    #             if __name__ ==  '__main__':
            #         print('Preparing to convert' {eval_examples_len} examples..')
            #         print(f'Spawning {process_count} processes..')
                with Pool(process_count) as p:
                    eval_features = list(p.imap(convert_example_to_feature, eval_examples_for_processing))

                all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

                # Run prediction for full data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)


                # Load pre-trained model (weights)
                model = BertForSequenceClassification.from_pretrained(CACHE_DIR + BERT_MODEL, cache_dir=CACHE_DIR, num_labels=len(label_list))
                print(label_list)


                model.to(device)

                model.eval()
                eval_loss = 0
                nb_eval_steps = 0
                preds = []

                for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        logits = model(input_ids, segment_ids, input_mask, labels=None)

                    # create eval loss and other metric required by the task

                    loss_fct = CrossEntropyLoss()
                    tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

                    eval_loss += tmp_eval_loss.mean().item()
                    nb_eval_steps += 1
                    if len(preds) == 0:
                        preds.append(logits.detach().cpu().numpy())
                    else:
                        preds[0] = np.append(
                            preds[0], logits.detach().cpu().numpy(), axis=0)

                eval_loss = eval_loss / nb_eval_steps
                preds = preds[0]
                preds = np.argmax(preds, axis=1)
                df[Category]=preds

            del df['label']

            dc=df.to_dict('records')

            for doc in dc:
                doc['civility_class']=[]
                for c in categories:
                    if doc[c]==1:
                        doc['civility_class'].append('uncivil')
                    else:
                        doc['civility_class'].append('civil')
                    del doc[c]
            print(len(dc))
            print(dc[0])
            print("Pushing into DB")
            ct=0
            for doc in dc:
                ct+=1
                print(ct)
                col.update_many({"ad_creative_body":doc['ad_creative_body'],'civility_class':None},{'$set':{'civility_class':doc['civility_class']}},True)
            del data
        else:
            break
    return "Done"
marking_step = PythonOperator(
    task_id='predict',
    provide_context=True,
    python_callable=predict,
    dag=dag,
)

marking_civility_step = PythonOperator(
    task_id='predict_civility',
    provide_context=True,
    python_callable=predict_civility,
    dag=dag,
)



def predict_topic(ds, **kwargs):

    # collecting ads from fb_ads_dump db and appending it to a df
    data=[]
    for doc in col.distinct('ad_creative_body',{'marked_topic':0}):
        if len(data)<2000:
            data.append({'ad_creative_body':doc})
        else:
            break           

    if len(data)>0:
        df1=pd.DataFrame(data)



    # rename column in df
    df1.rename(columns={'ad_creative_body':'text'}, inplace=True)

    # creating lexicon cues all the 12 different topics
    dic1 = {'covid': ['caronavirus',
      'bed capacity',
      'see collapse economy',
      'panicbuying',
      'evacuee',
      'social distancing',
      'panicshop',
      'ncov',
      'protect working people',
      'world health organization',
      'ppe',
      'mers',
      'sflockdown',
      'national testing strategy',
      'china virus',
      'distancing',
      'ventilators',
      '14dayquarantine',
      'quarantinelife',
      'chloroquine',
      'chinesevirus',
      'masks',
      'quarantine',
      'ebola',
      'protective equipment',
      'panic shopping',
      'duringmy14dayquarantine',
      'canceleverything',
      'flatten the curve',
      'stay at home',
      'pandemic',
      'wet markets',
      'covidiot',
      'panic shop',
      'shelteringinplace',
      'inmyquarantinesurvivalkit',
      'kungflu',
      'koronavirus',
      'chinavirus',
      'liberate',
      'isolating',
      'ppeshortage',
      'stay home',
      'going see collapse',
      'coronavirus',
      'corona virus',
      'virus',
      'remdesivir',
      'panic buy',
      'corona',
      'sheltering in place',
      'disinfect',
      'flu',
      'n95',
      'quarentinelife',
      'trumppandemic',
      'covid19',
      'stayhomechallenge',
      'screening',
      'nonessential',
      'disinfectants',
      'infuenza',
      'coronavirus pandemic',
      'outbreak',
      'trump pandemic',
      'stayhome',
      'infected patients',
      'restrict travel',
      'vaccine',
      'sinophobia',
      'pandemie',
      'wuhancoronavirus',
      'getmeppe',
      'coronapocalypse',
      'socialdistancing',
      'coronakindness',
      'socialdistancingnow',
      'sars',
      'sanitize',
      'staysafestayhome',
      'stayathome',
      'sars cov 2',
      'sanitizer',
      'stay safe stay home',
      'wuhan',
      'chinese virus',
      'paycheck protection',
      'covidー19',
      'influenza',
      'coronials',
      'sanitiser',
      'stay home challenge',
      'isolation',
      'panic buying',
      'defense production act',
      'covd',
      'paycheckprotectionprogram',
      'dontbeaspreader',
      'saferathome',
      'epitwitter',
      'virtual town hall',
      'wuhanlockdown',
      'medical supplies',
      'cdc',
      'invoke',
      'lock down',
      'flattenthecurve',
      'hydroxychloroquine',
      'panicbuy',
      'self isolate',
      'covid',
      'covid 19',
      'testing'],
     'economic': ['declining income',
      'increasing income',
      'subprime mortgage',
      'cutting spending',
      'hour wage',
      'deregulate',
      'million decent paying',
      'fund billionaires',
      'huge financial',
      'hourly job',
      'overtime pay',
      'interest deduction',
      'talking corporate',
      'capital requirements',
      'corporate america',
      'overtime work',
      'jobs created',
      'shadow banking',
      'creating jobs',
      'stocks',
      'business owner',
      'flat rate',
      'save jobs',
      'tax rate',
      'banker',
      'top 1%',
      'mergers',
      'manufacturing nation',
      'transportation',
      'creates jobs',
      'vocational training',
      'getting poorer',
      'deregulation',
      'wealth tax',
      'auto industry',
      'flat tax plan',
      'ordinary income',
      'taxed',
      'combined business',
      'bailouts',
      'believe capital',
      'billion surplus',
      'stronger job',
      'economists supporting',
      'multinationals',
      'enterprise system',
      'corporate power',
      'income scale',
      'regulate swaps',
      'taxing',
      'job creation',
      'capital gains',
      'dodd frank',
      'corporate loopholes',
      'manufacturing sector',
      'american products',
      'discretionary spending',
      'bankruptcy protection',
      'zero based budgeting',
      'market basket',
      'subprime mortgages',
      'capitalism',
      'i.r.s',
      'merge investor',
      'protection financial',
      'pro growth',
      'debt free',
      'gilmore growth code',
      'large corporations',
      'fund managers',
      'largest financial',
      'protect consumers',
      'economy isn t',
      'exemptions',
      'market segment',
      'super rich',
      'higher wage',
      'paying jobs',
      'percent income',
      'manufacturer',
      'pro worker',
      'jobs right',
      'wage right',
      'stagnation',
      'vat',
      'manufacturing plan',
      'median wage',
      'federal reserve',
      'street banks',
      'create growth',
      'made in america',
      'women workers',
      'income inequality',
      'interest deductions',
      'inflation',
      'deregulated',
      'fairtax',
      'grow jobs',
      'debt ceiling',
      'consumers',
      'banking guys f1is',
      'american companies',
      'economists',
      'lose jobs',
      'buy american',
      'corporation',
      'highest earning people',
      'spending cuts',
      'net result',
      'treasurer',
      'alan greenspan',
      'revenue neutral',
      'low wage',
      'corporate inversions',
      'foreign goods',
      'job growth',
      'job creating',
      'home mortgage',
      'stagnant',
      'bondholders',
      'fight for 15',
      'recession',
      'raising incomes',
      'middle income',
      'taxes unless',
      'taxation',
      'tax code',
      'jobs lost',
      'financial protection',
      'swaps',
      'collar',
      'barney frank',
      'capitalist',
      'gotten tax',
      'tax system',
      'disposable income',
      'exemption',
      'credit rating',
      'regulatory reform',
      'incomes',
      'growth code',
      'deregulate swaps',
      'tax capital',
      'monopolistic',
      'mortgages',
      'youth unemployment',
      'important jobs',
      'stagnated',
      'jobs open',
      'u.s. companies',
      'undertaxed',
      'mega bank',
      'moving jobs',
      'business opportunity',
      'keeping jobs',
      'upward mobility',
      'offshore',
      'exporting products',
      'growth rate',
      'fair tax',
      'six financial',
      'foreclosing',
      'hewlett packard',
      'index fell',
      'highest taxed',
      'company bought',
      'regressive',
      'wage raised',
      'productivity',
      'over taxing',
      'finance world',
      'massive redistribution',
      'derivatives',
      'budget deal',
      'job killers',
      'overtime',
      'favored banks',
      'fiscally',
      'food stamps',
      'deficit',
      'regulatory environment',
      'financial murder',
      'foreclosure rate',
      'decent paying jobs',
      'businessman',
      'create jobs',
      'workers earn',
      'lower tax',
      'jobs overseas',
      'bailout money',
      'spur economic',
      'job creators',
      'stimulus package',
      'foreclosure',
      'corporatist',
      'build jobs',
      'exempts',
      'alan krueger',
      'teacher pay',
      'growing jobs',
      'bring jobs',
      'bailout',
      'tom perkins',
      'internal revenue',
      'automakers',
      'entrepreneurial',
      'commodity',
      'supplier companies',
      'corp profits',
      'lower income',
      'pay equity',
      'automobile workers',
      'bernanke',
      'lower wages',
      'raise the wage',
      'industry jobs',
      'level playing',
      'payrolls',
      'savings',
      'commodities future',
      'modern day glass ',
      'irs',
      'overtaxed',
      'already overburdened',
      'raise incomes',
      'off shore',
      'created jobs',
      'vat tax',
      'bond rating',
      'income gap',
      'market rate',
      'hank paulson',
      'dividends',
      'percent poverty',
      'personal property taxes',
      'government spending',
      'give tax',
      'low income parents',
      'federal labor law',
      'requiring capital',
      'fed',
      'job loss',
      'state local',
      'paying taxes',
      'free enterprise',
      'antitrust',
      'mortgage companies',
      'pensions',
      'insolvency',
      'billionaire hedge',
      'budget committee',
      'business around',
      'small business',
      'exporting jobs',
      'bank posed',
      'supported corporate',
      'corporate inversion',
      'tax laws',
      'made banks',
      'economic ones',
      'growth agenda',
      'living wage',
      'personal income',
      'hedge fund',
      'bankrupting',
      'glass steagall',
      'deduction',
      'monopolistic practices',
      'employing people',
      'budgeting',
      'fair market',
      'futures trading',
      'treasury secretaries',
      'workplace protections',
      'lender',
      'business channel',
      'labor law',
      'devaluing',
      'american worker',
      'oil prices',
      'adjusting monetary',
      'entrepreneurial nation',
      'tax plan',
      'rich guys',
      'wall street ceos',
      'jobs come',
      'êthe glass steagall',
      'exploit poor',
      'obama clinton economy',
      'point economic',
      'economic imperative',
      'low wages',
      'aaa bond',
      'reinstating glass ',
      'low tax',
      'kill jobs',
      'wage increase',
      'mfg day',
      'growth strategy',
      'million jobs',
      'bring manufacturing',
      'preferred income',
      'mortgage interest',
      'glass stegall',
      'billion deficit',
      'street economy',
      'family income',
      'american household',
      'value added tax',
      'enterprise',
      'wages rise',
      'higher income',
      'surplus',
      'accountants',
      'stop spending',
      'suppress wages',
      'business model',
      'currencies',
      'less money',
      'subsidies',
      'consumer protection',
      'currency',
      'depositors',
      'discretionary',
      'tax free',
      'cash flow',
      'import bank',
      'surpluses',
      'employee',
      'monetary',
      'refinance',
      'deficits',
      'countrywide mortgage',
      'increase incomes',
      'state employees',
      'credit records',
      'american corporations',
      'plane manufacturers',
      'job exporting',
      'financial markets',
      'bretton woods',
      'taxpayers',
      'century glass steagall',
      'job market',
      'large financial',
      'poverty level',
      'verizon workers',
      'household income',
      'tax reform',
      'private sector',
      'automobile industry',
      'g.d.p',
      'after tax',
      'auto supplier',
      'cut spending',
      'devaluations',
      'anti monopoly',
      'larry summers',
      'making money',
      'federal minimum',
      'business leaders',
      'american workers',
      'banking',
      'banks',
      'ceos',
      'economic',
      'economy',
      'equal pay',
      'jobs',
      'manufacturing',
      'middle class',
      'millionaires',
      'prosperity',
      'shareholders',
      'stock',
      'tax',
      'taxes',
      'wages',
      'wage',
      'wealth',
      'workers'],
     'education': ['tutor',
      'debt financed education',
      'english language learners',
      'minority serving institutions',
      'students low income',
      'school climate',
      'borrowers',
      'pre school',
      'student loan borrowers',
      'special education',
      'hbcus',
      'school infrastructure',
      'cancel student loan debt',
      'students color',
      'default',
      'early education',
      'public education',
      'immigrant students',
      'k 12 education',
      'student debt',
      'students schools',
      'supports teachers',
      'school service personnel',
      'graduation',
      'free college tuition',
      'lgbtq students',
      'education',
      'loans',
      'public schools',
      'school',
      'student',
      'teachers'],
     'environment': ['renewables matter',
      'tides rising',
      'solar energy',
      'climate crisis',
      'alternative sources',
      'rising ocean',
      'ending fracking',
      'tested lead',
      'putting co2',
      'lee anne walters',
      'developing oil',
      'refineries',
      'using solar',
      'drought',
      'wind producing',
      'floods',
      'arctic oil drilling',
      'waste water',
      'defeat climate',
      'cheaper energy',
      'hydropower',
      'gases',
      'lower carbon',
      'expand fracking',
      'delivering water',
      'combat greenhouse',
      'fracking technology',
      'light bulbs',
      'heat trapping',
      'sustainable cities',
      'green new',
      'tap water',
      'rfs',
      'nuclear power',
      'coal fields',
      'scientific community',
      'national energy',
      'electric energy',
      'pro offshore drilling',
      'dig coal',
      'lower energy',
      'fewer greenhouse',
      'renewables',
      'alternative energy',
      'environmental protection',
      'supporting flint',
      'promote fracking',
      'sea levels',
      'planetary',
      'pro offshore',
      'temperatures',
      'water source',
      'solar power',
      'higher sea',
      'releases heat trapping',
      'wind',
      'epa',
      'frack',
      'solar panels',
      'power plant',
      'environmental legislation',
      'building coal',
      'energy independence',
      'pro coal',
      'clean power',
      'energy system',
      'pipes delivering',
      'climatedebate',
      'polluted',
      'carbon producer',
      'energy plan',
      'energy affordable',
      'carbon emissions',
      'water rates',
      'environmental committee',
      'climate changing',
      'man made problem',
      'global environment',
      'weather disturbances',
      'lead leaked',
      'million solar',
      'greatest polluters',
      'clean air',
      'earth warmer',
      'rising sea',
      'green buildings',
      'enable ethanol',
      'remove lead',
      'replace coal',
      'largest carbon',
      'coal power',
      'ozone layer',
      'environmental crisis',
      'clean environment',
      'support fracking',
      'paris agreement',
      'cut carbon',
      'coal plants',
      'crude oil',
      'degrees warmer',
      'ethanol mandate',
      'produce solar',
      'refineries blend',
      'tremendous energy',
      'sustainable energy',
      'environmental pollution',
      'thorium',
      'water corroded',
      'man aggravated',
      'public water',
      'blend biofuels',
      'energy independent',
      'causing seas',
      'fuel companies',
      'change watch',
      'energy sector',
      'epa director',
      'crude oil export ban',
      'carbon',
      'fracking',
      'natural gas',
      'greenhouse gas',
      'energy costs',
      'conservation',
      'save planet',
      'gas mileage',
      'develop renewables',
      'green building',
      'water systems',
      'extracting natural',
      'energy supply',
      'reactors',
      'global environmental',
      'fracks',
      'flint journal',
      'geothermal',
      'fuel standard',
      'flint waterworks',
      'polluting',
      'methane',
      'environment clean',
      'destroyed flint',
      'drilling',
      'oil production',
      'cut emissions',
      'energy grid',
      'reduced emissions',
      'ozone',
      'fuel standards',
      'keystone xl',
      'clean electric',
      'solyndra',
      'polluted cities',
      'energy production',
      'hydro',
      'environment clean up',
      'green jobs',
      'gas drilling',
      'gas prices',
      'power plants',
      'poison water',
      'environmental stringent',
      'thorium reactors',
      'gas emissions',
      'natural resources',
      'oppose ethanol',
      'water plants',
      'energy sources',
      'service pipes',
      'fighting human caused',
      'river water',
      'renewable fuel',
      'heat trapping gases',
      'flint trust',
      'environmental policy',
      'polluters',
      'energy self sufficiency',
      'energy efficiency',
      'water problems',
      'green new deal',
      'water system',
      'extreme weather',
      'fuels releases',
      'mandatory ethanol',
      'keystonexl',
      'supported fracking',
      'devastating flooding',
      'atmosphere',
      'pipes',
      'sustainable technologies',
      'wind energy',
      'lead paint',
      'biofuels',
      'solar rooftops',
      'ethanol',
      'clean renewable',
      'fight climate',
      'hydroelectric power',
      'alternate energy',
      'renewable',
      'residential pipes',
      'bryn mickle',
      'flint river',
      'human caused climate',
      'green efforts',
      'sea level',
      'pro keystone pipeline',
      'domestic fracking',
      'strong environmental',
      'hydroelectric',
      'greenhouse',
      'energy problems',
      'arctic',
      'fossil fuels',
      'ocean levels',
      'rebuilding flint',
      'energy source',
      'ethanol standard',
      'pro keystone',
      'renewable energy',
      'co2',
      'climate',
      'climate change',
      'climatechange',
      'emissions',
      'environment',
      'fossil',
      'fossil fuel',
      'fuel',
      'new deal'],
     'foreign policy': ['south koreans',
      'international atomic',
      'send diplomats',
      'bashar assad',
      'terrorists',
      'sunni arabs',
      'radical jihadis',
      'palestinian',
      'terrorist act',
      'bds',
      'israel left',
      'states army',
      'servicemen',
      'bernardino',
      'hagel',
      'syrianrefugees',
      'created barbaric',
      'guerrilla',
      'destroy radical',
      'u.s. relationship',
      'radicalislam',
      'muslim leaders',
      'embassy personnel',
      'sunnis',
      'missile technology',
      'civil wars',
      'arabs',
      'salvador aliende',
      'libyans turned',
      'advocated arming',
      'tiananmen square',
      'jordanian king',
      'stop tpp',
      'sending u.s.',
      'anti israeli',
      'sandinista government',
      'abdullah',
      'brits',
      'jihadism',
      'shias',
      'expansionist',
      'secretary kerry',
      'forces using',
      'aliende',
      'qassem',
      'fighting isis',
      'trade agreement',
      'bashar al assad',
      'kosovo',
      'muslim world',
      'ground combat',
      'middle eastern',
      'additional troops',
      'defense operations',
      'persian gulf',
      'muslim countries',
      'u.s. combat',
      'state department',
      'carpet bombing',
      'carpet bomb',
      'alawites',
      'intelligence indicating',
      'south asia',
      'tankers',
      'islamic world',
      'larger force',
      'soviet union',
      'north korean',
      'no fly zones',
      'ballistic',
      'radical clerics',
      'europe',
      'apocalyptic vision',
      'attacking russian',
      'benghai',
      'inviting iranian',
      'service officers',
      'wealthy countries',
      'isolated country',
      'inside iraq',
      'finland',
      'combatant',
      'european allies',
      'china sea',
      'militias',
      'diplomats',
      'little force',
      'kurd',
      'resurgence',
      'wartime',
      'khmer rouge',
      'lebanon',
      'enemies',
      'alliances',
      'murderous dictator',
      'u.n. security',
      'churchill',
      'anti israeli policy',
      'iranian troops',
      'overthrow assad',
      'nation states',
      'pro israel',
      'castros',
      'fight assad',
      'counterterrorism',
      'global power',
      'dirty bombs',
      'iranian deal',
      'isil',
      'benghazi',
      'security strategy',
      'sectarian',
      'threats facing',
      'committing massacres',
      'warfare within',
      'european union',
      'share intelligence',
      'emir',
      'rockets',
      'dabiq',
      'across russia',
      'september 11th',
      'u.k.',
      'terrorist attacks',
      'unilateral action',
      'gilmore commission',
      'medvedev',
      'deny isis',
      'save saeed',
      'strong anti american',
      'arab movement',
      'islam',
      '250,000 plus dead',
      'kingdom',
      'brutally executed',
      'sixth fleet',
      'shi a sunni',
      'hezbollah',
      'sectarian violence',
      'diplomatically',
      'saudi arabia',
      'american interests',
      'encampment',
      'defeating jihadism',
      'interrogation methods',
      'el sisi',
      'ash carter',
      'toughest sanctions',
      'significant threat',
      'long range rocket',
      'military strategy',
      'mass destruction',
      'reduce nuclear',
      'allah',
      'american boots',
      'ayatollah khomeini',
      'rebel forces',
      'bomb isis',
      'moammar qadafi',
      'countries across',
      'destroy isis',
      'powell',
      'aleppo',
      'egyptians',
      'vigilant',
      'globalization',
      'deploy u.s.',
      'minister barat',
      'terrorist organization',
      'barrel bombs',
      'regime changes',
      'overthrow saddam',
      'two terrorists',
      'two world',
      'western democracy',
      'kamatsu',
      'bigger war',
      'safe zones',
      'destabilizing areas',
      'protect syrians',
      'arab nations',
      'afghan army',
      'iraqi army',
      'democracy promoter',
      'seoul',
      'kuwaitis',
      'judea',
      'moderate rebels',
      'two state solution',
      'protect eastern',
      'american allied',
      'irandeal',
      'refugee crisis',
      'shiites',
      'edward snowden',
      'religious war',
      'rockets raining',
      'mexican government',
      'al assad',
      'san bernardino',
      'strong commander in chief',
      'jihadists',
      'nations around',
      'sinai',
      'ashraf',
      'extreme terrorism',
      'iranian government',
      'israel invites',
      'weapon system',
      'american led air',
      'inquisitions',
      'national intelligence',
      'arabians',
      'american servicemen',
      'tehran',
      'banning muslims',
      'operations forces',
      'sirte',
      'sunni led',
      'radical islamist',
      'james foleys',
      'muslim nations',
      'bombed assad',
      'annexed',
      'israels',
      'alawite',
      'air space',
      'sunni tribes',
      'jihadist',
      'iraqi military',
      'somalia',
      'libya moving',
      'ground force',
      'u n sanctions',
      'barat',
      'bombers',
      'nafta',
      'ebola',
      'recruit americans',
      'territory',
      'vladmir putin',
      'islamic radical',
      'jihadis',
      'defense committee',
      'isolationists',
      'diplomatic relations',
      'aggressive actions',
      'american help',
      'leader kim',
      'bahrainis',
      'american planes',
      'single dictator',
      'king abdullah',
      'khmer',
      'secretary rumsfeld',
      'geneva convention',
      'pursue justice',
      'lone wolves',
      'gadhafi',
      'kurdish forces',
      'saddam hussein',
      'difficult world',
      'geopolitical',
      'arabs fight',
      'nuclear weapon',
      'airstrikes',
      'islamic terrorism',
      'radical shia',
      'exiles regarding',
      'military commanders',
      'iranian general',
      'provided weapons',
      'topple assad',
      'crippling sanctions',
      'defense officials',
      'submarines',
      'another war',
      'exportation',
      'khomeini',
      'radical islam',
      'no fly zone',
      'russian people',
      'remembering iran',
      'american combat',
      'israeli government',
      'public servant america',
      'security council',
      'trade agreements',
      'troops inside',
      'economic sanctions',
      'arab countries',
      'alawite shias',
      'kashif',
      'worst foreign',
      'incoming rockets',
      'iranians getting',
      'fight isis',
      'anti isis',
      'continued threat',
      'nicaraguan',
      'americans murdered',
      'james foley',
      'hydrogen bomb',
      'nato support',
      'dangerous jihadist',
      'policy doctrine',
      'regimes',
      'percent tariff',
      'isis grow',
      'world religion',
      'invading',
      'tariffed',
      'missile capability',
      'rumsfeld',
      'cairo',
      'free trader',
      'fly zone',
      'putin right',
      'ramallah',
      'negotiate khamenei',
      'american principles',
      'allende',
      'isis using',
      'mass genocide',
      'drones',
      'arms deal',
      'build coalitions',
      'kim jong un',
      'prevents iran',
      'iranian aggression',
      'israeli',
      'palestinian authority',
      'smallest navy',
      'radical sunni',
      'rallying defense',
      'assad regime',
      'cuban americans',
      'militaristic',
      'bergdahl',
      'letting china',
      'air attacks',
      'venezuela',
      'defense system',
      'human intelligence',
      'khamenei',
      'syrian people',
      'invaded',
      'proliferation',
      'humanitarian corridors',
      'reliable ally',
      'encrypted communication',
      'baltics',
      'cia',
      'nuclear agreement',
      'armed services',
      'koreans',
      'cuba stops',
      'military response',
      'qaida',
      'space based missile',
      'turkish',
      'broad coalitions',
      'free amir',
      'kurdish fighters',
      'bombing american',
      'american action',
      'military facility',
      'sivakumar',
      'air power',
      'baltic countries',
      'security issues',
      'guantanamo',
      'geneva',
      'u.s. involvement',
      'strongest allies',
      'intelligence agent',
      'nuclear material',
      'paranoid country',
      'paris attacks',
      'combat troops',
      'strategic mistake',
      'war made',
      'geopolitical challenge',
      'defeated militarily',
      'islamists',
      'soviets',
      'multinational trade',
      'al nusra',
      'massacres',
      'former defense',
      'libyans',
      'theocracy',
      'sectarianized',
      'al sisi',
      'chinese government',
      'arabia',
      'islamic terrorist',
      'mentioned russia',
      'military operational',
      'jordanian',
      'italy',
      'israeli officials',
      'al maliki',
      'baltic states',
      'planes bombing',
      'million refugees',
      'arab spring',
      'commander in chief question',
      'pay ransom',
      'saudis',
      'military intervention',
      'troop level',
      'eastern ukraine',
      'belgium',
      'air flights',
      'terrorists taking',
      'henry kissinger',
      'drop barrel',
      'trade issues',
      'burkina faso',
      'chile',
      'secretary carter',
      'american blood',
      'coalition abroad',
      'give arms',
      'attack coming',
      'international affairs',
      'jihad',
      'world court',
      'nuclear deal',
      'humanitarian catastrophe',
      'american soldiers',
      'bengazni',
      'enemy combatant',
      'transatlanic',
      'security side',
      'raul castro',
      'opening cuba',
      'freeamir',
      'colin powell',
      'cambodia',
      'winston churchill',
      'radical terrorism',
      'send u.s.',
      'iran came',
      'war iran',
      'democratic syria',
      'covert cia',
      'barbaric group',
      'training centers',
      'burma',
      'emirates',
      'belligerence',
      'foreign relations',
      'special operators',
      'korean troops',
      'chinese people',
      'aggressive action',
      'radical islamic',
      'gilmorecommission',
      'marine officer',
      'afghan territory',
      'stable democracies',
      'genocidal threats',
      'cyber warfare',
      'course israel',
      'confront evil',
      'russian plane',
      'osama bin',
      'war fever',
      'affairs abroad',
      'latin america',
      'ground troops',
      'anti assad',
      'waterboarding',
      'cyber war',
      'south korea',
      'enabled states',
      'enemies fighting',
      'ghani',
      'development aid',
      'britain',
      'arafat',
      'terror network',
      'freejason',
      'kuwait',
      'unnecessary loss',
      'savesaeed',
      'cease fire',
      'deployment',
      'islamic law',
      'ally israel',
      'armies',
      'nuclear facilities',
      'war iranheld hostage',
      'conduct nuclear',
      'gadhafi regime',
      'genocidal',
      'ballistic missiles',
      'communist dictatorship',
      'terrorist foothold',
      'foreign money',
      'missile launch',
      'terrorist tool',
      'asia pacific',
      'asia pacific region',
      'targeting terrorists',
      'u.s. enemies',
      'bravest people',
      'croaker cardin',
      'bahrain',
      'israel palestine',
      'iran tomorrow',
      'sandinista',
      'isis thought',
      'dictator topples',
      'military tool',
      'foreign aid',
      'guatemala',
      'force reservist',
      'caliph',
      'aviv',
      'dictator falls',
      'civilizations',
      'offensive weapons',
      'invasions',
      'terror attacks',
      'proliferate',
      'bombing syrian',
      'legitimate caliphate',
      'ramadi',
      'jihadist group',
      'south china',
      'brutal murdering',
      'anti israel',
      'security committee',
      'jihadi',
      'atomic',
      'cuban people',
      'muslim troops',
      'small countries',
      'military exercises',
      'islamic terrorists',
      'patriot act',
      'entice syrians',
      'rouhani',
      'exports',
      'violent extremism',
      'battlefield',
      'warplanes',
      'quds forces',
      'allied fighters',
      'special operations',
      'taliban',
      'mentioned syria',
      'assad crossed',
      'defeat radical',
      'gulf war',
      'sunni allies',
      'assad',
      'u n',
      'tiananmen',
      'jordan',
      'netanyahu',
      'islamic state',
      'successful military',
      'international conflict',
      'ship rockets',
      'diplomatic solution',
      'north africa',
      'sunni cities',
      'strategic failings',
      'robert levinson',
      'libya',
      'libyan people',
      'air forces',
      'mossadegh',
      'damascus',
      'overthrow dictators',
      'restore diplomatic',
      'taylor force',
      'moammar gadhafi',
      'intelligence gathering',
      'terrorist groups',
      'jong un',
      'defense ash',
      'sanctions imposed',
      'overwhelming force',
      'chemical weapons',
      'international guerrilla',
      'shia cleric',
      'outsource foreign',
      'u.s. force',
      'safe havens',
      'destabilize',
      'arab country',
      'internal trade',
      'douglas macarthur',
      'destroy terrorism',
      'indonesia',
      'iraq war',
      'east crisis',
      'quds',
      'embassy',
      'current military',
      'london',
      'coalition building',
      'dictator named',
      'reimpose sanctions',
      'export',
      'comey says',
      'bombing',
      'poor syrians',
      'hussein',
      'libya move',
      'australians',
      'robert gates',
      'turks',
      'iranian revolution',
      'russian airliners',
      'turkey shot',
      'interrogation techniques',
      'north korea',
      'lethal weapons',
      'crimea',
      'prince sihanouk',
      'missiles flying',
      'sihanouk',
      'defend europe',
      'iran deal',
      'isis threat',
      'limited arms',
      'gaza area',
      'deploying',
      'destroying isis',
      'estonian',
      'nuclear power',
      'genocide',
      'arm syrian',
      'enhanced interrogation',
      'nato allies',
      'worst genocides',
      'russian planes',
      'u.s. ground',
      'north koreans',
      'colombia',
      'ceasefire',
      'instability',
      'b 52',
      'member countries',
      'vietnam era',
      'ruthless enemy',
      'military expert',
      'place missile',
      'cross border enforcement',
      'arm directly',
      'qaeda',
      'security threat',
      'american sailors',
      'foreign affairs',
      'fighting assad',
      'u.s. embassy',
      'failed foreign',
      'arabs standing',
      'dissidents',
      'prevent attacks',
      'air coalition',
      'tanzania',
      'ayatollah',
      'religious nazis',
      'entered world',
      'sunni shia',
      'create isis',
      'renewed airstrikes',
      'europeans',
      'arab boots',
      'destabilized',
      'sunni',
      'humanitarian relief',
      'u.s. forces',
      'u.s. trained',
      'trade deficit',
      'command center',
      'syrians',
      'security failing',
      'wage jihad',
      'sponsoring terrorism',
      'terrorist group',
      'american leadership',
      'salinas',
      'include arabs',
      'sunni world',
      'anwar',
      'sunni arab forces',
      'saturation bombing',
      'making militarily',
      'bangladesh',
      'military experts',
      'tel aviv',
      'strongholds inside',
      'nuclear treaty',
      'no nukes for iran',
      'missiles',
      'american neighbors',
      'b 52s',
      'another country',
      'missile defense',
      'sunni arab',
      'world trade',
      'israeli cabinet',
      'bring peace',
      'training camps',
      'provide security',
      'nicaragua',
      'estonia',
      'rebel',
      'palestinian people',
      'golan',
      'jerusalem',
      'invade cuba',
      'strikes',
      'al awlaki',
      'impose sanctions',
      'ambushes',
      'terror group',
      'theocratic',
      'regional army',
      'strongholds',
      'evade u.n.',
      'sudan',
      'arab leaders',
      'entire region',
      'ballistic missile',
      'toward diplomatic',
      'crush isis',
      'minister netanyahu',
      'special ops',
      'bombings',
      'hamas',
      'russian air',
      'jewish state',
      'fidel castro',
      'cufi',
      'arab allies',
      'service man',
      'reykjavik',
      'greatest military',
      'embargo',
      'sunni fighters',
      'trusting iran',
      'sunni activities',
      'belgians',
      'hamass',
      'islamic countries',
      'attacks abroad',
      'qadafi',
      'innocent civilians',
      'war began',
      'sunnis represented',
      'free jason',
      'intelligence briefings',
      'muslim allies',
      'dozen countries',
      'isis underestimated',
      'iran released',
      'sunni forces',
      'naval presence',
      'chad',
      'assad tomorrow',
      'dislodge isis',
      'supporting troops',
      'retaliate',
      'muslim ban',
      'u.s. troops',
      'american embassy',
      'benefit isis',
      'preemptive strike',
      'plane bombing',
      'isis infrastructure',
      'lithuania',
      'syrian refugees',
      'promote democracy',
      'emirati',
      'risk war',
      'robertlevinson',
      'embassies',
      'salvador allende',
      'used chemical',
      'muslims bad',
      'warning signal',
      'iran nuclear',
      'toppled gadhafi',
      'poland right',
      'military adventurism',
      'perpetual warfare',
      'russians',
      'thaad',
      'recruiting center',
      'rebels',
      'save assad',
      'important counterterrorism',
      'libyan',
      'yazidi genocide',
      'non israelis',
      'mediterranean',
      'havana',
      'general keane',
      'cafta',
      'rockets shipped',
      'crusades',
      'foment terror',
      'syrian fighters',
      'little island',
      'terrorist networks',
      'daniel ortega',
      'ataturk',
      'stabilize somalia',
      'military attention',
      'destabilizing governments',
      'isis capital',
      'radical jihadists',
      'destroying isil',
      'air campaign',
      'diplomatic intelligence',
      'caliphate',
      'ayatollah khamenei',
      'diplomat',
      'korean leader',
      'kenya',
      'ground forces',
      'iraq invasion',
      'confront russia',
      'bombardment',
      'isis territory',
      'arab world',
      'israelis',
      'kissinger',
      'persian',
      'moderate opposition',
      'foreign funding',
      'secretary colin',
      'apocalyptic version',
      'toppling gadhafi',
      'ground fighting',
      'enforcement consequence',
      'islam hates',
      'fighting isil',
      'shah',
      'egyptian',
      'american troops',
      'muslims coming',
      'syrian',
      'international terror',
      'beheaded',
      'jordanians',
      'u.n. resolutions',
      'tackle isis',
      'greatest military operational',
      'army since',
      'egypt',
      'fighting radical',
      'smallest army',
      'military terms',
      'trade center',
      'guerrilla war',
      'rockets within',
      'palestinians',
      'ukrainians',
      'current trade',
      'tribal sheiks',
      'shiite world',
      'u s intelligence',
      'beirut',
      'kill americans',
      'djibouti',
      'qatar',
      'tactical decision',
      'insurgents',
      'general soleimani',
      'diplomacy',
      'battle tested',
      'raqqa',
      'suleimani',
      'saddam',
      'arab',
      'sheiks',
      'constant threat',
      'sweden',
      'obama foreign',
      'usaid',
      'arab partners',
      'move cuba',
      'defense program',
      'general suleimani',
      'fighting force',
      'america boots',
      'israel',
      'bayonets',
      'eastern europe',
      'gaddafi',
      'kim jong',
      'islamist',
      'international coalition',
      'radical islamists',
      'u.s. consulate',
      'special forces',
      'soviet',
      'peace solution',
      'services committee',
      'weapons program',
      'sharing intelligence',
      'nuclear iran',
      'send troops',
      'kurdish',
      'bengazi',
      'air strikes',
      'aggressive military',
      'russian aggression',
      'afghan',
      'u.s. special',
      'catastrophic iranian',
      'dethrone',
      'bomber',
      'shiite',
      'military action',
      'defense burden',
      'countries combined',
      'destroy isil',
      'palestine',
      'shia',
      'militarily',
      'air strike',
      'counterterrorism issues',
      'western civilization',
      'ruthless dictator',
      'gaza',
      'strongest military',
      'range missiles',
      'tunisia',
      'c.i.a',
      'diplomatic track',
      'american ground',
      'foreign policies',
      'general lloyd',
      'aipac',
      'launched operations',
      'anti terrorism',
      'genocidal threat',
      'iranians',
      'maliki',
      'europeans offered',
      'international terrorism',
      'defeating isis',
      'commander in  chief',
      'nuclear program',
      'iranian nuclear',
      'security interests',
      'genocides',
      'newest base',
      'bin laden',
      'syrian dictator',
      'al qaeda',
      'missile',
      'nation states fail',
      'cold war',
      'endless war',
      'foreign',
      'new cold war',
      'senseless',
      'usmca'],
     'governance': ['liberal appointee',
      'fundrasing goals',
      'members congress',
      'restore power people',
      'presidential vetoes',
      'out of control court',
      'nominees',
      'constitutional amendment',
      'cutting government',
      'constitutional duty',
      'limited government',
      'obstructionists',
      'federal entitlements',
      'elected prosecutor',
      'constitutional role',
      'jurist',
      'constitutional democracy',
      'unconstitutional illegal',
      'voting rights act',
      'scalia understood',
      'federal overreach',
      'elected representatives',
      'filibuster',
      'fix congress',
      'court nominations',
      'top prosecutor',
      'sign petition',
      'pro freedom',
      'states constitution',
      'fifth circuit',
      'us elections',
      'judiciary',
      'judicial philosophy',
      'justices nominated',
      'defend constitution',
      'shrink government',
      'roe wade',
      'voter suppression',
      'conservative justices',
      'scalia',
      'big government',
      'impeach the president',
      'polling center',
      'hold hearings',
      'campaign finance law',
      'court nominees',
      'expand voting rights',
      'governmental system',
      'blasey ford',
      'executive orders',
      'fundamental liberty',
      'democratic judges',
      'roe v. wade',
      'signing executive',
      'legislate',
      'appoint supreme',
      'voter turnouts',
      'abuses power',
      'current constitution',
      'find judges',
      'bold judges',
      'executive authority',
      'government power',
      'u.s. supreme',
      'dark money politics',
      'jurists',
      'picked justices',
      'judicial tyranny',
      'american law',
      'ending lobbying',
      'constitution says',
      'obstructionism',
      'diane sykes',
      'audit the fed',
      'federal lobbyists',
      'judicial conservative',
      'constitutional law',
      'anthony kennedy',
      'federal entitlement',
      'supreme court justice',
      'nominated',
      'replace justice',
      'court justices',
      'government interference',
      'due process',
      'right to vote',
      'united supreme',
      'appoint justices',
      'constitutional responsibilities',
      'tomorrow filibustering',
      'strengthening title vi',
      'power to the states',
      '10th amendment',
      'appointee',
      'political power',
      'campaignother candidates',
      'constitution',
      'principled jurists',
      'principled constitutionalists',
      'chief justice',
      'rigging elections',
      'rigged election',
      'cronyism',
      'enriching corporations',
      'second bill of rights',
      'self funding billionaire',
      'abe fortas',
      'merrick gaarland',
      'democratic judge',
      'strong judicial',
      'justice away',
      'justice scalia',
      '14th amendment',
      'congressional process',
      'nomination passes',
      'constitution tells',
      'beloved justice',
      'constitutional power',
      'liberal justice',
      'gerrymandered',
      'voting rights',
      'circuit court',
      'campaign finance reform',
      'legislator',
      'appellate judges',
      'legal history',
      'banning lobbyists',
      'legalized bribery',
      'gerrymandering',
      'expand government',
      'big government special interests',
      'house floor',
      'court judges',
      'justice roberts',
      'lifetime appointments',
      'independent counsel',
      'conservative justice',
      'unconstitutional gerrymandering',
      'pardoning',
      'william rehnquist',
      'legal basis',
      'rehnquist',
      'efficient government',
      'justice william',
      'size of government',
      'fair districts',
      'gerrymandered districts',
      'right vote',
      'constitutionalists',
      'appointing',
      'filibustering',
      'cap on contributions',
      'ballot box',
      'fourteenth amendment',
      'reform election system',
      'judge scalia',
      'constitutional right vote',
      'confirming supreme',
      'campaign finance',
      'grow government',
      'appointing supreme',
      'appellate',
      'justices',
      'civil rights act',
      'citizens united',
      'unalienable rights',
      'defeat justices',
      'impeach president',
      'federal judges',
      'fourth circuit',
      'dark moneygerrymandering',
      'nominate supreme',
      'nominations',
      'barr',
      'census',
      'federal workers',
      'impeach',
      'impeachment',
      'kavanaugh',
      'mueller report',
      'prosecute',
      'roe',
      'secretary of education',
      'shutdown',
      'supreme court'],
     'health': ['aids',
      'overdoses',
      'address obamacare',
      'reproductive justice',
      'reproductive health',
      'premiums',
      'reproductive rights',
      'suicides',
      'care costs',
      'top down obamacare',
      'kill obamacare',
      'saved obamacare',
      'medicaid expansion',
      'change obamacare',
      'trauma based care',
      'insulin',
      'repeal obamacare',
      'copays',
      'appeal obamacare',
      'affordable health care',
      'universal health care',
      'called obamacare',
      'scrap obamacare',
      'prevent death',
      'addiction',
      'affordable health',
      'victim addiction',
      'opposes obamacare',
      'drug prices',
      'disastrous obamacare',
      'deductibles',
      'social emotional learning',
      'reproductive care',
      'healthy eating',
      'healthcare',
      'food deserts',
      'health crisis',
      'suicide',
      'hiv',
      'physical wellbeing',
      'abortion access',
      'replacing obamacare',
      'obamacare took',
      'obamacare',
      'abortion',
      'big pharma',
      'health',
      'health care',
      'prescription',
      'prescription drug'],
     'immigration': ['million immigrants',
      'asylum laws',
      'immigration policies',
      'additional border',
      'pro american immigration',
      'immigrate',
      'defending dapa',
      'illegals',
      'securetheborder',
      'endfamilydetention',
      'guest worker programs',
      'e verify',
      'dapa',
      'guest worker',
      'effectively vet',
      'immigration proposal',
      'syrianrefugees',
      'exit entry system',
      'pedestrian fence',
      'deporting chief',
      'legal immigration',
      'solve immigration',
      'kate steinle',
      'undocumented people',
      'recent immigrants',
      'immigrants legal',
      'assimilation',
      'migration',
      'proven vetting',
      'legal status',
      'deportations',
      'permanently barred',
      'undocumented family',
      'become american',
      'illegal aliens',
      'rubio schumer gang',
      'vetting process',
      'employing illegal',
      'among illegals',
      'visa over stayers',
      'reunite families',
      'immigration reform',
      'refugees come',
      'rubio schumer amnesty',
      'guestworkers programs',
      'vetting mechanisms',
      'immigration system',
      'country illegally',
      'sanctuary cities',
      'illegal immigration',
      'e verify system',
      'secure border',
      'absorb syrian',
      'using h1b',
      'mexican immigrants',
      'trafficking networks',
      'dangerous immigration',
      'accept refugees',
      'syrian refugees',
      'hard working immigrants',
      'detailed immigration',
      'wire fence',
      'immigration debate',
      'entry/exit',
      'lawful immigration',
      '1200 mile border',
      'net immigration',
      'melting pot',
      'coming illegally',
      'tourist visas',
      'xenophobes',
      'visa waiver',
      'great wall',
      'hispandering',
      'entire border',
      'deportation policy',
      'undocumented',
      'divide families',
      'legalization amnesty',
      'birth certificate',
      'based borders',
      'making immigration',
      'seeking asylum',
      'crisis southern',
      'visa',
      'american citizenship',
      'citizenship question',
      'guestworker',
      'emigrate',
      'exit entry',
      'immigration makes',
      'overstayed',
      'dream act',
      'securing borders',
      'latino workers',
      'refugee camps',
      'immigration focusing',
      'broken immigration',
      'deport',
      'mexican born',
      'triple h 1bs',
      'immigrant experience',
      'break the cage',
      'immigration security',
      'guestworker provisions',
      'catch and release',
      'deportation plan',
      'cheap labor',
      'refugees fleeing',
      'mandatory e verify',
      'visa overstays',
      'accepting refugees',
      'deport children',
      'immigrants contributing',
      'kates law',
      'preserve birthright',
      'immigrants getting',
      'wants sanctuary',
      'childhood arrivals',
      'assist immigrants',
      'undocumented immigrants',
      'immigration bashing',
      'overstay',
      'pressing immigration',
      'biometric exit/entry',
      'overstays',
      'million undocumented',
      'comprehensive immigration',
      'border agents',
      '2020 census',
      'end family detention',
      'h1 b abuse',
      'immigrant labor',
      'vicente fox',
      'entry exit tracking',
      'secure the border',
      'birther movement',
      'citizenship',
      'physical border',
      'someone illegally',
      'work permit',
      'undocumented children',
      'towards citizenship',
      'bastardizing citizenship',
      'eastern refugees',
      'green cards',
      'migratory crises',
      'rubio/schumer',
      'legal immigrants',
      'immigration comes',
      'illegal immigrant',
      'citizenship cards',
      'immigrated',
      'h1b program',
      'refugee camp',
      'migratory reform',
      'immigration law',
      'record immigration',
      'permanent residents',
      'mexicans',
      'vetting people',
      'path towards',
      'permanent residency',
      'deportation policies',
      'stop deportations',
      'came legally',
      'unfreeze dapa',
      'dolores huerta',
      'vigilantes known',
      'immediately deport',
      'sanctuary',
      'refugee',
      'legalizing people',
      'deport violent',
      'hiring illegal',
      'guestworker programs',
      'syrian refugee',
      'guestworkers',
      'h 1b process',
      'immigration problem',
      'h2b',
      'illegal immigrants',
      'guest workers',
      'h 1b',
      'american immigrants',
      'tear families',
      'come illegally',
      'unfreezedapa',
      'green card',
      'children fled',
      'immigrant haters',
      'enhance border',
      'entered illegally',
      'rubio schumer',
      'h1b',
      'birthright',
      'non american',
      'deporting criminals',
      'toward citizenship',
      'increased vetting',
      'border illegally',
      'foreign workers',
      'implement immigration',
      'crisis southern border',
      'naturalized',
      'hardworking immigrants',
      'visa program',
      'foreigners',
      'entering legally',
      'detentions',
      'vetting',
      'keep illegals',
      'breaking up families',
      'normal refugee',
      'border control',
      'anti immigrant',
      'allowing refugees',
      'birthright citizenship',
      'breakthecage',
      'permanent immigrants',
      'guest work',
      'h1bs',
      'facing deportation',
      'aliens',
      'entry/exit tracking',
      'immigration policy',
      'immigration history',
      'smugglers',
      'h 1b visas',
      '14th amendment',
      'welcoming nation',
      'deporting',
      'born citizens',
      'kateslaw',
      'refugee crisis',
      'hardworking immigrant',
      'border security',
      'refugees',
      'aspiring americans',
      'patrol agents',
      'immigration reforms',
      'vigilantes',
      'immigrants drive',
      'recent deportation',
      'mexican government',
      'deferred action',
      'h1 b',
      "kate's law",
      'allow refugees',
      'building walls',
      'self deportation',
      'entry exit',
      'deport immigrants',
      'deport family',
      'new americans',
      'deportation',
      'come legally',
      'citizens born',
      'federal immigration',
      'ordered deported',
      'refugees coming',
      'deport people',
      'unite families',
      'deporter',
      'guest worker provisions',
      'tall wall',
      'immigrant entrepreneurs',
      'border crosser',
      'border crisis',
      'allowed citizenship',
      'asylum',
      'asylum seekers',
      'border',
      'borders',
      'cages',
      'census',
      'detention',
      'immigrants',
      'immigration',
      'xenophobia'],
     'military': ['generals',
      'selective service',
      'love veterans',
      'inspector general',
      'draft',
      'allow veterans',
      'military budget',
      'veterans deserve',
      'veterans affairs',
      'armed services',
      'gold star family',
      'enlist',
      'wounded warrior',
      'veterans per',
      'green beret',
      'veterans face',
      'military service',
      'veterans must',
      'military issue',
      'airman',
      'war veterans',
      'service uniforms',
      'know responsibility service',
      'defense secretary',
      'veterans service',
      'veterans legislation',
      'desert veterans',
      'becoming veterans',
      'veteran support',
      'fighting country',
      'duty service',
      'become veterans',
      'combat roles',
      'giving veterans',
      'stronger military',
      'military personnel',
      'veterans department',
      'vietnam war',
      'care veterans',
      'veterans community',
      'ashton carter',
      'responsibility service members',
      'returning soldier',
      'brothers served military',
      'base',
      'homeless veterans',
      'veterans organizations',
      'u s forces',
      'service combat',
      'depleted military',
      'veterans groups',
      'gold star families',
      'vietnam vets',
      'wounded veterans',
      'veterans choices',
      'naval',
      'veterans card',
      'american boots',
      'concerned veterans',
      'veterans administration',
      'veterans committee',
      'among veterans',
      'service country',
      'serving abroad',
      'veterans died',
      'americans serving',
      'smallest military',
      'prosthetics',
      'military spending',
      'weak militarily',
      'tremendous veteran',
      'special forces',
      'helping vets',
      'veterans military families',
      'marine battalions',
      'comprehensive v.a',
      'years service',
      'aggressive military',
      'veteran affairs',
      'career civil',
      'every veteran',
      'strong military',
      'veteran ought',
      'veterans organization',
      'naval ships',
      'navy seals',
      'veteran gets',
      'admirals',
      'procurement process',
      'veterans memorial',
      'returning veterans',
      'v a',
      'combat ready',
      'steady commander in chief',
      'vfw',
      'strongest military',
      'army veteran',
      'army brigades',
      'combat service',
      'v a ',
      'give veterans',
      'smallest navy',
      'military technology',
      'broken v.a',
      'ptsd',
      'soldiers',
      'veterans living',
      'veteran comes',
      'veterans issues',
      'veterans care',
      'voluntary military',
      'procurement',
      'american legion',
      'vets',
      'served military know',
      'active duty',
      'v.a',
      'military leaders',
      'warrior organization',
      'help wounded',
      'million veterans',
      'service members',
      'troops'],
     'safety': ['body cameras',
      'nra wrote',
      'federal agents',
      'hook families',
      'sue gun',
      'prohibited ammunition',
      'taken marijuana',
      'combat assault',
      'prescribing opiates',
      'increased surveillance',
      'violent crimes',
      'gun liability',
      'pierce policemen',
      'bernardino',
      'retrain police',
      'law abiding citizens.but',
      'opiates',
      'possessed marijuana',
      'cyber attacked',
      '2nd amendment ',
      'officers involved',
      'great police',
      'federal crime',
      'hook elementary',
      'tough gun',
      'policing commissioner',
      'capital punishment',
      'improve police',
      'state police.they',
      'mosquitoes',
      'oxycontin',
      'reason gun',
      'enforcement right',
      'inmate education',
      'opioids',
      'ending profiling',
      'gun control',
      'justice reform',
      'concealed weapon',
      'quarantine people',
      'bear arms',
      'guns evolved',
      'low level offenders',
      'pass gun',
      'increasing drug',
      'alcoholism',
      'drug treatment',
      'gun liable',
      'u.s. attorney',
      'gun companies',
      'tsarnaev brothers',
      'abigail kopf',
      'buy guns',
      'charleston loophole',
      'addiction issues',
      'encrypted',
      'dealers turned',
      'political argument',
      'heroin deaths',
      'prosecuted fairly',
      'incarcerate another',
      'abused heroin',
      'police officers',
      'ferguson effect',
      'gun shops',
      'banned assault',
      'over mass incarceration',
      'emphasized gun',
      'patriot act',
      'marijuana laws',
      'police chief',
      'unfairly incarcerate',
      'gun accountable',
      'gun manufacturing',
      'officers',
      'reduce violent',
      'data breach',
      'police custody',
      'viruses',
      'manufacturers legally',
      'legal strategy',
      'rifles',
      'heroin addiction',
      'controlled substance',
      'controlled substances act',
      'gun lobby',
      'broken criminal',
      'tamir',
      'highway patrol',
      'enforcement community',
      'addicted',
      'stricter gun',
      'officers across',
      'record police',
      'heinous crimes',
      'military ammunition',
      'bernardino killings',
      'american prisoners',
      'protected millions',
      'regulate guns',
      'increased scrutiny',
      'think gun free',
      'brightest prosecutors',
      'heroin epidemic',
      'murders',
      'mass killing',
      'san bernadino',
      'loop hole',
      'opioid addiction',
      'dylann roof',
      'training videos',
      'thorough trial',
      'tamir rice',
      'waiting periods',
      'intelligence agency',
      'lot safer',
      'des moins',
      'profiling people',
      'the gun',
      'using firearms',
      'bombing',
      'deadly force',
      'coroner ruled',
      'ending gun',
      'first time drug',
      'law officers',
      'minimal sentencing',
      'opm breach',
      'quarantine',
      'important nra',
      'domestic spying',
      'drug courts',
      'keep guns',
      'prosecutors investigate',
      'alcohol',
      'overdose deaths',
      'annie oakley',
      'drug lords',
      'commonsense gun',
      'hunting traditions',
      'exploit loopholes',
      'gun ',
      'federal jurisdiction',
      'stop gun violence',
      'deer',
      '2nd amendment',
      'allows lawlessness',
      'individual right',
      'criminal backgrounds',
      'ferguson',
      'terrible gun',
      'arming',
      'crime rate',
      'violent crime',
      'trooper',
      'overdose',
      'safest cities',
      'force sellers',
      'zika',
      'public safety',
      '12 year old tamir',
      'licensed dealers',
      'bernardino.and',
      'giving guns',
      'american law',
      'drug problem',
      'additional attacks',
      '6th amendment',
      'comprehensive gun',
      'force gun',
      'committing crimes',
      'emmett',
      'americans afraid',
      'bernardino happen',
      'drug epidemic',
      'drug girl',
      'assault weapon',
      'respecting deer',
      'gun  free',
      'milwaukee police',
      'mexican drug',
      'gravely injured',
      'people murdered',
      'comprehensive background',
      'commit violence',
      'toy gun',
      'mass shooting',
      'cybersecurity ',
      'mandated background',
      'mcveigh received',
      'gun legislation',
      'child safety',
      'automatically trigger',
      'incarceration rates',
      'private prison',
      'buy combat',
      'rampage',
      'private server',
      'enforcement officer',
      'substance use',
      'safety locks',
      'legal weapon',
      'legally responsible',
      'legally purchase',
      'san bernardino',
      'nra gave',
      'end the patriot act',
      'enforcement understands',
      'intelligence officials',
      'passed gun',
      'incarceration rate',
      'police shootings',
      'purchase guns',
      'getting killed',
      'devastating attack',
      'crime stats',
      'penalty aspects',
      'respect law',
      'crack house',
      'shield gun',
      'safety legislation',
      'criminal record',
      'police force',
      'online loophole',
      'quarantined',
      'kill people',
      'gun free',
      'near ferguson',
      'oppose gun',
      'fbi made',
      'prison population',
      'killers',
      'carry guns',
      'selling guns',
      'cut juvenile',
      'so called ferguson',
      'people possessed',
      'felons',
      'bernardino case',
      'investigate cases',
      'keeping law',
      'regular law',
      'strong safety',
      'nra went',
      'increased security',
      'law abiding people',
      'evidentiary proof',
      'special protection',
      'people killed',
      'private prisons',
      'ebola',
      'unidentified male',
      'americans safer',
      'naloxone',
      'joint intelligence',
      'limiting gun',
      'brought guns',
      'jail/prison',
      'oppressive force',
      'nra saw',
      'suicide',
      'show loophole',
      'gun rights',
      'show loop',
      'recidivism rate',
      'fierce fighter',
      'lone wolves',
      'horrible crimes',
      'background check',
      'death row',
      'intelligence communities',
      'selling weapons',
      'timothy mcveigh',
      'people dead',
      'stop gun',
      'parenthood murders',
      'killed cops',
      'community policing',
      'newtown massacre',
      'federal gun',
      'protect americans',
      'sell automatic',
      'shooting people',
      'control guns',
      'sandy hook',
      'wiretap',
      'police violence',
      'tsarnaev',
      'local prosecutors',
      'quarantining',
      'prison sentences',
      'shot everyone',
      'buys guns',
      'san bernardino.and',
      'crack encrypted',
      'told gun',
      'hunting season',
      'nra legislation',
      'wayne lapierre',
      'gunmakers',
      'ban assault',
      'gun free zones',
      'criminal enforcement',
      'drug rehabilitation',
      'cybersecurity',
      'follow gun',
      'restore law',
      'zika virus',
      'reduced violent',
      'nsa',
      'horrific violent',
      'mcveigh',
      'versus heller',
      'jails',
      'shooting rampage',
      'drug offender',
      'court defending',
      'nra position',
      'imposed tougher',
      'opioid overdose',
      'violent acts',
      'victims announced',
      'barbaric acts',
      'homicides last',
      'loophole',
      'active viruses',
      'gun issues',
      'injuring two',
      'fbi crime',
      'gun shows',
      'weapons designed',
      'deer hunter',
      'officers become',
      'encryption',
      'weapons tomorrow',
      'concealed weapon permit',
      'gun store',
      'bullets went',
      'killed responding',
      'policing needs',
      'fbi investigation',
      'civilian review boards',
      'incarcerate',
      'gangs inside',
      'criminals running',
      'police department',
      'country safe',
      'abiding citizen',
      'concealed carry',
      'ame church',
      'terrible crimes',
      'called charleston',
      'stand your ground',
      'encrypted communication',
      'jailed',
      'heroin addicted',
      'reclassify marijuana',
      'police',
      'fbi agents',
      'bernardino two',
      'guns recovered',
      'non  violent',
      'commit suicide',
      'cops',
      'massacre',
      'encryption problem',
      'gun people',
      'shooting unarmed',
      'shooting gallery',
      'seven actions',
      'gun makers',
      'think crime',
      'selling military style',
      'gun policy',
      'monitoring mosques',
      'non  indictment',
      'black market',
      'safety regulations',
      'chicago police',
      'drug overdose',
      'police.they',
      'rising crime',
      'semi automatic rifles',
      'heroin overdose',
      'paris attacks',
      'bring law',
      'prosecute criminals',
      'heroin',
      'require police',
      'buying guns',
      'solitary confinement',
      'pop gun',
      'support banning',
      'year old tamir',
      'intelligence agencies',
      'tough law',
      'bring guns',
      'officers killed',
      'officer breaks',
      'guns safer',
      'sandra bland',
      'mass killings',
      'country police',
      'drug addicted',
      'automatic weapon',
      'metadata program',
      'banning firearms',
      'gun owner',
      'pipe bombs',
      'precincts',
      'gun manufacturer',
      'allows guns',
      'eric holder',
      'suspicious activity',
      'start killing',
      'death penalty',
      'alcohol addiction',
      'gun dealers',
      'violence within',
      'brutality',
      'horrible violence',
      'unarmed people',
      'church killings',
      'pop store',
      'officer wants',
      'dealers accountable',
      'deadly heroin',
      'toughest gun',
      'prevent guns',
      'automatic weapons',
      'heroin addicted drug',
      'instant background',
      'police officer',
      'drug dealers',
      'popular firearms',
      'sensible gun',
      'state prison',
      'hook victims',
      'policemen',
      'controversial police',
      'treatment funding',
      'purchasing issue',
      'kalamazoo',
      'steal guns',
      'domestic abusers',
      'bernardino couple',
      'police departments',
      'young officer',
      'marathon bombing',
      'bernardino attack',
      'lethal force',
      'virginia tech',
      'nra priority',
      'enforcement officers',
      'gun shop',
      'pain pills',
      'guns play',
      'tougher prison',
      'among death',
      'sue remington',
      'hate crimes',
      'emanuel ame',
      'sell guns',
      'safety protection',
      'sells legally',
      'addiction',
      'assault',
      'bail',
      'criminal',
      'criminal justice',
      'criminal justice system',
      'domestic violence',
      'drugs',
      'gun',
      'gun violence',
      'guns',
      'incarcerated',
      'incarceration',
      'marijuana',
      'nra',
      'prison',
      'shootings',
      'the nra'],
     'social and cultural': ['mothers',
      'religious opinion',
      'marriage laws',
      'islam.the',
      'women color',
      'redefine marriage',
      'parenthood funding',
      'abortion procedures',
      'so called ferguson',
      'discriminating',
      'hispanic churches',
      'white people',
      'theologian',
      'locker room talk',
      'im with kim',
      'abolished partial birth',
      'gay marriage',
      'privileged',
      'marriages',
      'choose life',
      'same sex couples',
      'gays',
      'protect children',
      'black male',
      'african  american',
      'often african americans',
      'prolife',
      'deeply held religious',
      'pro life',
      'defunded planned',
      'african american congressman',
      'virtuous',
      'islam.the fact',
      'disablity',
      'marriage dissenters',
      'heart beating',
      'tackling racial',
      'deep racial',
      'wade nationalized',
      'morning after pills',
      'fund planned',
      'little racist',
      'racial divides',
      'muslims',
      'impregnated',
      'parenthood videos',
      'muslim americans',
      'cervical cancer',
      'bullying people',
      'pro life person',
      'freedom coalition',
      'white person',
      'abortion law',
      'vra',
      'kim davis',
      'defending planned',
      'religion supersedes',
      'iowa faith and freedom coaltion',
      'islamic extremists',
      'faith says',
      'pro life governor',
      'abortion legal',
      'fund abortions',
      'trayvon',
      'religious bearing',
      'klux',
      'address systemic',
      'sharia',
      'defend religious',
      'parenthood tapes',
      'islamic groups',
      'race relations',
      'hbcus',
      'abortion factor',
      'anti women',
      'zealot',
      'converts',
      'little babies',
      'abortion illegal',
      'hyde amendment',
      'god gives',
      'rights activism',
      'harvesting organs',
      'capable unborn',
      'sex marriage',
      'say her name',
      'less abortions',
      'gospel',
      'young african ',
      'suburb',
      'roe v wade',
      'american muslims',
      'since roe',
      'partial birth abortions',
      'controls their bodies',
      'christians',
      'religious institutions',
      'adoption agencies',
      'communities color',
      'ferguson effect',
      'christian groups',
      'evangelicals',
      'protecting religious',
      'klu klux',
      'racial issues',
      'large mosques',
      'cair',
      'racial profiling',
      'people color',
      'versus wade',
      'deny god',
      'women national',
      'de funded planned',
      'wade',
      'kentucky clerk',
      'commandments',
      'infanticide',
      'stand with pp',
      'exaltation',
      'abortion case',
      'vetoed planned',
      'islamic communities',
      'restrict women',
      'late term',
      'crescent moons',
      'klan',
      'roe 43',
      'defund planned',
      'extremist',
      'african american man',
      'prayer shaming',
      'woman driven',
      'lesbian',
      'tamir',
      'pregnancy',
      'homosexuality',
      'roe versus',
      'racial blind',
      'predominantly african americans',
      'rosa parks',
      'abortion bans',
      'predominantly african american',
      'race war',
      'cash bail',
      'same sex marriage',
      'aborted',
      'black lives',
      'young african american',
      'antisemitism',
      'muslim  american',
      'cardinal timothy',
      'marriage equality',
      'planned parenthood',
      'partial birth',
      'freddie gray',
      'systematic discrimination',
      'religious belief',
      'african american graduates',
      'address race',
      'strong faith',
      'abortion techniques',
      'inner city chicago',
      'abortion practices',
      'pro life party',
      'muslim ',
      'jim crow',
      'pulpit',
      'inner cities',
      'confederate',
      'religious conscience',
      'lgbt community',
      'breast cancer',
      'conception act',
      'ghetto',
      'sex',
      'religious intoleration',
      'issue marriage',
      'lives matter',
      'religious rights',
      'defund planned parenthood',
      'believe deeply',
      'pregnancy centers',
      'timothy dolan',
      'tamir rice',
      'family planning',
      'roe',
      'parental notification',
      'crescent',
      'confederateflag',
      'parentlal leave act',
      'muslim communities',
      'made abortion',
      'cardinal',
      'late term abortion',
      'sermons',
      'violence lgbtq',
      'pro life position',
      'terrorists islamic',
      'medical marijuana',
      'issue same sex',
      'marriage licenses',
      'ten commandments',
      'transgender',
      'banning abortions',
      'actual abortion',
      'dissenters',
      'roberts care',
      'same sex marriages',
      'right to choose',
      'pro choice candidate',
      'ensure christians',
      'black communities',
      'genocide',
      'ms. davis',
      'provide morning after',
      'teenage',
      'churches',
      'sharia law',
      'mosques throughout',
      'pro choice voting',
      'mexicans',
      'african ',
      'mosques',
      'catholic',
      'partial birth abortion',
      'muslim',
      'gay couple',
      'disproportionately latino',
      'imams',
      'roe wade',
      'pro  choice',
      'ferguson',
      'scalia',
      "women's health",
      'radicals',
      'stand4life',
      'community groups',
      'birth abortion',
      'pastors',
      'violent jihad',
      'bus boycott',
      'abortion clinics',
      'crazy zealot',
      'weddings',
      'religious discrimination',
      'unborn children',
      'unborn human',
      'community based',
      'stopp',
      'pro life piece',
      'raped',
      'poor kids',
      'hope women',
      'think abortion',
      'sandra bland',
      'morning after',
      'parenthood right',
      'religious foundation',
      'pro choice',
      'preserve lives',
      'marriage legal',
      'defended planned',
      'partial  birth',
      'discriminatory policies',
      'declared same sex',
      'gay community',
      'mosque',
      'klux klan',
      'prejudice',
      'national urban league',
      'reproductive health',
      'black lives matter',
      'crazy zealots',
      'civil rights law',
      'faith alone',
      'sexist',
      'everyday muslim americans',
      'percent pro choice',
      'unborn child',
      'traditional marriage',
      'lgbt',
      'defunding planned',
      'life begins',
      'african american children',
      'african american kids',
      'segregated',
      'mosques preaching',
      'stand for life',
      'believer',
      'muslim detainees',
      'supremacy',
      'human child',
      'sanctity of life',
      'parenthood engages',
      'baby inside',
      'muslims serving',
      'islamophobia',
      'religious beliefs',
      'homophobe',
      'institutional racism',
      'working moms',
      "children's rights",
      'pro choice states',
      'jihad',
      'defund pp',
      'contraception',
      'poor black',
      'cervical',
      'allowing abortion',
      'bit racist',
      'youth minister',
      'especially mosques',
      'disenfranchised ',
      'organs',
      'fetus',
      'incest exception',
      'montgomery bus boycott',
      'anti bigotry',
      'overturn roe',
      'same sex couple',
      'african american woman',
      'exercise religious',
      'christian colleges',
      'often hispanics',
      'religious affairs',
      'teachings',
      'crisis pregnancy',
      'respect women',
      'turn abortion',
      'reverse roe',
      'protected religious',
      'ppfa',
      'parental leave act',
      'relgious liberty',
      'viable life',
      'school african american',
      'couples',
      'black america',
      'unborn',
      'preaching',
      'people of color',
      'matter movement',
      'young african',
      'anti muslim',
      'confederate flag',
      'michael brown',
      'investigate segregated',
      'conception',
      'white man',
      'same sex',
      'incest',
      'stand 4 life',
      'abortion',
      'abortions',
      'african',
      'african american',
      'african americans',
      'anti semitism',
      'black',
      'blacks',
      'black lives',
      'disabilities',
      'discrimination',
      'equal pay',
      'gay',
      'latino',
      'latinos',
      'racial',
      'racism',
      'racist',
      'religious freedom',
      'sexual orientation',
      'systemic',
      'systemic racism'],
     'social programs': ['charter schools',
      'out of pocket expenses',
      'health empowerment',
      'care taking',
      'debt free',
      'insurance discrimination',
      'means test',
      'care plan',
      'guarantee healthcare',
      'rent controlled',
      'school systems',
      'vaccines',
      'free college',
      'out of pocket costs',
      'childhood poverty',
      'access health care',
      'insurance program',
      'arnie duncan',
      'getting insurance',
      'maternity',
      'tremendous fever',
      'corporate welfare',
      'dismantle health',
      'price health',
      'insurers',
      'huge copayments',
      'medicare benefit',
      'care taking responsibilities',
      'reduce costs',
      'country health',
      'quality childcare',
      'non academic assets',
      'rent controlled apartment',
      'impoverished',
      'healthcare plan',
      'scholarships',
      'national marketplace',
      'home schools',
      'purchase health',
      'alzheimers',
      'pre existing',
      'single payer health',
      'school system',
      'study hard',
      'deductibles rising',
      'expand school',
      'medicare taxes',
      'expand medicaid',
      'blue shield',
      'address obamacare',
      'promising health',
      'universities tuition',
      'cancer survivor',
      'saved obamacare',
      'human services',
      'latino kids',
      'chip program',
      'address college',
      'reduce hospital',
      'clara barton',
      'cost of living',
      'improve health',
      'retirement program',
      'care policy',
      'poorest people',
      'rich kids',
      'million seniors',
      'medicaid reform',
      'free tuition',
      'hip replacement',
      'subsidy system',
      'fancy dormitories',
      'social security medicare',
      'payer program',
      'education standards',
      'medicaid for all',
      'premiums skyrocket',
      'children retire',
      'higher education',
      'health facilities',
      'doses',
      'emergency rooms',
      'cigarettes',
      'grant program',
      'medicare plan',
      'hillarycare',
      'single payer system',
      'unions protect',
      'insurance carrier',
      'fund health',
      'expand health',
      'care failed',
      'healthcare reform',
      'health economists',
      'pediatric',
      'catastrophic healthcare',
      'living longer',
      'subsidy',
      'universal pre k',
      'appeal obamacare',
      'income based',
      'care deductibles',
      'debilitating',
      'healthcare become',
      'rising student',
      'cancer drugs',
      'extend medicaid',
      'keep health',
      'decent nutrition',
      'expanded medicaid',
      'guarantee health',
      'wealthy children',
      'social programs',
      'ending social',
      'impose healthcare',
      'elderly people',
      'especially prescription',
      'enroll graduate',
      'college offer',
      'primary health',
      'federal health',
      'vocational training',
      'introduce vocational',
      'affordability',
      'essentially medicare',
      'demographics',
      'buttress social',
      'knee replacement',
      'health systems',
      'two parent',
      'guaranteed health',
      'kids jobs',
      'fewer doctors',
      'bad teachers',
      'eliminates medicare',
      'pharmaceutical industry',
      'childcare',
      'universal healthcare',
      'government benefits',
      'providing medicine',
      'mentally ill.',
      'medicare right',
      'universal child care',
      'need medicare',
      'single payer proposal',
      'provide debt free',
      'dorm business',
      'meds',
      'handouts',
      'biggest teachers',
      'pre existing conditions',
      'autistic',
      'measles outbreak',
      'city university',
      'medicare budget',
      'send education',
      'primary care',
      'health education',
      'expand entitlements',
      'education committee',
      'college graduates',
      'medicaid expansion',
      'sanders esque',
      'strengthen social',
      'youngest kids',
      'poorest',
      'insurer',
      'living conditions',
      'graduating scale',
      'on line education',
      'crumbling school',
      'food assistance',
      'beneficiaries',
      'senior citizens',
      'means disadvantaged',
      'catastrophic health',
      'single payer',
      'of life care',
      'preexisting',
      'lose welfare',
      'socialized medicine',
      'sicker',
      'billion hbcus',
      'schools work',
      'passive income',
      'pills',
      'co payments',
      'obamacare',
      'healthcare costs',
      'obamacare.you',
      'studies hard',
      'health coverage',
      'somoza',
      'gained health',
      'child care workers',
      'cutting social',
      'increase benefits',
      'security benefit',
      'young teachers',
      'all payer',
      'healthcare immigration',
      'education locally',
      'fee for  service',
      'insured',
      'keep preexisting',
      'guaranteeing health',
      'poor families',
      'poorest recipients',
      'taxable income',
      'insurance coverage',
      'includes pell',
      'social security',
      'college compact',
      'tuition',
      'called obamacare',
      'retirement age',
      'developmental disabled',
      'obama care',
      'virtually tuition free',
      'non academic',
      'changing entitlements',
      'school choice',
      'medical costs',
      'bright young',
      'curriculum',
      'medicaid program',
      'deductible',
      'test social',
      'top down obamacare',
      'aca',
      'education making',
      'college debt',
      'medical discoveries',
      'acute care',
      'fix social',
      'expand medicare',
      'eliminate medicare',
      'speculation tax',
      'school deeply',
      'percent coverage',
      'education swat',
      'security monthly',
      'statewide voucher',
      'preexisting conditions',
      '90000 doctor',
      'curricula',
      'repayment plan',
      'neurosurgeon',
      'medicare for all program',
      'family leave',
      'taking medicaid',
      '90,000 doctor shortage',
      'see healthcare',
      'call entitlements',
      'social promotion',
      'per student',
      'university tuition',
      'voucher program',
      'catastrophic insurance',
      'preexisting condition',
      'vaccinations',
      'insurance market',
      'retiring',
      'vocational education',
      'affordable housing',
      'college ',
      'premiums',
      'obamacare took',
      'welfare reform',
      'toward socialism',
      'diabetes',
      'giving tax breaks',
      'state tuition',
      'payer system',
      'called hillarycare',
      'insurance premiums',
      'universal health',
      'pre k',
      'dislike obamacare',
      'employer policy',
      'grad school',
      'child deserves',
      'community college',
      'poor kids',
      'vocational',
      'expand benefits',
      'autism',
      'entitlements',
      'abundant school',
      'dorm',
      'local school',
      'medicaid for all system',
      'pell grants',
      'tuition free',
      'and or career ready',
      'de link health',
      'private health',
      'low income seniors',
      'vaccinated',
      'education regardless',
      'summer programs',
      'blue cross',
      'free education',
      'pell grant',
      'scape goating teachers',
      'private health insurance',
      'poverty among',
      'expanding social',
      'ppfa',
      'copays',
      'cut benefits',
      'medicare for all',
      'children deserve',
      'view healthcare',
      'complete college',
      'save medicare',
      'debt free tuition',
      'curriculum reform',
      'young student',
      'provide health',
      'left right',
      'affordable price',
      'traditional health',
      'american students',
      'insurance companies',
      'entitlement reform',
      'great teachers',
      'extreme poverty',
      'insurance right',
      'security trust',
      'end  of life',
      'children inherit',
      'benefit check',
      'healthcare needs',
      'health savings',
      'federal subsidies',
      'pell',
      'college affordable',
      'education labor',
      'teacher shortage',
      'pre k.',
      'local curriculum',
      'less benefit',
      'broken system',
      'certified teachers',
      'sounds sanders esque',
      'reform medicare',
      'school degree',
      'faculty',
      'competitive schools',
      'physicians keep',
      'large out of pocket',
      'high deductibles',
      'universities tuition free',
      'change obamacare',
      'universal health care',
      'martin shkreli',
      'expanding benefits',
      'mentoring programs',
      'guarantees health',
      'childhood vaccines',
      'physicians',
      'public universities',
      'all payer system',
      'purchase insurance',
      'ensure low income',
      'decent housing',
      'chronic disease',
      'surprisingly college',
      'colleges free',
      'state school',
      'total enrollment',
      'college education',
      'scrap obamacare',
      'disadvantaged kids',
      'welfare benefits',
      'copayments',
      'insurance system',
      'measles',
      'characterized entitlement',
      "alzheimer's",
      'deductibles',
      'empowering patients',
      'school boards',
      'welfare programs',
      'affordable care act',
      'insurance lobbyists',
      'anastasia somoza',
      'defending social',
      'spouse dies',
      'keep patients',
      'school vouchers',
      'americans medicare',
      'payer health',
      'education department',
      'single payer program',
      'health reform',
      'childhood education',
      'pre kindergarten',
      'private employer',
      'medicate',
      'marketplace',
      'college tuition',
      'federal welfare',
      'retirement income',
      'entitlement',
      'existing student',
      'large deductibles',
      'outstanding student',
      'working woman',
      'education costs',
      'miami dade college',
      'schools system',
      'common core',
      'out of pocket',
      'graduates',
      'school programs',
      'medicaid budget',
      'basic income',
      'ubi',
      'expanding medicaid',
      'privatize social',
      'entitlement programs',
      'public colleges',
      'pay tuition',
      'medicine',
      'bad health',
      'insurance across',
      'private marketplace',
      'academic problems',
      'polio',
      'poorer',
      'crushing student',
      'medicare reform',
      'pediatricians',
      'retiree',
      'debt free college',
      'replace obamacare',
      'crippled',
      'give children',
      'fund social',
      'paid maternity',
      'college affordability',
      'public college',
      'universal coverage',
      'private market',
      'student achievement',
      'act score',
      'care hospitals',
      'teach districts',
      'care physicians',
      'grandchildren retire',
      'social welfare',
      'universal pre kindergarten',
      'heart disease',
      'federal education',
      'insurance people',
      'insurance company',
      'college free',
      'routine healthcare',
      'career ready',
      'socialized health',
      'fix medicare',
      'dormitories',
      'children s health',
      'enroll',
      'schools open',
      'hospital costs',
      'educational policy',
      'obama no care',
      'medicare available',
      'affordable college',
      'affordable health',
      'affordable health care',
      'health insurance',
      'medicare']}

    # check if ads text contains null values
    if df1['text'].isnull().sum():
        df1 = df1[df1['text'].notnull()]
        df1.reset_index(drop=True,inplace=True)

    # adding new column to df
    df1['ad_creative_body'] = df1['text']


    # function that filter the urls and symbols in the text 

    def filter_text(x):
        url = 'http[s]?://\S+'
        x = re.sub(url,'',x)
        x = re.sub("[^\w\s]",' ',x) # filter symbols
        x = re.sub("\s+",' ',x)

        ls=[w.lower() for w in x.split()] 

        return ' '.join(ls)

    df1['text'] = df1['text'].astype(str).apply(lambda x: filter_text(x))

    # the function that find the lexicon words in the text
    def find_words(x,lexicon):
        topics= lexicon.keys()  
        doc = nlp(x) # nlp() is spaCy 2.2 English language model 
        words= []
        for t in topics:
            terms= lexicon[t]
            patterns = [nlp.make_doc(text) for text in terms] 
            matcher.add("TerminologyList", None, *patterns) # spaCy2.2 phrase matcher
            matches = matcher(doc)
            for match_id, start,end in matches:
                span = doc[start:end]
                words.append(span.text)
        if words:
            words = list(set(words))
            return ','.join(words)
        else:
            return('no words')


    # tagging the topic in each message
    def find_topic(x,lexicon):
        topics= lexicon.keys()    
        if x=='no words':
            return ''    
        if x != 'no words': 
            words = x.split(',')
            labels = []        
            for t in topics:            
                terms = lexicon[t]
                if set(words)&set(terms):
                    labels.append(t)                

            return  ','.join(sorted(labels))




    df1['words'] = df1['text'].astype(str).apply(lambda x: find_words(x,dic1))


    df1['m_label'] = df1['words'].apply(lambda x: find_topic(x,dic1))


    df1['m_label'] = df1['m_label'].apply(lambda x: 'no topic' if x=='' else x)

    # splitting labels into list based on comma
    df1['m_label'] = df1['m_label'].apply(lambda x: x.split(','))

    # removing columns from df
    del df1['text']
    del df1['words']

    # creating a dictionary from df
    dc=df1.to_dict('records')

    # upserting topic tags for ads
    for doc in dc:
        for x in col.find({"ad_creative_body":doc['ad_creative_body'],'marked_topic':0}):
            x['marked_topic']=1
            x['topic']=doc['m_label']
            col.update_one({'_id': x['_id']},{"$set":x},True)    

    return "Done"






marking_topic_step = PythonOperator(
    task_id='predict_topic',
    provide_context=True,
    python_callable=predict_topic,
    dag=dag,
)







col1=db["fb_ads_dev"]
def push_data(ds, **kwargs):
    print("Pushing")
    ct=0
    for doc in db['fb_ads_dump'].find({'$or':[{'marked':1}, {'marked':0}, {'marked_topic':0},{'marked_topic':1}]},no_cursor_timeout=True):
        ct+=1
        if ct%1000==0:
            print(ct)

        datapt=col1.find_one({'id':doc['id']})
        if datapt:
            del doc['_id']
            col1.update_one({'_id':datapt['_id']},{'$set': doc}, True)
            print('row updated')
        else:
            col1.insert_one(doc)
            print('row inserted')
            
get_data_step = PythonOperator(
    task_id='get_data',
    provide_context=True,
    python_callable=get_data,
    dag=dag,
)


            


push_data_step = PythonOperator(
    task_id='push_data',
    provide_context=True,
    python_callable=push_data,
    dag=dag,
)



stop_op = DummyOperator(task_id='stop_task', dag=dag)


get_data_step >> marking_step >> marking_civility_step >> marking_topic_step >> push_data_step >> stop_op

