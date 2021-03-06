# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:40:43 2019

@author: sgupta27
"""

from __future__ import absolute_import, division, print_function

import csv
import os
import sys
import logging
logger = logging.getLogger()
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


import sys
import torch
import numpy as np
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
# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
# at=["EAABj9aBk8VwBAEIbKZBmZBM00ZC6Q2BHCQeJ4tH4E9cZAG8ZBNT862LoJeIU9o6ZCjhkItxLAYlCWcyUJekcJjuqecM7IUblk12XiXcVamZAMFOIfZALUuDyHQLENpZBj3ZALwNOauh3d8OjPGTzvHU6CyXXIigNAUjG8BBRhjXuJcFwNexhfZB0fsG"]

at=["EAABj9aBk8VwBAEBB5CtIdY8mCX0M02FFdk9INpoZAZCfZAKRsK5ZBTt9fmaAuVElXTZBtZCKghtl6ZCuJFjXITLtkVheX4WzcqUigZCj5tYCWSBpvdUIrL26ovzZCN9eDnDYsoCMZCIcXj4NQ3vSEfrzEL5w2ZB2UNCxlFeWIxZBEislLuV4CHZCQJcKQ"]
token=at[0]
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2020,4,17,0,0,0),
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
    'fb_posts_pipe',
    default_args=default_args,
    description='A dag to dump and mark our database with new facebook posts',
    schedule_interval='@daily',
)

dbase='POTUS_2020_DEV'
url1 = 'mongodb://%s:%s@denver.ischool.syr.edu'
mongoClient= pymongo.MongoClient(url1 % ('bitslab', '0rang3!'))
db = mongoClient[dbase]
col=db["fb_posts_dump"]
data=[]

def get_data(ds, **kwargs):
    global data
    global page_ids
    global token
    global at
    col.remove()
    fail=[]
    dt=date.today()
    dt5=(date.today()-timedelta(30))
    resp={}
    print("hellooo")
    t=time.time()
    for page in db['cand_info'].find(no_cursor_timeout=True):
        print(page['candidate_name'])
        drp_dt=page['drop_date']
        y=0
        if type(drp_dt)!=float:
            drp_dt=datetime.strptime(drp_dt, '%m/%d/%Y').date()
            if drp_dt>=dt5:
                y=1
                enddt=dt5
            if drp_dt>dt:
                startdt=drp_dt
            else:
                startdt=dt
        else:
            enddt=dt5
            startdt=dt
            y=1
#         ann_dt=page['announcement_date']
#         if type(ann_dt)==float:
#             ann_dt='01/01/2019'
#         ann_dt=datetime.strptime(ann_dt, '%m/%d/%Y').date()
        if y==1:
            params = (('fields', 'id,message,created_time,updated_time,shares,likes.limit(1).summary(true),comments.limit(1).summary(true)'),('since',enddt.__str__()),('until',startdt.__str__()),('include_hidden','true'),('limit',10),('access_token', token))
            resp = requests.get('https://graph.facebook.com/v3.1/' + page['page_id'] + '/posts', params=params)
            resp_json = json.loads(resp.text)
            x={}
            i=0
            ct=0
            while True:
                try:
                    ct=ct+1
                    print("page: ",ct)
                    j=0
                    print(resp_json)
                    for item in resp_json["data"]:
                        item['marked']=0
                        item['page_id']=page['page_id']
                        col.insert_one(item)
                        j+=1
                    print(j)
                    time.sleep(1)
                    try:
                        x=resp_json
                        resp = requests.get(resp_json['paging']['next'])
                        resp_json=json.loads(resp.text)
                    except KeyError:
                        break
                except KeyError:
                    print("token limit exceeded")
                    fail.append(page['candidate_name'])
                    break
            print("candidate completed! sleep!") 
            time.sleep(20)
        else:
            print("Candidate Not Collected!")
    print("Failed: ",fail)
    return "Done!"



def predict(ds, **kwargs):
    global db
    print(os.getcwd())
    predict=True
    if predict:
        data=[]
        ls=[]
        for doc in col.distinct('message',{'marked':0}):
            if len(data)<2000:
                data.append({'message':doc})
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
                'text': df['message'].replace(r'\n', ' ', regex=True)
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
            Data = 'FB20'
            DATA_DIR = "./home/jay/airflow/dags/data/"
            #categories = ["Attack", "Advocacy", "CTA", "Issue"]
            categories = ["Attack", "Advocacy", "Ceremonial", "CTA", "CI", "Image", "Issue"]
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
                for x in col.find({"message":doc['message'],'marked':0}):
                    x['marked']=1
                    x['class']=doc['class']
                    col.update_one({'_id': x['_id']},{"$set":x},True)    
        return "Done"
    

marking_step = PythonOperator(
    task_id='predict',
    provide_context=True,
    python_callable=predict,
    dag=dag,
)

def predict_civility(ds, **kwargs):
    global db
    print(os.getcwd())
    ls=[]
    civility = False
    while civility:
        data=[]
        for doc in col.distinct('message',{'civility_class':None}):
            if len(data)<2000:
                data.append({'message':doc})
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
                'text': df['message'].replace(r'\n', ' ', regex=True)
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
            Data = 'FB20'
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
                for x in col.find({"message":doc['message'],'marked':0}):
                   # x['marked']=1
                    x['civility_class']=doc['civility_class']
                    col.update_one({'_id': x['_id']},{"$set":x},True)    
    return "Done"

marking_civility_step = PythonOperator(
    task_id='predict_civility',
    provide_context=True,
    python_callable=predict_civility,
    dag=dag,
)


col1=db["fb_posts_dev"]
def push_data(ds, **kwargs):
    print("Pushing")
    ct=0
    #{'marked':1},
    for doc in db['fb_posts_dump'].find(no_cursor_timeout=True):
        ct+=1
        if ct%100==0:
            print(ct)
        datapt=col1.find_one({'id':doc['id']})
        if datapt:
            del doc['_id']
            col1.update_one({'_id':datapt['_id']},{'$set': doc}, True)
        else:
            col1.insert_one(doc)
            
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


get_data_step >> marking_step >> marking_civility_step >> push_data_step >> stop_op