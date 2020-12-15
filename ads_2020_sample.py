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
# from fuzzwuzzy import fuzz
# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization

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
    'execution_timeout': timedelta(minutes=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}
dag = DAG(
    'ads_2020_sample',
    default_args=default_args,
    description='sample 2020 facebook ads data',
    schedule_interval='@once',
)

dbase='POTUS_2020_DEV'
url1 = 'mongodb://%s:%s@denver.ischool.syr.edu'
mongoClient= pymongo.MongoClient(url1 % ('bitslab', '0rang3!'))
db = mongoClient[dbase]
col=db["fb_ads_dev"]
data=[]

def get_data(ds, **kwargs):
    sample=[]
    samp_text=[]
    start_date=Variable.get('sample_start_date')
    end_date=Variable.get('sample_end_date')
    sample_size=Variable.get('sample_size')
    print("hello")
    for doc in col.distinct('ad_creative_body',{"ad_delivery_start_time":{"$gte":start_date,"$lte":end_date}}):
        print(len(sample))
        if len(samp_text)<int(sample_size):
            samp_text.append({'ad_creative_body':doc})
        else:
            break
    for x in samp_text:
        dc=col.find_one({"ad_creative_body":x['ad_creative_body']})
        sample.append(dc.copy())
    
    print(len(sample))
    df=pd.DataFrame(sample[:int(sample_size)])
    df.to_csv("./home/jay/samples/facebook/facebook_ads_"+str(sample_size)+"_"+start_date+"--"+end_date+".csv")
    return "done"    
#     while True:
#         for doc in col.aggregate([ { "$match": {"ad_delivery_start_time":{"$gte":start_date,"$lte":end_date}} },{ "$sample": { "size": int(sample_size) } } ]):
#             print(len(sample))
#             if len(sample)>int(sample_size):
#                 break
#             try:
#                 i=1
#                 if doc['ad_creative_body'] not in samp_text:
#                     for y in samp_text:
#                         if fuzz.ratio(doc['ad_creative_body'],y)>50:
#                             i=1
#                             break
#                 if i==0:
#                     samp_text.append(doc['ad_creative_body'])
#                     sample.append(doc)
#             except:
#                 continue
#         if len(sample)>int(sample_size):
#                 break
#     return "done"

get_data_step = PythonOperator(
    task_id='get_data',
    provide_context=True,
    python_callable=get_data,
    dag=dag,
)



stop_op = DummyOperator(task_id='stop_task', dag=dag)


get_data_step >> stop_op


