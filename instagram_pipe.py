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

token='75K5noVVpbi7JorR1sOJVV8gcT7coHQZdfFxeshi'
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2020,4,9,2,0,0),
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
    'instagram_collection',
    default_args=default_args,
    description='A dag collect instagram data with crowdtangle api',
    schedule_interval='@daily',
)

dbase='POTUS_2020_DEV'
url1 = 'mongodb://%s:%s@denver.ischool.syr.edu'
mongoClient= pymongo.MongoClient(url1 % ('bitslab', '0rang3!'))
db = mongoClient[dbase]
col=db["ig_dump"]
def get_data(ds, **kwargs):
    global token
    col.remove()
    dt=(date.today()-timedelta(30)).strftime("%Y-%m-%d")
#     dt5=(date.today()-timedelta(5))
    resp={}
    print("hellooo")
    params = {'token':token,'count':100,'startDate':dt}
    resp=requests.get('https://api.crowdtangle.com/posts',params)
    resp_json = json.loads(resp.text)
    ct=0
    print("Starting Collection")
    while True:
        ct=ct+1
        print("Page: ",ct)
        try:
            posts=resp_json['result']['posts']
            for x in posts:
                col.insert_one(x)
            resp=requests.get(resp_json['result']['pagination']['nextPage'])
            resp_json = json.loads(resp.text)
        except KeyError:
            break
    return ct



col1=db["ig_posts_dev"]
def push_data(ds, **kwargs):
    print("Pushing")
    ct=0
    for doc in db['ig_dump'].find(no_cursor_timeout=True):
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


get_data_step >> push_data_step >> stop_op
