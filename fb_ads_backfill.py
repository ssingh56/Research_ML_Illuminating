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
at=['EAAML1IWIO8YBADDZAy2FnRo86v3yHeYN4deRSFP4nHebyZAgFH1ANhU4w0E3xhidZC4ZAidyBdXnyfjYZBUXqkwabU6rXJI8FoA4E1DZCfLaZCLuWsWICFgQtYSkY3woK49NyZAImfDn5xNTWe9XO3ZAOBiqDZCOlq6Tg6YpvrYuDICV1vAzodWqqT','EAAHZB2PDi8mwBAH4EWHSn9nKOC612ZAIy4njCB6wTN7JCWUopSCsPROB4rZAxnME1l867Xaxz0kpmFGSj62uqzbH61UbuA5eKWXGTqVGs1bDFQbbrwxIacc52X6ZBgpGtcjGcohr3ml4BJZBMhhmZCSxqBQ6zZC9aVek4pgTe5MOrlcsNl7kfnN']
token=at[1]
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2020,3,12,2,0,0),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=1),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    'execution_timeout': timedelta(minutes=900),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}
dag = DAG(
    'fb_ads_backfill',
    default_args=default_args,
    description='A dag to dump and mark our database with new ads',
    schedule_interval='@once',
)

dbase='POTUS_2020_DEV'
url1 = 'mongodb://%s:%s@denver.ischool.syr.edu'
mongoClient= pymongo.MongoClient(url1 % ('bitslab', '0rang3!'))
db = mongoClient[dbase]
col=db["fb_ads_backfillyang"]
data=[]

def get_data(ds, **kwargs):
    global data
    global page_ids
    global token
    global at
    col.remove()
    resp={}
    print("hellooo")
    t=time.time()
    for page in db['cand_info'].find({'backfill':None},no_cursor_timeout=True):
        print(page['candidate_name'])
        drp_dt=page['drop_date']
        ann_dt=page['announcement_date']
        d = date(2020, 2, 1)
        br=0
        y=1
        if type(drp_dt)!=float:
            if datetime.strptime(drp_dt, '%m/%d/%Y').date()<d:
                d=datetime.strptime(drp_dt, '%m/%d/%Y').date()
        ann_dt=datetime.strptime(ann_dt, '%m/%d/%Y').date()
        
#         if type(drp_dt)!=float:
#             drp_dt=datetime.strptime(drp_dt, '%m/%d/%Y').date()
#             if drp_dt>=dt:
#                 y=1
#         else:
#             y=1
        if y==1: 
            params = (('fields', 'ad_creation_time,ad_creative_body,ad_creative_link_caption,ad_creative_link_description,ad_creative_link_title,ad_delivery_start_time,ad_delivery_stop_time,ad_snapshot_url,currency,demographic_distribution,funding_entity,impressions,page_id,page_name,publisher_platforms,region_distribution,spend'),('ad_active_status','ALL'),('search_page_ids', page['page_id']),('ad_type', 'POLITICAL_AND_ISSUE_ADS'),('ad_reached_countries', '[\'US\']'),('access_token', token))
            resp = requests.get('https://graph.facebook.com/v5.0/ads_archive', params=params)
            resp_json = json.loads(resp.text)
            print(resp_json)
            x={}
            i=1
            ct=0
            collect=False
            while True:
                time.sleep(1)
                try:
                    ct=ct+1
                    if ct%100==0:
                        time.sleep(10)
                    try:
                        print("page: ",ct," date:",resp_json["data"][-1]["ad_delivery_start_time"][0:10])
                    except:
                        page['backfill']=1
                        db['cand_info'].update_one({'_id':page['_id']},{'$set': page}, True)
                        break
                    if not collect:
                        if datetime.strptime(resp_json['data'][0]["ad_delivery_start_time"][0:10], '%Y-%m-%d').date()<=d:
                            collect=True
                            print("started collecting")
                    if collect:
                        for item in resp_json["data"]:
                            item['marked']=0
                            col.insert_one(item)
                    try:
                        try:
                            if datetime.strptime(resp_json["data"][-1]["ad_delivery_start_time"][0:10], '%Y-%m-%d').date()<ann_dt:
                                print(1)
                                page['backfill']=1
                                db['cand_info'].update_one({'_id':page['_id']},{'$set': page}, True)
                                break
                        except KeyError:
                            print(resp_json["data"][-1])
                            if datetime.strptime(resp_json["data"][-1]["ad_delivery_start_time"][0:10], '%Y-%m-%d').date()<ann_dt:
                                print(2)
                                break
                    except IndexError:
                        try:
                            x["data"][-1]["ad_delivery_stop_time"][0:10]
                        except KeyError:
                            pass
                        pass
                    try:
                        x=resp_json
                        resp = requests.get(resp_json['paging']['next'])
                        resp_json=json.loads(resp.text)
                    except KeyError:
                        page['backfill']=1
                        db['cand_info'].update_one({'_id':page['_id']},{'$set': page}, True)
                        br=1
                        break
                except KeyError:
                    if time.time()-t<10:
                        break
                    t=time.time()
                    i=(i+1)%2
                    token=at[i]
                    new=x['paging']['next']
                    new=new.split('?')
                    atoken=new[1].split("&")
                    atoken[0]='access_token='+token
                    link=new[0]+'&'.join(atoken)
                    resp = requests.get(resp_json['paging']['next'])
                    resp_json=json.loads(resp.text)
                    print("token changed")
        else:
            print("Candidate Not Collected!")
    print(len(data))



            
get_data_step = PythonOperator(
    task_id='get_data',
    provide_context=True,
    python_callable=get_data,
    dag=dag,
)


        
stop_op = DummyOperator(task_id='stop_task', dag=dag)


get_data_step >> stop_op