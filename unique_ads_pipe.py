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

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2020,5,12,8,0,0),
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
    'execution_timeout': timedelta(hours=24),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}
dag = DAG(
    'fb_unique_ads_pipe',
    default_args=default_args,
    description='A dag to get unique ads',
    schedule_interval='@daily',
)
#schedule_interval='0 */6 * * *'
dbase='POTUS_2020_DEV'
url1 = 'mongodb://%s:%s@denver.ischool.syr.edu'
mongoClient= pymongo.MongoClient(url1 % ('bitslab', '0rang3!'))
db = mongoClient[dbase]
col=db["fb_ads_dev"]
unads=db["fb_ads_unique"]

def get_data(ds, **kwargs):
    ct=0
    for ad in col.find({'ad_delivery_start_time':{'$gte':'2020-02-01'},'agg':None},no_cursor_timeout=True):
        ct+=1
        if ct%500==0:
            print(ct)
        ad['agg']=1
        col.update_one({"_id":ad['_id']},{"$set": ad})
        try:
            datapt=unads.find_one({'ad_creative_body':ad['ad_creative_body']})
        except:
            continue
        if datapt:
            try:
                if ad['spend']['upper_bound']:
                        datapt['spend']+=((int(ad['spend']['lower_bound'])+int(ad['spend']['upper_bound']))/2)
                        sp=((int(ad['spend']['lower_bound'])+int(ad['spend']['upper_bound']))/2)
            except KeyError:
                datapt['spend']+=(int(ad['spend']['lower_bound']))
                sp=(int(ad['spend']['lower_bound']))
            try:
                if ad['impressions']['upper_bound']:
                    datapt['impressions']+=((int(ad['impressions']['lower_bound'])+int(ad['impressions']['upper_bound']))/2)
                    imp=((int(ad['impressions']['lower_bound'])+int(ad['impressions']['upper_bound']))/2)
            except:
                datapt['impressions']+=(int(ad['impressions']['lower_bound']))
                imp=(int(ad['impressions']['lower_bound']))
            try:
                for x in ad['demographic_distribution']:
                    x['spend']=sp*float(x['percentage'])
                for x in ad['demographic_distribution']:
                    x['impressions']=imp*float(x['percentage'])
                for x in ad['region_distribution']:
                    x['impressions']=imp*float(x['percentage'])
                for x in ad['region_distribution']:
                    x['spend']=sp*float(x['percentage']) 
            except:
                continue            
            datapt['demo_dist_arr'].append({'ad_delivery_start_time':ad['ad_delivery_start_time'],'dist':ad['demographic_distribution']})
            datapt['geo_dist_arr'].append({'ad_delivery_start_time':ad['ad_delivery_start_time'],'dist':ad['region_distribution']})
            datapt['first_created']=min(datapt['first_created'],ad['ad_delivery_start_time'])
            datapt['last_created']=max((datapt['last_created'],ad['ad_delivery_start_time']))
            for x in ad['publisher_platforms']:
                if x not in datapt['publisher_platforms']:
                    datapt['publisher_platforms'].append(x)  
            print(datapt['_id'])
            try:
                unads.update_one({"_id":datapt['_id']},{"$set": datapt})
            except:
                ad['agg']=0
                col.update_one({"_id":ad['_id']},{"$set": ad})
        else:
            datapt={}
            datapt['page_id']=ad['page_id']
            datapt['publisher_platforms']=ad['publisher_platforms']
            datapt['ad_creative_link_title']=''
            datapt['ad_creative_body']=''
            try:
                datapt['ad_creative_link_title']=ad['ad_creative_link_title']
            except:
                pass
            try:
                datapt['ad_creative_body']=ad['ad_creative_body']
            except:
                continue
            try:
                if ad['spend']['upper_bound']:
                    datapt['spend']=((int(ad['spend']['lower_bound'])+int(ad['spend']['upper_bound']))/2)
                    sp=((int(ad['spend']['lower_bound'])+int(ad['spend']['upper_bound']))/2)
            except KeyError:
                datapt['spend']=int(ad['spend']['lower_bound'])  
                sp=int(ad['spend']['lower_bound'])  
            try:
                if ad['impressions']['upper_bound']:
                    datapt['impressions']=((int(ad['impressions']['lower_bound'])+int(ad['impressions']['upper_bound']))/2)
                    imp=((int(ad['impressions']['lower_bound'])+int(ad['impressions']['upper_bound']))/2)
            except:
                datapt['impressions']=int(ad['impressions']['lower_bound'])
                imp==int(ad['impressions']['lower_bound'])
            try:
                for x in ad['demographic_distribution']:
                    x['spend']=sp*float(x['percentage'])
                for x in ad['demographic_distribution']:
                    x['impressions']=imp*float(x['percentage'])
                for x in ad['region_distribution']:
                    x['impressions']=imp*float(x['percentage'])
                for x in ad['region_distribution']:
                    x['spend']=sp*float(x['percentage'])
            except:
                continue
            datapt['demo_dist_arr']=[{'ad_delivery_start_time':ad['ad_delivery_start_time'],'dist':ad['demographic_distribution']}]
            datapt['geo_dist_arr']=[{'ad_delivery_start_time':ad['ad_delivery_start_time'],'dist':ad['region_distribution']}]
            datapt['first_created']=ad['ad_delivery_start_time']
            datapt['last_created']=ad['ad_delivery_start_time']
            datapt['first_ad_id']=ad['id']
            datapt['class']=ad['class']
            datapt['civility_class'] = ad['civility_class']
            datapt['topic'] = ad.get('topic',['no_topic'])
            unads.insert_one(datapt)
    
            
get_data_step = PythonOperator(
    task_id='get_data',
    provide_context=True,
    python_callable=get_data,
    dag=dag,
)




stop_op = DummyOperator(task_id='stop_task', dag=dag)


get_data_step >> stop_op
