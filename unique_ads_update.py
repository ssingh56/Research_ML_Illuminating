'''
bring the unique ads collection to current
'''

import datetime
import hashlib  # not needed if we have a better way to identify the creative
import json # only used to dump a test output
import pymongo

# airflow imports
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator

## -----------------------------
# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
## -----------------------------
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime.datetime(2020,5,12,8,0,0),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': datetime.timedelta(minutes=1),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime.datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': datetime.timedelta(hours=2),
    'execution_timeout': datetime.timedelta(minutes=3000),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}
## -----------------------------

## -----------------------------
## -----------------------------
dag = DAG(
    'fb_unique_ads_pipe_new',
    default_args=default_args,
    description='A dag to get unique ads',
#     schedule_interval='0 */6 * * *'
    schedule_interval='@daily'
)
## -----------------------------
server_url = "mongodb://%s:%s@denver.ischool.syr.edu" 
database_name = "POTUS_2020_DEV"
mongo_client = pymongo.MongoClient(server_url % ("bitslab", "0rang3!"))
database = mongo_client[database_name]

#--------------------------------------------------------------

def get_data(ds, **kwargs):
    ## -----------------------------
    ## get a handle to the mongo database
    ## -----------------------------
#    def get_database():
#         server_url = "mongodb://%s:%s@denver.ischool.syr.edu" % ("bitslab", "0rang3!")
#         database_name = "POTUS_2020_DEV"
#         mongo_client = pymongo.MongoClient(server_url)
#         return mongo_client[database_name]
    ## -----------------------------

    ## -----------------------------
    ## -----------------------------
    def get_average(measure):
        return round(
            (
                float(measure.get("lower_bound", 0)) +
                float(measure.get("upper_bound", measure.get("lower_bound", 0)))
            ) / 2
        )
    ## -----------------------------

    ## -----------------------------
    ## -----------------------------
    def ensure_ad_in_uniques(ad_buy, unique_ads_dictionary):
        ## -----------------------------
        # ensure this ad_buy is represented in the unique_ads_dictionary
        ## -----------------------------
        creative_body = ad_buy.get("ad_creative_body", "untitled ad from %s" % ad_buy["page_id"]).strip()
        creative_id = hashlib.sha256(creative_body.encode("utf-8")).hexdigest()

        if creative_id in unique_ads_dictionary:
            return (creative_id, unique_ads_dictionary)

        new_creative = {
            "page_id": ad_buy["page_id"],
            "publisher_platforms": ad_buy["publisher_platforms"],
            "ad_creative_link_title": ad_buy.get("ad_creative_link_title", "untitled ad from %s" % ad_buy["page_id"]),
            "ad_creative_body": creative_body,
            "ad_buys": 0,
            "spend": 0,
            "impressions": 0,
            "first_ad_id": ad_buy["id"],
            "first_created": ad_buy["ad_delivery_start_time"],
            "last_created": ad_buy["ad_delivery_start_time"],
            "distributions": {}
        }
        unique_ads_dictionary[creative_id] = new_creative
        return (creative_id, unique_ads_dictionary)
    ## -----------------------------

    ## -----------------------------
    ## -----------------------------
    def merge_ad_with_uniques(ad_buy, unique_ads_dictionary, creative_id):
        creative = unique_ads_dictionary[creative_id]

        creative["ad_buys"] += 1
        creative["spend"] += get_average(ad_buy["spend"])
        creative["impressions"] += get_average(ad_buy["impressions"])
        creative["last_created"] = ad_buy["ad_delivery_start_time"]
        creative["is_civil"] = ad_buy.get("civility_class", ["civil"])[0] == "civil"

        ## -----------------------------
        # topic coding
        ## -----------------------------
        topic_lookup = {
            "governance": "is_topic_governance",
            "economic": "is_topic_economic",
            "safety": "is_topic_safety",
            "education": "is_topic_education",
            "social and cultural": "is_topic_social_and_cultural",
            "health": "is_topic_health",
            "social programs": "is_topic_social_programs",
            "immigration": "is_topic_immigration",
            "environment": "is_topic_environment",
            "military": "is_topic_military",
            "covid": "is_topic_covid",
            "foreign policy": "is_topic_foreign_policy"
        }
        for key, value in topic_lookup.items():
            creative[value] = False

        for topic in ad_buy.get("topic", []):
            topic_id = topic_lookup.get(topic)
            if topic_id is not None:
                creative[topic_id] = True
        ## -----------------------------

        ## -----------------------------
        # type coding
        ## -----------------------------
        message_type_lookup = {
            "Advocacy": "is_message_type_advocacy",
            "Attack": "is_message_type_attack",
            "CTA": "is_message_type_call_to_action",
            "Call To Action": "is_message_type_call_to_action",
            "Image": "is_message_type_image",
            "Issue": "is_message_type_issue"
        }

        for key, value in message_type_lookup.items():
            creative[value] = False

        for message_type in ad_buy.get("class", []):
            message_type_id = message_type_lookup.get(message_type)
            if message_type_id is not None:
                creative[message_type_id] = True
        ## -----------------------------

        ## -----------------------------
        # demographics
        ## -----------------------------
        for demographic in ad_buy["demographic_distribution"]:
            for state in ad_buy["region_distribution"]:
                state_id = state["region"].lower().replace(" ", "_").replace("washington,_district_of_columbia", "district_of_columbia")
                pct = float(demographic["percentage"]) * float(state["percentage"])

                distribution = (
                    creative["distributions"]
                    .setdefault(demographic["gender"], {})
                    .setdefault(demographic["age"], {})
                    .setdefault(state_id, {"spend": 0, "impressions": 0})
                )
                distribution["spend"] += round(get_average(ad_buy["spend"]) * pct)
                distribution["impressions"] += round(get_average(ad_buy["impressions"]) * pct)
        ## -----------------------------
        
        return unique_ads_dictionary
    ## -----------------------------

    ## -----------------------------
    ## -----------------------------
    ad_buy_collection_name = "fb_ads_dev"
    ad_buy_collection_row_count = 0
    ad_buy_candidates = ("153080620724", "7860876103")
    ad_buy_delivery_start_time = "2020-06-01"
    ad_buy_delivery_end_time = "2021-09-21"
    ad_buy_collection_query = {
        "$and": [
            {"page_id": {"$in": ad_buy_candidates}},
            {"ad_delivery_start_time": {"$gte": ad_buy_delivery_start_time, "$lt": ad_buy_delivery_end_time}}
        ]
    }
    ## -----------------------------

    ## -----------------------------
    ## -----------------------------
    unique_ads_dictionary = {}
    unique_ads_collection_name = "jonsg_ads_unique"
    ## -----------------------------


    ## -----------------------------
    # Read each ad buy in the detail collection to construct a dictionary of "unique ads" with summary data.
    ## -----------------------------
    print("Reading ad buy collection:")
    for ad_buy in database[ad_buy_collection_name].find(ad_buy_collection_query, no_cursor_timeout=True):
        ad_buy_collection_row_count += 1
        print("\r\tcount: %s ad buy: %s" % (ad_buy_collection_row_count, ad_buy["id"]), flush=True, end="")

        ## -----------------------------
        # reject incomplete ads here
        ## -----------------------------
        if ad_buy.get("impressions", 0) == 0:
            print("")
            print("\t\tSkipping ad buy with no impressions")
            continue

        if ad_buy.get("demographic_distribution") is None:
            print("")
            print("\t\tSkipping ad buy with no distribution: " + ad_buy["ad_id"])
            continue

        if ad_buy.get("region_distribution") is None:
            print("")
            print("\t\tSkipping row with missing distribution: " + ad_buy["ad_id"])
            continue
        ## -----------------------------

        ## -----------------------------
        # ensure this ad_buy is represented in the unique_ads_dictionary
        ## -----------------------------
        creative_id, unique_ads_dictionary = ensure_ad_in_uniques(ad_buy, unique_ads_dictionary)
        ## -----------------------------

        ## -----------------------------
        # merge this ad_buy details into the unique_ads_dictionary
        ## -----------------------------
        unique_ads_dictionary = merge_ad_with_uniques(ad_buy, unique_ads_dictionary, creative_id)
        ## -----------------------------

    print("")
    print("Done processing ad_buys")
    ## -----------------------------

    ## -----------------------------
    # replace the mongo collection with the unique_ads_dictionary
    ## -----------------------------
    print("replacing unique ads collection")
    database[unique_ads_collection_name].drop()
    database[unique_ads_collection_name].insert_many([creative for creative_key, creative in unique_ads_dictionary.items()])
    print("done.")
    ## -----------------------------

    ## -----------------------------
    # testing
    ## -----------------------------
    #with open("test_results.json", mode="wt", encoding="utf-8") as destination_file:
    #    json.dump(unique_ads_dictionary, destination_file, sort_keys=True, indent=2, ensure_ascii=False, default=str)
    ## -----------------------------

#get_data(None)

get_data_step = PythonOperator(task_id='get_data', provide_context=True, python_callable=get_data, dag=dag)
stop_op = DummyOperator(task_id='stop_task', dag=dag)
get_data_step >> stop_op

