## Graduate Research Assistantship - Machine Learning Engineer - AirflowDags

• Developed computational journalism analytics solution using Airflow pipelines to empower journalists on US Presidential Elections

• Managed Mongo databases for easy tracking of what candidates say on social media via their campaign accounts and paid ads

• Implemented machine learning models including SVM & BERT to classify/tag social media data according to its Topic & Civility

### Airflow Dags description:

#Facebook_2020_sample.py: A dag to collect sample 2020 facebook data

#ads_2020_sample.py: A dag to collect sample 2020 facebook ads data

#civil_tagger.py: A dag to dump and mark our database with new ads with their civility

#fb_ads_backfill.py: A dag to backfill and dump and mark our database with new ads

#fb_ads_cand_bylines_pacs_extended_pipe.py: A dag to update our database with fb ads for candidates running ads that have any activity in last N days on other pages bylines

#fb_ads_cand_bylines_pacs_pipe.py: A dag to update our database with fb ads for candidates running ads on other pages bylines

#fb_ads_extended_pipe.py: A dag to dump and mark our database with ads that have any activity in last N days

#fb_ads_pipe.py: A dag to dump and mark our database with new ads

#fb_posts.py: A dag to dump and mark our database with new facebook posts

#fb_posts_gaps_senate_2018_pipe.py: A dag to dump and mark our database with missing senate 2018 facebook posts

#fb_published_posts_pipe.py: A dag to dump and mark our database with new facebook posts

#image_tagger.py: A dag to dump and mark our database with new ads along with the message type as image

#instagram_pipe.py: A dag collect instagram data with crowdtangle api

#marking_script.py: A dag to dump and mark our database with new ads along with the message type

#twitter_2020_posts.py: A dag to dump and mark our database with new twitter posts

#twitter_2020_sample.py: A dag to collect sample 2020 twitter data

#unique_ads_pipe.py: A dag to get unique ads

#unique_ads_update.py: An updated dag to get unique ads
