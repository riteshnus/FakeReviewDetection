from __future__ import print_function
import datetime
import numpy as np
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import pandas as pd
import webhoseio
import time
import logging

default_dag_args = {'start_date': datetime.datetime.now() - datetime.timedelta(minutes=5)}

with DAG('fake_review', schedule_interval=datetime.timedelta(minutes =2), default_args=default_dag_args) as dag:
    def extract_data_from_webhose():
        webhoseio.config(token="7ad89131-980e-48c3-b588-e68adb7c1be0")
        s = int(time.time()) - 500
        query_params = {"q": "language:english site:amazon.com site_category:shopping spam_score:>0.7", "ts": "{}".format(s), "sort": "crawled"}
        output = webhoseio.query("filterWebContent", query_params)
        logging.info(output)
        key = []
        reviewname = []
        productname = []
        reviewdate = []
        rating = []
        label = []
        sourcetype = []
        runtype = []
        spam_score = []
        text = []
        for i in range(0,1):
            logging.info(i)
            logging.info(output)
            key.append(i)
            reviewname.append(output['posts'][i]['author'])
            productname.append(output['posts'][i]['thread']['title'])
            reviewdate.append(output['posts'][i]['thread']['published'])
            rating.append(output['posts'][i]['thread']['rating'])
            tt = output['posts'][i]['text']
            text.append(tt)
            ss = output['posts'][i]['thread']['spam_score']
            spam_score.append(ss)
        df= pd.DataFrame()
        df['key'] = key
        df['reviewname'] = reviewname
        df['productname'] = productname
        df['reviewdate'] = reviewdate
        df['rating'] = rating
        df['label'] = 'fake'
        df['sourcetype'] = 'amazon'
        df['runtype'] = 'near_real_time'
        df['text'] = text  
        df['snapshot_time'] = s
		
        df.to_gbq('webhoseDB.staging_table', 'wehosestream', if_exists='append', verbose=False)
    
    write_python = PythonOperator(task_id='pywritebq', python_callable=extract_data_from_webhose)
