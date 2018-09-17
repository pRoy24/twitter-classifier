import time
import schedule
import os
import math
import socket
from classify_batch import start_classify_batch

from cluster_batch import start_cluster_batch
import warnings
warnings.filterwarnings("always")


def job():
    print("Running classify batch")
    start_classify_batch()
    print("Running cluster into buckets")
    start_cluster_batch()


schedule.every(10).minutes.do(job)


while True:
    schedule.run_pending()
    time.sleep(1)