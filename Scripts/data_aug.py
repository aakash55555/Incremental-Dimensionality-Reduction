import csv
import os
import random
import string
from random import seed
from random import random, choice

# seed(1)

def random_generator(size=9, chars=string.ascii_uppercase + string.digits + string.ascii_lowercase):
    return ''.join(choice(chars) for x in range(size))

with open(os.getcwd() + '/../Data/doc_topic.csv', mode='w', newline='') as csv_file:
    employee_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(["Document", "Topic0" ,"Topic1", "Topic2", "Topic3", "Topic4", "Topic5", "Topic6", "Topic7", "Topic8", "Topic9", "Dominant_topic"])
    for i in range(500):
        topics = []
        for _ in range(10):
            value = random()
            topics.append(round(value, 2))
        topics.append(topics.index(max(topics)))
        topics.insert(0, 'Doc'+str(i))
        employee_writer.writerow(topics)