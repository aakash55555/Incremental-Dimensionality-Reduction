#!/usr/bin/env python
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from tempfile import NamedTemporaryFile
import shutil
import csv
import numpy as np
import warnings
from random import random, choice
from sklearn.manifold import TSNE

# warnings.simplefilter(action='ignore', category=FutureWarning)

# HTTPRequestHandler class
class GibbsClassifier(BaseHTTPRequestHandler):

    def do_OPTIONS(self):           
        self.send_response(200, "ok")       
        self.send_header('Access-Control-Allow-Origin', '*')                
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With") 
    # GET
    def do_GET(self):
        query = urlparse(self.path).query
        query_components = dict(qc.split("=") for qc in query.split("&"))
        id = query_components["id"]
        label_id = query_components["label_id"]
        posno = query_components["posno"]
        new_label = query_components["newlabel"]
        ds = query_components["ds"]
        print(ds)
        if ds == '1':
            print('here')
            topic_label = './../Data/Topic_Label1.csv'
            label_url = './../Data/Anc_label1.csv'
        else:
            topic_label = './../Data/Topic_Label2.csv'
            label_url = './../Data/Anc_label2.csv'
        top_lab_field = ["Label", "Topic1", "Topic2", "Topic3", "Topic4", "Topic5", "Topic6", "Topic7", "Topic8", "Topic9", "Topic10"]
        fields = ['id', 'Label','Annotated Positive Document','Correct Annotated Positive Document','Annotated Negative Document','Correct Annotated Negative Document', 'summary', 'Document']
        if new_label == 'yes':
            with open(topic_label, mode='a', newline='') as csvfile2:
                writer_h = csv.writer(csvfile2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                new_row = []
                new_row.append(label_id)
                for _ in range(10):
                    value = random()
                    new_row.append(round(value, 2))
                writer_h.writerow(new_row)
        # Send response status code
        print(id, label_id)
        randomnums = np.random.randint(1, 500, 50)
        randomnums2 = [str(randomnums[k]) for k in range(len(randomnums))]
        print(randomnums2)
        tempfile = NamedTemporaryFile(mode='w', delete=False, newline='')
        tempfile2 = NamedTemporaryFile(mode='w', delete=False, newline='')
        if new_label == 'no':
            with open(topic_label, mode='r') as csvfile3, tempfile2:
                reader = csv.DictReader(csvfile3, fieldnames=top_lab_field)
                writer = csv.DictWriter(tempfile2, fieldnames=top_lab_field)
                for row in reader:
                    if row["Label"] == label_id:
                        for i in range(10):
                            row[str("Topic"+str(i))] = round(random(), 2)
                    row = {'Label': row["Label"], 'Topic1': row["Topic1"], 'Topic2': row["Topic2"], 'Topic3': row["Topic3"], 'Topic4': row["Topic4"], 'Topic5': row["Topic5"], 'Topic6': row["Topic6"], 'Topic7': row["Topic7"], 'Topic8': row["Topic8"], 'Topic9': row["Topic9"], 'Topic10': row["Topic10"]}
                    writer.writerow(row)
            shutil.move(tempfile2.name, topic_label)
        with open(label_url, 'r') as csvfile, tempfile:
            reader = csv.DictReader(csvfile, fieldnames=fields)
            writer = csv.DictWriter(tempfile, fieldnames=fields)
            for row in reader:
                if row["id"] == id:
                    print('UPDATING ROW HERE', row['id'])
                    if posno == 'yes':
                        row['Label'], row['Correct Annotated Positive Document'], row['Annotated Positive Document'], row['Correct Annotated Negative Document'], row['Annotated Negative Document'] = label_id, label_id, '', '', ''
                    elif posno == 'no':
                        row['Label'], row['Correct Annotated Negative Document'], row['Annotated Positive Document'], row['Correct Annotated Positive Document'], row['Annotated Negative Document'] = label_id, label_id, '', '', ''
                
                elif row["id"] in randomnums2:
                    print('UPDATING ROW HERE2', row['id'])
                    if posno == 'yes':
                        row['Label'], row['Annotated Positive Document'], row['Correct Annotated Positive Document'], row['Annotated Negative Document'], row['Correct Annotated Negative Document'] = label_id, label_id, '', '', ''
                    elif posno == 'no':
                        row['Label'], row['Annotated Negative Document'], row['Correct Annotated Positive Document'], row['Annotated Positive Document'], row['Correct Annotated Negative Document'] = label_id, label_id, '', '', ''
                row = {'id': row['id'], 'Label': row['Label'], 'Annotated Positive Document': row['Annotated Positive Document'], 'Correct Annotated Positive Document': row['Correct Annotated Positive Document'], 'Annotated Negative Document': row['Annotated Negative Document'], 'Correct Annotated Negative Document': row['Correct Annotated Negative Document'], 'summary': row['summary'], 'Document': row['Document']}
                writer.writerow(row)
        shutil.move(tempfile.name, label_url)
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.end_headers()
        message = 'success'
        self.wfile.write(message.encode('utf-8'))
        return

def run():
  print('starting server...')

  # Server settings
  # Choose port 8080, for port 80, which is normally used for a http server, you need root access
  server_address = ('127.0.0.1', 8081)
  httpd = HTTPServer(server_address, GibbsClassifier)
  print('running server...')
  httpd.serve_forever()


run()