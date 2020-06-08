import csv
import numpy as np
from sklearn.manifold import TSNE

topic_mat = np.diag(np.array([1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00]))
print(topic_mat)
topic_label = np.array(["Topic1","Topic2","Topic3","Topic4","Topic5","Topic6","Topic7","Topic8","Topic9","Topic10"])

with open('./../Data/Anc_docTopic2FoodReview.csv') as csv_file:
    all_math = []
    all_lab = []
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(", ".join(row))
            line_count += 1
        else:
            all_math.append(np.array([row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10]]))
            all_lab.append(row[11])
            line_count += 1
    all_math = np.asarray(all_math)
    all_lab = np.asarray(all_lab)
    topic_mat = np.append(all_math, topic_mat, axis=0)
    topic_label = np.append(all_lab, topic_label)
    print(topic_mat)
    print(topic_label)

    model = TSNE(n_components=2, random_state=0)

    tsne_data = model.fit_transform(topic_mat)

    tsne_data = np.vstack((tsne_data.T, topic_label)).T

    with open('./../Data/tsne3.csv', mode='w', newline='') as csv_file:
        data_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data_writer.writerow(["X", "Y" ,"Topic"])
        for i in range(len(tsne_data)):
            data_writer.writerow([tsne_data[i][0], tsne_data[i][1], tsne_data[i][2]])
        
