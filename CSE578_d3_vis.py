# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 15:55:50 2019

@author: anchi
"""

import pickle
import time
import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from nltk.corpus import stopwords
from pylab import *
from bs4 import BeautifulSoup
import random
from nltk.stem.porter import PorterStemmer
import re
import nltk
from nltk.corpus import stopwords 
import string
from nltk.tag import pos_tag
from sklearn import metrics
from textblob import TextBlob as tb
from collections import OrderedDict

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

d=OrderedDict()
#d={}
text_blob=[]



###############################################################################
def tf(word, blob):
    return blob.words.count(word) / len(blob.noun_phrases)#.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.noun_phrases)#.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

def tokenize(text):
    text= str(text).lower()    
    text = text.replace("'", "")          
    text = re.sub('(\d{1,4})[/.-:](\d{1,2})[/.-:](\d{0,4})', ' ', text).strip()      
    text = re.sub(r"\b[-+]?[\d.]+\b", ' ', text)    
    text = re.sub('[^A-Za-z0-9]+', ' ', text)        
#     words = [word for word in nltk.word_tokenize(text)] #for sent in nltk.sent_tokenize(text)      
#     words = [w for w in words if w not in stop_words and len(w)>=2 ]          
#     words = [stemmer.stem(i) for i in words]   
    return text#words
###############################################################################

def build_LDA_Model(tfs):
    lda_model = LatentDirichletAllocation(n_topics=8,               # Number of topics
                                          max_iter=10,               
    # Max learning iterations
                                          learning_method='online',   
                                          random_state=100,          
    # Random state
                                          batch_size=128,            
    # n docs in each learning iter
                                          evaluate_every = -1,       
    # compute perplexity every n iters, default: Don't
                                          n_jobs = -1,               
    # Use all available CPUs
                                         )
    lda_output = lda_model.fit_transform(tfs)
    print(lda_model)  # Model attributes
    
    #Analyzed model performance with per
    

def classify_doc_topics(docnames,tfs, feature_names,best_lda_model):
    # Create Document — Topic Matrix
      = best_lda_model.transform(tfs)
    # column names
    topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_topics)]
    # index names
#     docnames = ["Doc" + str(i) for i in range(len(data))]
    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic
    
    print('df_document_topics: ',df_document_topic)

    # Topic-Keyword Matrix
    df_topic_keywords = pd.DataFrame(best_lda_model.components_)
    # Assign Column and Index
    df_topic_keywords.columns = feature_names
    df_topic_keywords.index = topicnames
    
    df_document_topic.to_csv('doc_topic.csv')
    df_topic_keywords.to_csv('topic_keywords.csv')
    # View
    print(df_topic_keywords.head())
    
    return df_document_topic,topicnames,lda_output    

def Gridserachcv_LDA(tfs):
    # Define Search Param
    search_params = {'n_topics': [8], 'learning_decay': [0.2,0.3,.4,.5,.6, .7, .9]}#3,5,6,7,8
    # Init the Model
    lda = LatentDirichletAllocation(max_iter=5, learning_method='online', learning_offset=50.,random_state=0)
    # Init Grid Search Class
    model = GridSearchCV(lda, param_grid=search_params)
    # Do the Grid Search
    model.fit(tfs)
#     GridSearchCV(cv=None, error_score='raise',
#            estimator=LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
#                  evaluate_every=-1, learning_decay=0.7, learning_method=None,
#                  learning_offset=10.0, max_doc_update_iter=100, max_iter=10,
#                  mean_change_tol=0.001, n_topics=10, n_jobs=1,
# #                  n_topics=None, perp_tol=0.1, random_state=None,
#                  topic_word_prior=None, total_samples=1000000.0, verbose=0),
#            fit_params=None, iid=True, n_jobs=1,
#            param_grid={'n_topics': [2,3,45,6,10, 15, 20, 25, 30], 'learning_decay': [0.5, 0.7, 0.9]},
#            pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
#            scoring=None, verbose=0)
    
    # Best Model
    best_lda_model = model.best_estimator_
    # Model Parameters
    print("Best Model's Params: ", model.best_params_)
    # Log Likelihood Score
    print("Best Log Likelihood Score: ", model.best_score_)
    # Perplexity
    print("Model Perplexity: ", best_lda_model.perplexity(tfs))
    return best_lda_model




def text_blob_process(corpus_lst,df_result):
    # print('corpus_lst: ', corpus_lst)
    for i, blob in enumerate(corpus_lst):
        print("Top words in document {}".format(i + 1))
    #     for word in blob.noun_phrases:#words:
    #         print(word)
        print(type(blob))
        scores = {word: tfidf(word, (blob), corpus_lst) for word in (blob).words}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words[:15]:
#             d[(i,word)]=score  #Kalpana
            d[(df_result.iloc[i]['label'],word)]=score
            text_blob.append(word)
            #print("\tWord: {}, TF-IDF: {}".format(word, round(score, 15)))
    return text_blob,d

# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

def tb_conversion_df(d):
    #Kalpana
    print('Dictionary start: ')
    lst_colname=[]
    lst_idx=[]
    lst_colname.append('doc')
    for key, value in d.items():
        lst_colname.append(key[1])
        lst_idx.append(key[0])
        print (key[0], " : " , key[1], " : ", value)     
    
    df = pd.DataFrame(columns=list(set(lst_colname)), index= lst_idx)
    i=0
    for key, value in d.items():    
        df.iloc[i][key[1]] = value    
        i = i + 1
    df=df.fillna(0)  
    return df
    
def svm_model(df_result, text_blob):
    
    tfidf_vectorizer = TfidfVectorizer(vocabulary = list(set(text_blob)),ngram_range=(1, 5))#,tokenizer=tokenize, stop_words=stop_words
    X = tfidf_vectorizer.fit_transform(df_result['text_clean'])
    y = df_result['label']
    
    ovr = OneVsRestClassifier(LinearSVC(random_state=0,C=0.6)).fit(X, y)
    distance_to_decision_boundary = ovr.decision_function(X)
    print('X: ',list(distance_to_decision_boundary))
    df_lst = pd.DataFrame(distance_to_decision_boundary) #, columns=['Label0','Label1','Label2','Label3']
    df_lst.to_csv('svm_distance.csv')
    print(df_lst)
#     print('y: ',y)

    prediction=ovr.predict(X)
    df_result['prediction'] = prediction
    df_result.to_csv('df_result_svm_v2.csv')
    # model accuracy for X_test   
    accuracy = ovr.score(X, y) 
    print('accuracy: ', accuracy)  
    # creating a confusion matrix 
    cm = confusion_matrix(y, prediction) 
    print(cm)
    
def gridsearch_svm_model(df_result, text_blob):
    
    print('text_blob: ', text_blob)
#     df_result['text_clean'] = df_result['sentiment'].apply(lambda x: remove_stopwords(x))
    tfidf_vectorizer = TfidfVectorizer(vocabulary = list(set(text_blob)),ngram_range=(1, 5))#,tokenizer=tokenize, stop_words=stop_words
    xtrain_tfidf = tfidf_vectorizer.fit_transform(df_result['text_clean'])
    ytrain = df_result['label']
    
    svm_model(xtrain_tfidf, ytrain, df_result)
    print('ytrain: ', ytrain)
    parameters = {'estimator__kernel':('linear', 'rbf', 'poly'), 'estimator__C':[0.1,0.01,0.001,0.2,0.02,0.3,0.03,0.4,0.04,0.5,0.05,0.6,0.7,0.8, 1]}
    svr = OneVsRestClassifier(SVC(probability=True))
    clf = GridSearchCV(svr, parameters)
    clf.fit(xtrain_tfidf,ytrain)
    
    prediction=clf.predict(xtrain_tfidf)
    print('prediction: ', prediction)
    df_result['prediction'] = prediction
    df_result.to_csv('D:/df_result.csv')
    # model accuracy for X_test   
    accuracy = clf.score(xtrain_tfidf, ytrain) 
    print('accuracy: ', accuracy)  
    # creating a confusion matrix 
    cm = confusion_matrix(ytrain, prediction) 
    print(cm)
    
    distance_to_decision_boundary = clf.decision_function(xtrain_tfidf)
    print('X: ',list(distance_to_decision_boundary))
    df_lst = pd.DataFrame(distance_to_decision_boundary)#, columns=['Label0','Label1','Label2','Label3']
    print('distance: ',df_lst)
#     print('y: ',y)

#     prediction=clf.predict(corpus_lst)
#     print('prediction: ',prediction)
    print('best_param: ',clf.best_params_) 

#     # print how our model looks after hyper-parameter tuning 
#     print(clf.best_score_) 

#     # print classification report 
#     print(classification_report(y, prediction)) 

file_name=[]
corpus_lst=[]
def read_dir_into_df_txt(fpath):
    
    dirListing = os.listdir(fpath)
    bloblist = []
    stop_words = set(stopwords.words('english')) 
    simi_result=[]
    df=pd.DataFrame()
    corpus = []
    for item in dirListing:
        if ".txt" in item:
            text_file = (fpath+'\\'+item)
            file_name.append(item)
            corp_txt=''
            text1=''
            with open (text_file) as f:
                corp_txt = f.read()
                
                #my1 = [word for word in tokenize(f.read()) if word not in stopwords.words('english')]
                text1 = ' '.join(corp_txt)
                corpus.append(f.read() ) #kalpana                
#                 corpus_lst.append(f.read())
                f.close()
            print('corp_txt: ',corp_txt)
    df_text = pd.DataFrame(corpus,columns=['text']) 
    df_text['doc_name'] = list(file_name)
    df_text.set_index('doc_name', inplace=True)
    print(file_name)
    return df_text,file_name, corpus_lst


def sentiment_calc(text):
    try:
#         text = tokenize(text)
        return tb(text)#.sentiment
    except:
        return None




def read_dir_into_df_csv(filepath):
    df_csv = pd.read_json(filepath+"food")
    
    df_csv['text_clean'] = df_csv['text'].apply(lambda x: remove_stopwords(x))
    df_csv['sentiment'] = df_csv['text_clean'].apply(sentiment_calc)
    corpus_lst=df_csv['sentiment']
    return df_csv,corpus_lst

def read_dir_into_df_json(filepath):
    papers = pd.read_json(filepath+'arxivData.json')
    papers['paper_summary_processed'] = papers['summary'].apply(lambda x: remove_stopwords(x))
    papers['paper_summary_processed'] = papers['summary'].map(lambda x: re.sub('[,\.!?]', '', x))
    papers['paper_summary_processed'] = papers['summary'].map(lambda x: x.lower())
   # summary['sentiment'] = df_csv['text_clean'].apply(sentiment_calc)
    #corpus_lst=df_csv['sentiment']
    return papers

class LabelRanker(object):
    """
    
    """
    def __init__(self,
                 apply_intra_topic_coverage=True,
                 apply_inter_topic_discrimination=True,
                 mu=0.7,
                 alpha=0.9):
        self._coverage = apply_intra_topic_coverage
        self._discrimination = apply_inter_topic_discrimination
        self._mu = mu
        self._alpha = alpha

    def label_relevance_score(self,
                              topic_models,
                              pmi_w2l):
        """
        Calculate the relevance scores between each label and each topic
        Parameters:
        ---------------
        topic_models: numpy.ndarray(#topics, #words)
           the topic models
        pmi_w2l: numpy.ndarray(#words, #labels)
           the Point-wise Mutual Information(PMI) table of
           the form, PMI(w, l | C)
        
        Returns;
        -------------
        numpy.ndarray, shape (#topics, #labels)
            the scores of each label on each topic
        """
        assert topic_models.shape[1] == pmi_w2l.shape[0]
        return np.asarray(np.asmatrix(topic_models) *
                          np.asmatrix(pmi_w2l))
        
    def label_discriminative_score(self,
                                   relevance_score,
                                   topic_models,
                                   pmi_w2l):
        """
        Calculate the discriminative scores for each label
        
        Returns:
        --------------
        numpy.ndarray, shape (#topics, #labels)
            the (i, j)th element denotes the score
            for label j and all topics *except* the ith
        """
        assert topic_models.shape[1] == pmi_w2l.shape[0]
        k = topic_models.shape[0]
        return (relevance_score.sum(axis=0)[None, :].repeat(repeats=k, axis=0)
                - relevance_score) / (k-1)
        
    def label_mmr_score(self,
                        which_topic,
                        chosen_labels,
                        label_scores,
                        label_models):
        """
        Maximal Marginal Relevance score for labels.
        It's computed only when `apply_intra_topic_coverage` is True
        Parameters:
        --------------
        which_topic: int
            the index of the topic
        
        chosen_labels: list<int>
           indices of labels that are already chosen
        
        label_scores: numpy.ndarray<#topic, #label>
           label scores for each topic
        label_models: numpy.ndarray<#label, #words>
            the language models for labels
        Returns:
        --------------
        numpy.ndarray: 1D of length #label - #chosen_labels
            the scored label indices
        numpy.ndarray: same length as above
            the scores
        """
        chosen_len = len(chosen_labels)
        if chosen_len == 0:
            # no label is chosen
            # return the raw scores
            return (np.arange(label_models.shape[0]),
                    label_scores[which_topic, :])
        else:
            kl_m = np.zeros((label_models.shape[0]-chosen_len,
                             chosen_len))
            
            # the unchosen label indices
            candidate_labels = list(set(range(label_models.shape[0])) -
                                    set(chosen_labels))
            candidate_labels = np.sort(np.asarray(candidate_labels))
            for i, l_p in enumerate(candidate_labels):
                for j, l in enumerate(chosen_labels):
                    kl_m[i, j] = kl_divergence(label_models[l_p],
                                               label_models[l])
            sim_scores = kl_m.max(axis=1)
            mml_scores = (self._alpha *
                          label_scores[which_topic, candidate_labels]
                          - (1 - self._alpha) * sim_scores)
            return (candidate_labels, mml_scores)

    def combined_label_score(self, topic_models, pmi_w2l,
                             use_discrimination, mu=None):
        """
        Calculate the combined scores from relevance_score
        and discrimination_score(if required)
        Parameter:
        -----------
        use_discrimination: bool
            whether use discrimination or not
        mu: float
            the `mu` parameter in the algorithm
        Return:
        -----------
        numpy.ndarray, shape (#topics, #labels)
            score for each topic and label pair
        """
        rel_scores = self.label_relevance_score(topic_models, pmi_w2l)
        
        if use_discrimination:
            assert mu != None
            discrim_scores = self.label_discriminative_score(rel_scores,
                                                             topic_models,
                                                             pmi_w2l)
            label_scores = rel_scores - mu * discrim_scores
        else:
            label_scores = rel_scores

        return label_scores

    def select_label_sequentially(self, k_labels,
                                  label_scores, label_models):
        """
        Return:
        ------------
        list<list<int>>: shape n_topics x k_labels
        """
        n_topics = label_scores.shape[0]
        chosen_labels = []

        # don't use [[]] * n_topics !
        for _ in xrange(n_topics):
            chosen_labels.append(list())
            
        for i in xrange(n_topics):
            for j in xrange(k_labels):
                inds, scores = self.label_mmr_score(i, chosen_labels[i],
                                                    label_scores,
                                                    label_models)
                chosen_labels[i].append(inds[np.argmax(scores)])
        return chosen_labels

    def top_k_labels(self,
                     topic_models,
                     pmi_w2l,
                     index2label,
                     label_models=None,
                     k=5):
        """
        Parameters:
        ----------------
        
        index2label: dict<int, object>
           mapping from label index in the `pmi_w2l`
           to the label object, which can be string
        label_models: numpy.ndarray<#label, #words>
            the language models for labels
            if `apply_intra_topic_coverage` is True,
            then it's must be given
        Return:
        ---------------
        list<list of (label, float)>
           top k labels as well as scores for each topic model
        """

        assert pmi_w2l.shape[1] == len(index2label)

        label_scores = self.combined_label_score(topic_models, pmi_w2l,
                                                 self._discrimination,
                                                 self._mu)

        if self._coverage:
            assert isinstance(label_models, np.ndarray)
            # TODO: can be parallel
            chosen_labels = self.select_label_sequentially(k, label_scores,
                                                           label_models)
        else:
            chosen_labels = np.argsort(label_scores, axis=1)[:, :-k-1:-1]
        return [[index2label[j]
                 for j in topic_i_labels]
                for topic_i_labels in chosen_labels]
            
    def print_top_k_labels(self, topic_models, pmi_w2l,
                           index2label, label_models, k):
        res = u"Topic labels:\n"
        for i, labels in enumerate(self.top_k_labels(
                topic_models=topic_models,
                pmi_w2l=pmi_w2l,
                index2label=index2label,
                label_models=label_models,
                k=k)):
            res += u"Topic {}: {}\n".format(
                i,
                ', '.join(map(lambda l: ' '.join(l),
                              labels))
            )
        return res

def main():
    file_path=r'C:/Users/anchi/Downloads/arxivdataset/'
     df_result,file_name,corpus_lst = read_dir_into_df_txt(file_path)
    #df_result = read_dir_into_df_json(file_path)
    
#     print(df_result)
#     corpus_lst=[document1,document2,document3,document4]
    text_blob,d = text_blob_process(corpus_lst,df_result)
    df = tb_conversion_df(d)  
    df.to_csv(file_path+'df_result_hmm.csv')
#     build_LDA_Model(df)
    best_lda_model = Gridserachcv_LDA(df)
    doc_topic,topicnames,lda_output = classify_doc_topics(df.index,df, df.columns, best_lda_model)
#     print('doc_topic: ', list(doc_topic))
#     print('lda_output: ', lda_output)
#     x = df#lda_output
#     print(x.shape)
#     print('lda_output: ', lda_output.shape)
#     y = df.index #df_result['label']#
#     print(y.shape)
    
    
#     gridsearch_svm_model(df_result, text_blob)
    #svm_model(df_result, text_blob)
#     svm_model(x,y)
    
main()  

