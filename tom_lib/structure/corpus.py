# coding: utf-8
# from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import networkx as nx
import itertools
import pandas
from networkx.readwrite import json_graph
from scipy import spatial
import re

import sqlite3 as lite
import datetime

__author__ = "Adrien Guille, Pavel Soriano"
__email__ = "adrien.guille@univ-lyon2.fr"


class Corpus:

    def __init__(self,
                 source_file_path,
                 enqueueTime=1504224000000, # Default Values, whole semester contained
                 dequeueTime=1513555200000,
                 language=None,
                 n_gram=1,
                 vectorization='tfidf',
                 max_relative_frequency=1.,
                 min_absolute_frequency=0,
                 max_features=2000,
                 sample=None):

        self._source_file_path = source_file_path
        self._language = language
        self._n_gram = n_gram
        self._vectorization = vectorization
        self._max_relative_frequency = max_relative_frequency
        self._min_absolute_frequency = min_absolute_frequency

        self.max_features = max_features

        try:
            con = lite.connect(source_file_path)

            query = "SELECT s.name as Student, ta.name as TA, enqueueTime, dequeueTime, QUESTION as Question " \
            "FROM QUEUEHISTORY q " \
            "JOIN Students s " \
            "ON q.NetId = s.NetId " \
            "JOIN TAS ta " \
            "ON q.removedBy = ta.NetId " \
            "WHERE PASSOFF = 'false' " \
            "AND enqueueTime BETWEEN " + str(enqueueTime) + " AND " + str(dequeueTime)
            # Not testing this stuff yet
            # The enqueue dequeue stuff
            # MAKE SURE TO UNCOMMENT THE SLASH IN THE PLACE ABOVE LEL

            self.data_frame = pandas.read_sql_query(query, con)
        except lite.Error as e:
            print("Error %s:" % e.args[0])
        finally:
            if con:
                con.close()

        # self.data_frame = pandas.read_csv(source_file_path, sep='\t', encoding='utf-8')
        if sample:
            self.data_frame = self.data_frame.sample(frac=0.8)
        self.data_frame.fillna(' ')
        self.size = self.data_frame.count(0)[0]

        # making changes here to help the stopwords function correctly
        # Hard coded kekekeke
        stop_words = [
        'i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they',
        'them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do',
        'does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during',
        'before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all',
        'any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don',
        'should','now','d','ll','m','o','re','ve','y','ain','aren','couldn','didn','doesn','hadn','hasn','haven','isn','ma','mightn','mustn','needn','shan','shouldn','wasn',
        'weren','won','wouldn','help','question', 'homework', 'questions', 'hw', 'lab', '1', '2', '3', '4', '5', '6', '7', '8', '9', '57', '59', '68', '69', '73', '85', '88']
        # if language is not None:
        #     stop_words = stopwords.words(language)
        if vectorization == 'tfidf':
            vectorizer = TfidfVectorizer(ngram_range=(1, n_gram),
                                         max_df=max_relative_frequency,
                                         min_df=min_absolute_frequency,
                                         max_features=self.max_features,
                                         stop_words=stop_words)
        elif vectorization == 'tf':
            vectorizer = CountVectorizer(ngram_range=(1, n_gram),
                                         max_df=max_relative_frequency,
                                         min_df=min_absolute_frequency,
                                         max_features=self.max_features,
                                         stop_words=stop_words)
        else:
            raise ValueError('Unknown vectorization type: %s' % vectorization)
        self.sklearn_vector_space = vectorizer.fit_transform(self.data_frame['Question'].tolist())
        self.gensim_vector_space = None
        vocab = vectorizer.get_feature_names()
        self.vocabulary = dict([(i, s) for i, s in enumerate(vocab)])

    def export(self, file_path):
        self.data_frame.to_csv(path_or_buf=file_path, sep='\t', encoding='utf-8')

    def full_text(self, doc_id):
        return self.data_frame.iloc[doc_id]['Question']

    def title(self, doc_id):
        return self.data_frame.iloc[doc_id]['Question']

    def date(self, doc_id):
        return self.data_frame.iloc[doc_id]['enqueueTime']

    def author(self, doc_id):
        aut_str = str(self.data_frame.iloc[doc_id]['Student'])
        return aut_str.split(', ')

    def affiliation(self, doc_id):
        aff_str = self.data_frame.iloc[doc_id]['TA']
        if self.data_frame.iloc[doc_id]['Student'] == self.data_frame.iloc[doc_id]['TA']:
            aff_str = 'Student'
        # aff_str = str(self.data_frame.iloc[doc_id]['Student'])
        return aff_str.split(', ')

    def documents_by_author(self, author, date=None):
        ids = []
        potential_ids = range(self.size)
        if date:
            potential_ids = self.doc_ids(date)
        for i in potential_ids:
            if self.is_author(author, i):
                ids.append(i)
        return ids

    def all_authors(self):
        author_list = []
        for doc_id in range(self.size):
            author_list.extend(self.author(doc_id))
        return list(set(author_list))

    def is_author(self, author, doc_id):
        return author in self.author(doc_id)

    def docs_for_word(self, word_id):
        ids = []
        for i in range(self.size):
            vector = self.vector_for_document(i)
            if vector[word_id] > 0:
                ids.append(i)
        return ids

    def doc_ids(self, date):
        return self.data_frame[self.data_frame['enqueueTime'] == date].index.tolist()

    def vector_for_document(self, doc_id):
        vector = self.sklearn_vector_space[doc_id]
        cx = vector.tocoo()
        weights = [0.0] * len(self.vocabulary)
        for row, word_id, weight in itertools.zip_longest(cx.row, cx.col, cx.data):
            weights[word_id] = weight
        return weights

    def word_for_id(self, word_id):
        return self.vocabulary.get(word_id)

    def id_for_word(self, word):
        for i, s in self.vocabulary.items():
            if s == word:
                return i
        return -1

    def similar_documents(self, doc_id, num_docs):
        doc_weights = self.vector_for_document(doc_id)
        similarities = []
        for a_doc_id in range(self.size):
            if a_doc_id != doc_id:
                similarity = 1.0 - spatial.distance.cosine(doc_weights, self.vector_for_document(a_doc_id))
                similarities.append((a_doc_id, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:num_docs]

    def collaboration_network(self, doc_ids=None, nx_format=False):
        nx_graph = nx.Graph(name='')
        for doc_id in doc_ids:
            authors = self.author(doc_id)
            for author in authors:
                nx_graph.add_node(author)
            for i in range(0, len(authors)):
                for j in range(i+1, len(authors)):
                    nx_graph.add_edge(authors[i], authors[j])
        bb = nx.betweenness_centrality(nx_graph)
        nx.set_node_attributes(nx_graph, 'betweenness', bb)
        if nx_format:
            return nx_graph
        else:
            return json_graph.node_link_data(nx_graph)
