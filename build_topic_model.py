# coding: utf-8
import os
import shutil
import tom_lib.utils as utils
from flask import Flask, render_template
from tom_lib.nlp.topic_model import NonNegativeMatrixFactorization
from tom_lib.structure.corpus import Corpus

from random import *

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"



def outputTest():
    MYDIR = os.path.dirname(__file__)

    with open(os.path.join(MYDIR, "output/random.txt"), "w") as f:
        f.write(str(random()))

# Eventually we'll want this to take in parameters for the date range
# But for now, it's just good enough to load like we've always been doing
# It would also probably take in a class prefix parameter
#       like  CS240 or CS236
#       That way we could know what directory to save things into
#       And we'll have a way to provide the caller with the topic model file path
def getCorpus(className, startTime, endTime):
    # Parameters
    max_tf = 0.8
    min_tf = 4
    num_topics = 7
    vectorization = 'tfidf'

    MYDIR = os.path.dirname(__file__)

    # Should do whole semester by default
    return Corpus(source_file_path=os.path.join(MYDIR, getDataPathForClass(className)),   # Our own dataset!
                    language='english',     # The english language let's go
                    vectorization=vectorization,
                    enqueueTime=startTime,
                    dequeueTime=endTime,
                    max_relative_frequency=max_tf,
                    min_absolute_frequency=min_tf)

def buildTopicModel(className, startTime, endTime):
    print("Building Topic Model in build_topic_model.py")
    # Parameters
    max_tf = 0.8
    min_tf = 4
    num_topics = 7
    vectorization = 'tfidf'

    MYDIR = os.path.dirname(__file__)

    # Load corpus
    corpus = getCorpus(className, startTime, endTime)
    print('corpus size:', corpus.size)
    print('vocabulary size:', len(corpus.vocabulary))

    # Infer topics
    topic_model = NonNegativeMatrixFactorization(corpus=corpus)
    topic_model.infer_topics(num_topics=num_topics)
    topic_model.print_topics(num_words=10)

    # Save the topic model for reference
    # We'll just use a placeholder path for now
    utils.save_topic_model(topic_model, os.path.join(MYDIR, getTopicModelPath(className)))


def buildBrowser(className, startTime, endTime):
    # Parameters
    max_tf = 0.8
    min_tf = 4
    num_topics = 7
    vectorization = 'tfidf'

    MYDIR = os.path.dirname(__file__)

    # Load corpus
    corpus = getCorpus(className, startTime, endTime)
    print('corpus size:', corpus.size)
    print('vocabulary size:', len(corpus.vocabulary))

    # Infer topics
    topic_model = NonNegativeMatrixFactorization(corpus=corpus)
    topic_model.infer_topics(num_topics=num_topics)
    topic_model.print_topics(num_words=10)

    # Save the topic model for reference
    # We'll just use a placeholder path for now
    utils.save_topic_model(topic_model, os.path.join(MYDIR, getTopicModelPath(className)))

    MYDIR = os.path.dirname(__file__)

    # Clean the data directory
    if os.path.exists(os.path.join(MYDIR, 'browser/static/data')):
        shutil.rmtree(os.path.join(MYDIR, 'browser/static/data'))
    os.makedirs(os.path.join(MYDIR, 'browser/static/data'))

    # Export topic cloud
    utils.save_topic_cloud(topic_model, os.path.join(MYDIR,'browser/static/data/topic_cloud.json'))

    # Export details about topics
    for topic_id in range(topic_model.nb_topics):
        utils.save_word_distribution(topic_model.top_words(topic_id, 20),
                                     os.path.join(MYDIR, 'browser/static/data/word_distribution') + str(topic_id) + '.tsv')
        utils.save_affiliation_repartition(topic_model.affiliation_repartition(topic_id),
                                           os.path.join(MYDIR, 'browser/static/data/affiliation_repartition') + str(topic_id) + '.tsv')
        # evolution = []
        # for i in range(2012, 2016):
        #     evolution.append((i, topic_model.topic_frequency(topic_id, date=i)))
        # utils.save_topic_evolution(evolution, os.path.join(MYDIR, 'browser/static/data/frequency') + str(topic_id) + '.tsv')

    # Export details about documents
    for doc_id in range(topic_model.corpus.size):
        utils.save_topic_distribution(topic_model.topic_distribution_for_document(doc_id),
                                      os.path.join(MYDIR, 'browser/static/data/topic_distribution_d') + str(doc_id) + '.tsv')

    # Export details about words
    for word_id in range(len(topic_model.corpus.vocabulary)):
        utils.save_topic_distribution(topic_model.topic_distribution_for_word(word_id),
                                      os.path.join(MYDIR, 'browser/static/data/topic_distribution_w') + str(word_id) + '.tsv')

    # Associate documents with topics
    topic_associations = topic_model.documents_per_topic()

    # Export per-topic author network
    # The stuff below is giving us errors :(
    # I think it has something to do with anaconda python
    # Ultimate feelsbadman


    # for topic_id in range(topic_model.nb_topics):
    #     utils.save_json_object(corpus.collaboration_network(topic_associations[topic_id]),
    #                            'browser/static/data/author_network' + str(topic_id) + '.json')

def getTopicModelPath(className):
    return 'output/' + str(className) + 'topics.tom'

def getCorpusPath(className):
    return 'input/' + str(className) + 'help_queue_formatted_TOM.csv'

def getDataPathForClass(className):
    return 'input/' + str(className) + 'database.sqlite'


if __name__ == '__main__':
    buildBrowser('cs240', 1504224000000, 1513555200000)
    # buildTopicModel('cs240')
