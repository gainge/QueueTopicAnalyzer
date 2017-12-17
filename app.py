# coding: utf-8
import os
import shutil
import tom_lib.utils as utils
from flask import Flask, render_template, request, jsonify
from tom_lib.nlp.topic_model import NonNegativeMatrixFactorization
from tom_lib.structure.corpus import Corpus
import build_topic_model as builder

from random import *
import datetime

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"

# Flask Web server
app = Flask(__name__, static_folder='browser/static', template_folder='browser/templates')

# Hate that we hav to have these here
# Parameters
max_tf = 0.8
min_tf = 4
num_topics = 7
vectorization = 'tfidf'

DEFAULT_CLASS = 'cs240'
DEFAULT_START = 1504224000000
DEFAULT_END = 1513555200000

className = DEFAULT_CLASS

# builder.buildTopicModel(className, DEFAULT_START, DEFAULT_END)

corpus = builder.getCorpus(className, DEFAULT_START, DEFAULT_END)

MYDIR = os.path.dirname(__file__)

topic_model = utils.load_topic_model(os.path.join(MYDIR, builder.getTopicModelPath(className)))


def randomHelper():
    rng = []
    for i in range(3):
        rng.append(randint(0, 9))

    print("randomized url append: " + str(rng))
    return rng


def trimTimeStamp(timeStamp):
    return 0 # Stub

def getDateForTimestamp(timeStamp):
    timeStamp = timeStamp / 1000
    return datetime.datetime.fromtimestamp(timeStamp).strftime('%m/%d')



@app.route('/')
def index():
    return render_template('index.html',
                           topic_ids=range(topic_model.nb_topics),
                           doc_ids=range(corpus.size),
                           method=type(topic_model).__name__,
                           corpus_size=corpus.size,
                           vocabulary_size=len(corpus.vocabulary),
                           max_tf=max_tf,
                           min_tf=min_tf,
                           vectorization=vectorization,
                           num_topics=num_topics)

@app.route('/index.html')
def actualIndex():
    return index()


# So we'll actually probably have to make this like... a one for all method
# Like, it'll take in the date range and class, tell the builder to update the files
#   Then get even more data from the builder in order to update our own stuff (variables)
#   Then we'd have to re-render index.html?
#   Maybe in the html, on the button press we can have the page display a waiting thing or something
@app.route("/classChange/", methods=['POST'])
def changeClass():
    print("Yo Momma!")
    message = request.form['message']
    print(message)
    # Attempt at testing file output stuff
    builder.outputTest()

    # Attempt to read the random number we just generated
    output = "cool"
    with open(os.path.join(MYDIR, "output/random.txt"), "r") as f:
        output = f.read()


    # Let's just try to refresh the topic model for now
    # print("Telling Builder to Build the browser!!!")
    # builder.buildBrowser(className)
    # print("Reading corpus from builder")
    # corpus = builder.getCorpus(className)
    # print("Reading topic model from output directory")
    # topic_model = utils.load_topic_model(os.path.join(MYDIR, builder.getTopicModelPath(className)))

    return jsonify({ 'class': output }) # This is just json :)

@app.route("/newData/", methods=["POST"])
def newData():
    print("Request for new data... beginning calculations")
    className = request.form['className']
    startTime = request.form['startTime']
    endTime = request.form['endTime']

    print(className + '\t' + str(startTime) + '\t' + str(endTime))

    print("Telling Builder to Build the browser!!!")
    builder.buildBrowser(className, startTime, endTime)

    print("Reading corpus from builder")
    global corpus
    corpus = builder.getCorpus(className, startTime, endTime)

    print("Reading topic model from output directory")
    global topic_model
    topic_model = utils.load_topic_model(os.path.join(MYDIR, builder.getTopicModelPath(className)))

    return jsonify({ 'class': className }) # This is just json :)


@app.route('/topic_cloud.html')
def topic_cloud():
    return render_template('topic_cloud.html',
                           topic_ids=range(topic_model.nb_topics),
                           doc_ids=range(corpus.size))


@app.route('/vocabulary.html')
def vocabulary():
    word_list = []
    for i in range(len(corpus.vocabulary)):
        word_list.append((i, corpus.word_for_id(i)))
    splitted_vocabulary = []
    words_per_column = int(len(corpus.vocabulary)/5)
    for j in range(5):
        sub_vocabulary = []
        for l in range(j*words_per_column, (j+1)*words_per_column):
            sub_vocabulary.append(word_list[l])
        splitted_vocabulary.append(sub_vocabulary)
    return render_template('vocabulary.html',
                           topic_ids=range(topic_model.nb_topics),
                           doc_ids=range(corpus.size),
                           splitted_vocabulary=splitted_vocabulary,
                           vocabulary_size=len(word_list))


@app.route('/topic/<tid>.html')
def topic_details(tid):
    ids = topic_model.documents_per_topic()[int(tid)]
    documents = []
    for document_id in ids:
        documents.append((corpus.title(document_id),        # Removed .capitalize from this line (the title)
                          ', '.join(corpus.author(document_id)),
                          getDateForTimestamp(corpus.date(document_id)), document_id))
    return render_template('topic.html',
                           topic_id=tid,
                           frequency=round(topic_model.topic_frequency(int(tid))*100, 2),
                           documents=documents,
                           topic_ids=range(topic_model.nb_topics),
                           doc_ids=range(corpus.size))


@app.route('/document/<did>.html')
def document_details(did):
    vector = topic_model.corpus.vector_for_document(int(did))
    word_list = []
    for a_word_id in range(len(vector)):
        word_list.append((corpus.word_for_id(a_word_id), round(vector[a_word_id], 3), a_word_id))
    word_list.sort(key=lambda x: x[1])
    word_list.reverse()
    documents = []
    for another_doc in corpus.similar_documents(int(did), 5):
        documents.append((corpus.title(another_doc[0]),     # Removed .capitalize from this line (the title)
                          ', '.join(corpus.author(another_doc[0])),
                          getDateForTimestamp(corpus.date(another_doc[0])), another_doc[0], round(another_doc[1], 3)))

    date = corpus.date(int(did))
    return render_template('document.html',
                           doc_id=did,
                           words=word_list[:21],
                           topic_ids=range(topic_model.nb_topics),
                           doc_ids=range(corpus.size),
                           documents=documents,
                           authors=', '.join(corpus.author(int(did))),
                           year=getDateForTimestamp(date),
                           short_content=corpus.title(int(did)))


@app.route('/word/<wid>.html')
def word_details(wid):
    documents = []
    for document_id in corpus.docs_for_word(int(wid)):
        documents.append((corpus.title(document_id),    # Removed .capitalize from this line (the title)
                          ', '.join(corpus.author(document_id)),
                          corpus.date(document_id), document_id))
    return render_template('word.html',
                           word_id=wid,
                           word=topic_model.corpus.word_for_id(int(wid)),
                           topic_ids=range(topic_model.nb_topics),
                           doc_ids=range(corpus.size),
                           documents=documents)

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

if __name__ == '__main__':
    # Access the browser at http://localhost:2016/
    app.run(debug=True, host='localhost', port=8080)
