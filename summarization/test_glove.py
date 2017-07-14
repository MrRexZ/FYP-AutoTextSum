#Sample code on how to perform training

import summarization.glove.tf_glove as tf_glove
import os
from settings import APP_ROOT

#1. Training using word-to-word coocurrence matrix as input :
crawl_output_dir = os.path.join(APP_ROOT, 'summarization', 'glove', 'crawl', 'output')
tf_glove.start_training(crawl_output_dir, 100, 'sparkcrawl')

#2. Training using corpus as input:
#corpus_file_path = os.path.join(APP_ROOT, 'path_to_file')
#tf_glove.start_training(corpus_file_path, 100, 'corpus')

# file = open(os.path.join(os.path.abspath(os.path.dirname(__file__)),"glove","crawl","input","test-01.txt"))
# model = tf_glove.GloVeModel(embedding_size=50, context_size=10)
# model.fit_to_corpus(file)
# model.train(num_epochs=100)
# print(model.embedding_for("This"))
# model.generate_tsne()

# file = open(os.path.join(os.path.abspath(os.path.dirname(__file__)),"glove","crawl","input","test-02.txt"))
# model = tf_glove.GloVeModel(embedding_size=50, context_size=10)
# model.fit_to_corpus(file)
# model.train(num_epochs=100)
# print(model.embedding_for("This"))
#
#
# file = open(os.path.join(os.path.abspath(os.path.dirname(__file__)),"glove","crawl","input","test-01.txt"))
# model = tf_glove.GloVeModel(embedding_size=50, context_size=10)
# model.fit_to_corpus(file)
# model.train(num_epochs=100)
# print(model.embedding_for("This"))