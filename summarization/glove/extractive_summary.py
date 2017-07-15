import gensim
import nltk
import numpy
import string
import gcs_helper
import os, platform
from settings import APP_ROOT

from summarization.glove import glovegensim

def load_model(type):
    if type.lower() == 'word2vec':
        return load_word2vec_model()
    elif type.lower() == 'glove':
        return load_glove_model()
    else:
        raise FileNotFoundError


def load_word2vec_model():
    #TODO: Fix destination path
    model_fname = gcs_helper.download_blob("glove-tf-model", "glove_model.txt",
                                           os.path.join(APP_ROOT, "glove_model.txt"))
    gensim_file = model_fname
    #use_existing_glove_dataset(gensim_file)
    model = gensim.models.KeyedVectors.load_word2vec_format(gensim_file, binary=False)
    return model

def use_existing_glove_dataset(gensim_file):
    glove_file = "glove.6B.50d.txt"
    _, tokens, dimensions, _ = glove_file.split('.')
    num_lines = glovegensim.check_num_lines_in_glove(glove_file)
    dims = int(dimensions[:-1])
    num_lines = glovegensim.check_num_lines_in_glove(glove_file)
    gensim_first_line = "{} {}".format(num_lines, dims)
    if platform == "linux" or platform == "linux2":
        glovegensim.prepend_line(glove_file, gensim_file, gensim_first_line)
    else:
        glovegensim.prepend_slow(glove_file, gensim_file, gensim_first_line)


def load_glove_model():
    import summarization.glove.tf_glove as tf_glove
    model = tf_glove.GloVeModel(embedding_size=50, context_size=10)
    return model


def get_ex_sum(glove_model, doc):
    #Defaulting dim size to 50
    dim = 50
    document = doc
    sentences_tokens = nltk.tokenize.sent_tokenize(document)
    doc_vectors = _convert_document_to_sentences_vectors(glove_model, sentences_tokens, dim)
    gbl_median, sentences_median = _calc_median(doc_vectors)
    summarized_sentences = _sentences_above_median(doc_vectors, sentences_tokens, sentences_median, gbl_median)
    return summarized_sentences


def _sentences_above_median(doc_vectors, sentences_tokens, sentences_median, gbl_median):
    summarizedSentence = []
    numOfSentence = range(len(doc_vectors))
    for index in numOfSentence:
        if sentences_median[index] >= gbl_median:
            summarizedSentence.append(sentences_tokens[index])
    return summarizedSentence


def _calc_median(sentencevector):
    import scipy.spatial.distance
    senvecrange = range(len(sentencevector))
    sentences_median = numpy.empty(len(sentencevector), dtype=float)
    for f_senVecIndex in senvecrange:
        curSecVecComparison = numpy.empty(len(sentencevector), dtype=float)
        for s_senVecIndex in senvecrange:
            if s_senVecIndex != f_senVecIndex:
                cos_dist = scipy.spatial.distance.cosine(sentencevector[f_senVecIndex],
                                                                       sentencevector[s_senVecIndex])
                if not numpy.isnan(cos_dist):
                    curSecVecComparison[s_senVecIndex] = 1 - cos_dist
                else:
                    curSecVecComparison[s_senVecIndex] = 0
            else:
                curSecVecComparison[s_senVecIndex] = 1
        sentences_median[f_senVecIndex] = numpy.median(curSecVecComparison)

    globalMedian = numpy.median(sentences_median)
    if numpy.isnan(globalMedian):
        print(sentences_median)
        print(globalMedian)
        print("YEAH")
    return (globalMedian, sentences_median)


def _convert_document_to_sentences_vectors(glove_model, sentence, dim):
    sentences_vector = numpy.zeros((len(sentence), dim))
    wordsInSentence = numpy.empty((len(sentence)), dtype=object)
    for index in range(len(sentence)):
        removepunctuationtranslator = str.maketrans('', '', string.punctuation)
        currentsentence = nltk.word_tokenize(sentence[index].translate(removepunctuationtranslator))
        wordsInSentence[index] = currentsentence;
        for innerindex in range(len(currentsentence)):
            try:
                sentences_vector[index] = sentences_vector[index] + glove_model.word_vec(currentsentence[innerindex])
            except KeyError as error:
                pass
                print('Vocab not found for word : ', error.args,)
    return sentences_vector
