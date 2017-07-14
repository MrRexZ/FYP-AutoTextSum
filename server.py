import logging
from flask import Flask, jsonify
from webargs import fields
from webargs.flaskparser import use_args
from summarization.glove.glovegensim import *
from summarization.glove import extractive_summary as ext_sum
import nltk

nltk.data.path.append(os.getcwd()+'/nltk-data')

app = Flask(__name__)

userinputargs = {
    'sentences': fields.Str(required=True)
}

model = ext_sum.load_model('glove')

@app.route('/')
def index():
    return "For FYP Sunway University"

@app.route('/getExtractiveSummary')
@use_args(userinputargs)
def getUserSummary(args):
    document = args['sentences']
    summary = ext_sum.get_ex_sum(model, document)
    return jsonify(ExtractedSentences=summary)


@app.route('/tokenizeSentences')
@use_args(userinputargs)
def tokenizeSentence(args):
    document = args['sentences']
    tokenized_sentences = nltk.sent_tokenize(document)
    return jsonify(tokenized_sentences)


@app.route('/tokenizeWords')
@use_args(userinputargs)
def tokenizeWord(args):
    document = args['sentences']
    tokenized_sentences = nltk.tokenize.sent_tokenize(document)
    tokenized_words = nltk.word_tokenize(tokenized_sentences[0])
    return jsonify(tokenized_words)

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8001, debug=True)
