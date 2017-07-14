from __future__ import division
from collections import Counter, defaultdict
import os
from random import shuffle
import tensorflow as tf
import pickle
import os.path
import numpy as np
import collections
import math
import numpy
from pathlib import Path


SUCCESS_HEADER = 'S_'

class NotTrainedError(Exception):
    pass

class NotFitToCorpusError(Exception):
    pass

class GloVeModel():
    EMBEDDINGS = "embeddings"
    M_ID = 0
    SAVE_DIR_NAME = "model"
    SAVE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), SAVE_DIR_NAME)
    E_DIR_NAME = "embeddings"
    TF_CP_DIR_NAME = "tf_checkpoints"
    OLD_MODEL_BASEDIR = os.path.join(SAVE_DIR, SAVE_DIR_NAME + str(M_ID))
    NEW_MODEL_BASEDIR = os.path.join(SAVE_DIR, SAVE_DIR_NAME + str(M_ID + 1))
    TF_MODEL_NAME = "tf_model.ckpt"
    has_TF_Model = False

    def __init__(self, embedding_size, context_size, max_vocab_size=100000, min_occurrences=1,
                 scaling_factor=3 / 4, cooccurrence_cap=100, batch_size=50, learning_rate=0.05):
        self.embedding_size = embedding_size
        if isinstance(context_size, tuple):
            self.left_context, self.right_context = context_size
        elif isinstance(context_size, int):
            self.left_context = self.right_context = context_size
        else:
            raise ValueError("`context_size` should be an int or a tuple of two ints")
        self.max_vocab_size = max_vocab_size
        self.min_occurrences = min_occurrences
        self.scaling_factor = scaling_factor
        self.cooccurrence_cap = cooccurrence_cap
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.find_latest_train_iter()
        self.create_or_load_model()

    def find_latest_train_iter(self):
        folder_cont = os.listdir(self.SAVE_DIR)
        if len(folder_cont) is not 0:
            self.M_ID = self.__get_latest_file( self.SAVE_DIR_NAME , folder_cont)
        self.OLD_MODEL_BASEDIR = os.path.join(self.SAVE_DIR, self.SAVE_DIR_NAME + str(self.M_ID))
        self.NEW_MODEL_BASEDIR = os.path.join(self.SAVE_DIR, self.SAVE_DIR_NAME + str(self.M_ID + 1))

    def __get_latest_file(self, base_filename ,folder_contents):
        max = 0
        for file in folder_contents:
            cur_id = int(file.replace(base_filename, ""))
            if cur_id > max:
                max = cur_id
        return max

    def create_or_load_model(self):
        # self.__word_to_id = self.load_obj(self.WORD_TO_ID)
        self.__embeddings = self.load_obj(self.EMBEDDINGS)
        self.__words = [word for word, _ in self.__embeddings.items()] if (self.__embeddings is not None) else None
        self.__new_words = None
        self.__existing_words_count = len(self.__words) if (self.__words is not None) else 0
        self.__cooccurrence_matrix = None
        self.__word_counts = Counter()

    def fit_to_corpus(self, data, in_type="corpus"):
        self.__fit_to_corpus(data, in_type, self.max_vocab_size, self.min_occurrences,
                             self.left_context, self.right_context)
        self.__graph = tf.Graph()
        self.__create_graph(self.__graph)

    def __fit_to_corpus(self, data, in_type, vocab_size, min_occurrences, left_size, right_size):
        if in_type.lower() == 'corpus':
            cooccurrence_counts = self.__build_coocur_mat_from_corpus(data, left_size, right_size)
            self.__new_words = [word for word, count in self.__word_counts.most_common(vocab_size)
                                if (count >= min_occurrences and (
                self.__words is None or word not in self.__embeddings.keys()))]
        elif in_type.lower() == 'sparkcrawl':
            cooccurrence_counts = self.__convert_csv_to_coocur_dict(data)
        else:
            raise ValueError
        if len(cooccurrence_counts) == 0:
            raise ValueError("No coccurrences in corpus. Did you try to reuse a generator?")

        if self.__words != None:
            self.__words = self.__words + [word for word in self.__new_words]
        else:
            self.__words = [word for i, word in enumerate(self.__new_words)]
        # TODO : Check if word id is in coocurence matrix
        self.__cooccurrence_matrix = {
            (self.__words.index(words[0]), self.__words.index(words[1])): count
            for words, count in cooccurrence_counts.items()
            if words[0] in self.__words and words[1] in self.__words}

    def __convert_csv_to_coocur_dict(self, res_folder):
        import csv, win32api
        import platform
        cooccurrence_counts = defaultdict(float)
        new_words = []
        res_folder_gen = [label_folder for label_folder in os.listdir(res_folder) if label_folder[:2] != SUCCESS_HEADER]
        for label_folder in res_folder_gen:
            csv_gen = [csv_fname for csv_fname in os.listdir(os.path.join(res_folder, label_folder)) if csv_fname[-3:] == 'csv']
            for csv_fname in csv_gen:
                if any(platform.win32_ver()):
                    csv_file =  win32api.GetShortPathName(os.path.join(win32api.GetShortPathName(res_folder), label_folder, csv_fname))
                else:
                    csv_file = os.path.join(res_folder, label_folder, csv_fname)
                reader = csv.DictReader(open(csv_file), fieldnames=['tgt_word', 'ctx_word', 'coor_val'])
                for row in reader:
                    target_word = row['tgt_word']
                    context_word = row['ctx_word']
                    print(row['tgt_word'])
                    if (self.__embeddings is None or target_word not in self.__embeddings.keys()) and target_word not in new_words:
                        new_words.append(target_word)
                    if (self.__embeddings is None or context_word not in self.__embeddings.keys()) and context_word not in new_words:
                        new_words.append(context_word)
                    cooccurrence_counts[(target_word, context_word)] = row['coor_val']
        self.__new_words = new_words
        return cooccurrence_counts

    def __build_coocur_mat_from_corpus(self, corpus, left_size, right_size):
        cooccurrence_counts = defaultdict(float)
        for formerRegion in corpus:
            region = formerRegion.split()
            self.__word_counts.update(region)
            for l_context, word, r_context in _context_windows(region, left_size, right_size):
                for i, context_word in enumerate(l_context[::-1]):
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
                for i, context_word in enumerate(r_context):
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
        return cooccurrence_counts

    def save_obj(self, obj, name):
        new_save_dir = os.path.join(self.NEW_MODEL_BASEDIR, self.E_DIR_NAME)
        os.makedirs(new_save_dir)
        with open(os.path.join((new_save_dir), name) + '.pkl' , 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, name):
        try:
            with open(os.path.join(self.OLD_MODEL_BASEDIR, self.E_DIR_NAME , name) + '.pkl', 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

    def restore_vars(self, saver, sess, chkpt_dir):
        tf.global_variables_initializer().run()
        checkpoint_dir = chkpt_dir

        if not os.path.exists(checkpoint_dir):
            try:
                return False
            except OSError:
                pass

        path = tf.train.get_checkpoint_state(checkpoint_dir)
        print(checkpoint_dir, "path = ", path)
        if path is None:
            return False
        else:
            saver.restore(sess, path.model_checkpoint_path)
            return True

    def tf_checkpoints_available(self, checkpoint_dir):
        path = tf.train.get_checkpoint_state(checkpoint_dir)
        if path is None:
            return False
        else:
            self.has_TF_Model = True
            return True


    def __recreate_graph(self, __recreated_graph, old_focal_embeddings, old_context_embeddings, old_focal_biases,
                         old_context_biases):
        with __recreated_graph.as_default(), __recreated_graph.device(_device_for_node):
            count_max = tf.constant([self.cooccurrence_cap], dtype=tf.float32,
                                    name='max_cooccurrence_count')
            scaling_factor = tf.constant([self.scaling_factor], dtype=tf.float32,
                                         name="scaling_factor")
            self.__focal_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                name="focal_words")
            self.__context_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                  name="context_words")
            self.__cooccurrence_count = tf.placeholder(tf.float32, shape=[self.batch_size],
                                                       name="cooccurrence_count")


            self.focal_embeddings = tf.Variable(np.concatenate((old_focal_embeddings, np.random.uniform(-1, 1, (
            self.new_vocab_size, self.embedding_size)).astype(np.float32)), axis=0),
                                                name="focal_embeddings")
            self.context_embeddings = tf.Variable(np.concatenate((old_context_embeddings, np.random.uniform(-1, 1, (
            self.new_vocab_size, self.embedding_size)).astype(np.float32)), axis=0),
                                                  name="context_embeddings")
            self.focal_biases = tf.Variable(
                np.concatenate((old_focal_biases, np.random.uniform(-1, 1, self.new_vocab_size).astype(np.float32)),
                               axis=0),
                name='focal_biases')
            self.context_biases = tf.Variable(
                np.concatenate((old_context_biases, np.random.uniform(-1, 1, self.new_vocab_size).astype(np.float32)),
                               axis=0),
                name="context_biases")

            focal_embedding = tf.nn.embedding_lookup([self.focal_embeddings], self.__focal_input)
            context_embedding = tf.nn.embedding_lookup([self.context_embeddings], self.__context_input)
            focal_bias = tf.nn.embedding_lookup([self.focal_biases], self.__focal_input)
            context_bias = tf.nn.embedding_lookup([self.context_biases], self.__context_input)

            weighting_factor = tf.minimum(
                1.0,
                tf.pow(
                    tf.div(self.__cooccurrence_count, count_max),
                    scaling_factor))

            embedding_product = tf.reduce_sum(tf.multiply(focal_embedding, context_embedding), 1)

            log_cooccurrences = tf.log(tf.to_float(self.__cooccurrence_count))

            distance_expr = tf.square(tf.add_n([
                embedding_product,
                focal_bias,
                context_bias,
                tf.negative(log_cooccurrences)]))

            single_losses = tf.multiply(weighting_factor, distance_expr)
            self.__total_loss = tf.reduce_sum(single_losses)
            tf.summary.scalar("GloVe_loss", self.__total_loss)
            self.__optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(
                self.__total_loss)
            self.__summary = tf.summary.merge_all()
            self.__combined_embeddings = tf.add(self.focal_embeddings, self.context_embeddings,
                                                name="combined_embeddings")
            self.saver = tf.train.Saver()

    def __create_graph(self, graph):
        with graph.as_default(), graph.device(_device_for_node):
            count_max = tf.constant([self.cooccurrence_cap], dtype=tf.float32,
                                    name='max_cooccurrence_count')
            scaling_factor = tf.constant([self.scaling_factor], dtype=tf.float32,
                                         name="scaling_factor")

            self.__focal_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                name="focal_words")
            self.__context_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                  name="context_words")
            self.__cooccurrence_count = tf.placeholder(tf.float32, shape=[self.batch_size],
                                                       name="cooccurrence_count")

            # TODO CHECK IF OTHER TENSORFLOW VARIABLES ARE ACTUALLY TRAINED
            if self.tf_checkpoints_available(os.path.join(self.OLD_MODEL_BASEDIR, self.TF_CP_DIR_NAME)):
                self.focal_embeddings = tf.Variable(
                    tf.random_uniform([self.vocab_size, self.embedding_size], 1.0, -1.0),
                    name="focal_embeddings")
                self.context_embeddings = tf.Variable(
                    tf.random_uniform([self.vocab_size, self.embedding_size], 1.0, -1.0),
                    name="context_embeddings")
                self.focal_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0),
                                                name='focal_biases')
                self.context_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0),
                                                  name="context_biases")
            else:
                self.focal_embeddings = tf.Variable(
                    tf.random_uniform([self.new_vocab_size, self.embedding_size], 1.0, -1.0),
                    name="focal_embeddings")
                self.context_embeddings = tf.Variable(
                    tf.random_uniform([self.new_vocab_size, self.embedding_size], 1.0, -1.0),
                    name="context_embeddings")
                self.focal_biases = tf.Variable(tf.random_uniform([self.new_vocab_size], 1.0, -1.0),
                                                name='focal_biases')
                self.context_biases = tf.Variable(tf.random_uniform([self.new_vocab_size], 1.0, -1.0),
                                                  name="context_biases")

            focal_embedding = tf.nn.embedding_lookup([self.focal_embeddings], self.__focal_input)
            context_embedding = tf.nn.embedding_lookup([self.context_embeddings], self.__context_input)
            focal_bias = tf.nn.embedding_lookup([self.focal_biases], self.__focal_input)
            context_bias = tf.nn.embedding_lookup([self.context_biases], self.__context_input)

            weighting_factor = tf.minimum(
                1.0,
                tf.pow(
                    tf.div(self.__cooccurrence_count, count_max),
                    scaling_factor))

            embedding_product = tf.reduce_sum(tf.multiply(focal_embedding, context_embedding), 1)
            log_cooccurrences = tf.log(tf.to_float(self.__cooccurrence_count))
            distance_expr = tf.square(tf.add_n([
                embedding_product,
                focal_bias,
                context_bias,
                tf.negative(log_cooccurrences)]))

            single_losses = tf.multiply(weighting_factor, distance_expr)
            self.__total_loss = tf.reduce_sum(single_losses)
            tf.summary.scalar("GloVe_loss", self.__total_loss)
            self.__optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(
                self.__total_loss)
            self.__summary = tf.summary.merge_all()
            self.__combined_embeddings = tf.add(self.focal_embeddings, self.context_embeddings,
                                                name="combined_embeddings")
            self.saver = tf.train.Saver()

    def train(self, num_epochs, log_dir=None, summary_batch_interval=1000,
              tsne_epoch_interval=None):
        should_write_summaries = log_dir is not None and summary_batch_interval
        should_generate_tsne = log_dir is not None and tsne_epoch_interval
        batches = self.__prepare_batches()
        total_steps = 0
        with tf.Session(graph=self.__graph) as session:
            if should_write_summaries:
                print("Writing TensorBoard summaries to {}".format(log_dir))
                summary_writer = tf.summary.FileWriter(log_dir, graph=session.graph)
            else:
                summary_writer = None
            self.restore_vars(self.saver, session, os.path.join(self.OLD_MODEL_BASEDIR, self.TF_CP_DIR_NAME))
            if self.has_TF_Model:
                __recreated_graph = tf.Graph()
                old_focal_embeddings = self.focal_embeddings.eval()
                old_context_embeddings = self.context_embeddings.eval()
                old_focal_biases = self.focal_biases.eval()
                old_context_biases = self.context_biases.eval()
                self.__recreate_graph(__recreated_graph, old_focal_embeddings, old_context_embeddings, old_focal_biases,
                                      old_context_biases)
                with tf.Session(graph=__recreated_graph) as innersess:
                    tf.global_variables_initializer().run()
                    self.__inner_train(innersess, num_epochs, batches, summary_batch_interval,
                                       should_write_summaries, total_steps, should_generate_tsne,
                                       summary_writer, tsne_epoch_interval, log_dir)
            else:
                self.__inner_train(session, num_epochs, batches, summary_batch_interval,
                                   should_write_summaries, total_steps, should_generate_tsne,
                                   summary_writer, tsne_epoch_interval, log_dir)
            self.__existing_words_count = len(self.__words)
            self.save_obj(self.__embeddings, self.EMBEDDINGS)

    def __inner_train(self, session, num_epochs, batches, summary_batch_interval,
                      should_write_summaries, total_steps, should_generate_tsne,
                      summary_writer, tsne_epoch_interval, log_dir):
        for epoch in range(num_epochs):
            shuffle(batches)
            for batch_index, batch in enumerate(batches):
                i_s, j_s, counts = batch
                if len(counts) != self.batch_size:
                    continue
                feed_dict = {
                    self.__focal_input: i_s,
                    self.__context_input: j_s,
                    self.__cooccurrence_count: counts}
                session.run([self.__optimizer], feed_dict=feed_dict)
                if should_write_summaries and (total_steps + 1) % summary_batch_interval == 0:
                    summary_str = session.run(self.__summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, total_steps)
                total_steps += 1
            if should_generate_tsne and (epoch + 1) % tsne_epoch_interval == 0:
                current_embeddings = self.__combined_embeddings.eval()
                output_path = os.path.join(log_dir, "epoch{:03d}.png".format(epoch + 1))
                self.generate_tsne(output_path, embeddings=current_embeddings)
        self.__embeddings = self.__combined_embeddings.eval()
        self.__embeddings = collections.OrderedDict(
            [(self.__words[i], embedding) for i, embedding in enumerate(self.__embeddings)])
        if should_write_summaries:
            summary_writer.close()
        self.save_tf_model(session)

    def save_tf_model(self,session):
        new_save_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), self.SAVE_DIR_NAME, self.SAVE_DIR_NAME + str(self.M_ID + 1) , self.TF_CP_DIR_NAME)
        os.makedirs(new_save_dir)
        save_path = self.saver.save(session, os.path.join(new_save_dir,  self.TF_MODEL_NAME))
        print("Model saved in file: %s" % save_path)

    def copy_embeddings(self, loaded_embeddings):
        self.__embeddings = loaded_embeddings
        self.__recreate_graph()



    def embedding_for(self, word_str_or_id):
        if isinstance(word_str_or_id, str):
            return self.__embeddings[word_str_or_id]

    def __prepare_batches(self):
        if self.__cooccurrence_matrix is None:
            raise NotFitToCorpusError(
                "Need to fit model to corpus before preparing training batches.")
        cooccurrences = [(word_ids[0], word_ids[1], count)
                         for word_ids, count in self.__cooccurrence_matrix.items()]
        i_indices, j_indices, counts = zip(*cooccurrences)
        return list(_batchify(self.batch_size, i_indices, j_indices, counts))

    @property
    def vocab_size(self):
        return self.__existing_words_count

    @property
    def new_vocab_size(self):
        return len(self.__new_words)

    @property
    def words(self):
        if self.__words is None:
            raise NotFitToCorpusError("Need to fit model to corpus before accessing words.")
        return self.__words

    @property
    def embeddings(self):
        if self.__embeddings is None:
            raise NotTrainedError("Need to train model before accessing embeddings")
        return self.__embeddings

    def generate_tsne(self, path="glove/model/model", size=(100, 100), word_count=1000, embeddings=None):
        if embeddings is None:
            embeddings = self.embeddings
        from sklearn.manifold import TSNE
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs = tsne.fit_transform(numpy.asarray(list(embeddings.values())))
        labels = self.words[:word_count]
        return _plot_with_labels(low_dim_embs, labels, path, size)

    def word_vec(self, word):
        return self.embedding_for(word)

    def find_similarity(self, f_word, s_word):
        f_word_v = self.embedding_for(f_word)
        s_word_v = self.embedding_for(s_word)
        cos_sim = numpy.dot(f_word_v, s_word_v) / (numpy.linalg.norm(f_word_v) * numpy.linalg.norm(s_word_v))
        return cos_sim


def _context_windows(region, left_size, right_size):
    for i, word in enumerate(region):
        start_index = i - left_size
        end_index = i + right_size
        left_context = _window(region, start_index, i - 1)
        right_context = _window(region, i + 1, end_index)
        yield (left_context, word, right_context)


def start_training(file, num_epochs, input_type = 'corpus'):
    model = GloVeModel(embedding_size=50, context_size=10)
    model.fit_to_corpus(file, input_type)
    model.train(num_epochs=num_epochs)
    model.generate_tsne()

def _window(region, start_index, end_index):
    """
    Returns the list of words starting from `start_index`, going to `end_index`
    taken from region. If `start_index` is a negative number, or if `end_index`
    is greater than the index of the last word in region, this function will pad
    its return value with `NULL_WORD`.
    """
    last_index = len(region) + 1
    selected_tokens = region[max(start_index, 0):min(end_index, last_index) + 1]
    return selected_tokens


def _device_for_node(n):
    if n.type == "MatMul":
        return "/gpu:0"
    else:
        return "/cpu:0"


def _batchify(batch_size, *sequences):
    for i in range(0, len(sequences[0]), batch_size):
        yield tuple(sequence[i:i + batch_size] for sequence in sequences)


def _plot_with_labels(low_dim_embs, labels, path, size):
    import matplotlib.pyplot as plt
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    figure = plt.figure(figsize=size)  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right',
                     va='bottom')
    if path is not None:
        figure.savefig(path)
        plt.close(figure)
