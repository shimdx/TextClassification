# -*- coding: utf-8 -*-
import os
import sys
import csv
import time
import json
import datetime
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.contrib import learn
import pandas as pd
import data_helper
from rnn_classifier import rnn_clf
from cnn_classifier import cnn_clf
from clstm_classifier import clstm_clf
from cnn_classifier_w2v import cnn_clf_w2v
from clstm_classifier_w2v import clstm_clf_w2v
from rnn_classifier_w2v import rnn_clf_w2v

from sklearn.naive_bayes import GaussianNB
from sklearn import datasets, metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


try:
    from sklearn.model_selection import train_test_split
except ImportError as e:
    error = "Please install scikit-learn."
    print(str(e) + ': ' + error)
    sys.exit()

# Show warnings and errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

# Parameters
# =============================================================================
# Data_parameters setting
data_path = './data/bot_dataset_all_train.csv'
data_language = 'ko'
num_classes = 2

# Model choices
tf.flags.DEFINE_string('clf', 'clstm', "Type of classifiers. Default: cnn. You have four choices: [cnn, lstm, blstm, clstm]")

# Data parameters
tf.flags.DEFINE_string('data_file', data_path, 'Data file path')
tf.flags.DEFINE_string('stop_word_file', None, 'Stop word file path')
tf.flags.DEFINE_string('language', data_language, "Language of the data file. You have two choices: [ch, en]")
tf.flags.DEFINE_integer('min_frequency', 0, 'Minimal word frequency')
tf.flags.DEFINE_integer('num_classes', num_classes, 'Number of classes')
tf.flags.DEFINE_integer('max_length', 0, 'Max document length')
tf.flags.DEFINE_integer('vocab_size', 0, 'Vocabulary size')
tf.flags.DEFINE_float('test_size', 0.1, 'Cross validation test size')

# Model hyperparameters
tf.flags.DEFINE_integer('embedding_size', 256, 'Word embedding size. For CNN, C-LSTM.')
tf.flags.DEFINE_string('filter_sizes', '3, 4, 5', 'CNN filter sizes. For CNN, C-LSTM.')
tf.flags.DEFINE_integer('num_filters', 128, 'Number of filters per filter size. For CNN, C-LSTM.')
tf.flags.DEFINE_integer('hidden_size', 128, 'Number of hidden units in the LSTM cell. For LSTM, Bi-LSTM')
tf.flags.DEFINE_integer('num_layers', 2, 'Number of the LSTM cells. For LSTM, Bi-LSTM, C-LSTM')
tf.flags.DEFINE_float('keep_prob', 0.5, 'Dropout keep probability')  # All
tf.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')  # All
tf.flags.DEFINE_float('l2_reg_lambda', 0.001, 'L2 regularization lambda')  # All

# Training parameters
tf.flags.DEFINE_integer('batch_size', 32, 'Batch size')
tf.flags.DEFINE_integer('num_epochs', 50, 'Number of epochs')
tf.flags.DEFINE_float('decay_rate', 1, 'Learning rate decay rate. Range: (0, 1]')  # Learning rate decay
tf.flags.DEFINE_integer('decay_steps', 100000, 'Learning rate decay steps')  # Learning rate decay
tf.flags.DEFINE_integer('evaluate_every_steps', 100, 'Evaluate the model on validation set after this many steps')
tf.flags.DEFINE_integer('save_every_steps', 500, 'Save the model after this many steps')
tf.flags.DEFINE_integer('num_checkpoint', 10, 'Number of models to store')

#w2v model parameters
tf.flags.DEFINE_bool('is_w2v', False, 'Apply pre-trained word2vector mode')
#post tagging parameters
tf.flags.DEFINE_bool('is_post_tagged', False, 'Apply post_tagged words mode')
tf.flags.DEFINE_bool('is_noun', False, 'Allow noun only for training words')

FLAGS = tf.app.flags.FLAGS

if FLAGS.clf == 'lstm':
    FLAGS.embedding_size = FLAGS.hidden_size
elif FLAGS.clf == 'clstm':
    FLAGS.hidden_size = len(FLAGS.filter_sizes.split(",")) * FLAGS.num_filters

# Output files directory
timestamp = str(int(time.time()))
outdir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp+"_"+FLAGS.clf))
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Load and save data

# =============================================================================

data, labels, lengths, vocab_processor = data_helper.load_data(file_path=FLAGS.data_file,
                                                               sw_path=FLAGS.stop_word_file,
                                                               min_frequency=FLAGS.min_frequency,
                                                               max_length=FLAGS.max_length,
                                                               language=FLAGS.language,
                                                               shuffle=True, is_w2v=FLAGS.is_w2v, is_post_tagged=FLAGS.is_post_tagged, is_noun = FLAGS.is_noun)
# Save vocabulary processor
vocab_processor.save(os.path.join(outdir, 'vocab'))

FLAGS.vocab_size = len(vocab_processor.vocabulary_._mapping)

FLAGS.max_length = vocab_processor.max_document_length

params = FLAGS.flag_values_dict()



# pretrained w2v model 미 적용시, 연관 parameter 삭제 / embedding dim size는 pretrained의 크기 대로 (200)
if not params['is_w2v']:
    del params['is_w2v']
else :
    if ('lstm' or 'blstm') in params['clf']:
        print('lstm with w2v')
        FLAGS.hidden_size = 200
    FLAGS.embedding_size = 200

# Print parameters
model = params['clf']
if 'cnn' in model :
    del params['hidden_size']
    del params['num_layers']
elif 'lstm' in model or 'blstm' in model:
    del params['num_filters']
    del params['filter_sizes']
    params['embedding_size'] = params['hidden_size']
elif 'clstm' in model:
    params['hidden_size'] = len(list(map(int, params['filter_sizes'].split(",")))) * params['num_filters']

params_dict = sorted(params.items(), key=lambda x: x[0])
print('Parameters:')
for item in params_dict:
    print('{}: {}'.format(item[0], item[1]))
print('')

# Save parameters to file
params_file = open(os.path.join(outdir, 'params.pkl'), 'wb')
pkl.dump(params, params_file, True)
params_file.close()


# Simple Cross validation
x_train, x_valid, y_train, y_valid, train_lengths, valid_lengths = train_test_split(data,
                                                                                    labels,
                                                                                    lengths,
                                                                                    test_size=FLAGS.test_size,
                                                                                    random_state=22)
# Batch iterator
train_data = data_helper.batch_iter(x_train, y_train, train_lengths, FLAGS.batch_size, FLAGS.num_epochs)

# Train
# =============================================================================
classifier_GNB = GaussianNB()
classifier_SVM = svm.SVC()
classifier_RFC = RandomForestClassifier()
for idx, train_input in enumerate(train_data):
    input_x, input_y, sequence_length = train_input
    # input_conv = np.zeros((len(input_x), FLAGS.max_length, FLAGS.embedding_size), dtype=np.float32)
    # for idx, item in enumerate(input_x):
    #     if len(item) < FLAGS.max_length:
    #         for i, word in enumerate(item):
    #             input_conv[idx][i] = word
    #     else :
    #         input_conv[idx] = item[:FLAGS.max_length]
    # input_x = input_conv

    classifier_GNB.fit(input_x, input_y)
    classifier_SVM.fit(input_x, input_y)
    classifier_RFC.fit(input_x, input_y)

    score_GNB = metrics.accuracy_score(input_y, classifier_GNB.predict(input_x))
    score_SVM = metrics.accuracy_score(input_y, classifier_SVM.predict(input_x))
    score_RFC = metrics.accuracy_score(input_y, classifier_RFC.predict(input_x))
    print("%dth Accuracy_GNB: %f" % (idx, score_GNB))
    print("%dth Accuracy_SVM: %f" % (idx, score_SVM))
    print("%dth Accuracy_RFC: %f" % (idx, score_RFC))

test_path = './data/bot_dataset_all_test.csv'
test_x, test_y, test_lengths, _ = data_helper.load_data(file_path=test_path,
                                                 sw_path=params['stop_word_file'],
                                                 min_frequency=params['min_frequency'],
                                                 max_length=params['max_length'],
                                                 language=params['language'],
                                                 vocab_processor=vocab_processor,
                                                 shuffle=False, is_w2v=FLAGS.is_w2v, is_post_tagged=FLAGS.is_post_tagged, is_noun = FLAGS.is_noun)

score_GNB = metrics.accuracy_score(test_y, classifier_GNB.predict(test_x))
score_SVM = metrics.accuracy_score(test_y, classifier_SVM.predict(test_x))
score_RFC = metrics.accuracy_score(test_y, classifier_RFC.predict(test_x))
print("Test Accuracy_GNB: %f" % score_GNB)
print("Test Accuracy_SVM: %f" % score_SVM)
print("Test Accuracy_RFC: %f" % score_RFC)

test_names = ['dialog','trained']
cm_GNB = pd.DataFrame(confusion_matrix(test_y, classifier_GNB.predict(test_x)), columns=test_names, index=test_names)
print(cm_GNB)
cm_SVM = pd.DataFrame(confusion_matrix(test_y, classifier_SVM.predict(test_x)), columns=test_names, index=test_names)
print(cm_SVM)
cm_RFC = pd.DataFrame(confusion_matrix(test_y, classifier_RFC.predict(test_x)), columns=test_names, index=test_names)
print(cm_RFC)
# sns.heatmap(cm, annot=True)
# plt.show()

# with tf.Graph().as_default():
#     with tf.Session() as sess:
#         if FLAGS.clf == 'cnn':
#             classifier = cnn_clf(FLAGS)
#         elif FLAGS.clf == 'lstm' or FLAGS.clf == 'blstm':
#             classifier = rnn_clf(FLAGS)
#         elif FLAGS.clf == 'clstm':
#             classifier = clstm_clf(FLAGS)
#         elif FLAGS.clf == 'cnn_w2v':
#             classifier = cnn_clf_w2v(FLAGS)
#         elif FLAGS.clf == 'clstm_w2v':
#             classifier = clstm_clf_w2v(FLAGS)
#         elif FLAGS.clf == 'lstm_w2v' or FLAGS.clf == 'blstm_w2v':
#             classifier = rnn_clf_w2v(FLAGS)
#         else:
#             raise ValueError('clf should be one of [cnn, lstm, blstm, clstm]')
#
#         # Train procedure
#         global_step = tf.Variable(0, name='global_step', trainable=False)
#         # Learning rate decay
#         starter_learning_rate = FLAGS.learning_rate
#         learning_rate = tf.train.exponential_decay(starter_learning_rate,
#                                                    global_step,
#                                                    FLAGS.decay_steps,
#                                                    FLAGS.decay_rate,
#                                                    staircase=True)
#         optimizer = tf.train.AdamOptimizer(learning_rate)
#         grads_and_vars = optimizer.compute_gradients(classifier.cost)
#         train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
#
#         # Summaries
#         loss_summary = tf.summary.scalar('Loss', classifier.cost)
#         accuracy_summary = tf.summary.scalar('Accuracy', classifier.accuracy)
#
#         # Train summary
#         train_summary_op = tf.summary.merge_all()
#         train_summary_dir = os.path.join(outdir, 'summaries', 'train')
#         train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
#
#         # Validation summary
#         valid_summary_op = tf.summary.merge_all()
#         valid_summary_dir = os.path.join(outdir, 'summaries', 'valid')
#         valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)
#
#         saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoint)
#
#         sess.run(tf.global_variables_initializer())
#
#
#         def run_step(input_data, is_training=True):
#             """Run one step of the training process."""
#             input_x, input_y, sequence_length = input_data
#
#             # w2v 적용을 위한 array 정형화
#             if FLAGS.is_w2v:
#                 input_conv = np.zeros((len(input_x), FLAGS.max_length, FLAGS.embedding_size), dtype=np.float32)
#                 for idx, item in enumerate(input_x):
#                     if len(item) < FLAGS.max_length:
#                         for i, word in enumerate(item):
#                             input_conv[idx][i] = word
#                     else :
#                         input_conv[idx] = item[:FLAGS.max_length]
#                 input_x = input_conv
#
#             fetches = {'step': global_step,
#                        'cost': classifier.cost,
#                        'accuracy': classifier.accuracy,
#                        'learning_rate': learning_rate}
#             feed_dict = {classifier.input_x: input_x,
#                          classifier.input_y: input_y}
#
#
#
#             if  'cnn' not in FLAGS.clf:
#                 fetches['final_state'] = classifier.final_state
#                 feed_dict[classifier.batch_size] = len(input_x)
#                 feed_dict[classifier.sequence_length] = sequence_length
#
#             if is_training:
#                 fetches['train_op'] = train_op
#                 fetches['summaries'] = train_summary_op
#                 feed_dict[classifier.keep_prob] = FLAGS.keep_prob
#             else:
#                 fetches['summaries'] = valid_summary_op
#                 feed_dict[classifier.keep_prob] = 1.0
#
#             vars = sess.run(fetches, feed_dict)
#             step = vars['step']
#             cost = vars['cost']
#             accuracy = vars['accuracy']
#             summaries = vars['summaries']
#
#             # Write summaries to file
#             if is_training:
#                 train_summary_writer.add_summary(summaries, step)
#             else:
#                 valid_summary_writer.add_summary(summaries, step)
#
#             time_str = datetime.datetime.now().isoformat()
#             print("{}: step: {}, loss: {:g}, accuracy: {:g}".format(time_str, step, cost, accuracy))
#
#             return accuracy
#
#
#         print('Start training ...')
#
#         for train_input in train_data: # 다른 점은 word의 idx로 들어가느냐, vector로 들어가느냐
#             run_step(train_input, is_training=True)
#             current_step = tf.train.global_step(sess, global_step)
#
#             if current_step % FLAGS.evaluate_every_steps == 0:
#                 print('\nValidation')
#                 run_step((x_valid, y_valid, valid_lengths), is_training=False)
#                 print('')
#
#             if current_step % FLAGS.save_every_steps == 0:
#                 save_path = saver.save(sess, os.path.join(outdir, 'model/clf'), current_step)
#
#         print('\nAll the files have been saved to {}\n'.format(outdir))
