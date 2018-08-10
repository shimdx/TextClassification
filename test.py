# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.contrib import learn

import data_helper

# Show warnings and errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
# Parameter setting
data_path = './data/bot_dataset_all_new_test.csv'
run_dir = './runs/1533865660_clstm_w2v'
checkpoint = 'clf-2500'

# File paths
tf.flags.DEFINE_string('data_file', data_path, 'Test data file path')
tf.flags.DEFINE_string('run_dir', run_dir, 'Restore the model from this run')
tf.flags.DEFINE_string('checkpoint', checkpoint, 'Restore the graph from this checkpoint')

# Test batch size
tf.flags.DEFINE_integer('batch_size', 20, 'Test batch size')

#w2v model parameters
tf.flags.DEFINE_bool('is_w2v', True, 'Apply pre-trained word2vector mode')
tf.flags.DEFINE_integer('max_length', 28, 'Max document length')
tf.flags.DEFINE_integer('embedding_size', 200, 'Word embedding size. For CNN, C-LSTM.')

#post tagging parameters
tf.flags.DEFINE_bool('is_post_tagged', True, 'Apply post_tagged words mode')
tf.flags.DEFINE_bool('is_noun', True, 'Allow noun only for training words')

FLAGS = tf.app.flags.FLAGS

# Restore parameters
with open(os.path.join(FLAGS.run_dir, 'params.pkl'), 'rb') as f:
    params = pkl.load(f, encoding='bytes')

# Restore vocabulary processor
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(os.path.join(FLAGS.run_dir, 'vocab'))

# Load test data
data, labels, lengths, _ = data_helper.load_data(file_path=FLAGS.data_file,
                                                 sw_path=params['stop_word_file'],
                                                 min_frequency=params['min_frequency'],
                                                 max_length=params['max_length'],
                                                 language=params['language'],
                                                 vocab_processor=vocab_processor,
                                                 shuffle=False, is_w2v=FLAGS.is_w2v, is_post_tagged=FLAGS.is_post_tagged, is_noun = FLAGS.is_noun)
labels = [label - 1 for label in labels]
target = [idx for idx, label in enumerate(labels) if label > -1]
data = [data[idx] for idx in target]
labels = [labels[idx] for idx in target]
lengths = [lengths[idx] for idx in target]
print(set(labels))
# Restore graph
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    # Restore metagraph
    saver = tf.train.import_meta_graph('{}.meta'.format(os.path.join(FLAGS.run_dir, 'model', FLAGS.checkpoint)))
    # Restore weights
    saver.restore(sess, os.path.join(FLAGS.run_dir, 'model', FLAGS.checkpoint))
    # print([node.name for node in tf.get_default_graph().as_graph_def().node])
    # Get tensors
    input_x = graph.get_tensor_by_name('input_x:0')
    input_y = graph.get_tensor_by_name('input_y:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    predictions = graph.get_tensor_by_name('softmax/predictions:0')
    accuracy = graph.get_tensor_by_name('accuracy/accuracy:0')

    # Generate batches
    batches = data_helper.batch_iter(data, labels, lengths, FLAGS.batch_size, 1)

    num_batches = int(len(data)/FLAGS.batch_size)
    all_predictions = []
    sum_accuracy = 0

    # Test
    for batch in batches:
        x_test, y_test, x_lengths = batch

        if FLAGS.is_w2v:
            input_conv = np.zeros((len(x_test), FLAGS.max_length, FLAGS.embedding_size), dtype=np.float32)
            for idx, item in enumerate(x_test):
                if len(item) < FLAGS.max_length:
                    for i, word in enumerate(item):
                        input_conv[idx][i] = word
                else:
                    input_conv[idx] = item[:FLAGS.max_length]
            x_test = input_conv

        if 'cnn' in params['clf']:
            feed_dict = {input_x: x_test, input_y: y_test, keep_prob: 1.0}
            batch_predictions, batch_accuracy = sess.run([predictions, accuracy], feed_dict)
        else:
            batch_size = graph.get_tensor_by_name('batch_size:0')
            sequence_length = graph.get_tensor_by_name('sequence_length:0')
            feed_dict = {input_x: x_test, input_y: y_test, batch_size: FLAGS.batch_size, sequence_length: x_lengths, keep_prob: 1.0}

            batch_predictions, batch_accuracy = sess.run([predictions, accuracy], feed_dict)

        sum_accuracy += batch_accuracy
        all_predictions = np.concatenate([all_predictions, batch_predictions])
    # print(all_predictions)
    # print(y_test)
    final_accuracy = sum_accuracy / num_batches
# Print test accuracy
print('Test accuracy: {}'.format(final_accuracy))

# Save all predictions
with open(os.path.join(FLAGS.run_dir, 'predictions.csv'), 'w', encoding='utf-8', newline='') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(['True class', 'Prediction'])
    for i in range(len(all_predictions)):
        csvwriter.writerow([labels[i], all_predictions[i]])
    print('Predictions saved to {}'.format(os.path.join(FLAGS.run_dir, 'predictions.csv')))
