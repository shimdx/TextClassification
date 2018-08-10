# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn import datasets, metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import data_helper

# Show warnings and errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# Parameter setting
data_path = './data/bot_dataset_all_new_test.csv'
run_mdl1_dir = './runs/1533872236_clstm_w2v'
run_mdl2_dir = './runs/1533875054_cnn_w2v'
checkpoint_md1 = 'clf-3000'
checkpoint_md2 = 'clf-2500'

# File paths
tf.flags.DEFINE_string('data_file', data_path, 'Test data file path')
tf.flags.DEFINE_string('run_dir', run_mdl1_dir, 'Restore the model from this run')
tf.flags.DEFINE_string('checkpoint', checkpoint_md1, 'Restore the graph from this checkpoint')

# Test batch size
tf.flags.DEFINE_integer('batch_size', 20, 'Test batch size')

#w2v model parameters
tf.flags.DEFINE_bool('is_w2v', True, 'Apply pre-trained word2vector mode')
tf.flags.DEFINE_integer('max_length', 28, 'Max document length')
tf.flags.DEFINE_integer('embedding_size', 200, 'Word embedding size. For CNN, C-LSTM.')

#post tagging parameters
tf.flags.DEFINE_bool('is_post_tagged', False, 'Apply post_tagged words mode')
tf.flags.DEFINE_bool('is_noun', False, 'Allow noun only for training words')

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
                                                 shuffle=False,
                                                 is_w2v=FLAGS.is_w2v,
                                                 is_post_tagged=FLAGS.is_post_tagged,
                                                 is_noun = FLAGS.is_noun)
labels_md1 = [1 if label > 1 else label for label in labels]

# Restore graph
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    # Restore metagraph
    saver = tf.train.import_meta_graph('{}.meta'.format(os.path.join(FLAGS.run_dir, 'model', checkpoint_md1)))
    # Restore weights
    saver.restore(sess, os.path.join(FLAGS.run_dir, 'model', checkpoint_md1))

    # Get tensors
    input_x = graph.get_tensor_by_name('input_x:0')
    input_y = graph.get_tensor_by_name('input_y:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    predictions_md1 = graph.get_tensor_by_name('softmax/predictions:0')
    logits_md1 = graph.get_tensor_by_name('softmax/logits:0')
    accuracy = graph.get_tensor_by_name('accuracy/accuracy:0')

    # Generate batches
    batches = data_helper.batch_iter(data, labels_md1, lengths, FLAGS.batch_size, 1)

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

        if 'cnn' in run_mdl1_dir:
            feed_dict = {input_x: x_test, input_y: y_test, keep_prob: 1.0}
            batch_predictions, batch_accuracy, _ = sess.run([predictions_md1, accuracy, logits_md1], feed_dict)
        else:
            batch_size = graph.get_tensor_by_name('batch_size:0')
            sequence_length = graph.get_tensor_by_name('sequence_length:0')
            feed_dict = {input_x: x_test, input_y: y_test, batch_size: FLAGS.batch_size, sequence_length: x_lengths, keep_prob: 1.0}

            batch_predictions, batch_accuracy, _ = sess.run([predictions_md1, accuracy, logits_md1], feed_dict)

        sum_accuracy += batch_accuracy
        all_predictions = np.concatenate([all_predictions, batch_predictions])

    final_accuracy = sum_accuracy / num_batches
# Print test accuracy
print('Model 1 =========================================================')
print('Test accuracy: {}'.format(final_accuracy))

passed = [idx for idx, val in enumerate(all_predictions) if val == 1]

cm_md1 = pd.DataFrame(confusion_matrix(labels_md1, all_predictions), columns=['exception', 'normal'], index=['exception', 'normal'])
print(cm_md1)

print('=================================================================')
print('Total Number of passed(md1) : %d' % len(passed))
print('Model 2 =========================================================')
data_md2 = [data[idx] for idx in passed]
labels_md2 = [(labels[idx]-1) for idx in passed]
lengths_md2 = [lengths[idx] for idx in passed]
#
# data_md2 = data
# labels_md2 = [label-1 for label in labels]
# lengths_md2 = lengths

# Restore graph
graph_md2 = tf.Graph()
with graph_md2.as_default():
    sess_md2 = tf.Session()
    # Restore metagraph
    saver_md2 = tf.train.import_meta_graph('{}.meta'.format(os.path.join(run_mdl2_dir, 'model',checkpoint_md2)))
    # Restore weights
    saver_md2.restore(sess_md2, os.path.join(run_mdl2_dir, 'model', checkpoint_md2))

    # Get tensors
    input_x_md2 = graph_md2.get_tensor_by_name('input_x:0')
    input_y_md2 = graph_md2.get_tensor_by_name('input_y:0')
    keep_prob_md2 = graph_md2.get_tensor_by_name('keep_prob:0')
    predictions_md2 = graph_md2.get_tensor_by_name('softmax/predictions:0')
    logits_md2 = graph_md2.get_tensor_by_name('softmax/logits:0')
    accuracy_md2 = graph_md2.get_tensor_by_name('accuracy/accuracy:0')

    # Generate batches
    batches_md2 = data_helper.batch_iter(data_md2, labels_md2, lengths_md2, FLAGS.batch_size, 1)

    num_batches_md2 = int(len(data_md2)/FLAGS.batch_size)
    all_predictions_md2 = []
    sum_accuracy_md2 = 0

    # Test
    for batch in batches_md2:
        x_test_md2, y_test_md2, x_lengths_md2 = batch

        if FLAGS.is_w2v:
            input_conv_md2 = np.zeros((len(x_test_md2), FLAGS.max_length, FLAGS.embedding_size), dtype=np.float32)
            for idx, item in enumerate(x_test_md2):
                if len(item) < FLAGS.max_length:
                    for i, word in enumerate(item):
                        input_conv_md2[idx][i] = word
                else:
                    input_conv_md2[idx] = item[:FLAGS.max_length]
            x_test_md2 = input_conv_md2

        if 'cnn' in run_mdl2_dir:
            feed_dict_md2 = {input_x_md2: x_test_md2, input_y_md2: y_test_md2, keep_prob_md2: 1.0}
            batch_predictions_md2, batch_accuracy_md2, _ = sess_md2.run([predictions_md2, accuracy_md2, logits_md2], feed_dict_md2)
        else:
            batch_size_md2 = graph_md2.get_tensor_by_name('batch_size:0')
            sequence_length_md2 = graph_md2.get_tensor_by_name('sequence_length:0')
            feed_dict_md2 = {input_x_md2: x_test_md2, input_y_md2: y_test_md2, batch_size_md2: min(FLAGS.batch_size, len(y_test_md2)), sequence_length_md2: x_lengths_md2, keep_prob_md2: 1.0}

            batch_predictions_md2, batch_accuracy_md2, _ = sess_md2.run([predictions_md2, accuracy_md2, logits_md2], feed_dict_md2)

        sum_accuracy_md2 += batch_accuracy_md2
        all_predictions_md2 = np.concatenate([all_predictions_md2, batch_predictions_md2])

    final_accuracy_md2 = sum_accuracy_md2 / num_batches_md2
# length = -(len(labels_md2)%20)

cm_md2 = pd.DataFrame(confusion_matrix(labels_md2, all_predictions_md2), columns=["dialog", "life", "bot", "shp", "tour"], index=["dialog", "life", "bot", "shp", "tour"])
print(cm_md2)

# Print test accuracy
print('Test accuracy: {}'.format(final_accuracy_md2))


#Sample Test
def sample_test(text=""):
    chat_x_input, chat_x_len = data_helper.load_text(text=text,
                                           sw_path=params['stop_word_file'],
                                           min_frequency=params['min_frequency'],
                                           max_length=params['max_length'],
                                           language=params['language'],
                                           vocab_processor=vocab_processor,
                                           shuffle=False, is_w2v=FLAGS.is_w2v,
                                           is_post_tagged=FLAGS.is_post_tagged, is_noun=FLAGS.is_noun)
    print(chat_x_input, chat_x_len)

    if FLAGS.is_w2v:
        chat_input_conv = np.zeros((len(chat_x_input), FLAGS.max_length, FLAGS.embedding_size), dtype=np.float32)
        for idx, item in enumerate(chat_x_input):
            if len(item) < FLAGS.max_length:
                for i, word in enumerate(item):
                    chat_input_conv[idx][i] = word
            else:
                chat_input_conv[idx] = item[:FLAGS.max_length]
        chat_x_input = chat_input_conv

    if 'cnn' in run_mdl1_dir:
        chat_feed_dict = {input_x: chat_x_input, keep_prob: 1.0}
        chat_prediction_md1, chat_logits_md1 = sess.run([predictions_md1, logits_md1], chat_feed_dict)
    else:
        chat_feed_dict = {input_x: chat_x_input, batch_size: 1, sequence_length: chat_x_len,
                     keep_prob: 1.0}

        chat_prediction_md1, chat_logits_md1 = sess.run([predictions_md1, logits_md1], chat_feed_dict)

    if 'cnn' in run_mdl2_dir:
        chat_feed_dict_md2 = {input_x_md2: chat_x_input, keep_prob_md2: 1.0}
        chat_prediction_md2, chat_logits_md2 = sess_md2.run([predictions_md2, logits_md2], chat_feed_dict_md2)
    else:
        chat_feed_dict_md2 = {input_x_md2: chat_x_input,
                         batch_size_md2: 1, sequence_length_md2: chat_x_len,
                         keep_prob_md2: 1.0}

        chat_prediction_md2, chat_logits_md2 = sess_md2.run([predictions_md2, logits_md2], chat_feed_dict_md2)

    idx2res = {"life":0,"svc":1,"shp":2,"tour":3}

    # print("model1", chat_prediction_md1, chat_logits_md1)
    # print("model2", chat_prediction_md2, chat_logits_md2)
    return chat_prediction_md1, chat_prediction_md2, chat_logits_md2

def get_bot_name(bot_id):
    if bot_id == "svc":
        return "전자서비스"
    elif bot_id == "life":
        return "보험"
    elif bot_id == "tour":
        return "여행"
    else :
        return "쇼핑"

#Chatbot
while True:
    user_input_text = input("질문을 입력하세요")
    user_input_text.replace("질문을 입력하세요","")
    if user_input_text == "quit":
        print("종료합니다.")
        break
    result_md1, result_md2, softmax_vec,  = sample_test(text=user_input_text)
    sv_list = softmax_vec[0]
    _max = max(sv_list)
    _argmax = np.argmax(sv_list)

    if result_md1==0:
        print("학습되지 않은 분야의 질문입니다. 보험/전자서비스/여행/쇼핑 관련 질문을 해 주세요")
    if _max > 0.97:
        idx2res = {0:"life", 1:"svc", 2:"shp", 3:"tour"}
        print(get_bot_name(idx2res[result_md2]) + "봇에 연결합니다.")
    else:
        # softmax_vec.sort(reverse =True)

        second_max = 0
        second_max_idx = -1
        for i, v in enumerate(sv_list):
            if i != _argmax:
                if v > second_max:
                    second_max = v
                    second_max_idx = i

        print(get_bot_name(idx2res[result_md2]) + "/" + get_bot_name(idx2res[second_max_idx]) + "봇 중에 선택해주세요")