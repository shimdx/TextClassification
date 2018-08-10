# -*- coding: utf-8 -*-
import re
import os
import sys
import csv
import time
import json
import collections
from _hangul import normalize
## w2v 모델 적용
import gensim
from gensim.test.utils import datapath
## 형태소 분석 후 명사/동사 학습
import konlpy
from konlpy.tag import Kkma, Mecab
import math
import numpy as np
from tensorflow.contrib import learn
from collections import Counter

def load_data(file_path, sw_path=None, min_frequency=0, max_length=0, language='ch', vocab_processor=None, shuffle=True, is_w2v=False, is_post_tagged=False, is_noun=False):
    """
    Build dataset for mini-batch iterator
    :param file_path: Data file path
    :param sw_path: Stop word file path
    :param language: 'ch' for Chinese and 'en' for English
    :param min_frequency: the minimal frequency of words to keep
    :param max_length: the max document length
    :param vocab_processor: the predefined vocabulary processor
    :param shuffle: whether to shuffle the data
    :return data, labels, lengths, vocabulary processor
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        print('Building dataset ...')
        start = time.time()
        incsv = csv.reader(f)
        header = next(incsv)  # Header
        label_idx = header.index('label')
        content_idx = header.index('content')
        print(file_path)
        labels = []
        sentences = []

        if sw_path is not None:
            sw = _stop_words(sw_path)
        else:
            sw = None

        for line in incsv:
            sent = line[content_idx].strip()

            if language == 'ch':
                sent = _tradition_2_simple(sent)  # Convert traditional Chinese to simplified Chinese
            elif language == 'en':
                sent = sent.lower()
            elif language == 'ko':
                sent = sent
            else:
                raise ValueError('language should be one of [ch, en, ko].')

            sent = _clean_data(sent, sw, language=language)  # Remove stop words and special characters

            if len(sent) < 1:
                continue

            if language == 'ch':
                sent = _word_segmentation(sent)
            sentences.append(sent)

            if int(line[label_idx]) < 0:
                labels.append(2)
            else:
                labels.append(int(line[label_idx]))

    labels = np.array(labels)
    # Real lengths
    lengths = np.array(list(map(len, [sent.strip().split(' ') for sent in sentences])))
    counter = Counter(labels)
    print(counter.most_common(len(set(labels))))

    if shuffle : # 현재 우리는 이미 random shuffle 하여 input 제공하므로,..
        shuffle_indices = np.random.permutation(np.arange(len(sentences)))
        sentences = [sentences[idx] for idx in  shuffle_indices]
        labels = [labels[idx] for idx in  shuffle_indices]
        lengths = [lengths[idx] for idx in  shuffle_indices]

    if max_length == 0:
        max_length = max(lengths)

    # Extract vocabulary from sentences and map words to indices
    if vocab_processor is None:
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_length, min_frequency=min_frequency)
        data = np.array(list(vocab_processor.fit_transform(sentences)))
    else:
        data = np.array(list(vocab_processor.transform(sentences)))

    # Change data as word2vector form
    if is_post_tagged:
        sentences = [" ".join(sentence) for sentence in _keyword_list_extractor(sentences, is_noun)]
    print('post-tagging: (%r), example : %s' % (is_post_tagged, sentences[0]))
    if is_w2v:
        w2v_model = gensim.models.Word2Vec.load(datapath("/home/alice/yeongmin/dataset/ko.bin"))
        data = _vectorize_sentence_list(w2v_model=w2v_model, docs = sentences)

    data_size = len(data)


    end = time.time()

    print('Dataset has been built successfully.')
    print('Run time: {}'.format(end - start))
    print('Number of sentences: {}'.format(len(data)))
    print('Vocabulary size: {}'.format(len(vocab_processor.vocabulary.mapping) if not is_w2v else w2v_model.vocabulary.__sizeof__())) # word 2vector에 맞춰 변경
    print('Max document length: {}\n'.format(vocab_processor.max_document_length)) # 추가 변경 해줘야 될 부분

    return data, labels, lengths, vocab_processor


def load_text(text, sw_path=None, min_frequency=0, max_length=0, language='ch', vocab_processor=None, shuffle=True,
              is_w2v=False, is_post_tagged=False, is_noun=False):
    """
    Build dataset for mini-batch iterator
    :param file_path: Data file path
    :param sw_path: Stop word file path
    :param language: 'ch' for Chinese and 'en' for English
    :param min_frequency: the minimal frequency of words to keep
    :param max_length: the max document length
    :param vocab_processor: the predefined vocabulary processor
    :param shuffle: whether to shuffle the data
    :return data, labels, lengths, vocabulary processor
    """

    # print('Building text dataset ...')
    start = time.time()
    sentences = []

    if sw_path is not None:
        sw = _stop_words(sw_path)
    else:
        sw = None

    sent = text.strip()

    if language == 'ch':
        sent = _tradition_2_simple(sent)  # Convert traditional Chinese to simplified Chinese
    elif language == 'en':
        sent = sent.lower()
    elif language == 'ko':
        sent = sent
    else:
        raise ValueError('language should be one of [ch, en, ko].')

    sent = _clean_data(sent, sw, language=language)  # Remove stop words and special characters

    if language == 'ch':
        sent = _word_segmentation(sent)
    sentences.append(sent)
    # print("sentences",sentences)

    # Real lengths
    lengths = np.array(list(map(len, [sent.strip().split(' ') for sent in sentences])))

    if max_length == 0:
        max_length = max(lengths)

    # Extract vocabulary from sentences and map words to indices
    if vocab_processor is None:
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_length, min_frequency=min_frequency)
        data = np.array(list(vocab_processor.fit_transform(sentences)))
    else:
        data = np.array(list(vocab_processor.transform(sentences)))
    # print("data.shape",data.shape)
    # Change data as word2vector form
    if is_post_tagged:
        sentences = [" ".join(sentence) for sentence in _keyword_list_extractor(sentences, is_noun)]
    # print('post-tagging: (%r), example : %s' % (is_post_tagged, sentences[0]))
    if is_w2v:
        # print("sentences after pos tagging", sentences)
        w2v_model = gensim.models.Word2Vec.load(datapath("/home/alice/yeongmin/dataset/ko.bin"))
        data = _vectorize_sentence_list(w2v_model=w2v_model, docs=sentences)
    end = time.time()
    # print("len(data)",len(data))
    # print("data",data)
    #
    # print('Dataset has been built successfully.')
    # print('Run time: {}'.format(end - start))
    # print('Number of sentences: {}'.format(len(data)))
    # print('Vocabulary size: {}'.format(len(
    #     vocab_processor.vocabulary.mapping) if not is_w2v else w2v_model.vocabulary.__sizeof__()))  # word 2vector에 맞춰 변경
    # print('Max document length: {}\n'.format(vocab_processor.max_document_length))  # 추가 변경 해줘야 될 부분

    return data, lengths


def batch_iter(data, labels, lengths, batch_size, num_epochs):
    """
    A mini-batch iterator to generate mini-batches for training neural network
    :param data: a list of sentences. each sentence is a vector of integers
    :param labels: a list of labels
    :param batch_size: the size of mini-batch
    :param num_epochs: number of epochs
    :return: a mini-batch iterator
    """
    assert len(data) == len(labels) == len(lengths)

    data_size = len(data)
    epoch_length = math.ceil(data_size / batch_size)
    # epoch_length = data_size // batch_size

    for _ in range(num_epochs):
        for i in range(epoch_length):
            start_index = i * batch_size
            end_index = min(start_index + batch_size, len(data))
            # end_index = start_index + batch_size


            xdata = data[start_index: end_index]
            ydata = labels[start_index: end_index]
            sequence_length = lengths[start_index: end_index]

            yield xdata, ydata, sequence_length

# --------------- Private Methods ---------------

def _tradition_2_simple(sent):
    """ Convert Traditional Chinese to Simplified Chinese """
    # Please download langconv.py and zh_wiki.py first
    # langconv.py and zh_wiki.py are used for converting between languages
    try:
        import langconv
    except ImportError as e:
        error = "Please download langconv.py and zh_wiki.py at "
        error += "https://github.com/skydark/nstools/tree/master/zhtools."
        print(str(e) + ': ' + error)
        sys.exit()

    return langconv.Converter('zh-hans').convert(sent)


def _word_segmentation(sent):
    """ Tokenizer for Chinese """
    import jieba
    sent = ' '.join(list(jieba.cut(sent, cut_all=False, HMM=True)))
    return re.sub(r'\s+', ' ', sent)


def _stop_words(path):
    with open(path, 'r', encoding='utf-8') as f:
        sw = list()
        for line in f:
            sw.append(line.strip())

    return set(sw)


def _clean_data(sent, sw, language='ch'):
    """ Remove special characters and stop words """
    if language == 'ch':
        sent = re.sub(r"[^\u4e00-\u9fa5A-z0-9！？，。]", " ", sent)
        sent = re.sub('！{2,}', '！', sent)
        sent = re.sub('？{2,}', '！', sent)
        sent = re.sub('。{2,}', '。', sent)
        sent = re.sub('，{2,}', '，', sent)
        sent = re.sub('\s{2,}', ' ', sent)
    if language == 'en':
        sent = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sent)
        sent = re.sub(r"\'s", " \'s", sent)
        sent = re.sub(r"\'ve", " \'ve", sent)
        sent = re.sub(r"n\'t", " n\'t", sent)
        sent = re.sub(r"\'re", " \'re", sent)
        sent = re.sub(r"\'d", " \'d", sent)
        sent = re.sub(r"\'ll", " \'ll", sent)
        sent = re.sub(r",", " , ", sent)
        sent = re.sub(r"!", " ! ", sent)
        sent = re.sub(r"\(", " \( ", sent)
        sent = re.sub(r"\)", " \) ", sent)
        sent = re.sub(r"\?", " \? ", sent)
        sent = re.sub(r"\s{2,}", " ", sent)
    if language == 'ko':
        sent = normalize(sent, english=True, number=True, punctuation=False)
    if sw is not None:
        sent = "".join([word for word in sent if word not in sw])

    return sent

def vectorize_word_list(w2v_model, sentence):  # sentence(word list)의 벡터화
    word_list = []
    for word in sentence :
        try:
            word_list.append(list(w2v_model.wv.get_vector(word)))
        except KeyError:  # 단어가 없는 경우, 아무것도 하지 않고 프로세스 진행
            pass
    return word_list

def _vectorize_sentence_list(w2v_model, docs): # document(sentence list)의 벡터화
    list = []
    for sentence in docs :
        result = vectorize_word_list(w2v_model, sentence)
        list.append(result)
    return list


def _post_tagger (sentence):
    tokenizer = Kkma()
    return tokenizer.pos(sentence)

def _keyword_extractor (sentence, is_noun): # 중요 단어만 추리기 (Noun, Verb 위주)
    key_tagger = ['NNG', 'NNP', 'NNB', 'NNM', 'NR', 'NP', 'VV', 'VA']  # 추후 Tagger 중 중요한 품사 추가하거나, 덜 중요한 품사 제외
    if is_noun:
        key_tagger = ['NNG', 'NNP', 'NNB', 'NNM', 'NR', 'NP']  # 명사만 적용
    words = [word for word, tag in _post_tagger(sentence) if tag in key_tagger]
    return words

def _keyword_list_extractor (dataset, is_noun):
    keword_list = []
    criteria = int(len(dataset)/10)
    percent = 0
    for idx, item in enumerate(dataset) :
        if criteria > 0:
            if idx % criteria == 0 :
                print("%d%% of sentence's has been post-tagged" % percent)
                percent += 10
        keword_list.append(_keyword_extractor(item, is_noun)) # konlpy로 분석해서 형태소별로 중요한 단어만 남기기
    return keword_list