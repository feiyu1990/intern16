import os
import json
import argparse
from random import shuffle, seed
import string
# import h5py
import numpy as np
from scipy.misc import imread, imresize
import nltk
from collections import defaultdict
import operator
from collections import Counter


def tokeninze_phrase(sentence_pre, priority_phrase, start_phrase):
    tokenized = []
    idx = 0
    while idx < len(sentence_pre):
        if sentence_pre[idx] == '-':
            idx+=1
        elif sentence_pre[idx] in start_phrase:
            if ' '.join(sentence_pre[idx:idx+2]) in priority_phrase:
                tokenized.append(' '.join(sentence_pre[idx:idx+2]))
                idx += 2
                # print tokenized
            elif ' '.join(sentence_pre[idx:idx+3]) in priority_phrase:
                tokenized.append(' '.join(sentence_pre[idx:idx+3]))
                idx += 3
            else:
                tokenized.append(sentence_pre[idx])
                idx += 1
        else:
            tokenized.append(sentence_pre[idx])
            idx += 1
    return tokenized


def create_phrase_dict():
    phrases = set()
    with open('Keywords.txt') as f:
        for line in f:
            phrases.add(line.split('\n')[0])
    words = json.load(open('word_count_neuraltalk.json'))
    count = 0
    priority_phrase = []
    start_phrase = set()
    len_phrase = []
    for i in phrases:
        if i not in words:
            len_phrase.append(len(i.split(' ')))
            if len(i.split(' ')) in {2, 3}:
                count += 1
                priority_phrase.append(i)
                start_phrase.add(i.split(' ')[0])
    print count
    print priority_phrase
    print Counter(len_phrase)
    json.dump({'phrase': priority_phrase, 'start': list(start_phrase)}, open('priority_phrase.json', 'w'))


def phrase_encoding(tokenize='nltk', threshold=50, len_threshold=3):
    b = json.load(open('../fotolia/train_raw.json'))
    word_count = defaultdict(int)
    caption_list = []
    punc = '!#$%&()*+,./:;<=>?@[\\]^`{|}~'
    caption_length = []
    info = []
    temp = json.load(open('priority_phrase.json'))
    priority_phrase = set(temp['phrase'])

    for idx, i in enumerate(b):
        i['file_path'] = ('/').join(i['file_path'].split('/')[6:])
        try:
            str(i['captions'][0])
        except:
            continue
        if tokenize == 'nltk':
            sentence = str(i['captions'][0]).lower().translate(None, punc)
            token = nltk.word_tokenize(sentence)
        elif tokenize == 'neuraltalk':
            sentence = str(i['captions'][0]).lower().translate(None, string.punctuation).strip()
            token = sentence.split()
        else:
            sentence = str(i['captions'][0]).lower().translate(None, punc)
            start_phrase = set(temp['start'])
            token = tokeninze_phrase(sentence.split(), priority_phrase, start_phrase)
        if idx % 10000 == 0:
            print token, sentence
        if len(token) == 0:
            continue
        for word in token:
            word_count[word] += 1
        caption_list.append(token)
        caption_length.append(len(token))
        i['tokenized'] = token
        info.append(i)
    print 'number of valid images:', len(info)

    temp = sorted(word_count.items(), key=operator.itemgetter(1), reverse=True)
    print temp[:100]
    min_occur_list = []
    for sent in caption_list:
        min_occur = np.min([word_count[i] for i in sent])
        min_occur_list.append(min_occur)
    # print Counter(min_occur_list)
    # print np.histogram(min_occur_list, bins=30)
    print 'caption length:', sorted(Counter(caption_length).items(), key=operator.itemgetter(0))
    json.dump(word_count, open('word_count_' + tokenize + '.json', 'w'))
    json.dump(info, open('../fotolia/train_raw_' + tokenize + '.json', 'w'))

    info_cleaned = []
    rare_words = set([i for i,j in word_count.items() if j<threshold])
    for i in info:
        if len(i['tokenized']) < len_threshold:
            continue
        dump = False
        for word in i['tokenized']:
            if word in rare_words:
                dump = True
                break
        if not dump:
            info_cleaned.append(i)
    print 'rare words #', len(rare_words)
    print 'info cleaned #', len(info_cleaned)
    json.dump(info_cleaned, open('../fotolia/train_raw_'+tokenize+'_cleaned.json', 'w'))


def phrase_encoding_():
    b = json.load(open('../fotolia/train_raw_neuraltalk_cleaned.json'))
    # temp = json.load(open('../fotolia/training/train_minlen2_50w.json'))['images']
    # img_ids = set([i['id'] for i in temp])
    # print len(img_ids)
    word_count = defaultdict(int)
    caption_list = []
    punc = '!#$%&()*+,./:;<=>?@[\\]^`{|}~'
    caption_length = []
    info = []
    temp = json.load(open('priority_phrase.json'))
    priority_phrase = set(temp['phrase'])
    start_phrase = set(temp['start'])
    idx = 0
    for i in b:
        # if i['id'] not in img_ids:
        #     continue
        sentence = str(i['captions'][0]).lower().translate(None, punc)
        token = tokeninze_phrase(sentence.split(), priority_phrase, start_phrase)
        if idx % 10000 == 0:
            print idx, token, sentence
        if len(token) == 0:
            continue
        for word in token:
            word_count[word] += 1
        caption_list.append(token)
        caption_length.append(len(token))
        i['tokenized'] = token
        info.append(i)
        idx += 1

    print 'number of valid images:', len(info)

    tokenize = 'phrase'
    temp = sorted(word_count.items(), key=operator.itemgetter(1), reverse=True)
    print temp[:100]
    min_occur_list = []
    for sent in caption_list:
        min_occur = np.min([word_count[i] for i in sent])
        min_occur_list.append(min_occur)
    # print Counter(min_occur_list)
    # print np.histogram(min_occur_list, bins=30)
    print 'caption length:', sorted(Counter(caption_length).items(), key=operator.itemgetter(0))
    json.dump(word_count, open('word_count_' + tokenize + '.json', 'w'))
    json.dump(info, open('../fotolia/train_raw_' + tokenize + '.json', 'w'))


if __name__ == "__main__":
    phrase_encoding_()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--method', required=True)
    # args = parser.parse_args()
    # params = vars(args) # convert to ordinary dict
    # phrase_encoding(params['method'])
