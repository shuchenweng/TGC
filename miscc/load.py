import os
import struct
import torchtext
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
from miscc.config import cfg
from nltk.tokenize import RegexpTokenizer
import pickle
from collections import defaultdict

# spacy_en = spacy.load('en_core_web_sm')

def en_tokenizer(text): # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]

def translate_glove(captions, wordtoix):
    captions_new = []
    for t in captions:
        rev = []
        for w in t:
            if w in wordtoix:
                rev.append(wordtoix[w])
        # rev.append(0)  # do not need '<end>' token
        captions_new.append(rev)
    return captions_new

def load_glove_vocab(dataset_path):
    TEXT = torchtext.legacy.data.Field(sequential=True, lower=True)
    tab_dataset = torchtext.legacy.data.TabularDataset(
        path=dataset_path, format='tsv',
        fields=[('TEXT', TEXT)]
    )
    TEXT.build_vocab(tab_dataset, vectors="glove.6B.%dd"%(cfg.TEXT.GLOVE_EMBEDDING_DIM))
    return TEXT.vocab

def load_glove_emb(data_dir, split, train_names, test_names, embeddings_num):
    filepath = os.path.join(data_dir, 'captions_glove.pickle')
    if not os.path.isfile(filepath):
        train_path = '%s/glove/train.txt' % (data_dir)
        test_path = '%s/glove/test.txt' % (data_dir)
        if not os.path.exists(train_path):
            create_glove_txt(data_dir, train_names, train_path, embeddings_num)
        if not os.path.exists(test_path):
            create_glove_txt(data_dir, test_names, test_path, embeddings_num)
        train_vocab = load_glove_vocab(train_path)
        test_vocab = load_glove_vocab(test_path)
        train_captions = load_captions(data_dir, train_names, embeddings_num)
        test_captions = load_captions(data_dir, test_names, embeddings_num)
        train_captions = translate_glove(train_captions, train_vocab.stoi)
        test_captions = translate_glove(test_captions, test_vocab.stoi)
        with open(filepath, 'wb') as f:
            pickle.dump([train_captions, test_captions, train_vocab, test_vocab], f, protocol=2)
            print('Save to: ', filepath)
    else:
        with open(filepath, 'rb') as f:
            x = pickle.load(f)
            train_captions, test_captions = x[0], x[1]
            train_vocab, test_vocab = x[2], x[3]
            del x
            print('Load from: ', filepath)

    # print(len(train_vocab.itos), len(train_vocab.stoi), len(test_vocab.stoi), len(test_vocab.itos))
    if split == 'train':
        # a list of list: each list contains
        # the indices of words in a sentence
        captions = train_captions
        glove_embed = nn.Embedding(len(train_vocab.stoi), cfg.TEXT.GLOVE_EMBEDDING_DIM)
        glove_embed.weight.data.copy_(train_vocab.vectors)
        ixtoword, wordtoix = train_vocab.itos, train_vocab.stoi
    else:  # split=='test'
        captions = test_captions
        glove_embed = nn.Embedding(len(test_vocab.stoi), cfg.TEXT.GLOVE_EMBEDDING_DIM)
        glove_embed.weight.data.copy_(test_vocab.vectors)
        ixtoword, wordtoix = test_vocab.itos, test_vocab.stoi

    return captions, ixtoword, wordtoix, glove_embed

def load_captions(data_dir, filenames, embeddings_num):
    all_captions = []
    for i in range(len(filenames)):
        cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
        with open(cap_path, "r", encoding='utf-8') as f:
            captions = f.read().split('\n')
            cnt = 0
            for cap in captions:
                if len(cap) == 0:
                    continue
                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(cap.lower())
                # print('tokens', tokens)
                if len(tokens) == 0:
                    print('cap', cap)
                    continue

                tokens_new = []
                for t in tokens:
                    t = t.encode('ascii', 'ignore').decode('ascii')
                    if len(t) > 0:
                        tokens_new.append(t)
                all_captions.append(tokens_new)
                cnt += 1
                if cnt == embeddings_num:
                    break
            if cnt < embeddings_num:
                print('ERROR: the captions for %s less than %d'
                      % (filenames[i], cnt))
    return all_captions

def create_glove_txt(data_dir, names, path,embeddings_num):
    f_out = open(path, "w", encoding='utf-8')
    caps = load_captions(data_dir, names, embeddings_num)
    for cap in caps:
        for i, word in enumerate(cap):
            if i == 0: f_out.write(word)
            else: f_out.write(' '+word)
        f_out.write('\n')
    f_out.close()

def load_acts_data(data_dir):
    filepath = os.path.join(data_dir, '%s_acts.pickle'% 'test')
    if not os.path.isfile(filepath):
        print('Error: no such a file %s'%(filepath))
        return None
    else:
        with open(filepath, 'rb') as f:
            x = pickle.load(f)
            acts_dict = x[0]
            del x
            print('Load from: ', filepath)

    return acts_dict

def load_class_id(data_dir, total_num):
    if os.path.isfile(data_dir + '/class_info.pickle'):
        with open(data_dir + '/class_info.pickle', 'rb') as f:
            class_id = pickle.load(f,encoding='bytes')
    else:
        class_id = np.arange(total_num)
    return class_id

def load_filenames(data_dir, split):
    filepath = '%s/%s/filenames.pickle' % (data_dir, split)
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
    else:
        filenames = []
    return filenames

def load_bbox(data_dir):
    bbox_path = os.path.join(data_dir, 'bounding_boxes.txt')
    df_bounding_boxes = pd.read_csv(bbox_path,
                                    delim_whitespace=True,
                                    header=None).astype(int)
    #
    filepath = os.path.join(data_dir, 'images.txt')
    df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
    filenames = df_filenames[1].tolist()
    print('Total filenames: ', len(filenames), filenames[0])
    #
    filename_bbox = {img_file[:-4]: [] for img_file in filenames}
    numImgs = len(filenames)
    for i in range(0, numImgs):
        # bbox = [x-left, y-top, width, height]
        bbox = df_bounding_boxes.iloc[i][1:].tolist()

        key = filenames[i][:-4]
        filename_bbox[key] = bbox
    #
    return filename_bbox

def build_dictionary(train_captions, test_captions):
    word_counts = defaultdict(float)
    captions = train_captions + test_captions
    # cnt = 0
    for sent in captions:
        for word in sent:
            word_counts[word] += 1

    vocab = [w for w in word_counts if word_counts[w] >= 0]

    ixtoword = {}
    ixtoword[0] = '<end>'
    wordtoix = {}
    wordtoix['<end>'] = 0
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    train_captions_new = []
    for t in train_captions:
        rev = []
        for w in t:
            if w in wordtoix:
                rev.append(wordtoix[w])
        # rev.append(0)  # do not need '<end>' token
        train_captions_new.append(rev)

    test_captions_new = []
    for t in test_captions:
        rev = []
        for w in t:
            if w in wordtoix:
                rev.append(wordtoix[w])
        # rev.append(0)  # do not need '<end>' token
        test_captions_new.append(rev)
    return [train_captions_new, test_captions_new,
            ixtoword, wordtoix, len(ixtoword)]

def load_text_data(data_dir, split, train_names, test_names, embeddings_num):
    filepath = os.path.join(data_dir, 'captions.pickle')
    if not os.path.isfile(filepath):
        train_captions = load_captions(data_dir, train_names, embeddings_num)
        test_captions = load_captions(data_dir, test_names, embeddings_num)
        train_captions, test_captions, ixtoword, wordtoix, n_words = build_dictionary(train_captions, test_captions)
        with open(filepath, 'wb') as f:
            pickle.dump([train_captions, test_captions, ixtoword, wordtoix], f, protocol=2)
            print('Save to: ', filepath)
    else:
        with open(filepath, 'rb') as f:
            x = pickle.load(f)
            train_captions, test_captions = x[0], x[1]
            ixtoword, wordtoix = x[2], x[3]
            del x
            n_words = len(ixtoword)
            print('Load from: ', filepath)
    if split == 'train':
        # a list of list: each list contains
        # the indices of words in a sentence
        captions = train_captions
        filenames = train_names
    else:  # split=='test'
        captions = test_captions
        filenames = test_names

    return filenames, captions, ixtoword, wordtoix, n_words

def load_part_label(data_dir, glove_wordtoix):
    part_labels, part_label_lens = [], []
    part_label_path = '%s/parts/parts.txt' % (data_dir)
    with open(part_label_path, "r", encoding='utf-8') as f:
        raw_parts = f.read().split('\n')
        for raw_part in raw_parts:
            if len(raw_part) == 0:
                continue
            raw_part = raw_part.replace("\ufffd\ufffd", " ")
            # picks out sequences of alphanumeric characters as tokens
            # and drops everything else
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(raw_part.lower())
            tokens = tokens[1:]
            # print('tokens', tokens)
            if len(tokens) == 0:
                print('raw_part', raw_part)
                continue

            tokens_new = []
            for t in tokens:
                if t == 'left' or t == 'right': continue
                t = t.encode('ascii', 'ignore').decode('ascii')
                if len(t) > 0:
                    tokens_new.append(glove_wordtoix[t])
            part_labels.append(tokens_new)
            part_label_lens.append(len(tokens_new))

    max_len = max(part_label_lens)
    assert (max_len == 1)
    new_part_labels = np.zeros((len(part_labels), max_len), dtype='int64')
    for i in range(len(part_labels)):
        new_part_labels[i,:part_label_lens[i]] = np.array(part_labels[i])

    part_labels = torch.from_numpy(new_part_labels)
    part_label_lens = torch.LongTensor(part_label_lens)

    sorted_part_label_lens, sorted_part_label_indices = torch.sort(part_label_lens, 0, True)

    sorted_part_labels = part_labels[sorted_part_label_indices]
    _, resorted_part_label_indices = torch.sort(sorted_part_label_indices, 0, False)

    sorted_part_labels = Variable(sorted_part_labels).cuda()
    sorted_part_label_lens = Variable(sorted_part_label_lens).cuda()
    resorted_part_label_indices = Variable(resorted_part_label_indices).cuda()
    return sorted_part_labels, sorted_part_label_lens, resorted_part_label_indices

################################### load img data #################################
def write_bytes(data_dir, filenames, filepath, modality, postfix):
    # modality: 'images' / 'semantic_segmentations'
    # postfix: '.jpg' / 'npz'
    with open(filepath, 'wb') as wfid:
        for index in range(len(filenames)):
            if index % 500 == 0:
                print('%07d / %07d'%(index, len(filenames)))

            file_name = os.path.join(data_dir, modality, '%s%s'%(filenames[index], postfix))
            with open(file_name, 'rb') as fid:
                fbytes = fid.read()

            wfid.write(struct.pack('i', len(fbytes)))
            wfid.write(fbytes)

def read_bytes(data_dir, filenames, filepath):
    fbytes = []
    print('start loading bigfile (%0.02f GB) into memory' % (os.path.getsize(filepath)/1024/1024/1024))
    with open(filepath, 'rb') as fid:
        for index in range(len(filenames)):
            fbytes_len = struct.unpack('i', fid.read(4))[0]
            fbytes.append(fid.read(fbytes_len))

    return fbytes

def load_bytes_data(data_dir, split, filenames, modality, postfix):
    filepath = os.path.join(data_dir, '%s_%s.bigfile'%(split, modality))

    if not os.path.isfile(filepath):
        print('writing %s files'%(split))
        write_bytes(data_dir, filenames, filepath, modality, postfix)

    fbytes = read_bytes(data_dir, filenames, filepath)

    return fbytes