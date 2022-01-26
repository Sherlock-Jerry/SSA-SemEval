import sys
from collections import OrderedDict

import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
#import nltk
#nltk.download('averaged_perceptron_tagger')
#from nltk.tokenize import SpaceTokenizer

from tqdm import tqdm

'''
def BERTPOS(sent_file, bert_pos_file):
    sents = open(sent_file).readlines()
    bert_model_name = 'bert-base-uncased'
    #bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_basic_tokenize=False)
    writer = open(bert_pos_file, 'w')
    for sent in sents:
        sent = sent.strip()
        seq = list()
        tk = SpaceTokenizer()
        text = tk.tokenize(sent)
        tags = nltk.pos_tag(text)
        for t in tags:
            t1 = t[1]
            seq.append(t1)

        bert_seq = []
        tokens = sent.split(' ')
        for j in range(len(tokens)):
            sub_tokens = bert_tokenizer.wordpiece_tokenizer.tokenize(tokens[j].lower())
            if len(sub_tokens) == 0:
                sub_tokens = [tokens[j]]
            for k in range(len(sub_tokens)):
                bert_seq.append(seq[j])
        writer.write(' '.join(bert_seq) + '\n')
    writer.close()
'''

def BERTData(sent_file, pointer_file, pos_file, bert_sent_file, bert_pointer_file, bert_pos_file):#only changed here
    bert_model_name = 'bert-base-cased'
    sents = open(sent_file).readlines()
    pos_lists = open(pos_file).readlines()
    pointer_lines = open(pointer_file).readlines()
    #bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_basic_tokenize=False)

    writer1 = open(bert_sent_file, 'w+')
    writer2 = open(bert_pointer_file, 'w+')
    writer3 = open(bert_pos_file, 'w+')
    max_len = 0
    max_len_tok = 0 
    max_trg_len=0
    for i in tqdm(range(len(sents))):
        sent = sents[i].strip()#sentence
        pos_seq = pos_lists[i].strip().split()#pos tag list
        #print(pos_seq)
        #pos_seq = list()
        #tk = SpaceTokenizer()
        #tags = nltk.pos_tag(tk.tokenize(sent))
        #for t in tags:
            #t1 = t[1]
            #pos_seq.append(t1)#all the corresponding pos tags

        tokens = sent.split(' ')#token list
        bert_tokens = []###
        token_map = OrderedDict()
        bert_idx = 0
        bert_pos_seq = []###
        for j in range(len(tokens)):
            sub_tokens = bert_tokenizer.tokenize(tokens[j])
            if len(sub_tokens) == 0:
                sub_tokens = [tokens[j]]
            bert_tokens += sub_tokens
            token_map[j] = (bert_idx, bert_idx + len(sub_tokens) - 1)
            bert_idx += len(sub_tokens)
            bert_pos_seq += [pos_seq[j] for k in range(len(sub_tokens))]

        #print(bert_pos_seq)
        assert len(bert_tokens) == len(bert_pos_seq)
        if max_len < len(bert_pos_seq):
            max_len = len(bert_pos_seq)

        if max_len_tok < len(bert_tokens):
            max_len_tok = len(bert_tokens)

        bert_pointers = []
        
        pointer_line = pointer_lines[i].strip()
        pointers = pointer_line.split(' | ')
        for p in pointers:
            t_s, t_e, ev, a_s, a_e, ar, role = p.split()
            new_p = [str(token_map[int(t_s)][0]), str(token_map[int(t_e)][1]), ev, str(token_map[int(a_s)][0]), str(token_map[int(a_e)][1]), ar, role]
            bert_pointers.append(' '.join(new_p))
        #print(bert_pointers)
        if max_trg_len < len(bert_pointers):
            max_trg_len = len(bert_pointers)
        bert_sent = ' '.join(bert_tokens)
        bert_pos = ' '.join(bert_pos_seq)
        bert_pointer_line = ' | '.join(bert_pointers)
        # print(sent)
        # print(pointer_line)
        # print(bert_sent)
        # print(bert_pointer_line)
        # print('\n\n')
        writer1.write(bert_sent + '\n')
        writer2.write(bert_pointer_line + '\n')
        writer3.write(bert_pos + '\n')
    writer1.close()
    writer2.close()
    writer3.close()
    return max_len_tok, max_len, max_trg_len


def DepDist():
    print()


if __name__ == "__main__":
    sent_file='dev_oct.sent'#print(sys.argv[1])
    point_file='dev_trim_oct.pointer'#print(sys.argv[2])
    pos_file='dev_oct.pos'#print(sys.argv[3])
    bert_sent_file='dev_bert.sent'#print(sys.argv[4])
    bert_point_file='dev_bert.pointer'#print(sys.argv[5])
    bert_pos_file='dev_bert.pos'
    #BERTData(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    max1, max2, max3 = BERTData(sent_file, point_file, pos_file, bert_sent_file, bert_point_file, bert_pos_file)
    print('{},{},{}'.format(max1, max2, max3))
    # BERTPOS(sys.argv[1], sys.argv[2])
