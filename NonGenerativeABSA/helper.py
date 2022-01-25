import sys
from collections import OrderedDict

from pytorch_transformers import BertTokenizer
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import SpaceTokenizer


def BERTPOS(sent_file, bert_pos_file):
    sents = open(sent_file).readlines()
    bert_model_name = 'bert-base-uncased'
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)
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


def BERTData(sent_file, pointer_file, bert_sent_file, bert_pointer_file, bert_pos_file):
    bert_model_name = 'bert-base-uncased'
    sents = open(sent_file).readlines()
    pointer_lines = open(pointer_file).readlines()
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)

    writer1 = open(bert_sent_file, 'w')
    writer2 = open(bert_pointer_file, 'w')
    writer3 = open(bert_pos_file, 'w')

    for i in range(len(sents)):
        sent = sents[i].strip()

        pos_seq = list()
        tk = SpaceTokenizer()
        tags = nltk.pos_tag(tk.tokenize(sent))
        for t in tags:
            t1 = t[1]
            pos_seq.append(t1)

        tokens = sent.split(' ')
        bert_tokens = []
        token_map = OrderedDict()
        bert_idx = 0
        bert_pos_seq = []
        for j in range(len(tokens)):
            sub_tokens = bert_tokenizer.wordpiece_tokenizer.tokenize(tokens[j].lower())
            if len(sub_tokens) == 0:
                sub_tokens = [tokens[j]]
            bert_tokens += sub_tokens
            token_map[j] = (bert_idx, bert_idx + len(sub_tokens) - 1)
            bert_idx += len(sub_tokens)
            bert_pos_seq += [pos_seq[j] for k in range(len(sub_tokens))]

        bert_pointers = []
        pointer_line = pointer_lines[i].strip()
        pointers = pointer_line.split(' | ')
        for p in pointers:
            ap_s, ap_e, op_s, op_e, sentiment = p.split(' ')
            new_p = [str(token_map[int(ap_s)][0]), str(token_map[int(ap_e)][1]),
                     str(token_map[int(op_s)][0]), str(token_map[int(op_e)][1]), sentiment]
            bert_pointers.append(' '.join(new_p))
        bert_sent = ' '.join(bert_tokens)
        bert_pointer_line = ' | '.join(bert_pointers)
        # print(sent)
        # print(pointer_line)
        # print(bert_sent)
        # print(bert_pointer_line)
        # print('\n\n')
        writer1.write(bert_sent + '\n')
        writer2.write(bert_pointer_line + '\n')
        writer3.write(' '.join(bert_pos_seq) + '\n')
    writer1.close()
    writer2.close()
    writer3.close()


def DepDist():
    print()


if __name__ == "__main__":
    print(sys.argv[1])
    print(sys.argv[2])
    print(sys.argv[3])
    print(sys.argv[4])
    print(sys.argv[5])
    BERTData(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    # BERTPOS(sys.argv[1], sys.argv[2])

