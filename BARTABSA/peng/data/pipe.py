from typing import OrderedDict
from fastNLP.io import Pipe, DataBundle, Loader
import os
import json
from fastNLP import DataSet, Instance
from transformers import AutoTokenizer
import numpy as np
from itertools import chain
from functools import cmp_to_key
from collections import OrderedDict


def cmp_aspect(v1, v2):
    if v1[0]['from']==v2[0]['from']:
        return v1[1]['from'] - v2[1]['from']
    return v1[0]['from'] - v2[0]['from']

def cmp_opinion(v1, v2):
    if v1[1]['from']==v2[1]['from']:
        return v1[0]['from'] - v2[0]['from']
    return v1[1]['from'] - v2[1]['from']


class BartBPEABSAPipe(Pipe):
    def __init__(self, tokenizer='facebook/bart-base', opinion_first=True, dataset = None):
        super(BartBPEABSAPipe, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.mapping = OrderedDict()
        
        with open(os.path.join(dataset, 'train_convert.json'), 'r') as f:
            tmp_data = json.load(f)
        pol, inten = [], []
        for row in tmp_data:
            # print(json.dumps(row, indent=4))
            for ori_opi in row['aspects']:
                pol.append(ori_opi['Polarity'])
                inten.append(ori_opi['Intensity'])
        pol, inten = list(set(pol)), list(set(inten))
        for label in pol+inten:
            self.mapping[label] = f'<<{label.lower()}>>'
        # self.mapping['<<none>>'] = "<<none>>"
        
        self.opinion_first = opinion_first  # 是否先生成opinion

        cur_num_tokens = self.tokenizer.vocab_size
        self.cur_num_token = cur_num_tokens

        tokens_to_add = sorted(list(self.mapping.values()), key=lambda x:len(x), reverse=True)
        unique_no_split_tokens = self.tokenizer.unique_no_split_tokens
        sorted_add_tokens = sorted(list(tokens_to_add), key=lambda x:len(x), reverse=True)
        for tok in sorted_add_tokens:
            assert self.tokenizer.convert_tokens_to_ids([tok])[0]==self.tokenizer.unk_token_id
        self.tokenizer.unique_no_split_tokens = unique_no_split_tokens + sorted_add_tokens
        self.tokenizer.add_tokens(sorted_add_tokens)
        self.mapping2id = {}
        self.mapping2targetid = {}

        for key, value in self.mapping.items():
            key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
            assert len(key_id) == 1, value
            assert key_id[0] >= cur_num_tokens
            self.mapping2id[key] = key_id[0]
            self.mapping2targetid[key] = len(self.mapping2targetid)

        print("mapping2targetid", '\n', json.dumps(self.mapping2targetid, indent=4))
        print()
        print("mapping2id", '\n', json.dumps(self.mapping2id, indent=4))
        print()

        import pickle
        dictionary_data = self.mapping2targetid 
        a_file = open("/content/mapping2targetid.pkl", "wb")
        pickle.dump(dictionary_data, a_file)
        a_file.close()
        

    def process(self, data_bundle: DataBundle) -> DataBundle:
        """
        words: List[str]
        aspects: [{
            'index': int
            'from': int
            'to': int
            'polarity': str
            'intensity' : str
            'term': List[str]
        }],
        opinions: [{
            'index': int
            'from': int
            'to': int
            'term': List[str]
        }]
        holders: [{
            'index': int
            'from': int
            'to': int
            'term': List[str]
        }]

        output: [o_s, o_e, a_s, a_e, h_s, h_e, pol, int], # [a_s, a_e, o_s, o_e, h_s, h_e pol, int]
        :param data_bundle:
        :return:
        """
        target_shift = len(self.mapping) + 2  # sos，eos

        # if target_shift != 11:
        #     print(self.mapping)
        # else:
        #     print("Print nhi hua!!")
        #     # exit()
        
        def prepare_target(ins):
            raw_words = ins['raw_words']
            word_bpes = [[self.tokenizer.bos_token_id]]
            for word in raw_words:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                word_bpes.append(bpes)
            word_bpes.append([self.tokenizer.eos_token_id])

            lens = list(map(len, word_bpes))
            cum_lens = np.cumsum(list(lens)).tolist()
            target = [0]  # 特殊的开始
            target_spans = []
            _word_bpes = list(chain(*word_bpes))

            aspects_opinions = [(a, o, h) for a, o, h in zip(ins['aspects'], ins['opinions'], ins['holders'])] # -- new code --
            if self.opinion_first:
                aspects_opinions = sorted(aspects_opinions, key=cmp_to_key(cmp_opinion))
            else:
                aspects_opinions = sorted(aspects_opinions, key=cmp_to_key(cmp_aspect))

            for aspects, opinions, holders in aspects_opinions:  # 预测bpe的start
                assert aspects['index'] == opinions['index'] == holders['index']
                try: 
                    a_start_bpe = cum_lens[aspects['from']]  # 因为有一个sos shift
                    a_end_bpe = cum_lens[aspects['to']-1]  # 这里由于之前是开区间，刚好取到最后一个word的开头
                except:
                    a_start_bpe = -1 
                    a_end_bpe =-1
                
                try:
                    o_start_bpe = cum_lens[opinions['from']]  # 因为有一个sos shift
                    o_end_bpe = cum_lens[opinions['to']-1]  # 因为有一个sos shift
                except:
                    o_start_bpe = -1
                    o_end_bpe =-1
                try: 
                    h_start_bpe = cum_lens[holders['from']]  # 因为有一个sos shift
                    h_end_bpe = cum_lens[holders['to']-1]  # 因为有一个sos shift
                except:
                    h_start_bpe = -1
                    h_end_bpe =-1
                # 这里需要evaluate
                # if ins["raw_words"][0] == "At" and ins['raw_words'] == "first":
                ###### word_bpes have multiple items in list, making cum_lens not continuous #####
                # print()
                # print(word_bpes)
                # print([self.tokenizer.convert_ids_to_tokens(i) for i in word_bpes])
                # print(lens)
                # print(ins.items())
                # print(cum_lens, len(cum_lens))
                # print(ins['raw_words'],a_start_bpe,a_end_bpe,o_start_bpe,o_end_bpe, target_shift)
                # x123 = input("Enter num") 
                for idx, word in zip((o_start_bpe, o_end_bpe, a_start_bpe, a_end_bpe, h_start_bpe, h_end_bpe),
                                     (opinions['term'][0], opinions['term'][-1], aspects['term'][0], aspects['term'][-1],holders['term'][0], holders['term'][-1])):
                    try: assert _word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True)[:1])[0] or \
                           _word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True)[-1:])[0]
                    except: pass

                if self.opinion_first:
                    target_spans.append([o_start_bpe+target_shift, o_end_bpe+target_shift,
                                         a_start_bpe+target_shift, a_end_bpe+target_shift,
                                         h_start_bpe+target_shift, h_end_bpe+target_shift])
                else:
                    target_spans.append([a_start_bpe+target_shift, a_end_bpe+target_shift,
                                         o_start_bpe+target_shift, o_end_bpe+target_shift,
                                         h_start_bpe+target_shift, h_end_bpe+target_shift])
                target_spans[-1].append(self.mapping2targetid[aspects['Polarity']]+2)   # 前面有sos和eos
                target_spans[-1].append(self.mapping2targetid[aspects['Intensity']]+2)   # 前面有sos和eos
                target_spans[-1] = tuple(target_spans[-1])
            target.extend(list(chain(*target_spans)))
            target.append(1)  # append 1是由于特殊的eos

            return {'tgt_tokens': target, 'target_span': target_spans, 'src_tokens': list(chain(*word_bpes))}

        data_bundle.apply_more(prepare_target, use_tqdm=True, tqdm_desc='Pre. tgt.')

        data_bundle.set_ignore_type('target_span')
        data_bundle.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
        data_bundle.set_pad_val('src_tokens', self.tokenizer.pad_token_id)

        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
        data_bundle.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len')
        data_bundle.set_target('tgt_tokens', 'tgt_seq_len', 'target_span')

        return data_bundle

    def process_from_file(self, paths, demo=False) -> DataBundle:
        """

        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.ConllLoader` 的load函数。
        :return: DataBundle
        """
        # 读取数据
        data_bundle = ABSALoader(demo=demo).load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle


class ABSALoader(Loader):
    def __init__(self, demo=False):
        super().__init__()
        self.demo = demo

    def _load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ds = DataSet()
        for ins in data:
            tokens = ins['words']
            aspects = ins['aspects']
            opinions = ins['opinions']
            holders = ins['holder']
            assert len(aspects)==len(opinions)==len(holders)
            ins = Instance(raw_words=tokens, aspects=aspects, opinions=opinions, holders=holders)
            ds.append(ins)
            if self.demo and len(ds)>30:
                break
        return ds


if __name__ == '__main__':
    data_bundle = BartBPEABSAPipe().process_from_file('pengb/16res')
    print(data_bundle)

