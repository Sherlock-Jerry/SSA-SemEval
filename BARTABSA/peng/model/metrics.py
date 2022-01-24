from fastNLP import MetricBase
from fastNLP.core.metrics import _compute_f_pre_rec
from collections import Counter
import os, json
from tqdm import tqdm

class Seq2SeqSpanMetric(MetricBase):
    def __init__(self, eos_token_id, num_labels, opinion_first=True, data_bundle=None, tokenizer = None, mapping2id = None, dataset = None):
        super(Seq2SeqSpanMetric, self).__init__()
        self.eos_token_id = eos_token_id
        self.num_labels = num_labels
        self.word_start_index = num_labels + 2  # +2, shift for sos and eos
        self.pred_spans = []
        self.null_token = 'NA '
        self.results_write = 0

        self.ae_oe_fp = 0
        self.ae_oe_tp = 0
        self.ae_oe_fn = 0
        self.triple_fp = 0
        self.triple_tp = 0
        self.triple_fn = 0
        self.em = 0
        self.invalid = 0
        self.total = 0
        self.ae_sc_fp = 0
        self.ae_sc_tp = 0
        self.ae_sc_fn = 0
        assert opinion_first is False, "Current metric only supports aspect first"
        self.data_bundle = data_bundle
        self.tokenizer = tokenizer
        self.mapping2id = mapping2id
        self.dataset = f'../final_data/{dataset}'
        # self.tgt_token_list = []
        self.opinin_first = opinion_first
        self.sentence_id_list = []

    def make_phrase(self, start, end, tokenized_sentence):
        phrase = tokenized_sentence[start-len(self.mapping2id)-3: end-len(self.mapping2id)-3+1]
        if phrase == []: return [], []
        sent = self.tokenizer.convert_tokens_to_string(tokenized_sentence)
        phrase = self.tokenizer.convert_tokens_to_string(phrase).strip()
        if phrase == 'NA' or phrase == []: return [], []

        start_idx = sent.find(phrase) - len(self.null_token)
        if start_idx + len(self.null_token) == -1:
            print()
            print(sent)
            print(phrase)
            print(start, end, len(self.mapping2id), self.mapping2id)
            print(tokenized_sentence)
            return [], []
        # assert(start_idx+len(self.null_token)!=-1)
        end_idx = start_idx+len(phrase)

        return [phrase], [f'{start_idx}:{end_idx}']

    def make_pol_int(self, start):
        rev_map = {}
        for k,v in self.mapping2id.items(): rev_map[v] = k
        minima = min(self.mapping2id.values())
        return rev_map[start+minima-2]

    def make_result(self, pred_spans):
        # offset = len(self.mapping2id)
        var_len = len(pred_spans)
        category = 'test'
        req_data_len = len(self.data_bundle.get_dataset(category))

        if var_len == req_data_len:
            print()
            print('*'*20, f'This is test with {var_len} {req_data_len}', '*'*20)

        elif var_len == len(self.data_bundle.get_dataset('dev')):
            print()
            category = 'dev'
            req_data_len = len(self.data_bundle.get_dataset(category))
            print('*'*20, f'This is dev with {var_len} {req_data_len}', '*'*20)

        else:
            print(f'Different Lengths -> Req {req_data_len} \t Len {var_len}')
            return

        self.sentence_id_list = self.sentence_id_list[-req_data_len:]
        assert(len(self.sentence_id_list) == len(pred_spans) == req_data_len)

        final_data_bundle = self.data_bundle.get_dataset(category)
        master_preds = []

        for i in tqdm(range(req_data_len), desc=category):
            idx = list(final_data_bundle['sent_id']).index(self.sentence_id_list[i])
            tokenized_sent_with_NA = self.tokenizer.tokenize(final_data_bundle['text'][idx])
            opinions = []

            for pair in list(set(pred_spans[i])):
                aspect_s, aspect_e, opinion_s, opinion_e, holder_s, holder_e, polarity, intensity = pair
                
                aspect_phrase, aspect_phrase_idx = self.make_phrase(aspect_s, aspect_e, tokenized_sent_with_NA)
                opinion_phrase, opinion_phrase_idx = self.make_phrase(opinion_s, opinion_e, tokenized_sent_with_NA)
                holder_phrase, holder_phrase_idx = self.make_phrase(holder_s, holder_e, tokenized_sent_with_NA)

                polarity_phrase = self.make_pol_int(polarity)
                intensity_phrase = self.make_pol_int(intensity)

                if aspect_phrase == aspect_phrase_idx == holder_phrase == holder_phrase_idx == opinion_phrase == opinion_phrase_idx == []:
                    # [['something'], ['2:5']]
                    # opinions = []
                    continue

                if (polarity_phrase!=self.null_token.strip()) and (intensity_phrase!=self.null_token.strip()):
                    single_set = {
                        "Source": [holder_phrase, holder_phrase_idx],
                        "Target": [aspect_phrase, aspect_phrase_idx],
                        "Polar_expression": [opinion_phrase, opinion_phrase_idx],
                        "Polarity": polarity_phrase,
                        "Intensity": intensity_phrase,
                    }
                    opinions.append(single_set)
                else:
                    # opinions = []
                    # break
                    continue

            master_preds.append({
                'sent_id': self.sentence_id_list[i],
                'text':final_data_bundle['text'][idx].replace(self.null_token, ''),
                'opinions':opinions
            })

        category = '_'.join(category)
        epoch = self.results_write // 2 + 1
        self.results_write += 1

        results_dir_path = f'{self.dataset}/outputs/'
        if not os.path.exists(results_dir_path): os.makedirs(results_dir_path)
        print('Saving to File', f'{results_dir_path}preds_{category}_epoch{epoch}.json')
        with open(f'{results_dir_path}preds_{category}_epoch{epoch}.json','w') as f:
            f.write(json.dumps(master_preds, indent=4))

    def evaluate(self, target_span, pred, tgt_tokens, sent_id):
        
        for i in sent_id: self.sentence_id_list.append(i)

        raghav = False
        self.total += pred.size(0)

        pred_eos_index = pred.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()
        target_eos_index = tgt_tokens.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()
        pred = pred[:, 1:]  # delete </s>
        tgt_tokens = tgt_tokens[:, 1:]
        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1)  # bsz
        pred_seq_len = (pred_seq_len - 2).tolist()
        target_seq_len = target_eos_index.flip(dims=[1]).eq(target_eos_index[:, -1:]).sum(dim=1)  # bsz
        target_seq_len = (target_seq_len - 2).tolist()
        pred_spans = []
        for i, (ts, ps) in enumerate(zip(target_span, pred.tolist())):
            if raghav:
                print('*'*30)
                print("i",self.total-16+i+1)
                print('ts', ts)
            em = 0
            ps = ps[:pred_seq_len[i]]
            if pred_seq_len[i] == target_seq_len[i]:
                em = int(
                    tgt_tokens[i, :target_seq_len[i]].eq(pred[i, :target_seq_len[i]]).sum().item() == target_seq_len[i])
            self.em += em
            invalid = 0
            pairs = []
            cur_pair = []

            if len(ps):
                for index, j in enumerate(ps):
                    cur_pair.append(j)
                    if len(cur_pair) < 8: continue
                    
                    if cur_pair[0] > cur_pair[1] or cur_pair[2] > cur_pair[3] \
                        or cur_pair[4] > cur_pair[5] or cur_pair[6] > cur_pair[7] \
                        or cur_pair[-1]>=self.word_start_index or cur_pair[-2]>=self.word_start_index:
                        invalid = 1
                    else: 
                        pairs.append(tuple(cur_pair))

                    cur_pair = []


            if raghav: print('pairs', pairs)

            pred_spans.append(pairs.copy())
            self.invalid += invalid

            oe_ae_target = [tuple(t[:6]) for t in ts]
            oe_ae_pred = [p[:6] for p in pairs]

            oe_ae_target_counter = Counter(oe_ae_target)
            oe_ae_pred_counter = Counter(oe_ae_pred)
            tp, fn, fp = _compute_tp_fn_fp(set(list(oe_ae_pred_counter.keys())),
                                           set(list(oe_ae_target_counter.keys())))
            self.ae_oe_fn += fn
            self.ae_oe_fp += fp
            self.ae_oe_tp += tp

            # note aesc
            ae_sc_target = [(t[0], t[1], t[-2], t[-1]) for t in ts]
            ae_sc_pred = [(p[0], p[1], p[-2], p[-1]) for p in pairs]
            asts = set([tuple(t) for t in ae_sc_target])
            asps = set(ae_sc_pred)
            for p in list(asps):  # pairs is a 5-tuple
                if p in asts:
                    asts.remove(p)
                    self.ae_sc_tp += 1
                else:
                    self.ae_sc_fp += 1
            self.ae_sc_fn += len(asts)

            ts = set([tuple(t) for t in ts])
            ps = set(pairs)

            for p in list(ps):
                if p in ts:
                    ts.remove(p)
                    self.triple_tp += 1
                else:
                    self.triple_fp += 1
            self.triple_fn += len(ts)

        self.pred_spans += pred_spans



    def get_metric(self, reset=True):
        res = {}
        f, pre, rec = _compute_f_pre_rec(1, self.triple_tp, self.triple_fn, self.triple_fp)
        print('\ntp, fp, fn', self.triple_tp, self.triple_fp, self.triple_fn)
        print('Total', self.triple_tp + self.triple_fp + self.triple_fn)
        print('f, pre, rec', f, pre, rec)

        res['triple_f'] = round(f, 4)*100
        res['triple_rec'] = round(rec, 4)*100
        res['triple_pre'] = round(pre, 4)*100

        f, pre, rec = _compute_f_pre_rec(1, self.ae_oe_tp, self.ae_oe_fn, self.ae_oe_fp)

        res['oe_ae_he_f'] = round(f, 4)*100
        res['oe_ae_he_rec'] = round(rec, 4)*100
        res['oe_ae_he_pre'] = round(pre, 4)*100

        f, pre, rec = _compute_f_pre_rec(1, self.ae_sc_tp, self.ae_sc_fn, self.ae_sc_fp)
        res["ae_pol_int_f"] = round(f, 4)*100
        res["ae_pol_int_rec"] = round(rec, 4)*100
        res["ae_pol_int_pre"] = round(pre, 4)*100

        res['em'] = round(self.em / self.total, 4)
        res['invalid'] = round(self.invalid / self.total, 4)
        if reset:
            self.ae_oe_fp = 0
            self.ae_oe_tp = 0
            self.ae_oe_fn = 0
            self.triple_fp = 0
            self.triple_tp = 0
            self.triple_fn = 0
            self.em = 0
            self.invalid = 0
            self.total = 0
            self.ae_sc_fp = 0
            self.ae_sc_tp = 0
            self.ae_sc_fn = 0
            self.make_result(self.pred_spans)
            self.pred_spans = []
        return res


def _compute_tp_fn_fp(ps, ts):
    ps = ps.copy()
    tp = 0
    fp = 0
    fn = 0
    if isinstance(ts, set):
        ts = {key: 1 for key in list(ts)}
    if isinstance(ps, set):
        ps = {key: 1 for key in list(ps)}
    for key in ts.keys():
        t_num = ts[key]
        if key not in ps:
            p_num = 0
        else:
            p_num = ps[key]
        tp += min(p_num, t_num)
        fp += max(p_num - t_num, 0)
        fn += max(t_num - p_num, 0)
        if key in ps:
            ps.pop(key)
    fp += sum(ps.values())
    return tp, fn, fp
