import sys
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from config import FLAGS


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def inv_dict(dictionary):
    return {v: k for k, v in dictionary.iteritems()}

def write_logs(fn, step, loss, score):
    fd = open(fn, 'a+')
    line = '{}\t{}\t{}\n'.format(step, loss, score)
    fd.write(line)
    fd.close()

def get_status(n_events, multi_dec_logits, multi_dec_target, word2ix):
    event_sentences = dict(gold=[], hypo=[])

    for k in range(n_events):
        event_sentences['gold'].append(' '.join([inv_dict(word2ix)[ix] for ix in multi_dec_target[k][0] if ix != FLAGS.PAD]))
        event_sentences['hypo'].append(' '.join([inv_dict(word2ix)[ix] for ix in multi_dec_logits[k][0].argmax(axis=-1) if ix != FLAGS.PAD]))

    return event_sentences

def calc_bleu(multi_dec_logits, multi_dec_target, word2ix):
    event_sentences = get_status(n_events=len(multi_dec_target),
                                 multi_dec_logits=multi_dec_logits, multi_dec_target=multi_dec_target, word2ix=word2ix)

    cur_scores = []

    def get_eos_ix(tokens):
        for ix, tok in enumerate(tokens):
            if tok=='<EOS>':
                return ix
        # if <EOS> is not found, return None
        return None

    for ix, (gold, hypo) in enumerate(zip(event_sentences['gold'], event_sentences['hypo'])):
        gold = gold.strip().encode('utf-8').split()
        hypo = hypo.strip().encode('utf-8').split()

        gold = gold[:get_eos_ix(gold)]
        hypo = hypo[:get_eos_ix(hypo)]

        smoothie = SmoothingFunction().method3
        cur_scores.append(sentence_bleu([gold], hypo, smoothing_function=smoothie))

    return np.mean(cur_scores)      # return avg scores for all event


def generate_eval_data(sess, ph, g, step_func,
                       batcher, word2ix):
    datasetGold = dict(annotations=[])
    datasetHypo = dict(annotations=[])

    score_list = []

    cnt = 0
    while (True):
        vid, multi_dec_logits, multi_dec_target = step_func(sess, ph, fetches=g['multi_dec_logits'], batcher=batcher)

        score_list.append(calc_bleu(multi_dec_logits, multi_dec_target, word2ix))

        event_sentences = get_status(n_events=len(multi_dec_target),
                                     multi_dec_logits=multi_dec_logits, multi_dec_target=multi_dec_target, word2ix=word2ix)

        cnt += 1

        llprint('[test #{}] : {}\n'.format(cnt, vid))

        for ix, (gold, hypo) in enumerate(zip(event_sentences['gold'], event_sentences['hypo'])):
            gold = gold.encode('utf-8')
            hypo = hypo.encode('utf-8')

            gold = gold[:gold.find('<EOS>')].strip()
            hypo = hypo[:hypo.find('<EOS>')].strip()

            datasetGold['annotations'].append(dict(sentence_id=vid + ':' + str(ix), caption=gold))
            datasetHypo['annotations'].append(dict(sentence_id=vid + ':' + str(ix), caption=hypo))

            llprint(u"\tEvent : {}\n".format(ix))
            llprint("\t[gold] : {}\n".format(gold))
            llprint("\t[hypo] : {}\n".format(hypo))

        if batcher.epoch > 0:
            # if end of batch, then break loop!
            break

    print("\nAvg. BLEU : {}".format(sum(score_list)/float(len(score_list))))

    return datasetGold, datasetHypo


def generate_JSONDEMO(sess, ph, g, step_func,
                      batcher, word2ix):
    JSONDemo = {}

    cnt = 0
    while (True):
        vid, duration, multi_timestamps, multi_dec_logits, multi_dec_target = step_func(sess, ph,
                                                                                        fetches=g['multi_dec_logits'],
                                                                                        batcher=batcher)
        event_sentences = get_status(n_events=len(multi_dec_target),
                                     multi_dec_logits=multi_dec_logits, multi_dec_target=multi_dec_target, word2ix=word2ix)

        cnt += 1

        llprint('[test #{}] : {}\n'.format(cnt, vid))

        hypo_sentences = []
        for ix, hypo in enumerate(event_sentences['hypo']):
            _start, _end = multi_timestamps[ix][0], multi_timestamps[ix][1]

            hypo = hypo.encode('utf-8')
            hypo = hypo[:hypo.find('<EOS>')].strip()
            hypo_sentences.append(hypo.decode('utf-8'))

        JSONDemo[vid] = {u'duration': duration,
                         u'sentences': hypo_sentences,
                         u'timestamps': multi_timestamps}

        if batcher.epoch > 0:
            # if end of batch, then break loop!
            break

    return JSONDemo
