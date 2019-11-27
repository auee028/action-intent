import numpy as np
from pprint import pprint

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge

from config_v7 import FLAGS


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def inv_dict(dictionary):
    return {v: k for k, v in dictionary.items()}

def get_status(n_events, multi_dec_logits, multi_dec_target, word2ix):
    multi_dec_logits = multi_dec_logits.cpu().detach().numpy()
    multi_dec_target = multi_dec_target.detach().numpy()

    event_sentences = dict(output=[], target=[])
    # print(np.shape(multi_dec_logits), np.shape(multi_dec_target))       # (100, 16, 10403) (100, 16)
    # print(np.shape(multi_dec_logits[0].argmax(axis=-1)))        # (16,)

    for k in range(n_events):
        event_sentences['target'].append(' '.join([inv_dict(word2ix)[ix] for ix in multi_dec_target[k] if ix != FLAGS.PAD]))
        # event_sentences['output'].append(' '.join([inv_dict(word2ix)[ix] for ix in multi_dec_logits[k].argmax(axis=-1) if ix != FLAGS.PAD]))
        event_sentences['output'].append(' '.join([inv_dict(word2ix)[ix] for ix in multi_dec_logits[k] if ix != FLAGS.PAD]))

    return event_sentences

def calc_bleu(output, trg_real, word2ix):
    event_sentences = get_status(n_events=trg_real.shape[0], multi_dec_logits=output, multi_dec_target=trg_real, word2ix=word2ix)
    # print("1st output: {}".format(event_sentences['output'][0]))
    # print("1st target: {}".format(event_sentences['target'][0]))
    cur_scores = []

    def get_eos_ix(tokens):
        for ix, tok in enumerate(tokens):
            if tok=='<EOS>':
                return ix
        # if <EOS> is not found, return None
        return None

    for ix, (output, target) in enumerate(zip(event_sentences['output'], event_sentences['target'])):
        gold = target.strip().encode('utf-8').split()
        hypo = output.strip().encode('utf-8').split()

        gold = gold[:get_eos_ix(gold)]
        hypo = hypo[:get_eos_ix(hypo)]

        smoothie = SmoothingFunction().method3
        bleu_1 = sentence_bleu([gold], hypo, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu_2 = sentence_bleu([gold], hypo, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        bleu_3 = sentence_bleu([gold], hypo, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
        bleu_4 = sentence_bleu([gold], hypo, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        scores = [bleu_1, bleu_2, bleu_3, bleu_4]
        cur_scores.append(scores)

    return tuple(np.mean(cur_scores, axis=0))      # return avg scores for all event

def calc_scores(cur_events, output, trg_real, word2ix):
    event_sentences = get_status(n_events=trg_real.shape[0], multi_dec_logits=output, multi_dec_target=trg_real, word2ix=word2ix)

    def get_eos_ix(tokens):
        for ix, tok in enumerate(tokens):
            if tok=='<EOS>':
                return ix
        # if <EOS> is not found, return None
        return None

    gts = {}
    res = {}
    for ix, (output, target) in enumerate(zip(event_sentences['output'], event_sentences['target'])):
        gold = target.strip().split()
        hypo = output.strip().split()

        # cut when the first '<EOS>' comes out
        gold = gold[:get_eos_ix(gold)]
        hypo = hypo[:get_eos_ix(hypo)]

        # print(gold)
        # print(hypo)

        gts[ix] = [' '.join([token for token in gold])]
        res[ix] = [' '.join([token for token in hypo])]

        # print(gts[ix])
        # print(res[ix])

        print(cur_events[ix])
        print("\toutput: {}".format(gts[ix][0]))
        print("\ttarget: {}\n".format(res[ix][0]))

    # calculate scores
    # http://opennmt.net/OpenNMT-py/vid2text.html
    scorers = {
        "Bleu": Bleu(4),
        "Meteor": Meteor(),
        "Cider": Cider(),
        "Rouge": Rouge()
    }

    scores = {}
    for name, scorer in scorers.items():
        score, all_scores = scorer.compute_score(gts, res)
        if isinstance(score, list):
            for i, sc in enumerate(score, 1):
                scores[name + str(i)] = sc
        else:
            scores[name] = score
    # pprint(scores)

    return scores
