import numpy as np

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from config_v4_4 import FLAGS


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def inv_dict(dictionary):
    return {v: k for k, v in dictionary.items()}

def get_status(n_events, multi_dec_logits, multi_dec_target, word2ix):
    event_sentences = dict(output=[], target=[])
    # print(np.shape(multi_dec_logits), np.shape(multi_dec_target))       # (100, 16, 10403) (100, 16)
    # print(np.shape(multi_dec_logits[0].argmax(axis=-1)))        # (16,)

    for k in range(n_events):
        event_sentences['target'].append(' '.join([inv_dict(word2ix)[ix] for ix in multi_dec_target[k] if ix != FLAGS.PAD]))
        event_sentences['output'].append(' '.join([inv_dict(word2ix)[ix] for ix in multi_dec_logits[k].argmax(axis=-1) if ix != FLAGS.PAD]))

    return event_sentences

def calc_bleu(output, trg_real, word2ix):
    event_sentences = get_status(n_events=len(trg_real),
                                 multi_dec_logits=output.cpu().detach().numpy(), multi_dec_target=trg_real.cpu().detach().numpy(), word2ix=word2ix)
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
