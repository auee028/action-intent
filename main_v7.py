import os
import glob
import numpy as np
import math
import datetime
import time
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as torchdata
from torch.utils.tensorboard import SummaryWriter

import utils_v7 as utils
from data_loader_v7 import FrameDataset, FeatsDataset
from model_v7 import Transformer
from config_v7 import FLAGS


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    with_cuda = True
    device_ids = None
    print("------- GPU Working -------")
    print("[Current GPU]:" + str(torch.cuda.get_device_name(0)))
else:
    with_cuda = False
    device_ids = None
    print("------- CPU Working -------")


mode = FLAGS.mode
now = datetime.datetime.now()

# log file path
logs_dir = FLAGS.logs_dir
loss_dir = os.path.join(logs_dir, "loss")
score_dir = os.path.join(logs_dir, "score")
if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)

loss_file = os.path.join(logs_dir, "loss", mode, "{}_loss_{}_v{}".format(mode, now.strftime('%Y-%m-%d'), FLAGS.version))
score_file = os.path.join(logs_dir, "score", mode, "{}_score_{}_v{}".format(mode, now.strftime('%Y-%m-%d'), FLAGS.version))
with open(loss_file, "a") as f:
    f.write("*******************************************************************************************\n")
    f.write("learning_rate = {}\n\n".format(FLAGS.learning_rate))
if mode == "train":
    with open(score_file, "a") as f:
        f.write("*******************************************************************************************\n")
        f.write("learning_rate = {}\n\n".format(FLAGS.learning_rate))

# checkpoint
ckpt_dir = ''
if mode == "train":
    ckpt_dir = os.path.join(logs_dir, "ckpt", 'ckpt_{}_v{}'.format(now.strftime('%Y-%m-%d'), FLAGS.version))
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
else:
    ckpt_date = sorted(glob.glob(os.path.join(logs_dir, "ckpt/*")))[-1]
    ckpt_dir = sorted(glob.glob(ckpt_date + "/*"))[-1]
    # ckpt_dir = ckpt_date + "/model-80000.pt"

# tensorboard writer dir
writer_dir = os.path.join(FLAGS.writer_dir, "transformer_{}_v{}".format(now.strftime('%Y-%m-%d'), FLAGS.version))


# training setup
assert((FLAGS.frame_or_feats == "frame") or (FLAGS.frame_or_feats == "feats"))
if FLAGS.frame_or_feats == "frame":
    data_batcher = FrameDataset(mode=mode, dataset=FLAGS.dataset)
else:
    data_batcher = FeatsDataset(mode=mode, dataset=FLAGS.dataset)


word2ix = data_batcher.word2ix
pad_idx = word2ix['<PAD>']
vocab_len = len(word2ix)
dataset_len = len(data_batcher.event_list)        # size of whole dataset

# eval_type = 'rgb'       # help='rgb, flow, or joint'
# imagenet_pretrained = 'true'    # help='true or false'
# i3d_finetuning = True

# model
# model = I3D_Transformer(vocab_len=vocab_len)
model = Transformer(vocab_len=vocab_len, feats_dim=data_batcher.feats_dim)
model.to(device)

learning_rate = FLAGS.learning_rate

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
# criterion = nn.NLLLoss()

# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=FLAGS.lr_decay_step, gamma=FLAGS.gamma)

# torch.save(model.get_idx(), os.path.join(ckpt_dir, 'idx_{}_.pt'.format(FLAGS.batch_size, FLAGS.batch_size, now.strftime('%Y-%m-%d'))))
print(f'Model parameters : {utils.count_parameters(model):,}')

writer = SummaryWriter(writer_dir)   # Note that this line alone creates a writer_dir folder.
# writer.add_graph(model=model)

def train():
    print("Start training !\n")
    model.train()

    epochs = FLAGS.epochs
    global_step = 0
    start_time = time.time()

    DISP_INTERVAL = 100
    SAVE_INTERVAL = 5000

    # if not i3d_finetuning:
    #     for param in model.i3d.parameters():
    #         param.requires_grad = False
    #     model.i3d.eval()

    ckpt_dir = None
    # ckpt_dir = "/media/pjh/2e8b4b6c-7754-4bf3-b610-7e52704614af/dense-captioning/logs/ckpt/ckpt_2019-11-15_v4_4/model-80000.pt"
    if ckpt_dir:
        model.load_state_dict(torch.load(ckpt_dir))
        print("Checkpoint loaded : {}".format(ckpt_dir))

    for epoch in range(epochs):

        epoch += 1
        train_loss = 0

        iterations = dataset_len // FLAGS.num_events  # the remainder is considered at shuffling
        for n in range(iterations):
            global_step += 1

            s, t = data_batcher._collate_fn()
            # print(np.shape(s), np.shape(t))

            # src_ = torch.FloatTensor(np.transpose(s, (0, 4, 1, 2, 3))).to(device)   # torch.Size([4, 3, 64, 224, 224])
            src_ = torch.FloatTensor(s).to(device)

            trg_ = []
            trg_real = []
            for tokenized_sentence in t:
                if len(tokenized_sentence) <= (FLAGS.trg_max_seq_len - 2):
                    padded_sentence = [word2ix['<PAD>'] for _ in range(FLAGS.trg_max_seq_len-1)]
                    padded_sentence[:len(tokenized_sentence)] = tokenized_sentence
                    # print(padded_sentence, len(padded_sentence))
                    trg_.append([word2ix['<SOS>']] + padded_sentence)
                    trg_real.append(padded_sentence + [word2ix['<EOS>']])
                else:
                    cropped_sentence = tokenized_sentence[:FLAGS.trg_max_seq_len-1]
                    # cropped_sentence = [word2ix['<SOS>']] + cropped_sentence + [word2ix['<EOS>']]
                    # print(cropped_sentence, len(cropped_sentence))
                    trg_.append([word2ix['<SOS>']] + cropped_sentence)
                    trg_real.append(cropped_sentence + [word2ix['<EOS>']])

            trg_ = torch.LongTensor(trg_).to(device)
            trg_real = torch.LongTensor(trg_real)    # NLLLoss: target value have to be type of Long

            output = model(src_, trg_, device)  # torch.Size([2, 16, 10403])
            output_argmax = torch.argmax(output, dim=-1)

            # It takes much time to process to print 1st output and 1st target below.. (gpu -> cpu)
            event_sentences = utils.get_status(n_events=len(trg_real), multi_dec_logits=output_argmax, multi_dec_target=trg_real, word2ix=word2ix)
            print("\t1st output: {}".format(event_sentences['output'][0]))
            print("\t1st target: {}\n".format(event_sentences['target'][0]))

            # print(output.transpose(1, 2).size())      # torch.Size([100, 10403, 16])

            # get loss and scores
            loss = criterion(output.transpose(1, 2), trg_real.to(device))      # output.transpose(1, 2) -> torch.Size([16, 100, 512])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            scores = utils.calc_scores(cur_events=data_batcher.selected_events, output=output_argmax, trg_real=trg_real,
                                       word2ix=word2ix)

            writer.add_scalar('loss', train_loss, global_step=global_step)
            writer.add_scalar('score_meteor', scores['Meteor'], global_step=global_step)

            if global_step % DISP_INTERVAL == 0:
                with open(loss_file, "a") as f:
                    if global_step == 0:
                        print('\tloss = {}'.format(train_loss))
                        f.write("epoch: {}\titer: {}\tloss: {}\n".format(epoch, n, train_loss))
                    else:
                        train_loss /= DISP_INTERVAL

                        print('\tloss = {}'.format(train_loss))
                        f.write("epoch: {}\titer: {}\tloss: {} (cumulative time: {} min)\n".format(epoch, n, train_loss, (
                                    time.time() - start_time) / 60))

                        train_loss = 0
                    f.write("\t1st output: {}\n".format(event_sentences['output'][0]))
                    f.write("\t1st target: {}\n".format(event_sentences['target'][0]))

                # # if (n % (FLAGS.iterations/10) == 0) & (n != 0):
                with open(score_file, "a") as f:
                    pprint(scores)
                    f.write("(epoch: {}, iter: {})\n{}".format(epoch, n, scores))

            # save model
            if global_step % SAVE_INTERVAL == 0:
                print("saving model ...")
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model-{}.pt'.format(global_step)))

                # np.save("frames_{}_{}".format(epoch,  n), s)
                # np.save("features_{}_{}".format(epoch, n), i3d_feature_maps.cpu().detach().numpy())
                # np.save("output_{}_{}".format(epoch, n), output.cpu().detach().numpy())

        # print("{} epoch time: {} min".format(epoch, (time.time() - start_time) / 60))
        with open(loss_file, "a") as f:
            f.write("\t{} epoch time: {} min\n".format(epoch, (time.time() - start_time) / 60))

        # if (epoch % interval == 0) or (epoch == 1):
        #     print("{} Loss: {:.4f}".format(phase, epoch_loss))


def val():
    print("Start validation !")

    test_loss = 0

    if ckpt_dir:
        state_dict = torch.load(ckpt_dir)
        model.load_state_dict(state_dict)
        print('model checkpoint restored')

        model.eval()

        with torch.no_grad():
            for step in range(dataset_len // FLAGS.batch_size + 1):
                s, t = data_batcher._collate_fn()
                # print(np.shape(s), np.shape(t))

                src_ = torch.FloatTensor(np.transpose(s, (0, 4, 1, 2, 3))).to(device)  # torch.Size([4, 3, 64, 224, 224])

                dec_in = [word2ix['<SOS>']]

                trg_ = []
                trg_real = []
                for tokenized_sentence in t:
                    trg_.append(dec_in)

                    if len(tokenized_sentence) <= (FLAGS.trg_max_seq_len - 2):
                        padded_sentence = [word2ix['<PAD>'] for _ in range(FLAGS.trg_max_seq_len - 1)]
                        padded_sentence[:len(tokenized_sentence)] = tokenized_sentence
                        padded_sentence = padded_sentence + [word2ix['<EOS>']]
                        # print(padded_sentence, len(padded_sentence))
                        trg_real.append(padded_sentence)
                    else:
                        cropped_sentence = tokenized_sentence[:FLAGS.trg_max_seq_len - 1]
                        cropped_sentence = cropped_sentence + [word2ix['<EOS>']]
                        # print(cropped_sentence, len(cropped_sentence))
                        trg_real.append(cropped_sentence)

                trg_ = torch.LongTensor(trg_).to(device)
                trg_real = torch.LongTensor(trg_real)

                # limit = 20
                for num in range(FLAGS.trg_max_seq_len):
                    output = model(src_, trg_, device)
                    # print("dec_out: {}".format(output.shape))
                    output_argmax = torch.argmax(output, dim=-1)
                    # print(output_argmax.shape)
                    trg_ = torch.cat([trg_, output_argmax[:,-1:]], dim=-1)
                    # print(trg_.shape)
                results = trg_[:,1:]
                # print(results.shape)
                # print(results)
                # print(trg_real)

                # It takes much time to process to print 1st output and 1st target below.. (gpu -> cpu)
                event_sentences = utils.get_status(n_events=len(trg_real), multi_dec_logits=results, multi_dec_target=trg_real, word2ix=word2ix)

                cur_events = data_batcher.selected_events
                print("***RESULTS***")
                for i in range(len(cur_events)):
                    # print("({})".format(cur_events[i]))
                    # print("\toutput: {}".format(event_sentences['output'][i]))
                    # print("\ttarget: {}\n".format(event_sentences['target'][i]))

                    with open(loss_file, "a") as f:
                        f.write("{}:  {}\n".format(step * FLAGS.batch_size + i + 1, cur_events[i]))
                        f.write("\toutput: {}\n".format(event_sentences['output'][i]))
                        f.write("\ttarget: {}\n\n".format(event_sentences['target'][i]))

                    # with open(score_file, "a") as f:
                        scores = utils.calc_scores(cur_events=cur_events, output=results, trg_real=trg_real, word2ix=word2ix)
                        pprint(scores)

                        # bleu_1, bleu_2, bleu_3, bleu_4 = scores["Bleu"]
                        # meteor = scores["Meteor"]
                        # cider = scores['Cider']
                        # rouge = scores["Rouge"]
                        '''
                        bleu_1, bleu_2, bleu_3, bleu_4 = utils.calc_bleu(output=results.cpu().detach().numpy(), trg_real=trg_real.detach().numpy(),
                                                                         word2ix=word2ix)
                        print('\nscores:  bleu_1={:.6f}, bleu_2={:.6f}, bleu_3={:.6f}, bleu_4={:.6f}\n'.format(bleu_1,
                                                                                                               bleu_2,
                                                                                                               bleu_3,
                                                                                                               bleu_4))
                                                                                                          
                        '''
                        # print(
                        #     'scores:  bleu_1={:.6f}, bleu_2={:.6f}, bleu_3={:.6f}, bleu_4={:.6f}\n'.format(bleu_1,
                        #                                                                                    bleu_2,
                        #                                                                                    bleu_3,
                        #                                                                                    bleu_4))
                        # f.write("{}:  {}\n".format(step * FLAGS.batch_size + i + 1, cur_events[i]))
                        # f.write(
                        #     "\tbleu_1={:.6f}, bleu_2={:.6f}, bleu_3={:.6f}, bleu_4={:.6f}\n\n".format(bleu_1, bleu_2, bleu_3, bleu_4))

                # print(output.transpose(1, 2).size())      # torch.Size([100, 10403, 16])
                loss = criterion(output.transpose(1, 2),
                                 trg_real.to(device))  # output.transpose(1, 2) -> torch.Size([16, 100, 512])

                test_loss += loss.item()

        print("**********************************************************************************")
        print("Avg. loss = {}".format(test_loss / dataset_len))
        with open(loss_file, "a") as f:
            f.write("**********************************************************************************\n")
            f.write("Avg. loss = {}\n".format(test_loss / dataset_len))


if __name__ == "__main__":
    if mode == "train":
        assert FLAGS.dataset == "train"
        train()
    elif mode == "val":
        # assert FLAGS.dataset == "val_1"
        val()
    elif mode == "test":
        assert FLAGS.dataset == "val_1"
        # test()
    else:
        print("mode configuration is wrong (train/val)")
