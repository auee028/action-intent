import os
import numpy as np
import math
import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as torchdata
from torch.utils.tensorboard import SummaryWriter

import utils_v4_2 as utils
from data_loader_v4_2 import FrameDataset
from model_v4_2 import I3D_Transformer
from config_v4_2 import FLAGS


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

# log file path
now = datetime.datetime.now()

logs_dir = FLAGS.logs_dir
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

loss_file = os.path.join(logs_dir, "loss", "{}_loss_{}".format(FLAGS.mode, now.strftime('%Y-%m-%d')))
score_file = os.path.join(logs_dir, "score", "{}_score_{}".format(FLAGS.mode, now.strftime('%Y-%m-%d')))

# checkpoint
ckpt_dir = os.path.join(logs_dir, "ckpt", 'ckpt_{}'.format(now.strftime('%Y-%m-%d')))
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# tensorboard writer dir
writer_dir = FLAGS.writer_dir


# training setup
train_batcher = FrameDataset(mode=FLAGS.mode, dataset=FLAGS.dataset)

word2ix = train_batcher.word2ix
pad_idx = word2ix['<PAD>']
vocab_len = len(word2ix)
dataset_len = len(train_batcher.event_list)        # size of whole dataset

epochs = FLAGS.epochs
learning_rate = FLAGS.learning_rate

eval_type = 'rgb'       # help='rgb, flow, or joint'
imagenet_pretrained = 'true'    # help='true or false'
i3d_finetuning = False

# model
model = I3D_Transformer(vocab_len=vocab_len)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=FLAGS.lr_decay_step, gamma=FLAGS.gamma)

torch.save(model.get_idx(), os.path.join(ckpt_dir, 'idx_{}_.pt'.format(FLAGS.batch_size, FLAGS.batch_size, now.strftime('%Y-%m-%d'))))
print(f'Model parameters : {utils.count_parameters(model):,}')


phase = 'Train'

if phase == 'Train':
    model.train()
    if not i3d_finetuning:
        for param in model.i3d.parameters():
            param.requires_grad = False
else:
    model.eval()

writer = SummaryWriter(writer_dir)   # Note that this line alone creates a writer_dir folder.
# writer.add_graph(model=model)

print("Training START!")
start_time = time.time()

for epoch in range(epochs):
    epoch += 1
    train_loss = 0

    iterations = int(dataset_len / FLAGS.batch_size)  # the remaining part is considered at shuffling
    for n in range(iterations):
        global_step = (epoch - 1) * iterations + n

        s, t = train_batcher._collate_fn()
        # print(np.shape(s), np.shape(t))

        src_ = torch.FloatTensor(np.transpose(s, (0, 4, 1, 2, 3))).to(device)   # torch.Size([4, 3, 64, 224, 224])

        trg_ = []
        for tokenized_sentence in t:
            if len(tokenized_sentence) <= (FLAGS.trg_max_seq_len - 2):
                padded_sentence = [word2ix['<PAD>'] for _ in range(FLAGS.trg_max_seq_len-1)]
                padded_sentence[:len(tokenized_sentence)] = tokenized_sentence
                padded_sentence = [word2ix['<SOS>']] + padded_sentence + [word2ix['<EOS>']]
                # print(padded_sentence, len(padded_sentence))
                trg_.append(padded_sentence)
            else:
                cropped_sentence = tokenized_sentence[:FLAGS.trg_max_seq_len-1]
                cropped_sentence = [word2ix['<SOS>']] + cropped_sentence + [word2ix['<EOS>']]
                # print(cropped_sentence, len(cropped_sentence))
                trg_.append(cropped_sentence)

        output = model(src_, trg_, device)      # torch.Size([2, 16, 10403])
        trg_real = torch.LongTensor(trg_)[:, 1:]    # NLLLoss: target value have to be type of Long

        # It takes much time to process to print 1st output and 1st target below.. (gpu -> cpu)
        event_sentences = utils.get_status(n_events=len(trg_real),
                                     multi_dec_logits=output.cpu().detach().numpy(),
                                     multi_dec_target=trg_real.cpu().detach().numpy(), word2ix=word2ix)
        print("\t1st output: {}".format(event_sentences['output'][0]))
        print("\t1st target: {}\n".format(event_sentences['target'][0]))

        # print(output.transpose(1, 2).size())      # torch.Size([100, 10403, 16])
        loss = criterion(output.transpose(1, 2), trg_real.to(device))      # output.transpose(1, 2) -> torch.Size([16, 100, 512])

        if phase == 'Train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss += loss.item()

        display_interval = 100
        saving_interval = 1000
        if global_step % display_interval == 0:
            with open(loss_file, "a") as f:
                if global_step == 0:
                    writer.add_scalar('loss', train_loss, global_step=global_step)
                    print('\tloss = {}'.format(train_loss))
                    f.write("epoch: {}\titer: {}\tloss: {}\n".format(epoch, n, train_loss))
                else:
                    train_loss /= display_interval

                    writer.add_scalar('loss', train_loss, global_step=global_step)
                    print('\tloss = {}'.format(train_loss))
                    f.write("epoch: {}\titer: {}\tloss: {} (cumulative time: {} min)\n".format(epoch, n, train_loss, (time.time()-start_time)/60))

                    train_loss = 0
                f.write("\t1st output: {}\n".format(event_sentences['output'][0]))
                f.write("\t1st target: {}\n".format(event_sentences['target'][0]))

        # if (n % (FLAGS.iterations/10) == 0) & (n != 0):
            with open(score_file, "a") as f:
                bleu_1, bleu_2, bleu_3, bleu_4 = utils.calc_bleu(output=output, trg_real=trg_real, word2ix=word2ix)
                print('\nscores:  bleu_1={:.6f}, bleu_2={:.6f}, bleu_3={:.6f}, bleu_4={:.6f}\n'.format(bleu_1, bleu_2, bleu_3, bleu_4))
                f.write("(epoch: {}, iter: {})  bleu_1={:.6f}, bleu_2={:.6f}, bleu_3={:.6f}, bleu_4={:.6f}\n".format(epoch, n, bleu_1, bleu_2, bleu_3, bleu_4))
        if (global_step % saving_interval == 0) & (global_step != 0):
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
