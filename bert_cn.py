# https://mccormickml.com/2019/07/22/BERT-fine-tuning/
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import torch.nn
import json
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3, 4"


def loadcn(filename, classes, max_len=128, batch_size=32):
    sentences = []
    contexts = []
    contexts_labels = []
    labels = []
    with open(filename, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            jsondata = json.loads(line)
            classname = jsondata['category']
            if classname in classes:
                str = ' '.join(jsondata['context'].split())
                # str = re.sub(u"([a-zA-Z])", "", str)
                contexts.append(str)
                contexts_labels.append(classes.index(classname))
            line = f.readline()
            # if len(sentences) > 10:
            #     break
    for idx, context in enumerate(contexts):
        num = len(context) // max_len
        for i in range(num):
            sentences.append(context[i*max_len: i*max_len + max_len])
            labels.append(contexts_labels[idx])
    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", do_lower_case=True)
    token_text = [tokenizer.tokenize(sentence) for sentence in sentences]
    input_ids = [tokenizer.convert_tokens_to_ids(token) for token in token_text]
    print(token_text[0])
    # input_ids = pad_sequences(input_ids, maxlen=max_len, dtype='long', truncating="post", padding="post")
    input_ids = pad_sequences(input_ids, maxlen=max_len, dtype='long')
    print(input_ids.shape)
    attention_masks = input_ids.copy()
    for x in attention_masks:
        x[x > 0] = 1
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                                        random_state=2018,
                                                                                        test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2018, test_size=0.1)
    train_inputs, train_labels, train_masks = torch.tensor(train_inputs).long(), torch.tensor(train_labels).long(), torch.tensor(
        train_masks).float()
    validation_inputs, validation_labels, validation_masks = torch.tensor(validation_inputs).long(), torch.tensor(
        validation_labels).long(), torch.tensor(validation_masks).float()
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_loader = DataLoader(validation_data, shuffle=False, batch_size=batch_size)
    return train_dataloader, validation_loader


def pad_sequences(inputs, maxlen, dtype):
    length = len(inputs)
    output = np.zeros((length, maxlen), dtype=dtype)
    for idx, input in enumerate(inputs):
        if len(input) > maxlen:
            output[idx, :] = input[:maxlen]
        else:
            output[idx, :len(input)] = input[:]
    return output


def train(model, device, epoch, train_dataloader, dev_dataloader, optimzer, n_gpu):
    train_loss = []
    for _ in trange(epoch):
        model.train()
        tr_loss = 0
        num_examples, num_step = 0, 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, labels = batch
            optimzer.zero_grad()
            loss = model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=labels)
            if n_gpu > 1:
                loss = loss.mean()
            train_loss.append(loss.item())
            loss.backward()
            optimzer.step()

            tr_loss += loss.item()
            num_examples += input_ids.size(0)
            num_step += 1
        print("train loss : %f" % (tr_loss/num_step))


        model.eval()
        num_samples = 0
        correct = 0
        for step, batch in enumerate(dev_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, labels = batch
            with torch.no_grad():
                logits = model(input_ids, token_type_ids=None, attention_mask=input_mask)
            logits = logits.detach().cpu()
            labels = labels.cpu()
            pre = torch.max(logits, -1)[1]
            correct += torch.sum(pre.eq(labels)).item()
            num_samples += input_ids.size(0)
        print("validation accuracy: %f, (%d/%d)" % (correct/num_samples, correct, num_samples))
    plt.figure(figsize=(15, 8))
    plt.title("Training loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(train_loss)
    plt.show()
    plt.savefig("loss")





def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    # print(torch.cuda.get_device_name(0))
    datadir = '../cola_public/raw/'
    filename = '../cndata/cntext.json'
    classes = ['C31-Enviornment', 'C32-Agriculture', 'C34-Economy', 'C38-Politics', 'C39-Sports']
    # loadcn(filename, classes)
    MAX_LEN = 128
    batch_size = 32
    lr = 2e-5
    epoch = 4
    train_dataloader, dev_dataloader = loadcn(filename, classes, MAX_LEN, batch_size)
    model = BertForSequenceClassification.from_pretrained("../bert-base-chinese", num_labels=len(classes))
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    no_decay = ['bias', 'gamma', 'beta']
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0},
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr, warmup=0.1)
    train(model, device, epoch, train_dataloader, dev_dataloader, optimizer, n_gpu)
    torch.save(model.state_dict(), "../save/bert_cn%d" % epoch)


if __name__ == '__main__':
    main()