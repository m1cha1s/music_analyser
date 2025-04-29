#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import argparse as ap
from CNN import *

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
split = 0.9
batch_size = 1

genres = np.load("genres.npy").astype("float32")
samples = np.load("samples.npy").astype("float32")

np.random.shuffle(genres)
np.random.shuffle(samples)

n = int(len(genres)*split)

train_input = samples[:n]
val_input = samples[n:]

train_output = genres[:n]
val_output = genres[n:]

assert(len(train_input)+len(val_input) == len(samples))
assert(len(train_output)+len(val_output) == len(genres))

@torch.no_grad
def get_batch(*, train = False):
    input_data = train_input if train else val_input
    output_data = train_output if train else val_output
    ix = torch.randint(len(input_data), (batch_size,))
    iy = torch.randint(len(output_data), (batch_size,))
    x = torch.stack([torch.from_numpy(input_data[x]) for x in ix]).to(device)
    y = torch.stack([torch.from_numpy(output_data[y]) for y in iy]).to(device)

    x.unsqueeze_(1)
    #y.unsqueeze_(1)

    return x, y

eval_iters = 200

@torch.no_grad
def accuracy_function(output, target):
    # shape: (batch, labels)
    f_output = output.cpu()
    f_target = target.cpu()

    output_res = f_output.detach().numpy()
    target_res = f_target.detach().numpy()
    predicted_index = np.argmax(output_res, axis=1)

    target_index = np.argmax(target_res, axis=1)

    # counter
    correct = np.sum(predicted_index == target_index)
    accuracy = correct / (output.shape[0])
    return accuracy

@torch.no_grad
def estimate_loss(model):
    out = {}
    model.eval()
    for split in [True, False]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train=split)
            props, loss = model(X, Y)
            losses[k] = loss.item()
        out['train' if split else 'val'] = losses.mean()
    model.train()
    return out

#test with dummy input
if __name__ == "__main__":
    parser = ap.ArgumentParser(prog="Gym", description="Trains the CNN")
    parser.add_argument("-m", "--model")
    parser.add_argument("-lr", "--learning_rate", default=0.0001)
    parser.add_argument("-i", "--iters", default=100000)
    
    args = parser.parse_args()

    num_classes = 10
    learning_rate = float(args.learning_rate)
    eval_interval = 10
    max_iters = args.iters

    print(f"Training on {device}")

    m = None

    if args.model:
        m = torch.load(args.model, weights_only=False)
    else:
        m = CNNMusicRecogniser(num_classes=num_classes).to(device)
    optim = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    m.train()
    try:
        for iter in range(max_iters):
            xb, yb = get_batch(train=True)
            optim.zero_grad()
            props, loss = m(xb, yb)
            if iter % eval_interval == 0:
                losses = estimate_loss(m)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, accuracy {accuracy_function(props, yb)}")

            loss.backward()
            optim.step()
            
        m.eval()
        torch.save(m, f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.model")

        for _ in range(10):
            x, y = get_batch(train=True)
            output = m(x)


            print(f"Output: {output}")
            print(f"Expected output: {y}")

    #        plt.bar(range(10), output[0].cpu().detach().numpy()[0])
    #        plt.show()
    except KeyboardInterrupt:
        m.eval()
        torch.save(m, f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.model")
        
