import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sb
from sys import argv
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

m = torch.load(argv[1], weights_only=False, map_location=torch.device(device))

genres = torch.from_numpy(np.load("genres.npy").astype("float32")).to(device)
samples = torch.from_numpy(np.load("samples.npy").astype("float32")).to(device)

y_pred = []
y_true = []


for i in range(len(samples)):
    x = torch.stack([samples[i]])

    x.unsqueeze_(1)

    y, _ = m(x)

    output = (torch.max(torch.exp(y), 1)[1]).data.cpu().numpy()
    y_pred.extend(output)

    labels = (torch.max(torch.exp(torch.stack([genres[i]])), 1)[1]).cpu().numpy()
    y_true.extend(labels)

classes = ("blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock")

cf_matrix = confusion_matrix(y_true, y_pred)
df_cm     = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=classes, columns=classes)

plt.figure(figsize=(12,7))

sb.heatmap(df_cm, annot=True)

plt.savefig("cm.png")