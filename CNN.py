import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNNMusicRecogniser(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNMusicRecogniser, self).__init__()
        #1st convolutional block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #2nd convolutional block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        #3rd conv block
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=124, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        #128x128 input = map size 16x16
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, y=None):
        # print(x[0][0].shape, x)
        #conv block 1
        x = self.conv1(x) #applies first convolution
        print(x.shape)
        x = F.relu(x) # ReLU function increases the complexity of the neural network by introducing non-linearity, which allows the network to learn more complex representations of the data. The ReLU function is defined as f(x) = max(0, x), which sets all negative values to zero.
        x = self.pool(x) # reduces size using max pooling, (if we have 4x4 block of values, it makes 2x2 block where for every 2x2 array from 4x4 block takes the highest value)
        x = self.bn1(x) #normalizes the output

        #conv block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.bn2(x)

        #conv block 3
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.bn3(x)

        #flattenign the tensor
        x = x.view(x.size(0), -1)

        #connect the layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        if y is None:
            loss = None
        else:
            loss = F.cross_entropy(x, y)

        return x, loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'
split = 0.9
batch_size = 4

genres = np.load("genres.npy").astype("float32")
samples = np.load("samples.npy").astype("float32")

n = int(len(genres)*split)

train_input = samples[:n]
val_input = samples[n:]

train_output = genres[:n]
val_output = genres[n:]

assert(len(train_input)+len(val_input) == len(samples))
assert(len(train_output)+len(val_output) == len(genres))

def get_batch(*, train = False):
    input_data = train_input if train else val_input
    output_data = train_output if train else val_output
    ix = torch.randint(len(input_data), (batch_size,))
    iy = torch.randint(len(output_data), (batch_size,))
    x = torch.stack([torch.from_numpy(input_data[x]) for x in ix]).to(device)
    y = torch.stack([torch.from_numpy(output_data[y]) for y in iy]).to(device)

    return x, y

eval_iters = 200

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

def train(num_classes = 10,
          learning_rate = 0.01,
          eval_interval = 10,
          max_iters = 1000):

    m = CNNMusicRecogniser(num_classes=num_classes).to(device)
    optim = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    m.train()

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(m)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch(train=True)

        props, loss = m(xb, yb)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

    m.eval()

    torch.save(m, f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.model")

#test with dummy input
if __name__ == "__main__":
    train(num_classes = 10)

#dummy input tensor [batch_size, channels, height, width]
dummy_input = torch.randn(1, 1, 128, 128)
output = model(dummy_input)
print("Output shape:", output.shape) # [1, num_classes]
