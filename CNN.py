from tinygrad import Tensor, nn, Device
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class CNNMusicRecogniser:
    def __init__(self, num_classes=10):

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm(128)

        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = x.relu6() 
        x = x.max_pool2d((2,2), stride=2)
        x = self.bn1(x)

        x = self.conv2(x)
        x = x.relu6()
        x = x.max_pool2d((2,2), stride=2)
        x = self.bn2(x)

        x = self.conv3(x)
        x = x.relu6()
        x = x.max_pool2d((2,2), stride=2)
        x = self.bn3(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = x.relu6()
        x = x.dropout(p=0.5)
        x = self.fc2(x)

        x = x.softmax()

        return x 

device = Device.DEFAULT
split = 0.9
batch_size = 1

genres = Tensor(np.load("genres.npy").astype("float32"), device)
samples = Tensor(np.load("samples.npy").astype("float32"), device)

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
    ix = Tensor.randint(batch_size, high=input_data.shape[0])
    iy = Tensor.randint(batch_size, high=output_data.shape[0])
    x = Tensor.stack(*[input_data[x] for x in ix], dim=0)
    y = Tensor.stack(*[output_data[y] for y in iy], dim=0)

    x = x.unsqueeze(0)

    return x, y

eval_iters = 200

# @torch.no_grad
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
          learning_rate = 0.001,
          eval_interval = 10,
          max_iters = 2000):
    print(f"Training on {device}")

    m = CNNMusicRecogniser(num_classes=num_classes)
    optim = nn.optim.AdamW(nn.state.get_parameters(m), lr=learning_rate)

    def step():
        Tensor.training = True
        xb, yb = get_batch(train=True)

        optim.zero_grad()
        loss = m(xb).cross_entropy(yb).backward()
        optim.step()

#    torch.save(m, f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.model")
    import timeit

    print(timeit.repeat(step, repeat=5, number=1))

    return m

#test with dummy input
if __name__ == "__main__":
    m = train(num_classes = 10)

    #dummy input tensor [batch_size, channels, height, width]
    # dummy_input = torch.randn(1, 1, 128, 128).to(device)
    for _ in range(1):
        x, y = get_batch(train=False)
        output = m(x)
