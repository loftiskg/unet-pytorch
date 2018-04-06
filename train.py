import model
import data
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import os, shutil
from PIL import Image
import numpy as np
from sklearn.preprocessing import minmax_scale

PREDICTION_PATH = './pred'

cuda = True
model = model.UNet(1)
if cuda:
    model = model.cuda()

transform = transforms.Compose([data.Normalize(), data.ToTensor()])
dat = data.SampleDataDriver(transform=transform)
dataloader = DataLoader(dat, batch_size=1, shuffle=False, num_workers=1)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 3

def inference(model):
    if os.path.exists(PREDICTION_PATH):
        shutil.rmtree(PREDICTION_PATH)
    os.makedirs(PREDICTION_PATH)

    for i, sample in enumerate(dataloader):
        input = Variable(sample['input']).float()
        if cuda:
            input = input.cuda()
        y_pred = model(input)
        img = tensor_to_PIL(y_pred)
        img.save(os.path.join(PREDICTION_PATH, 'prediction_{:03d}.png'.format(i)))

def tensor_to_PIL(tensor):
    if tensor.is_cuda:
        tensor = tensor.data.cpu().numpy()
    else:
        tensor = tensor.data.numpy()
    tensor = np.squeeze(tensor)

    tensor = normalize(tensor)
    tensor = np.uint8(tensor * 255)
    img = Image.fromarray(tensor)
    return img

def train(epoch):
    model.train()
    for batch_idx, sample in enumerate(dataloader):
        input = Variable(sample['input']).float()
        target = Variable(sample['target']).float()
        if cuda:
            input = input.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model(input)
        #output_2 = output.permute(2, 3, 0, 1).contiguous().view(-1, 1)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.data[0]))

def normalize(array):
    array_shape = array.shape

    array = array.flatten()
    array_min = np.amin(array)
    array_max = np.amax(array)

    array_normalized = (array - array_min)/(array_max-array_min)

    array_normalized = array_normalized.reshape(array_shape)

    return array_normalized


def main():
    for epoch in range(epochs):
        train(epoch)
    inference(model)
    print("Complete")


if __name__ == '__main__':
    main()
