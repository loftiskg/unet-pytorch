import model
import data
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import os, shutil
from PIL import Image
import numpy as np

### Settings ###
PREDICTION_PATH = './pred'  # output directory of predicted segmentations
cuda = True    # Set to true if you want to use GPUs
num_classes = 1 # number of classes in target
epochs = 1
save_weights = True  # save model weights?
learning_rate = .0001
batch_size = 1
#############################





### Initialize Model, Data, and Optimizers
model = model.UNet(num_classes) # initialize model

if cuda:
    model = model.cuda() # converts model params into cuda tensors

# Create DataLoader
transform = transforms.Compose([data.Normalize(), data.ToTensor()])  # Define transformations to be done on data
dat = data.SampleDataDriver(transform=transform)
dataloader = DataLoader(dat, batch_size=1, shuffle=False, num_workers=1)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
##############################################################################

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

# takes a pytorch tensor and converts to a 0-255 PIL image
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
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.data[0]))

# Normalizes a 2D array for all values to be between 0 and 1
def normalize(array):
    array_shape = array.shape

    array = array.flatten()
    array_min = np.amin(array)
    array_max = np.amax(array)

    array_normalized = (array - array_min)/(array_max-array_min)
    array_normalized = array_normalized.reshape(array_shape)

    return array_normalized

# Saves model parameters to file
def save_model(model,path='./model/model_weigths.pt'):
    print("Saving Model a {}".format(path))
    torch.save(model.state_dict(), path)
    print('Model Saved')


def main():
    # Start Training
    print("Starting Training")
    for epoch in range(epochs):
        train(epoch)
    if save_weights:
        if not os.path.exists('./model'):
            os.makedirs('./model')
        save_model(model,path = './model/model.weights.pt')
    inference(model)
    print("Complete")


if __name__ == '__main__':
    main()
