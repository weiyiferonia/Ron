import os
import torch
import gzip
import torch.utils.data as data
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
import visdom
from scipy import ndimage
from mnist_net import *
# Hyper-parameters
image_size = 28
num_channels =1
pixel_depth = 255
input_size = 10
num_classes = 10
num_epochs = 500
batch_size = 400
learning_rate = 1e-3

class Loader_data(data.Dataset):
    def __init__(self,root,image_fn,label_fn,num_images,isTest = True):
        image_fn = os.path.join(root,image_fn)
        label_fn = os.path.join(root, label_fn)
        base_data = extract_data(image_fn, num_images)
        base_labels = extract_labels(label_fn, num_images)
        if root.split('/',-1)[-2] == 'train' and isTest==False:
            expend_datas = expend_training_data(base_data,base_labels)
            base_data = expend_datas[:, :-10]
            base_labels = expend_datas[:, -10:]
        data_base_list = []
        for ind,v in enumerate(base_data):
            base_labels_ = base_labels[ind].tolist().index(max(base_labels[ind]))
            data_base_list.append((v,base_labels_))
        self.data_base_list = data_base_list
    def __getitem__(self, item):
        base , label = self.data_base_list[item]
        return torch.Tensor(base) , torch.from_numpy(np.array([np.int(label)]))

    def __len__(self):
        return len(self.data_base_list)

def expend_training_data(images, labels):
    expanded_images = []
    expanded_labels = []
    j = 0 # counter
    for x, y in zip(images, labels):
        j = j+1
        if j%10000==0:
            print ('expanding data : %03d / %03d' % (j,np.size(images,0)))
        # register original data
        expanded_images.append(x)
        expanded_labels.append(y)
        # get a value for the background
        # zero is the expected value, but median() is used to estimate background's value
        bg_value = np.median(x) # this is regarded as background's value
        image = np.reshape(x, (-1, 28))
        for i in range(4):
            # rotate the image with random degree
            angle = np.random.randint(-15,15,1)
            new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)
            # shift the image with random distance
            shift = np.random.randint(-2, 2, 2)
            new_img_ = ndimage.shift(new_img,shift, cval=bg_value)
            # register new training data
            expanded_images.append(np.reshape(new_img_, 784))
            expanded_labels.append(y)
    # images and labels are concatenated for random-shuffle at each epoch
    # notice that pair of image and label should not be broken
    expanded_train_total_data = np.concatenate((expanded_images, expanded_labels), axis=1)
    np.random.shuffle(expanded_train_total_data)
    return expanded_train_total_data

def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(image_size * image_size * num_images * num_channels)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (pixel_depth / 2.0)) / pixel_depth
        data = data.reshape(num_images, image_size, image_size, num_channels)
        data = np.reshape(data, [num_images, -1])
    return data

# Extract the labels
def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        num_labels_data = len(labels)
        one_hot_encoding = np.zeros((num_labels_data,num_classes))
        one_hot_encoding[np.arange(num_labels_data),labels] = 1
        one_hot_encoding = np.reshape(one_hot_encoding, [-1, num_classes])
    return one_hot_encoding

# dataset (train and test)
trainPath = "/home/yaorongguo/PycharmProjects/mnist_tensorflow/minst_program/minst_data/train/"
testPath = "/home/yaorongguo/PycharmProjects/mnist_tensorflow/minst_program/minst_data/test/"
train_dataset = Loader_data(root = trainPath,image_fn = 'train-images-idx3-ubyte.gz',label_fn = 'train-labels-idx1-ubyte.gz',num_images=60000)
test_dataset = Loader_data(root = testPath,image_fn = 't10k-images-idx3-ubyte.gz',label_fn = 't10k-labels-idx1-ubyte.gz',num_images=10000)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False)
# model
model = Net()
print model

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)
# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
curr_lr = learning_rate
#use GPU
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
if use_cuda:
    model.cuda()
    criterion.cuda()
numacc = []#count the correct number
# model.load_state_dict(torch.load('/home/yaorongguo/PycharmProjects/mnist_tensorflow/'
#                                                                 'minst_program/models/Accuracy_99.72%.pt'))
isTest = True

if isTest is False:
    # Train the model
    for epoch in range(num_epochs):
        for i, (base, label) in enumerate(train_loader):
            base = base.view([-1,1,28,28])
            # Forward pass
            label = label.squeeze(1)
            base,label = Variable(base.type(dtype)),Variable(label.cuda())
            outputs = model(base)
            loss = criterion(outputs, label)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1)%150 == 0:
                print ('epoch:{} batch:{}/{} loss:{}'.format(epoch+1, i+1 ,len(train_loader),round(loss.data.cpu().numpy()[0],5)))
    # Test the model
        if(epoch+1)% 1 == 0:
            correct = 0
            total = 0
            model.eval()
            for idx,(test_base, test_label) in enumerate (test_loader):
                test_base = test_base.view([-1, 1, 28, 28])
                test_label = test_label.squeeze(1)
                test_base = Variable(test_base.type(dtype))
                outputs = model(test_base)
                _, predicted = torch.max(outputs.data,1)
                total += test_label.size(0)
                # correct += (predicted == test_label).sum()
                correct += (predicted.cpu().numpy() == test_label).sum()
            print("The machine correctly guessed " + str(correct) + "/"+str(total))
            numacc.append(correct)
            # Save the model checkpoint
            torch.save(model.state_dict(), '/home/yaorongguo/PycharmProjects/mnist_tensorflow/minst_program'
                                                                '/models/Accuracy_{}%.pt'.format((float(correct)/100)))
            vis = visdom.Visdom()
            trace = dict(x=len(numacc), y=numacc, mode="lines", type='custom',
                         marker={'color': 'blue', 'symbol': 104, 'size': "10"},
                         )
            layout = dict(title="", xaxis={'title': 'Iterations'}, yaxis={'title': 'AccNum'})
            vis._send({'data': [trace], 'layout': layout, 'win': 'Train error'})
    # Decay learning rate
        if (epoch+1) % 100 == 0 and epoch+1 <= 300:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)
else:
    model.load_state_dict(torch.load('/home/yaorongguo/PycharmProjects/mnist_tensorflow/'
                                                'minst_program/models/Accuracy_99.72%.pt'))
    correct = 0
    total = 0
    model.eval()
    for idx, (test_base, test_label) in enumerate(test_loader):
        test_base = test_base.view([-1, 1, 28, 28])
        test_label = test_label.squeeze(1)
        test_base = Variable(test_base.type(dtype))
        outputs = model(test_base)
        _, predicted = torch.max(outputs.data, 1)
        total += test_label.size(0)
        # correct += (predicted == test_label).sum()
        correct += (predicted.cpu().numpy() == test_label).sum()
    print("The machine correctly guessed " + str(correct) + "/" + str(total))