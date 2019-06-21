import csv
import os
import random

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from matplotlib.pyplot import imshow
from plotly.utils import numpy
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

CLASSES = 16
CUDA_ENABLED = torch.cuda.is_available()

stored_model = ""


def create_net(stored_model="", layers_to_freeze=4):
    if stored_model:
        model = torchvision.models.resnet50(pretrained=False)
        model.load_state_dict(torch.load(stored_model, map_location='cpu'))
    else:
        model = torchvision.models.resnet50(pretrained=True)

    actual_layer = 0
    for name, child in model.named_parameters():
        if "0.conv1" in name:
            actual_layer += 1
        child.requires_grad = layers_to_freeze < actual_layer

    if not stored_model:
        fc_layer_inputs = model.fc.in_features
        model.fc = nn.Linear(fc_layer_inputs, CLASSES)

    if CUDA_ENABLED:
        model.cuda()

    return model


def train_model(model,
                dataset_data,
                dataset_folder,
                seed=None,
                epochs=30,
                batch_size=25):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if seed is not None:
        random.seed(seed)

    train_data, test_data = split_data(dataset_data)

    data_transform = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomPerspective(),
            transforms.RandomRotation(90),
            transforms.RandomVerticalFlip()
        ]),
        torchvision.transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    for e in range(1,epochs+1):

        print(f"Epoch {e}")
        train_step(batch_size, criterion, data_transform, dataset_folder, exp_lr_scheduler, model, optimizer,
                   train_data)

        if e % 5:
            print(f"Test step (epoch {e})")
            test_step(batch_size, data_transform, dataset_folder, model, test_data)


def train_step(batch_size, criterion, data_transform, dataset_folder, exp_lr_scheduler, model, optimizer, train_data):
    exp_lr_scheduler.step()
    model.train(True)
    epoch_score = 0
    for batch_index in range(0, len(train_data), batch_size):
        batch = train_data[batch_index: batch_index + batch_size]

        input = []
        tags = []
        for img_name, _tag in batch:
            img = Image.open(f'{dataset_folder}/{img_name}.jpg')
            input.append(data_transform(img).numpy())
            tag = [0] * CLASSES
            tag[int(_tag)] = 1
            tags.append(int(_tag))

        input = numpy.array(input).astype(numpy.float32)
        tags = numpy.array(tags).astype(numpy.float32)

        input = Variable(torch.from_numpy(input), requires_grad=True)
        tags = Variable(torch.from_numpy(tags), requires_grad=True)

        if CUDA_ENABLED:
            input.cuda()
            tags.cuda()

        output = model(input)
        loss = criterion(output, tags)

        loss.backward()
        optimizer.step()

        _, preds = torch.max(output.data, 1)
        _, true_tags = torch.max(tags.data, 1)

        epoch_score += torch.sum(preds == true_tags)
    print(f"Epoch score {epoch_score / len(train_data) * 100}%")


def test_step(batch_size, data_transform, dataset_folder, model, test_data):
    model.train(False)
    test_score = 0
    for batch_index in range(0, len(test_data), batch_size):
        batch = test_data[batch_index: batch_index + batch_size]

        input = []
        tags = []
        for img_name, _tag in batch:
            img = Image.open(f'{dataset_folder}/{img_name}.jpg')
            input.append(data_transform(img).numpy())
            tag = [0] * CLASSES
            tag[int(_tag)] = 1
            tags.append(int(_tag))

        input = numpy.array(input).astype(numpy.float32)
        tags = numpy.array(tags).astype(numpy.float32)

        input = Variable(torch.from_numpy(input), requires_grad=True)
        tags = Variable(torch.from_numpy(tags), requires_grad=True)

        if CUDA_ENABLED:
            input.cuda()
            tags.cuda()

        output = model(input)

        _, preds = torch.max(output.data, 1)
        _, true_tags = torch.max(tags.data, 1)

        test_score += torch.sum(preds == true_tags)
    print(f"Test score {test_score / len(test_data) * 100}%")


def split_data(dataset_data):
    whole_data = []
    with open(dataset_data) as data:
        data.readline()
        for _, file, class_number in csv.reader(data):
            whole_data.append((file, class_number))
    random.shuffle(whole_data)
    train_size = len(whole_data) // 90
    train_data, test_data = whole_data[:train_size], whole_data[train_size:]
    return train_data, test_data


def run_training(data_tags, dataset_folder, store_path, model_path=""):
    model = create_net(model_path)
    train_model(model=model,
                dataset_data=data_tags,
                dataset_folder=dataset_folder)
    torch.save(model.state_dict(), store_path)


def imgshow(images, classes):
    import matplotlib.pyplot as plt

    inp = images.numpy().transpose((1, 2, 0))
    mean = numpy.array([0.485, 0.456, 0.406])
    std = numpy.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = numpy.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(classes)


TAGS_TRANSLATION = [
    "Bar",
    "Bathroom",
    "Bedroom",
    "Breakfast",
    "City View",
    "Dining",
    "Hotel Front",
    "Hotel Exterior View",
    "Hotel Interior View",
    "Kitchen",
    "Living Room",
    "Lobby",
    "Natural View",
    "Pool",
    "Recreation",
    "Sports"
]


def test_model(model_path, data_folder):
    images = os.listdir(data_folder)
    model = create_net(model_path)

    random.shuffle(images)
    images = images[:6]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    images = [transform(Image.open(f'{data_folder}/{img_name}')).numpy() for img_name in images]

    array = numpy.array(images).astype(numpy.float32)
    print(Variable(torch.from_numpy(array)).data)
    outs = model(Variable(torch.from_numpy(array)))


    _, preds = torch.max(outs.data, 1)

    images_so_far = 0
    fig = plt.figure()

    for j in range(len(images)):
        images_so_far += 1
        ax = plt.subplot(3, 2, images_so_far)
        ax.axis('off')
        ax.set_title('predicted: {}'.format(TAGS_TRANSLATION[preds[j]]))
        imshow(images[j].transpose((1, 2, 0)))
    plt.show()