import csv
import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from matplotlib.pyplot import imshow
from plotly.utils import numpy
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms

CLASSES = 16
CUDA_ENABLED = torch.cuda.is_available()

stored_model = ""


def create_net(stored_model="", layers_to_freeze=4):
    if stored_model:
        model = torchvision.models.resnext101_32x8d(pretrained=False)
        fc_layer_inputs = model.fc.in_features
        model.fc = nn.Linear(fc_layer_inputs, CLASSES)
        model.load_state_dict(torch.load(stored_model, map_location='cpu'))
    else:
        model = torchvision.models.resnext101_32x8d(pretrained=True)
        fc_layer_inputs = model.fc.in_features
        model.fc = nn.Linear(fc_layer_inputs, CLASSES)

    actual_layer = 0
    for name, child in model.named_parameters():
        if "0.conv1" in name:
            actual_layer += 1
        child.requires_grad = layers_to_freeze < actual_layer

    model.fc.requires_grad = True

    if CUDA_ENABLED:
        model.cuda()

    return model


def train_model(model,
                dataset_data,
                dataset_folder,
                store_path,
                seed=None,
                epochs=30,
                batch_size=15):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if seed is not None:
        random.seed(seed)

    train_data, test_data = split_data(dataset_data)

    data_transform = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomResizedCrop(297),
            transforms.RandomHorizontalFlip(),
            transforms.RandomPerspective(),
            transforms.ColorJitter(),
            transforms.RandomRotation(9),
            #transforms.RandomVerticalFlip()
        ]),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    for e in range(1, epochs + 1):
        random.shuffle(train_data)
        print(f"Epoch {e}")
        ac = train_step(batch_size + (e-1) * 5, criterion, data_transform, dataset_folder, exp_lr_scheduler, model, optimizer,
                   train_data)

        torch.save(model.state_dict(), f"{store_path}/state_epoch_{e}({int(ac*100)}).mdl")

        if e % 5:
            print(f"Test step (epoch {e})")
            #test_step(batch_size, data_transform, dataset_folder, model, test_data)


def train_step(batch_size, criterion, data_transform, dataset_folder, exp_lr_scheduler, model, optimizer, train_data):
    # exp_lr_scheduler.step()
    model.train(True)
    epoch_score = 0
    epoch_loss = 0
    total_elements = 0
    for batch_index in range(0, len(train_data), batch_size):
        batch = train_data[batch_index: batch_index + batch_size]

        input = []
        tags = []
        for img_name, _tag in batch:
            img = Image.open(f'{dataset_folder}/{img_name}.jpg').convert('RGB')
            try:
                input.append(data_transform(img).numpy())
            except:
                print(f"Wrong image {img_name}")
                continue
            tags.append(int(_tag))

            if tags[-1] == 2:
                continue

            input.append(data_transform(img).numpy())
            tags.append(int(_tag))

        # print(f"{batch_index} - [{img_name}|{TAGS_TRANSLATION[_tag]}] ({input[-1].shape})")

        input = numpy.array(input).astype(numpy.float32)
        tags = numpy.array(tags).astype(numpy.long)

        input = Variable(torch.from_numpy(input), requires_grad=True)
        tags = Variable(torch.from_numpy(tags), requires_grad=False)

        if CUDA_ENABLED:
            input = input.cuda()
            tags = tags.cuda()

        optimizer.zero_grad()

        output = model(input)
        loss = criterion(output, tags)

        loss.backward()
        optimizer.step()

        _, preds = torch.max(output.data, 1)
        true_tags = tags.data

        batch_ac = 0
        for i in range(len(true_tags)):
            batch_ac += 1 if preds[i] == true_tags[i] else 0
        batch_loss = loss.item()

        elements = len(tags)
        total_elements += elements
        print(f"Batch {batch_index // batch_size} - Ac: {batch_ac / elements} Loss: {batch_loss / elements}")
        print(f"{preds.cpu().numpy()}\n{true_tags.cpu().numpy()}")

        epoch_score += batch_ac
        epoch_loss += batch_loss

    print(f"Epoch score {epoch_score / total_elements * 100}%")
    print(f"Epoch loss  {epoch_loss / total_elements * 100}%")
    return epoch_score / total_elements * 100

def test_step(batch_size, data_transform, dataset_folder, model, test_data):
    model.train(False)
    test_score = 0
    for batch_index in range(0, len(test_data), batch_size):
        batch = test_data[batch_index: batch_index + batch_size]

        input = []
        tags = []
        for img_name, _tag in batch:
            img = Image.open(f'{dataset_folder}/{img_name}.jpg').convert('RGB')
            try:
                input.append(data_transform(img).numpy())
            except:
                print(f"Wrong image {img_name}")
                continue
            tags.append(int(_tag))

        input = numpy.array(input).astype(numpy.float32)
        tags = numpy.array(tags).astype(numpy.long)

        input = Variable(torch.from_numpy(input), requires_grad=True)
        tags = Variable(torch.from_numpy(tags), requires_grad=False)

        if CUDA_ENABLED:
            input = input.cuda()
            tags = tags.cuda()

        output = model(input)

        _, preds = torch.max(output.data, 1)
        _, true_tags = torch.max(tags.data, 0)

        test_score += torch.sum(preds == true_tags)
    print(f"Test score {test_score / len(test_data) * 100}%")


def split_data(dataset_data):
    whole_data = []
    with open(dataset_data) as data:
        data.readline()
        for _, file, class_number in csv.reader(data):
            whole_data.append((file, class_number))
    random.shuffle(whole_data)
    #train_size = len(whole_data) // 10
    test_data, train_data = [], whole_data
    return train_data, test_data


def run_training(data_tags, dataset_folder, store_path, model_path="",frozen_layers=4):
    model = create_net(model_path,frozen_layers)
    train_model(model=model,
                dataset_data=data_tags,
                dataset_folder=dataset_folder,
                store_path=store_path)
    torch.save(model.state_dict(), f"{store_path}/trained.mdl")


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

def generate_response(model_path, data_folder,csv_out):
    model = create_net(model_path)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    images = [f"{i}.jpg" for i in range(11100) if os.path.isfile(f"{data_folder}/{i}.jpg")]

    with open(csv_out,"w") as out:
        writer = csv.writer(out)
    
    for i in range(0,11100,35):
        try:
            batch_images = [transform(Image.open(f'{data_folder}/{img_name}').convert('RGB')).numpy() for img_name in images[i:i+35]]
        except IOError:
            continue
            
        if not batch_images:
            break
        
        array = numpy.array(batch_images).astype(numpy.float32)
        input_data = Variable(torch.from_numpy(array))

        if CUDA_ENABLED:
            input_data = input_data.cuda()

        outs = model(input_data)
        
        image_names = [ img_name.split(".")[0] for img_name in images[i:i+35] ]
        
        _, preds = torch.max(outs.data, 1)

        rows = []
        for i in range(len(image_names)):
            row = image_names[i],int(preds[i])
            print(row)
            rows.append(row)
            
        with open(csv_out,"a") as out:
            writer = csv.writer(out)
            writer.writerows(rows)
