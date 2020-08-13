# Python's native libraries
import os

# deep learning/vision libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import cv2 as cv  # OpenCV

# numeric and plotting libraries
import numpy as np

import copy
import random


def generate_content_from_noise(model, criterion, photo, activation, num_steps=1000):
    photo = photo.transpose(2, 0, 1)
    white_noise_img = np.random.uniform(-90., 90., photo.shape).astype(np.float32)
    noise_img_tensor = torch.from_numpy(white_noise_img)

    IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
    IMAGENET_MEAN_STD_NEUTRAL = [1, 1, 1]

    transformer = transforms.Compose([
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(IMAGENET_MEAN_255, IMAGENET_MEAN_STD_NEUTRAL)
    ])

    noise_img = torch.autograd.Variable(noise_img_tensor, requires_grad=True)

    photo = transformer(torch.from_numpy(photo)).float()

    model.eval()
    count = 0

    _ = model(image_loader_tensor(photo))
    original_activation = activation['conv4_2']

    def closure():
        nonlocal count

        optimizer.zero_grad()

        _ = model(image_loader_tensor(noise_img))
        noisy_activation = activation['conv4_2']

        loss = criterion(noisy_activation, original_activation)

        loss.backward()

        running_loss = loss.item()

        epoch_loss = running_loss

        count += 1

        if (count + 0) % 20 == 0:
            print(f' Loss: {epoch_loss:.4f}')

        return loss

    optimizer = optim.LBFGS([noise_img.requires_grad_()], max_iter=num_steps, line_search_fn='strong_wolfe')
    optimizer.step(closure)

    return noise_img


def generate_style_from_noise(model, criterion, photo, activation, num_steps=1000):
    photo = photo.transpose(2, 0, 1)

    white_noise_img = np.random.uniform(-90., 90., photo.shape).astype(np.float32)
    noise_img_tensor = torch.from_numpy(white_noise_img)

    IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
    IMAGENET_MEAN_STD_NEUTRAL = [1, 1, 1]

    transformer = transforms.Compose([
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(IMAGENET_MEAN_255, IMAGENET_MEAN_STD_NEUTRAL)
    ])

    noise_img = torch.autograd.Variable(noise_img_tensor, requires_grad=True)

    photo = transformer(torch.from_numpy(photo)).float()

    model.eval()
    count = 0

    layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

    _ = model(image_loader_tensor(photo))
    original_activations = {}

    for i in layers:
        original_activations[i] = copy.deepcopy(activation[i])

    def closure():
        nonlocal count

        optimizer.zero_grad()

        loss = 0

        _ = model(image_loader_tensor(noise_img))

        for i in layers:
            noisy_activaton = activation[i]
            original_activation = original_activations[i]

            loss += criterion(noisy_activaton, original_activation)

        loss.backward()

        running_loss = loss.item()

        epoch_loss = running_loss

        if (count + 0) % 20 == 0:
            print(f'Iteration: {count:03}, Loss: {epoch_loss:.4f}')

        count += 1

        return loss

    optimizer = optim.LBFGS((noise_img,), max_iter=num_steps, line_search_fn='strong_wolfe')
    optimizer.step(closure)

    return noise_img


def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :-1] - y[:, :, 1:])) + \
           torch.sum(torch.abs(y[:, :-1, :] - y[:, 1:, :]))


def neural_style_transfer(model, criterion_content, criterion_style, photo_content, photo_style, activation, num_steps=1000):
    photo_style = cv.resize(photo_style, (photo_content.shape[1], photo_content.shape[0]), interpolation=cv.INTER_CUBIC)
    photo_content = photo_content.transpose(2, 0, 1)
    photo_style = photo_style.transpose(2, 0, 1)

    content_copy_img = copy.deepcopy(photo_content)
    noise_img_tensor = torch.from_numpy(content_copy_img)

    IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
    IMAGENET_MEAN_STD_NEUTRAL = [1, 1, 1]

    transformer = transforms.Compose([
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(IMAGENET_MEAN_255, IMAGENET_MEAN_STD_NEUTRAL)
    ])

    noise_img_tensor = transformer(noise_img_tensor).float()
    noise_img = torch.autograd.Variable(noise_img_tensor, requires_grad=True)

    photo_content = transformer(torch.from_numpy(photo_content)).float()
    photo_style = transformer(torch.from_numpy(photo_style)).float()

    model.eval()

    count = 0

    layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

    _ = model(image_loader_tensor(photo_content))
    original_activation_content = activation['conv4_2']

    _ = model(image_loader_tensor(photo_style))
    original_activations_style = {}
    for i in layers:
        original_activations_style[i] = copy.deepcopy(activation[i])

    def closure():
        nonlocal count

        optimizer.zero_grad()

        style_loss_var = 0

        _ = model(image_loader_tensor(noise_img))

        for i in layers:
            noisy_activaton = activation[i]
            original_activation = original_activations_style[i]

            style_loss_var += criterion_style(noisy_activaton, original_activation)

        content_loss_var = criterion_content(activation['conv4_2'], original_activation_content)

        alpha = 1e3
        beta = 3e9
        gamma = 1e3
        tv_loss = total_variation(noise_img).to(device)
        total_loss = alpha * content_loss_var + beta * style_loss_var + gamma * tv_loss

        total_loss.backward()

        running_loss = total_loss.item()

        epoch_loss = running_loss

        if (count + 0) % 20 == 0:
            print('Content loss: ', content_loss_var)
            print('Style loss: ', style_loss_var)
            print('Total variation loss: ', tv_loss)
            print(f'Iteration: {count:03}, Loss: {epoch_loss:.4f}')

        count += 1

        return total_loss

    optimizer = optim.LBFGS((noise_img,), max_iter=num_steps, line_search_fn='strong_wolfe')
    optimizer.step(closure)

    return noise_img


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output

    return hook


def content_loss(noisy_img, original_img):
    loss = nn.MSELoss(reduction='sum')
    return loss(noisy_img.squeeze(axis=0), original_img.squeeze(axis=0)) / 2


def get_gm(layer):
    a, b, c, d = layer.size()
    features = layer.view(a * b, c * d)
    gram = torch.matmul(features, features.t())

    return gram


def style_loss(noisy_layer, original_layer):
    A = get_gm(original_layer)
    G = get_gm(noisy_layer)

    E = nn.MSELoss(reduction='sum')

    return 0.2 * E(A, G) / ((2 * noisy_layer.shape[1] * noisy_layer.shape[2] * noisy_layer.shape[3]) ** 2)


def unnormalize(tensor, mean, std):
    """
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
        Tensor: Normalized image.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
        # The normalize code -> t.sub_(m).div_(s)
    return tensor


def image_loader(image):
    return torch.from_numpy(image).unsqueeze(0).to(device, torch.float)


def image_loader_tensor(image_tensor):
    return image_tensor.unsqueeze(0).to(device, torch.float)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torchvision.models.vgg19(pretrained=True)
    model.features[1].register_forward_hook(get_activation('conv1_1'))
    model.features[6].register_forward_hook(get_activation('conv2_1'))
    model.features[11].register_forward_hook(get_activation('conv3_1'))
    model.features[20].register_forward_hook(get_activation('conv4_1'))
    model.features[22].register_forward_hook(get_activation('conv4_2'))
    model.features[29].register_forward_hook(get_activation('conv5_1'))

    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device)

    style_folder_name = os.getcwd() + "/data/style-images/"
    content_folder_name = os.getcwd() + "/data/content-images/"

    style_content = []

    for img_style_name in os.listdir(style_folder_name):
        for img_content_name in os.listdir(content_folder_name):
            style_content.append({img_style_name, img_content_name})

    random.shuffle(style_content)

    for img_style_name, img_content_name in style_content:
        print(img_style_name.split('.')[0], img_content_name.split('.')[0])

        flag = True
        c = 0
        failedFlag = False

        while flag:
            try:
                img_style = cv.imread(style_folder_name + img_style_name)
                dst_style = cv.cvtColor(img_style, cv.COLOR_BGR2RGB)
                dst_style = dst_style / 255

                img_content = cv.imread(content_folder_name + img_content_name)
                dst = img_content

                if img_content.shape[0] > 1200 and img_content.shape[1] > 1200:
                    if img_content.shape[0] > img_content.shape[1]:
                        ratio = img_content.shape[0] // 1000
                        dst = cv.resize(img_content, (1000, img_content.shape[1] // ratio))
                    else:
                        ratio = img_content.shape[1] // 1000
                        dst = cv.resize(img_content, (img_content.shape[0] // ratio, 1000))
                    dst = cv.resize(img_content, (1000, 650))
                elif img_content.shape[0] > 1200:
                    ratio = img_content.shape[0] // 1000
                    dst = cv.resize(img_content, (1000, img_content.shape[1] // ratio))
                elif img_content.shape[1] > 1200:
                    ratio = img_content.shape[1] // 1000
                    dst = cv.resize(img_content, (img_content.shape[0] // ratio, 1000))

                dst_content = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
                dst_content = dst_content / 255
                flag = False
            except cv.error:
                c += 1
                if c > 100:
                    failedFlag = True
                    break

        if failedFlag:
            continue

        base = neural_style_transfer(model, content_loss, style_loss, dst_content, dst_style, activation).detach()
        no_unorm = base.numpy()

        with_unorm = unnormalize(base, (123.675, 116.28, 103.53), (1, 1, 1)).numpy()
        with_unorm = with_unorm.transpose(1, 2, 0)

        with_unorm = cv.cvtColor(with_unorm, cv.COLOR_RGB2BGR)
        with_unorm = with_unorm.astype(int)

        out_name = img_style_name.split('.')[0] + '_' + img_content_name.split('.')[0]
        cv.imwrite("data/results/" + out_name + ".png", with_unorm.astype(int))
