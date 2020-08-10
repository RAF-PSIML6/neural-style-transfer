# Python's native libraries
import time
import os
import copy
from collections import defaultdict

# deep learning/vision libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import cv2 as cv  # OpenCV

# numeric and plotting libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
import scipy.ndimage

import copy


def train_photo(model, criterion, photo, layer_idx, activation, num_epochs=40):
    #     start_time = time.time()

    photo = photo.transpose(2, 0, 1)
    # noise_img_tensor = torch.rand(photo.shape)
    white_noise_img = np.random.uniform(-90., 90., photo.shape).astype(np.float32)
    noise_img_tensor = torch.from_numpy(white_noise_img)

    IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225])

    IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
    IMAGENET_MEAN_STD_NEUTRAL = [1, 1, 1]

    # loader = transforms.Compose([transforms.Normalize(IMAGENET_MEAN_1, IMAGENET_STD_1)])
    loader = transforms.Compose([
        transforms.Lambda(lambda x: x.mul(255)),
        # transforms.Normalize(IMAGENET_MEAN_1, IMAGENET_STD_1)
        transforms.Normalize(IMAGENET_MEAN_255, IMAGENET_MEAN_STD_NEUTRAL)
    ])

    # noise_img_tensor = loader(noise_img_tensor).float()
    noise_img = torch.autograd.Variable(noise_img_tensor, requires_grad=True)

    photo = loader(torch.from_numpy(photo)).float()
    # print(photo)
    # for epoch in range(num_epochs):
    #    if epoch % 20 == 0:
    #        print(f'Epoch {epoch}/{num_epochs - 1}')
    #        print('-' * 10)

    model.eval()

    count = 0

    _ = model(image_loader_tensor(photo))
    original_activation = activation['conv4_2']

    def closure():
        nonlocal count
        # torch.clamp(noise_img, 0, 1) # 255

        optimizer.zero_grad()

        # with torch.set_grad_enabled(False):

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

    optimizer = optim.LBFGS([noise_img.requires_grad_()], max_iter=350, line_search_fn='strong_wolfe')
    optimizer.step(closure)

    # time_elapsed = time.time() - start_time
    # print(f'Training complete in {(time_elapsed // 60):.0f}m {time_elapsed % 60:.0f}s')

    return noise_img


def train_photo2(model, criterion, photo, layer_idx, activation, num_epochs=40):
    #     start_time = time.time()

    photo = photo.transpose(2, 0, 1)

    white_noise_img = np.random.uniform(-90., 90., photo.shape).astype(np.float32)
    noise_img_tensor = torch.from_numpy(white_noise_img)

    # IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406])
    # IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225])

    IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
    IMAGENET_MEAN_STD_NEUTRAL = [1, 1, 1]

    loader = transforms.Compose([
        transforms.Lambda(lambda x: x.mul(255)),
        #transforms.Normalize(IMAGENET_MEAN_1, IMAGENET_STD_1)
        transforms.Normalize(IMAGENET_MEAN_255, IMAGENET_MEAN_STD_NEUTRAL)
    ])

    # noise_img_tensor = loader(noise_img_tensor).float()
    noise_img = torch.autograd.Variable(noise_img_tensor, requires_grad=True)

    photo = loader(torch.from_numpy(photo)).float()

    # print(photo)
    # for epoch in range(num_epochs):
    #    if epoch % 20 == 0:
    #        print(f'Epoch {epoch}/{num_epochs - 1}')
    #        print('-' * 10)

    model.eval()

    count = 0

    layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

    outputs2 = model(image_loader_tensor(photo))
    original_activations = {}
    for i in layers:
        original_activations[i] = copy.deepcopy(activation[i])

    def closure():
        nonlocal count
        # torch.clamp(noise_img, 0, 1) # 255

        optimizer.zero_grad()

        # with torch.set_grad_enabled(False):

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

    optimizer = optim.LBFGS((noise_img,), max_iter=350, line_search_fn='strong_wolfe')
    optimizer.step(closure)

    # time_elapsed = time.time() - start_time
    # print(f'Training complete in {(time_elapsed // 60):.0f}m {time_elapsed % 60:.0f}s')

    return noise_img

def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :-1] - y[:, :, 1:])) + \
           torch.sum(torch.abs(y[:, :-1, :] - y[:, 1:, :]))

def train_photo3(model, criterion_content, criterion_style, photo_content, photo_style, layer_idx, activation, fm_guides, photo_style2=None):
    #     start_time = time.time()

    photo_style = cv.resize(photo_style, (photo_content.shape[1], photo_content.shape[0]), interpolation=cv.INTER_CUBIC)
    photo_style2 = cv.resize(photo_style2, (photo_content.shape[1], photo_content.shape[0]), interpolation=cv.INTER_CUBIC)
    photo_content = photo_content.transpose(2, 0, 1)
    photo_style = photo_style.transpose(2, 0, 1)
    photo_style2 = photo_style2.transpose(2, 0, 1)

    # white_noise_img = np.random.uniform(-90., 90., photo_style.shape).astype(np.float32)
    # white_noise_img = np.random.uniform(size=photo_style.shape)
    content_copy_img = copy.deepcopy(photo_content)
    noise_img_tensor = torch.from_numpy(content_copy_img)

    # IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406])
    # IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225])

    IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
    IMAGENET_MEAN_STD_NEUTRAL = [1, 1, 1]

    loader = transforms.Compose([
        transforms.Lambda(lambda x: x.mul(255)),
        #transforms.Normalize(IMAGENET_MEAN_1, IMAGENET_STD_1)
        transforms.Normalize(IMAGENET_MEAN_255, IMAGENET_MEAN_STD_NEUTRAL)
    ])

    noise_img_tensor = loader(noise_img_tensor).float()
    noise_img = torch.autograd.Variable(noise_img_tensor, requires_grad=True)

    photo_content = loader(torch.from_numpy(photo_content)).float()
    photo_style = loader(torch.from_numpy(photo_style)).float()
    photo_style2 = loader(torch.from_numpy(photo_style2)).float()

    # print(photo)
    # for epoch in range(num_epochs):
    #    if epoch % 20 == 0:
    #        print(f'Epoch {epoch}/{num_epochs - 1}')
    #        print('-' * 10)

    model.eval()

    count = 0

    layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

    _ = model(image_loader_tensor(photo_content))
    original_activation_content = activation['conv4_2']

    _ = model(image_loader_tensor(photo_style))
    original_activations_style = {}
    for i in layers:
        original_activations_style[i] = copy.deepcopy(activation[i])

    _ = model(image_loader_tensor(photo_style2))
    original_activations_style2 = {}
    for i in layers:
        original_activations_style2[i] = copy.deepcopy(activation[i])

    def closure():
        nonlocal count
        # torch.clamp(noise_img, 0, 1) # 255

        optimizer.zero_grad()

        # with torch.set_grad_enabled(False):

        # loss = 0
        style_loss_var = 0
        style_loss_var2 = 0

        _ = model(image_loader_tensor(noise_img))

        for i in layers:
            noisy_activaton = activation[i]
            original_activation = original_activations_style[i]

            style_loss_var += criterion_style(noisy_activaton, original_activation, fm_guides, i, False)

        if photo_style2 is not None:
            for i in layers:
                noisy_activaton = activation[i]
                original_activation = original_activations_style2[i]

                style_loss_var2 += criterion_style(noisy_activaton, original_activation, fm_guides, i, True)

        content_loss_var = criterion_content(activation['conv4_2'], original_activation_content)
        # print(torch.all(torch.eq(activation['conv4_2'], original_activation_content)))

        alpha = 1e3
        beta = 3e9
        gamma = 1e3
        tv_loss = total_variation(noise_img).to(device)
        total_loss = alpha * content_loss_var + beta * style_loss_var + beta * style_loss_var2 + gamma * tv_loss

        total_loss.backward()

        running_loss = total_loss.item()

        epoch_loss = running_loss

        if (count + 0) % 20 == 0:
            print('Content ', content_loss_var)
            print('Style ', style_loss_var + style_loss_var2)
            print('Total variation', tv_loss)
            print(f'Iteration: {count:03}, Loss: {epoch_loss:.4f}')

        count += 1

        return total_loss

    optimizer = optim.LBFGS((noise_img,), max_iter=1000, line_search_fn='strong_wolfe')
    optimizer.step(closure)

    # time_elapsed = time.time() - start_time
    # print(f'Training complete in {(time_elapsed // 60):.0f}m {time_elapsed % 60:.0f}s')

    return noise_img

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output

    return hook


def content_loss(noisy_img, original_img):
    #     noisy_img = torch.reshape(noisy_img, (-1,))
    #     original_img = torch.reshape(original_img, (-1,))
    loss = nn.MSELoss(reduction='sum')
    return loss(noisy_img.squeeze(axis=0), original_img.squeeze(axis=0)) / 2


def get_gm(layer, fm_guide, layer_name, should_invert):
    #print(layer[0].shape)
    # layer[0] = torch.reshape(layer[0], (layer[0].shape[0], layer[0].shape[1] * layer[0].shape[2]))  # L*H*W -> L*M (M = H*W)
    a, b, c, d = layer.size()
    # e, f, g = fm_guide[layer_wname].size()
    # print(a, b, c, d)
    # print(fm_guide[layer_name].shape)
    if should_invert:
        # new_fm_guide = np.logical_xor(fm_guide[layer_name], np.ones(fm_guide[layer_name].shape)).astype(int)[0]
        new_fm_guide = (1 - fm_guide[layer_name])[0]
        # print(fm_guide[layer_name].shape)
        # print(new_fm_guide.shape)
    else:
        new_fm_guide = fm_guide[layer_name][0]
    dot_product = torch.mul(layer[0], torch.from_numpy(new_fm_guide).float().to(device))
    features = dot_product.view(a*b, c*d)
    gram = torch.matmul(features, features.t())
    # gram /= a * b * c * d

    return gram

def style_loss(noisy_layer, original_layer, fm_guide, layer_name, should_invert):
    A = get_gm(original_layer, fm_guide, layer_name, should_invert)
    G = get_gm(noisy_layer, fm_guide, layer_name, should_invert)

    E = nn.MSELoss(reduction='sum')

    # return 0.2 * E(A, G)
    return 0.2 * E(A, G) / ((2 * noisy_layer.shape[1] * noisy_layer.shape[2] * noisy_layer.shape[3]) ** 2)

def get_fm_guides(guides, model, mode='simple'):
    # img_content = cv.imread(os.getcwd() + "/data/content-images/" + "golden_gate.jpg")
    # dst = cv.resize(img_content, (1000, 650))
    # img = cv.imread(os.getcwd()+"/data/content-images/"+"tubingen.png")
    img_content = np.zeros(guides.transpose(2, 0, 1).shape)
    # dst_content = cv.cvtColor(img_content, cv.COLOR_BGR2RGB)
    # dst_content = dst_content / 255
    # dst_content = dst_content.transpose(2, 0, 1)
    _ = model(torch.from_numpy(img_content).unsqueeze(0).to(device, torch.float))
    layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

    fm_guides = {}
    if mode == 'simple':
        # probe_image = np.zeros((3,) + guides.shape[:-1])
        # probe_image += 1e2 * np.randn(*probe_image.shape)
        # feature_maps = get_activations(probe_image, caffe_model, layers=layers)
        for layer in layers:
            sf = np.asarray(activation[layer].shape[1:]).astype(float) / np.asarray(guides.transpose(2, 0, 1).shape)
            # print(activation[layer].shape)
            # print(guides.transpose(2, 0, 1).shape)
            sf[0] = 1
            # print(sf)
            fm_guides[layer] = scipy.ndimage.zoom(guides.transpose(2, 0, 1), sf, mode='nearest')
            # print(fm_guides[layer].shape)
    return fm_guides

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

# optimizer = optim.Adam(filter(lambda p: p.requires_grad, finetuned_model.parameters()))


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
    #     image = image.transpose((2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0).to(device, torch.float)


def image_loader_tensor(image_tensor):
    # image_tensor = image_tensor.transpose((2, 0, 1))
    return image_tensor.unsqueeze(0).to(device, torch.float)


img_style = cv.imread(os.getcwd()+"/data/style-images/"+"vg_starry_night.jpg")
# img = cv.imread(os.getcwd()+"/data/content-images/"+"tubingen.png")
dst_style = cv.cvtColor(img_style, cv.COLOR_BGR2RGB)
dst_style = dst_style / 255

img_style2 = cv.imread(os.getcwd()+"/data/style-images/"+"vg_la_cafe.jpg")
# img = cv.imread(os.getcwd()+"/data/content-images/"+"tubingen.png")
dst_style2 = cv.cvtColor(img_style2, cv.COLOR_BGR2RGB)
dst_style2 = dst_style2 / 255

img_content = cv.imread(os.getcwd()+"/data/video/"+"out58.png")
# dst = cv.resize(img_content, (1000, 650))
# img = cv.imread(os.getcwd()+"/data/content-images/"+"tubingen.png")
dst_content = cv.cvtColor(img_content, cv.COLOR_BGR2RGB)
dst_content = dst_content / 255

mask_content = cv.imread(os.getcwd()+"/maska2.png")
dst_mask = cv.cvtColor(mask_content, cv.COLOR_BGR2RGB)
dst_mask = dst_mask / 255
fm_guides = get_fm_guides(dst_mask, model)

# base = train_photo(model, content_loss, dst, 0, activation).detach()
# base = train_photo2(model, style_loss, dst, 0, activation).detach()
base = train_photo3(model, content_loss, style_loss, dst_content, dst_style, 0, activation, fm_guides, dst_style2).detach()
no_unorm = base.numpy()

print('min ', np.min(no_unorm),'max ', np.max(no_unorm))

with_unorm = unnormalize(base, (123.675, 116.28, 103.53), (1, 1, 1)).numpy()
# with_unorm = unnormalize(base, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).numpy()

print('min ', np.min(with_unorm), 'max ', np.max(with_unorm))

with_unorm = with_unorm.transpose(1, 2, 0)

# f, (ax1, ax2) = plt.subplots(1, 2)
# ax1.imshow(with_unorm)
# # ax1.imshow(no_unorm)
# ax2.imshow(dst)
# plt.show()

# with_unorm *= 255
# print('min ', np.min(with_unorm), 'max ', np.max(with_unorm))

# with_unorm_32 = np.float32(with_unorm)
with_unorm = cv.cvtColor(with_unorm, cv.COLOR_RGB2BGR)
with_unorm = with_unorm.astype(int)
print('min ', np.min(with_unorm), 'max ', np.max(with_unorm))
cv.imwrite("rezultat_final_350_normal29.png", with_unorm.astype(int))

# normal: alpha = 1, beta 10e3
# normal2: alpha = 1e5, beta = 3e8
# normal3: alpha = 1e5, beta = 3e7
# normal4: alpha = 1e5, beta = 3e9, tv_loss exists, gamma = 1
# normal5: alpha = 1e5, beta = 3e12, tv_loss exists, gamma = 1 Loss: nan
# normal6: alpha = 8, beta = 1e8, gamma = 1
# normal7: alpha = 1e5, beta = 1e10, gamma = 1
# normal8: alpha = 1e5, beta = 3e8, gamma = 1
# normal9: alpha = 1e5, beta = 3e8, gamma = 0
# normal10: alpha = 1e5, beta = 3e8, gamma = 1
# normal11: alpha = 1e5, beta = 3e8, gamma = 10
# normal12: alpha = 1e5, beta = 3e8, gamma = 1e3
# normal13: alpha = 1e5, beta = 3e8, gamma = 1e3, initialized with content image, no normalization, bad rgb
# normal14: alpha = 1e5, beta = 3e8, gamma = 1e3, initialized with content image, with normalization, bad rgb
# normal15: alpha = 1e5, beta = 3e9, gamma = 1e3, initialized with content image, with normalization
# normal16: alpha = 1e4, beta = 3e9, gamma = 1e3, initialized with content image, with normalization
# normal17: alpha = 1e4, beta = 3e9, gamma = 1e3, initialized with content image, with normalization, brigde with cafe
# normal18: alpha = 1e3, beta = 3e9, gamma = 1e3, initialized with content image, with normalization
# normal19: alpha = 1e3, beta = 3e9, gamma = 1e1, initialized with content image, with normalization
# normal20: alpha = 1e3, beta = 3e10, gamma = 1e3, initialized with content image, with normalization
# normal21: alpha = 1e3, beta = 3e10, initialized with content image, with normalization
# normal22: alpha = 1e3, beta = 3e10, gamma = 1e4 initialized with content image, with normalization
# normal23: alpha = 1e3, beta = 3e9, gamma = 1e3 initialized with content image, with normalization




