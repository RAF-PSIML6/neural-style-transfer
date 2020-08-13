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
import scipy
import scipy.ndimage

import copy

from nst_segmentation import segment_photo


def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :-1] - y[:, :, 1:])) + \
           torch.sum(torch.abs(y[:, :-1, :] - y[:, 1:, :]))


def neural_style_transfer(model, criterion_content, criterion_style, photo_content, photo_style, activation, fm_guides, photo_style2=None, num_steps=1000):
    photo_style = cv.resize(photo_style, (photo_content.shape[1], photo_content.shape[0]), interpolation=cv.INTER_CUBIC)
    photo_style2 = cv.resize(photo_style2, (photo_content.shape[1], photo_content.shape[0]), interpolation=cv.INTER_CUBIC)
    photo_content = photo_content.transpose(2, 0, 1)
    photo_style = photo_style.transpose(2, 0, 1)
    photo_style2 = photo_style2.transpose(2, 0, 1)

    content_copy_img = copy.deepcopy(photo_content)
    noise_img_tensor = torch.from_numpy(content_copy_img)

    IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
    IMAGENET_MEAN_STD_NEUTRAL = [1, 1, 1]

    loader = transforms.Compose([
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(IMAGENET_MEAN_255, IMAGENET_MEAN_STD_NEUTRAL)
    ])

    noise_img_tensor = loader(noise_img_tensor).float()
    noise_img = torch.autograd.Variable(noise_img_tensor, requires_grad=True)

    photo_content = loader(torch.from_numpy(photo_content)).float()
    photo_style = loader(torch.from_numpy(photo_style)).float()
    photo_style2 = loader(torch.from_numpy(photo_style2)).float()

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

        optimizer.zero_grad()

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

        alpha = 1e3
        beta = 3e9
        gamma = 1e3
        tv_loss = total_variation(noise_img).to(device)
        total_loss = alpha * content_loss_var + beta * style_loss_var + beta * style_loss_var2 + gamma * tv_loss

        total_loss.backward()

        running_loss = total_loss.item()

        epoch_loss = running_loss

        if (count + 0) % 20 == 0:
            print('Content loss: ', content_loss_var)
            print('Style loss: ', style_loss_var + style_loss_var2)
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


def get_gm(layer, fm_guide, layer_name, should_invert):
    a, b, c, d = layer.size()
    if should_invert:
        new_fm_guide = (1 - fm_guide[layer_name])[0]
    else:
        new_fm_guide = fm_guide[layer_name][0]
    dot_product = torch.mul(layer[0], torch.from_numpy(new_fm_guide).float().to(device))
    features = dot_product.view(a*b, c*d)
    gram = torch.matmul(features, features.t())

    return gram

def style_loss(noisy_layer, original_layer, fm_guide, layer_name, should_invert):
    A = get_gm(original_layer, fm_guide, layer_name, should_invert)
    G = get_gm(noisy_layer, fm_guide, layer_name, should_invert)

    E = nn.MSELoss(reduction='sum')

    return 0.2 * E(A, G) / ((2 * noisy_layer.shape[1] * noisy_layer.shape[2] * noisy_layer.shape[3]) ** 2)

def get_fm_guides(guides, model, mode='simple'):
    img_content = np.zeros(guides.transpose(2, 0, 1).shape)

    _ = model(torch.from_numpy(img_content).unsqueeze(0).to(device, torch.float))

    layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

    fm_guides = {}
    if mode == 'simple':
        for layer in layers:
            sf = np.asarray(activation[layer].shape[1:]).astype(float) / np.asarray(guides.transpose(2, 0, 1).shape)
            sf[0] = 1
            fm_guides[layer] = scipy.ndimage.zoom(guides.transpose(2, 0, 1), sf, mode='nearest')
    return fm_guides


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


if __name__ == "__main__":

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

    img_style = cv.imread(os.getcwd()+"/data/style-images/"+"mosaic.jpg")
    dst_style = cv.cvtColor(img_style, cv.COLOR_BGR2RGB)
    dst_style = dst_style / 255

    img_style2 = cv.imread(os.getcwd()+"/data/style-images/"+"vg_wheat_field.jpg")
    dst_style2 = cv.cvtColor(img_style2, cv.COLOR_BGR2RGB)
    dst_style2 = dst_style2 / 255

    img_content = cv.imread(os.getcwd()+"/data/content-images/"+"woman.jpeg")
    dst_content = cv.cvtColor(img_content, cv.COLOR_BGR2RGB)
    dst_content = dst_content / 255

    mask = segment_photo(dst_content)
    dst_mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)
    dst_mask = dst_mask / 255
    fm_guides = get_fm_guides(dst_mask, model)

    base = neural_style_transfer(model, content_loss, style_loss, dst_content, dst_style, activation, fm_guides, dst_style2).detach()
    no_unorm = base.numpy()

    with_unorm = unnormalize(base, (123.675, 116.28, 103.53), (1, 1, 1)).numpy()
    with_unorm = with_unorm.transpose(1, 2, 0)

    with_unorm = cv.cvtColor(with_unorm, cv.COLOR_RGB2BGR)
    with_unorm = with_unorm.astype(int)
    cv.imwrite("data/results/mosaic_vg_wheat_field_woman_segmentation.png", with_unorm.astype(int))




