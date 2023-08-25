#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

def change_color_gray(image,parsing,parts):
    b1,g1,r1 = [255,255,255]
    tar_color1 = np.zeros_like(image)
    tar_color1[:,:,0] = b1
    tar_color1[:,:,1] = g1
    tar_color1[:,:,2] = r1
    
    tar_color2 = np.zeros_like(image)
    parts = np.array(parts)
    tar_color2[np.isin(parsing,parts)] = tar_color1[np.isin(parsing,parts)]
    return tar_color2

def dilate_image(image,kernel_size=(5,5),iterate=1):
    dilation = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones(kernel_size,np.uint8)
    dilation = cv2.dilate(dilation,kernel,iterations=iterate)
    return dilation

def change_eyeshadow(dilation_image,parsing,parts):
    # grayscale_image = cv2.cvtColor(grayscale_image,cv2.COLOR_BGR2GRAY)
    tar_color = np.zeros_like(dilation_image)
    
    dilation_image[np.isin(parsing,parts)] = tar_color[np.isin(parsing,parts)]
    return dilation_image

def add_parsing_eyeshadow(image,parsing):
    index = np.where(image == 255)
    parsing[index[0],index[1]] = 19
    return parsing


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    grayscale_image = change_color_gray(vis_im,vis_parsing_anno,[4,5])
    dilation_image = dilate_image(grayscale_image,kernel_size=(5,5),iterate=5)
    eyeshadow_image = change_eyeshadow(dilation_image,vis_parsing_anno,[4,5])
    vis_parsing_add_eyeshadow = add_parsing_eyeshadow(eyeshadow_image,vis_parsing_anno)



    num_of_class = np.max(vis_parsing_add_eyeshadow)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_add_eyeshadow == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_add_eyeshadow)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im

def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('./cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            print(np.unique(parsing))

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))


if __name__ == "__main__":
    evaluate(dspth='./src', cp='79999_iter.pth')


