import os
import cv2
import toml
import argparse
import numpy as np
import time
import torch
from torch.nn import functional as F
import utils
from utils import CONFIG
import networks


def single_inference(model, image_dict):
    with torch.no_grad():
        image, trimap = image_dict['image'], image_dict['trimap']

        image = image.cuda()
        trimap = trimap.cuda()

        # run model
        pred, _, _ = model(image, trimap)
        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred[:, 0:1, :, :], pred[:, 1:2, :, :], pred[:, 2:3, :, :]

        # refinement
        alpha_pred = alpha_pred_os8.clone().detach()
        weight_os4 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width1,
                                                        train_mode=False)
        alpha_pred[weight_os4 > 0] = alpha_pred_os4[weight_os4 > 0]
        weight_os1 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width2,
                                                        train_mode=False)
        alpha_pred[weight_os1 > 0] = alpha_pred_os1[weight_os1 > 0]

        h, w = image_dict['alpha_shape']
        alpha_pred = alpha_pred[0, 0, ...].data.cpu().numpy() * 255
        alpha_pred = alpha_pred.astype(np.uint8)

        alpha_pred[np.argmax(trimap.cpu().numpy()[0], axis=0) == 0] = 0.0
        alpha_pred[np.argmax(trimap.cpu().numpy()[0], axis=0) == 2] = 255.

        alpha_pred = alpha_pred[64:h + 64, 64:w + 64]
        return alpha_pred


def generator_tensor_dict(image_path, trimap_path):
    # read images
    image = cv2.imread(image_path)
    trimap = cv2.imread(trimap_path, 0)

    sample = {'image': image, 'trimap': trimap, 'alpha_shape': (image.shape[0], image.shape[1])}

    # reshape
    h, w = sample["alpha_shape"]

    if h % 64 == 0 and w % 64 == 0:
        padded_image = np.pad(sample['image'], ((64, 64), (64, 64), (0, 0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((64, 64), (64, 64)), mode="reflect")

        sample['image'] = padded_image
        sample['trimap'] = padded_trimap

    else:
        target_h = h % 64
        target_w = w % 64
        pad_h = 64 - target_h
        pad_w = 64 - target_w
        padded_image = np.pad(sample['image'], ((64, pad_h + 64), (64, pad_w + 64), (0, 0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((64, pad_h + 64), (64, pad_w + 64)), mode="reflect")

        sample['image'] = padded_image
        sample['trimap'] = padded_trimap

    # ImageNet mean & std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # convert GBR images to RGB
    image, trimap = sample['image'][:, :, ::-1], sample['trimap']

    # swap color axis
    image = image.transpose((2, 0, 1)).astype(np.float32)

    # trimap configuration
    padded_trimap[padded_trimap < 85] = 0
    padded_trimap[padded_trimap >= 170] = 2
    padded_trimap[padded_trimap >= 85] = 1

    # normalize image
    image /= 255.

    # to tensor
    sample['image'], sample['trimap'] = torch.from_numpy(image), torch.from_numpy(trimap).to(torch.long)
    sample['image'] = sample['image'].sub_(mean).div_(std)

    # trimap to one-hot 3 channel
    sample['trimap'] = F.one_hot(sample['trimap'], num_classes=3).permute(2, 0, 1).float()

    # add first channel
    sample['image'], sample['trimap'] = sample['image'][None, ...], sample['trimap'][None, ...]

    return sample


def monitor_gpu_memory():
    max_memory_cached = torch.cuda.max_memory_cached()
    print(f"Max memory cached: {max_memory_cached / (1024 ** 2):.2f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/Composition1k.toml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help="path of checkpoint")

    # local
    parser.add_argument('--image-dir', type=str, default='../dataset/Transparent-460/composited_images/', help="input image dir")
    parser.add_argument('--mask-dir', type=str, default='../dataset/Transparent-460/trimap_copy/', help="input trimap dir")
    parser.add_argument('--trimap-dir', type=str, default='../dataset/Transparent-460/trimap_copy/', help="input trimap dir")
    parser.add_argument('--output', type=str, default='HRAMR-Matting-MDRDLoss/Transparent-460/', help="output dir")

    # parser.add_argument('--image-dir', type=str, default='../alphamatting/input/', help="input image dir")
    # parser.add_argument('--mask-dir', type=str, default='../alphamatting/trimap/', help="input trimap dir")
    # parser.add_argument('--trimap-dir', type=str, default='../alphamatting/trimap/', help="input trimap dir")
    # parser.add_argument('--output', type=str, default='HRAMR-Matting/alphamatting/', help="output dir")

    # Parse configuration
    args = parser.parse_args()
    with open(args.config) as f:
        utils.load_config(toml.load(f))

    # Check if toml config file is loaded
    if CONFIG.is_default:
        raise ValueError("No .toml config loaded.")

    utils.make_dir(os.path.join(args.output, 'pred_alpha'))

    # build model
    model = networks.get_generator(is_train=False)
    model.cuda()

    # load checkpoint
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)

    # inference
    model = model.eval()

    start = time.time()
    for i, image_name in enumerate(os.listdir(args.image_dir)):
        # assume image and mask have the same file name
        image_path = os.path.join(args.image_dir, image_name)
        trimap_path = os.path.join(args.trimap_dir, image_name)

        image_dict = generator_tensor_dict(image_path, trimap_path)
        alpha_pred = single_inference(model, image_dict)

        # save images
        _al = cv2.cvtColor(alpha_pred, cv2.COLOR_GRAY2RGB)

        cv2.imwrite(os.path.join(args.output, 'pred_alpha', image_name), _al)
        print('[{}/{}] inference done : {}'.format(i, len(os.listdir(args.image_dir)), os.path.join(args.output, 'pred_alpha', image_name)))
    end = time.time()
    monitor_gpu_memory()
    print('sum_latency:', end - start)
    print('avg_latency:', (end - start) / 1000)
