import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from torch import optim

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
from utils.dice_score import dice_loss, multiclass_dice_coeff, dice_coeff
from utils.reproducibility import set_all_lib_seed, set_seed_worker

from networks.segtran2d import Segtran2d, CONFIG
from config.config_parser import configure_parse
from config.settings import SEED

dir_img = Path('./data/test/imgs')
dir_mask = Path('./data/test/masks')
dir_checkpoint = Path('./checkpoints/')
dir_masks_pred = Path('./data/pred/masks')

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_parser():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')

    return parser


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    parser = get_parser()
    configure_parse(parser)
    args = parser.parse_args()

    args_dict = {
        'trans_output_type': 'private',
        'mid_type': 'shared',
        'in_fpn_scheme': 'AN',
        'out_fpn_scheme': 'AN',
    }
    for arg, v in args_dict.items():
        args.__dict__[arg] = v

    CONFIG.update_config(args)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(SEED)

    dataset = BasicDataset(str(dir_img), str(dir_mask), args.scale)
    num_workers = 2
    test_loader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=set_seed_worker,
        generator=dataloader_generator,
    )


    # in_files = args.input
    # out_files = get_output_filenames(args)




    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = None
    CONFIG.device = 'cuda'

    if args.net == 'unet':
        model = UNet(n_channels=args.channels, n_classes=args.classes,
                     bilinear=args.bilinear)
    else:
        CONFIG.n_channels = args.channels
        model = Segtran2d(CONFIG)

    model = model.to(memory_format=torch.channels_last)
    model.classes = args.classes
    model.n_classes = args.classes

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    model.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    model.load_state_dict(state_dict)

    logging.info('Model loaded!')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    criterion = nn.CrossEntropyLoss() if model.classes > 1 else \
        nn.BCEWithLogitsLoss()
    global_step = 0
    dice_score = 0
    num_val_batches = len(test_loader)

    for batch in test_loader:
        images, true_masks, names = batch['image'], batch['mask'], \
                                       batch['name']

        images = images.to(device=device, dtype=torch.float32,
                           memory_format=torch.channels_last)
        true_masks = true_masks.to(device=device, dtype=torch.long)

        with torch.autocast(device.type if device.type != 'mps' else 'cpu',
                            enabled=args.amp):
            masks_pred = model(images)
            for mask, name in masks_pred, names:
                out_filename = dir_masks_pred / name
                result = mask_to_image(mask, mask_values)
                result.save(out_filename)
                logging.info(f'Mask saved to {out_filename}')
            if model.n_classes == 1:
                loss = criterion(masks_pred.squeeze(1), true_masks.float())
                loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)),
                                  true_masks.float(), multiclass=False)
                masks_pred = (F.sigmoid(masks_pred) > 0.5).float()
                dice_score += dice_coeff(masks_pred, true_masks,
                                         reduce_batch_first=False)
            else:
                loss = criterion(masks_pred, true_masks)
                loss += dice_loss(
                    F.softmax(masks_pred, dim=1).float(),
                    F.one_hot(true_masks, model.n_classes).permute(0, 3, 1,
                                                                   2).float(),
                    multiclass=True
                )
                true_masks = F.one_hot(true_masks, model.n_classes).permute(0,
                                                                         3, 1,
                                                                        2).float()
                masks_pred = F.one_hot(masks_pred.argmax(dim=1),
                                      model.n_classes).permute(0, 3, 1,
                                                              2).float()
                dice_score += multiclass_dice_coeff(masks_pred[:, 1:],
                                                    true_masks[:, 1:],
                                                    reduce_batch_first=False)
        # if not args.no_save:
        #     for mask in masks_pred:
        #         result = mask_to_image(mask, mask_values)
        #         result.save(out_filename)
        #         logging.info(f'Mask saved to {out_filename}')
    dice_score = dice_score / max(num_val_batches, 1)
    print('Dice Score: {}'.format(dice_score))
    logging.info('Predict Dice score: {}'.format(dice_score))





    # for i, filename in enumerate(in_files):
    #     logging.info(f'Predicting image {filename} ...')
    #     img = Image.open(filename)
    #
    #     mask = predict_img(net=net,
    #                        full_img=img,
    #                        scale_factor=args.scale,
    #                        out_threshold=args.mask_threshold,
    #                        device=device)
    #
    #     if not args.no_save:
    #         out_filename = out_files[i]
    #         result = mask_to_image(mask, mask_values)
    #         result.save(out_filename)
    #         logging.info(f'Mask saved to {out_filename}')
    #
    #     if args.viz:
    #         logging.info(f'Visualizing results for image {filename}, close to continue...')
    #         plot_img_and_mask(img, mask)
