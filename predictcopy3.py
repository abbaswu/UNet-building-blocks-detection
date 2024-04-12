import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

from skimage.util import view_as_windows

W_SIZE = 500
PAD_PX = W_SIZE // 2

def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)

        # (H, W, C)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                # img: (C, H, W)
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

def generate_tiling(
    in_img: np.ndarray, # (H, W, C)
    w_size=W_SIZE
) -> np.ndarray:
    # Generate tiling images
    win_size = w_size
    pad_px = win_size // 2

    # Padding image
    img_pad = np.pad(in_img, [(pad_px,pad_px), (pad_px,pad_px), (0,0)], 'constant')
    tiles = view_as_windows(img_pad, (win_size,win_size,3), step=pad_px)
    
    # Iterate tiles by row and then by column
    # Append them to tiles_lst
    tiles_lst = []
    for row in range(tiles.shape[0]):
        for col in range(tiles.shape[1]):
            # This is one tile
            tt = tiles[row, col, 0, ...].copy()
            tiles_lst.append(tt)

    # (44200, 500, 3)
    # All the rows of the tiles
    # (ROW NO., W, C)
    tiles_array = np.concatenate(tiles_lst)

    # You must reshape the tiles_array into (batch_size, width, height, 3)
    # (884, 500, 500, 3)
    tiles_array = tiles_array.reshape(int(tiles_array.shape[0]/w_size), w_size, w_size, 3)
    return tiles_array

def reconstruct_from_patches(
        patches_images,
        patch_size,
        step_size,
        height_width_tuple: tuple[int, int],
        image_dtype: type=np.uint8
    ):
    '''Adjust to take patch images directly.
    patch_size is the size of the tiles
    step2_size should be patch_size//2
    height_width_tuple is the size of the original image
    image_dtype is the data type of the target image

    Most of this could be guessed using an array of patches
    (except step_size but, again, it should be should be patch_size//2)
    '''
    i_h, i_w = np.array(height_width_tuple[:2]) + (patch_size, patch_size)
    p_h = p_w = patch_size
    if len(patches_images.shape) == 4:
        img = np.zeros((i_h+p_h//2, i_w+p_w//2, 3), dtype=image_dtype)
    else:
        img = np.zeros((i_h+p_h//2, i_w+p_w//2), dtype=image_dtype)

    numrows = (i_h)//step_size-1
    numcols = (i_w)//step_size-1
    expected_patches = numrows * numcols
    if len(patches_images) != expected_patches:
        raise ValueError(f"Expected {expected_patches} patches, got {len(patches_images)}")

    patch_offset = step_size//2
    patch_inner = p_h-step_size
    for row in range(numrows):
        for col in range(numcols):
            tt = patches_images[row*numcols+col]
            tt_roi = tt[patch_offset:-patch_offset,patch_offset:-patch_offset]
            img[row*step_size:row*step_size+patch_inner,
                col*step_size:col*step_size+patch_inner] = tt_roi # +1?? 
    return img[step_size//2:-(patch_size+step_size//2),step_size//2:-(patch_size+step_size//2),...]

def predict_img(net,
        full_img: Image,
                device,
                scale_factor=1,
                out_threshold=0.5) -> np.ndarray:
    net.eval()

    # (C, H, W)
    img = torch.from_numpy(preprocess(None, full_img, scale_factor, is_mask=False).copy())
    
    # (N, C, H, W)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            # Unexplored branch
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    # returns (H, W)
    return mask[0].long().squeeze().numpy()




def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='./checkpoints/checkpoint_epoch5.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(
        mask: np.ndarray,
        mask_values # == [0, 1], or == [[...], ...]
):
    height = mask.shape[-2]
    width = mask.shape[-1]

    if isinstance(mask_values[0], list):
        out = np.zeros((height, width, len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((height, width), dtype=bool)
    else:
        out = np.zeros((height, width), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    
    # no problem here
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
       
        # TODO:
        # read image
        # then call generate_tiling
        # then convert image format
        # then pass adequate parameter img to predict_img below

        # Read image
        # >>> in_img.shape                                                                                                                                              
        # (6373, 8269, 3)
        # (H, W, C)
        in_img = np.array(Image.open(filename))

        height, width, channels = in_img.shape

        # (884, 500, 500, 3)
        # (884, H, W, 3)
        tiles_array_ = generate_tiling(in_img)

        # Store a list of masks generated from each tile 
        masks_list = []

        # (H, W, 3)
        for tile in tqdm(tiles_array_):
            # (W, H, 3)
            tile_ = Image.fromarray(tile)

            # (H, W)
            mask = predict_img(net=net,
                full_img=tile_,
                scale_factor=args.scale,
                out_threshold=args.mask_threshold,
                device=device
            )

            masks_list.append(mask)
            # TODO: can we reconstruct the image and the mask successfully? then let's work on other topics...

        # Reconstruct the whole mask from the list of masks
        masks_array = np.array(masks_list)
        reconstructed_mask = reconstruct_from_patches(masks_array, W_SIZE, PAD_PX, (height, width))

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(reconstructed_mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
