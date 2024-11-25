import torch
import numpy as np

from PIL import Image, ImageFilter
from ldm_patched.pfn.architecture.RRDB import RRDBNet as ESRGAN

import cv2
import math
import os
from collections import OrderedDict

from ldm_patched.contrib.external_upscale_model import ImageUpscaleWithModel

model = None
LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
opImageUpscaleWithModel = ImageUpscaleWithModel()


def pytorch_to_numpy(x):
    return [np.clip(255.0 * y.cpu().numpy(), 0, 255).astype(np.uint8) for y in x]


def numpy_to_pytorch(x):
    y = x.astype(np.float32) / 255.0
    y = y[None]
    y = np.ascontiguousarray(y.copy())
    y = torch.from_numpy(y).float()
    return y


def perform_upscale(img, path_upscale_models):
    global model
    print(f"Upscaling image with shape {str(img.shape)} ...")

    if model is None:
        print("loading upscale model")
        sd = torch.load(path_upscale_models)
        sdo = OrderedDict()
        for k, v in sd.items():
            sdo[k.replace("residual_block_", "RDB")] = v
        del sd
        model = ESRGAN(sdo)
        model.cpu()
        model.eval()

    img = numpy_to_pytorch(img)
    img = opImageUpscaleWithModel.upscale(model, img)[0]
    img = pytorch_to_numpy(img)[0]

    return img


def resample_image(im, width, height):
    im = Image.fromarray(im)
    im = im.resize((int(width), int(height)), resample=LANCZOS)
    return np.array(im)


def get_shape_ceil(h, w):
    return math.ceil(((h * w) ** 0.5) / 64.0) * 64.0


def get_image_shape_ceil(im):
    H, W = im.shape[:2]
    return get_shape_ceil(H, W)


def set_image_shape_ceil(im, shape_ceil):
    shape_ceil = float(shape_ceil)

    H_origin, W_origin, _ = im.shape
    H, W = H_origin, W_origin

    for _ in range(256):
        current_shape_ceil = get_shape_ceil(H, W)
        if abs(current_shape_ceil - shape_ceil) < 0.1:
            break
        k = shape_ceil / current_shape_ceil
        H = int(round(float(H) * k / 64.0) * 64)
        W = int(round(float(W) * k / 64.0) * 64)

    if H == H_origin and W == W_origin:
        return im

    return resample_image(im, width=W, height=H)


inpaint_head_model = None


class InpaintHead(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = torch.nn.Parameter(torch.empty(size=(320, 5, 3, 3), device="cpu"))

    def __call__(self, x):
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), "replicate")
        return torch.nn.functional.conv2d(input=x, weight=self.head)


current_task = None


def box_blur(x, k):
    x = Image.fromarray(x)
    x = x.filter(ImageFilter.BoxBlur(k))
    return np.array(x)


def max_filter_opencv(x, ksize=3):
    # Use OpenCV maximum filter
    # Make sure the input type is int16
    return cv2.dilate(x, np.ones((ksize, ksize), dtype=np.int16))


def morphological_open(x):
    # Convert array to int16 type via threshold operation
    x_int16 = np.zeros_like(x, dtype=np.int16)
    x_int16[x > 127] = 256

    for i in range(32):
        # Use int16 type to avoid overflow
        maxed = max_filter_opencv(x_int16, ksize=3) - 8
        x_int16 = np.maximum(maxed, x_int16)

    # Clip negative values to 0 and convert back to uint8 type
    x_uint8 = np.clip(x_int16, 0, 255).astype(np.uint8)
    return x_uint8


def up255(x, t=0):
    y = np.zeros_like(x).astype(np.uint8)
    y[x > t] = 255
    return y


def imsave(x, path):
    x = Image.fromarray(x)
    x.save(path)


def regulate_abcd(x, a, b, c, d):
    H, W = x.shape[:2]
    if a < 0:
        a = 0
    if a > H:
        a = H
    if b < 0:
        b = 0
    if b > H:
        b = H
    if c < 0:
        c = 0
    if c > W:
        c = W
    if d < 0:
        d = 0
    if d > W:
        d = W
    return int(a), int(b), int(c), int(d)


def compute_initial_abcd(x):
    indices = np.where(x)
    a = np.min(indices[0])
    b = np.max(indices[0])
    c = np.min(indices[1])
    d = np.max(indices[1])
    abp = (b + a) // 2
    abm = (b - a) // 2
    cdp = (d + c) // 2
    cdm = (d - c) // 2
    l = int(max(abm, cdm) * 1.15)
    a = abp - l
    b = abp + l + 1
    c = cdp - l
    d = cdp + l + 1
    a, b, c, d = regulate_abcd(x, a, b, c, d)
    return a, b, c, d


def solve_abcd(x, a, b, c, d, k):
    k = float(k)
    assert 0.0 <= k <= 1.0
    H, W = x.shape[:2]
    if k == 1.0:
        return 0, H, 0, W
    while True:
        if b - a >= H * k and d - c >= W * k:
            break

        add_h = (b - a) < (d - c)
        add_w = not add_h

        if b - a == H:
            add_w = True

        if d - c == W:
            add_h = True

        if add_h:
            a -= 1
            b += 1

        if add_w:
            c -= 1
            d += 1

        a, b, c, d = regulate_abcd(x, a, b, c, d)
    return a, b, c, d


def fooocus_fill(image, mask):
    current_image = image.copy()
    raw_image = image.copy()
    area = np.where(mask < 127)
    store = raw_image[area]

    for k, repeats in [
        (512, 2),
        (256, 2),
        (128, 4),
        (64, 4),
        (33, 8),
        (15, 8),
        (5, 16),
        (3, 16),
    ]:
        for _ in range(repeats):
            current_image = box_blur(current_image, k)
            current_image[area] = store

    return current_image

def create_soft_blend_mask(shape):
    rows, cols, _ = shape
    mask = np.zeros((rows, cols), dtype=np.float32)
    edge_num = 0.1


    center_x, center_y = cols // 2, rows // 2
    sigma = min(rows, cols) / 4 
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
            mask[i, j] = np.exp(-distance ** 2 / (2 * sigma ** 2))
    
    mask = edge_num + (1 - edge_num) * mask / np.max(mask)

    mask = np.dstack([mask] * 3)

    return mask

def soft_blend(image1, image2, mask):
    mask = mask.astype(np.float32)
    mask = np.clip(mask, 0, 1)
    blended_image = image1 * mask + image2 * (1 - mask)
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
    return blended_image

class InpaintWorker:
    def __init__(
        self,
        image,
        mask,
        path_upscale_models=None,
        use_fill=True,
        k=0.618,
        target_resolution=1024,
    ):
        a, b, c, d = compute_initial_abcd(mask > 0)
        a, b, c, d = solve_abcd(mask, a, b, c, d, k=k)
        # interested area
        self.interested_area = (a, b, c, d)
        self.interested_mask = mask[a:b, c:d]
        self.interested_image = image[a:b, c:d]

        # super resolution
        if (
            path_upscale_models is not None and get_image_shape_ceil(self.interested_image)
            < target_resolution
        ):
            self.interested_image = perform_upscale(
                self.interested_image, path_upscale_models
            )

        # resize to make images ready for diffusion
        self.interested_image = set_image_shape_ceil(
            self.interested_image, target_resolution
        )
        self.interested_fill = self.interested_image.copy()
        H, W, C = self.interested_image.shape

        # process mask
        self.interested_mask = up255(resample_image(self.interested_mask, W, H), t=127)

        # compute filling
        if use_fill:
            self.interested_fill = fooocus_fill(
                self.interested_image, self.interested_mask
            )

        # soft pixels
        self.mask = morphological_open(mask)
        self.image = image

        # ending
        self.latent = None
        self.latent_after_swap = None
        self.swapped = False
        self.latent_mask = None
        self.inpaint_head_feature = None
        return 

    def load_latent(self, latent_fill, latent_mask, latent_swap=None):
        self.latent = latent_fill
        self.latent_mask = latent_mask
        self.latent_after_swap = latent_swap
        return

    def swap(self):
        if self.swapped:
            return

        if self.latent is None:
            return

        if self.latent_after_swap is None:
            return

        self.latent, self.latent_after_swap = self.latent_after_swap, self.latent
        self.swapped = True
        return

    def unswap(self):
        if not self.swapped:
            return

        if self.latent is None:
            return

        if self.latent_after_swap is None:
            return

        self.latent, self.latent_after_swap = self.latent_after_swap, self.latent
        self.swapped = False
        return

    def color_correction(self, img):
        fg = img.astype(np.float32)
        bg = self.image.copy().astype(np.float32)
        w = self.mask[:, :, None].astype(np.float32) / 255.0
        y = fg * w + bg * (1 - w)
        return y.clip(0, 255).astype(np.uint8)
    
    def post_process(self, img, soft_blending=False):
        a, b, c, d = self.interested_area
        content = resample_image(img, d - c, b - a)
        result = self.image.copy()

        if soft_blending:
            mask = create_soft_blend_mask(content.shape)
            original_content = result[a:b, c:d]
            blended_content = soft_blend(content, original_content, mask)
            result[a:b, c:d] = blended_content
        else:
            result[a:b, c:d] = content

        result = self.color_correction(result)
        return result

    def visualize_mask_processing(self):
        return [self.interested_fill, self.interested_mask, self.interested_image]
