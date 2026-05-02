
from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F
from IPython import embed

def load_img(img_path):
	out_np = np.asarray(Image.open(img_path))
	if(out_np.ndim==2):
		out_np = np.tile(out_np[:,:,None],3)
	return out_np

def resize_img(img, HW=(256,256), resample=3):
	return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))

def preprocess_img(img_rgb_orig, HW=(256,256), resample=3):
	# return original size L and resized L as torch Tensors
	img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
	
	img_lab_orig = color.rgb2lab(img_rgb_orig)
	img_lab_rs = color.rgb2lab(img_rgb_rs)

	img_l_orig = img_lab_orig[:,:,0]
	img_l_rs = img_lab_rs[:,:,0]

	tens_orig_l = torch.Tensor(img_l_orig)[None,None,:,:]
	tens_rs_l = torch.Tensor(img_l_rs)[None,None,:,:]

	return (tens_orig_l, tens_rs_l)

def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
	# tens_orig_l 	1 x 1 x H_orig x W_orig
	# out_ab 		1 x 2 x H x W

	HW_orig = tens_orig_l.shape[2:]
	HW = out_ab.shape[2:]

	# call resize function if needed
	if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
		out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
	else:
		out_ab_orig = out_ab

	out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
	return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))


def preprocess_img_l_training(img_rgb_orig, HW=(256,256), resample=3):
    if img_rgb_orig.shape[-1] == 4:
        img_rgb_orig = img_rgb_orig[:, :, :3]

    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)

    img_lab_rs = color.rgb2lab(img_rgb_rs)
    img_l_rs   = img_lab_rs[:, :, 0]  # (H, W)

    # (H, W) → (1, H, W) — single channel dim, NO batch dim
    tens_rs_l = torch.Tensor(img_l_rs)[None, :, :]

    return tens_rs_l
    
def preprocess_img_ab(img_rgb_orig, HW=(256,256), resample=3):
    if img_rgb_orig.shape[-1] == 4:
        img_rgb_orig = img_rgb_orig[:, :, :3]

    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)

    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs   = color.rgb2lab(img_rgb_rs)

    img_ab_orig = img_lab_orig[:, :, 1:]
    img_ab_rs   = img_lab_rs[:, :, 1:]

    # (H, W, 2) → (2, H, W) — NO [None] batch dim here
    tens_orig_ab = torch.Tensor(img_ab_orig).permute(2, 0, 1)
    tens_rs_ab   = torch.Tensor(img_ab_rs).permute(2, 0, 1)

    return (tens_orig_ab, tens_rs_ab)

# takes as input l channel and ab channel (as torch tensors) and returns a lab format image
def lab_to_rgb(l, ab):
    lab = torch.cat([l, ab], dim=1)
    lab = lab.squeeze(0).permute(1, 2, 0).detach().numpy()
    return color.lab2rgb(lab)