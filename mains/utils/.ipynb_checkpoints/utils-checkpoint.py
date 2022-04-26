import torch
import numpy as np
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
'''
- [x] PSNR corrected
- [x] weight initialization applied
'''


def ssim(image,gt):
    '''
    input
    -----
        image: numpy.ndarray
        gt: numpy.ndarray
    output
    -----
        numpy scalar of ssim averaged for channels
    '''
    if not (isinstance(image,np.ndarray) and isinstance(gt,np.ndarray)):
        raise ValueError("both inputs should be in numpy,ndarray type")
    if not image.ndim == gt.ndim:
        raise ValueError("dimensiom of the inputs should be the same")
    data_range = np.max(gt) - np.min(gt)
    if image.ndim==4: # N,H,W,L
        return structural_similarity(image.transpose(1,2,3,0), gt.transpose(1,2,3,0), data_range = data_range,multichannel=True)
    elif image.ndim==3: # H,W,L Batch_size = 1
        return structural_similarity(image, gt, data_range = data_range, multichannel=True)


def psnr(image,gt):
    mse = np.mean((image - gt)**2)
    if mse == 0:
        return float('inf')
    #    data_range = np.max(gt) - np.min(gt)
    data_range= gt.max() - gt.min() # choose 1 if data in float type, 255 if data in int8
    return 20* np.log10(data_range) - 10*np.log10(mse)

def weights_init(m):
    if isinstance(m, torch.nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m,torch.nn.BatchNorm3d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
    
def vis_plot(tensor_img):
    fig,ax = plt.subplots(3,3,figsize=(10,10))
    axe = ax.ravel()
    img_thickness = int(tensor_img.size(2)/2)
    [axe[i].imshow(tensor_img[i,0,img_thickness].squeeze().detach().cpu().numpy(),cmap='gray') for i in range(tensor_img.size()[0])]
    
def make_coord(shape, ranges=None, flatten=True):
    """
    Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret