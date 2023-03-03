import numpy as np
# import ot
import torch
from skimage.metrics import structural_similarity


def ssim(a: np.array, b: np.array, **kwargs) -> float:
    '''
    Please make sure the data are provided in [H, W, C] shape.
    '''
    assert a.shape == b.shape

    H, W = a.shape[:2]

    if min(H, W) < 7:
        win_size = min(H, W)
        if win_size % 2 == 0:
            win_size -= 1
    else:
        win_size = None

    if len(a.shape) == 3:
        channel_axis = -1
    else:
        channel_axis = None

    return structural_similarity(a,
                                 b,
                                 channel_axis=channel_axis,
                                 win_size=win_size,
                                 **kwargs)


class ContrastiveNoAugLoss(torch.nn.Module):

    def __init__(self):
        super(ContrastiveNoAugLoss, self).__init__()

    def forward(self, z: torch.Tensor, x: torch.Tensor):
        '''
        z: embedding vectors
        x: images
        '''
        B, _ = z.shape
        z = torch.nn.functional.normalize(input=z, p=2, dim=1)

        # Wavelet transform on images.

        # Compute cosine distances on the embeddings and EMD on the images.
        z_cos_sim = torch.matmul(z, z.T)

        # x_np = x.cpu().detach().numpy().reshape(B, -1).astype(np.float64)
        x_np = np.moveaxis(x.cpu().detach().numpy(), 1, -1).astype(np.float64)
        # x_np = (x_np - x_np.min()) / (x_np.max() - x_np.min())
        x_ssim = np.empty((B, B))
        for i in range(B - 1):
            x_ssim[i, i] = 1
            for j in range(i + 1, B):
                # x_i = x_np[i] / x_np[i].sum()
                # x_j = x_np[j] / x_np[j].sum()
                # x_emd[i, j] = x_emd[j, i] = ot.emd2_1d(x_i, x_j)
                x_ssim[i, j] = x_ssim[j, i] = ssim(a=x_np[i], b=x_np[j], data_range=2)

        x_ssim = torch.from_numpy(x_ssim).type(torch.FloatTensor).to(x.device)

        # TODO: Loss function to preserve rank-order on embedding distances.
        loss = torch.nn.functional.mse_loss(x_ssim, z_cos_sim)

        return loss
