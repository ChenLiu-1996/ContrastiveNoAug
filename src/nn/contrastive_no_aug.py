import numpy as np
import torch
from scipy.stats import wasserstein_distance


class ContrastiveNoAugLoss(torch.nn.Module):

    def __init__(self, temperature: float = 0.5):
        super(ContrastiveNoAugLoss, self).__init__()
        self.temperature = temperature
        self.epsilon = 1e-7

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
        z_cosd = z_cos_sim.max() - z_cos_sim

        x_np = x.cpu().detach().numpy()
        x_emd = np.empty((B, B))
        for i in range(B - 1):
            x_emd[i, i] = 0
            for j in range(i + 1, B):
                x_i = x_np[i, ...].reshape(-1)
                x_j = x_np[j, ...].reshape(-1)
                x_emd[i, j] = x_emd[j, i] = wasserstein_distance(x_i, x_j)
        x_emd = torch.from_numpy(x_emd).type(torch.FloatTensor).to(x.device)

        # Loss function to preserve rank-order on embedding distances.
        loss = torch.nn.functional.mse_loss(x_emd, z_cosd)

        print(loss.item())
        return loss
