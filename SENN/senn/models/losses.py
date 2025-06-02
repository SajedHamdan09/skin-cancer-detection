import torch
import torch.nn.functional as F

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class cancer_robustness_loss(nn.Module):
    def __init__(self, num_features, num_concepts, num_classes):
        super().__init__()
        num_features = 147456
        self.projection = nn.Linear(num_features, num_concepts * num_classes)

    def forward(self, x, aggregates, concepts, relevances):
        """
        x: (batch_size, num_features) flattened input
        aggregates: (batch_size, num_classes, concept_dim)
        concepts: (batch_size, num_concepts, concept_dim)
        relevances: (batch_size, num_concepts, num_classes)
        """

        batch_size = x.size(0)
        device = x.device

        # Ensure input requires gradients for autograd
        x.requires_grad_(True)

        grad_tensor = torch.ones(batch_size, aggregates.size(1)).to(device)
        J_yx = torch.autograd.grad(outputs=aggregates, inputs=x, 
                                   grad_outputs=grad_tensor, create_graph=True, only_inputs=True)[0]

        # print("J_yx shape before flatten:", J_yx.shape)
        # J_yx = J_yx.view(batch_size, -1)
        # print("J_yx shape after flatten:", J_yx.shape)

        # print("Shape after: ", J_yx.shape)  # confirm last dimension matches num_features


        # Flatten J_yx if it has more than 2 dimensions (i.e., has spatial dims)
        if J_yx.dim() > 2:
            J_yx = J_yx.view(batch_size, -1)

        # Project flattened J_yx to (batch_size, num_concepts * num_classes)
        projected = self.projection(J_yx)

        # Reshape to (batch_size, num_concepts, num_classes)
        projected = projected.view(batch_size, relevances.size(1), relevances.size(2))

        # Compute difference and Frobenius norm
        robustness_loss = projected - relevances
        return robustness_loss.norm(p='fro')



def compas_robustness_loss(x, aggregates, concepts, relevances):
    """Computes Robustness Loss for the Compas data
    
    Formulated by Alvarez-Melis & Jaakkola (2018)
    [https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks.pdf]
    The loss formulation is specific to the data format
    The concept dimension is always 1 for this project by design

    Parameters
    ----------
    x            : torch.tensor
                 Input as (batch_size x num_features)
    aggregates   : torch.tensor
                 Aggregates from SENN as (batch_size x num_classes x concept_dim)
    concepts     : torch.tensor
                 Concepts from Conceptizer as (batch_size x num_concepts x concept_dim)
    relevances   : torch.tensor
                 Relevances from Parameterizer as (batch_size x num_concepts x num_classes)
   
    Returns
    -------
    robustness_loss  : torch.tensor
        Robustness loss as frobenius norm of (batch_size x num_classes x num_features)
    """
    batch_size = x.size(0)
    num_classes = aggregates.size(1)

    grad_tensor = torch.ones(batch_size, num_classes).to(x.device)
    J_yx = torch.autograd.grad(outputs=aggregates, inputs=x, \
                               grad_outputs=grad_tensor, create_graph=True, only_inputs=True)[0]
    # bs x num_features -> bs x num_features x num_classes
    J_yx = J_yx.unsqueeze(-1)

    # J_hx = Identity Matrix; h(x) is identity function
    robustness_loss = J_yx - relevances

    return robustness_loss.norm(p='fro')


def mnist_robustness_loss(x, aggregates, concepts, relevances):
    """Computes Robustness Loss for MNIST data
    
    Formulated by Alvarez-Melis & Jaakkola (2018)
    [https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks.pdf]
    The loss formulation is specific to the data format
    The concept dimension is always 1 for this project by design

    Parameters
    ----------
    x            : torch.tensor
                 Input as (batch_size x num_features)
    aggregates   : torch.tensor
                 Aggregates from SENN as (batch_size x num_classes x concept_dim)
    concepts     : torch.tensor
                 Concepts from Conceptizer as (batch_size x num_concepts x concept_dim)
    relevances   : torch.tensor
                 Relevances from Parameterizer as (batch_size x num_concepts x num_classes)
   
    Returns
    -------
    robustness_loss  : torch.tensor
        Robustness loss as frobenius norm of (batch_size x num_classes x num_features)
    """
    # concept_dim is always 1
    concepts = concepts.squeeze(-1)
    aggregates = aggregates.squeeze(-1)

    batch_size = x.size(0)
    num_concepts = concepts.size(1)
    num_classes = aggregates.size(1)

    # Jacobian of aggregates wrt x
    jacobians = []
    for i in range(num_classes):
        grad_tensor = torch.zeros(batch_size, num_classes).to(x.device)
        grad_tensor[:, i] = 1.
        j_yx = torch.autograd.grad(outputs=aggregates, inputs=x, \
                                   grad_outputs=grad_tensor, create_graph=True, only_inputs=True)[0]
        # bs x 1 x 28 x 28 -> bs x 784 x 1
        jacobians.append(j_yx.view(batch_size, -1).unsqueeze(-1))
    # bs x num_features x num_classes (bs x 784 x 10)
    J_yx = torch.cat(jacobians, dim=2)

    # Jacobian of concepts wrt x
    jacobians = []
    for i in range(num_concepts):
        grad_tensor = torch.zeros(batch_size, num_concepts).to(x.device)
        grad_tensor[:, i] = 1.
        j_hx = torch.autograd.grad(outputs=concepts, inputs=x, \
                                   grad_outputs=grad_tensor, create_graph=True, only_inputs=True)[0]
        # bs x 1 x 28 x 28 -> bs x 784 x 1
        jacobians.append(j_hx.view(batch_size, -1).unsqueeze(-1))
    # bs x num_features x num_concepts
    J_hx = torch.cat(jacobians, dim=2)

    # bs x num_features x num_classes
    robustness_loss = J_yx - torch.bmm(J_hx, relevances)

    return robustness_loss.norm(p='fro')


def BVAE_loss(x, x_hat, z_mean, z_logvar):
    """ Calculate Beta-VAE loss as in [1]

    Parameters
    ----------
    x : torch.tensor
        input data to the Beta-VAE

    x_hat : torch.tensor
        input data reconstructed by the Beta-VAE

    z_mean : torch.tensor
        mean of the latent distribution of shape
        (batch_size, latent_dim)

    z_logvar : torch.tensor
        diagonal log variance of the latent distribution of shape
        (batch_size, latent_dim)

    Returns
    -------
    loss : torch.tensor
        loss as a rank-0 tensor calculated as:
        reconstruction_loss + beta * KL_divergence_loss

    References
    ----------
        [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
        a constrained variational framework." (2016).
    """
    # recon_loss = F.binary_cross_entropy(x_hat, x.detach(), reduction="mean")
    recon_loss = F.mse_loss(x_hat, x.detach(), reduction="mean")
    kl_loss = kl_div(z_mean, z_logvar)
    return recon_loss, kl_loss

def mse_l1_sparsity(x, x_hat, concepts, sparsity_reg):
    """Sum of Mean Squared Error and L1 norm weighted by sparsity regularization parameter

    Parameters
    ----------
    x : torch.tensor
        Input data to the encoder.
    x_hat : torch.tensor
        Reconstructed input by the decoder.
    concepts : torch.Tensor
        Concept (latent code) activations.
    sparsity_reg : float
        Regularizer (xi) for the sparsity term.

    Returns
    -------
    loss : torch.tensor
        Concept loss
    """
    return F.mse_loss(x_hat, x.detach()) + sparsity_reg * torch.abs(concepts).sum()


def kl_div(mean, logvar):
    """Computes KL Divergence between a given normal distribution
    and a standard normal distribution

    Parameters
    ----------
    mean : torch.tensor
        mean of the normal distribution of shape (batch_size x latent_dim)

    logvar : torch.tensor
        diagonal log variance of the normal distribution of shape (batch_size x latent_dim)

    Returns
    -------
    loss : torch.tensor
        KL Divergence loss computed in a closed form solution
    """
    batch_loss = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).mean(dim=0)
    loss = batch_loss.sum()
    return loss


def zero_loss(*args, **kwargs):
    """Dummy loss that always returns zero.

        Parameters
        ----------
        args : list
            Can take any number of positional arguments (without using them).
        kwargs : dict
            Can take any number of keyword arguments (without using them).

        Returns
        -------
        loss : torch.tensor
            torch.tensor(0)
        """
    return torch.tensor(0)
