"""
VAE model definition
"""


import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2_logvar = nn.Linear(hidden_dim, latent_dim)
        self.l2_mu = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        tmp = self.relu(self.l1(x))
        mu = self.l2_mu(tmp)
        logvar = self.l2_logvar(tmp)
        sigma = torch.exp(0.5*logvar)
        return mu,sigma

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.l1 = nn.Linear(latent_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        res = self.relu(self.l1(x))
        res = self.sigmoid(self.l2(res))
        return res

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
    
    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        z = mu + eps*sigma
        return z
    
    def get_loss(self, x):
        mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        x_hat = self.decoder(z)
        
        batch_size = len(x)
        L1 = nn.MSELoss(reduction='sum')(x_hat, x)
        L2 = - torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)
        return (L1+L2)/batch_size

    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        x_hat = self.decoder(z)
        return x_hat, mu, sigma