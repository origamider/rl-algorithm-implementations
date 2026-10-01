import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class MDNRNN(nn.Module):
    """
    MDN-RNN。
    actions:(seq_len, batch_size, action_dim)
    zs:(seq_len, batch_size, latent_dim)
    
    (seq_len, batch_size, action_dim + latent_dim)
    -LSTM->(seq_len, batch_size, hidden_dim)
    -MDN->(seq_len, batch_size, (2*latent_dim+1)*gaussians_dim)
    
    MDNが予測するmu,sigma,piについて
    p(z|x)=sigma pi * N(z;mu,sigma)で、
    zは(latent_dim,)より、
    mu,sigmaは(gaussians_dim,latent_dim)。
    論文より、sigmaはdiagonal covariance matrix(対角共分散行列)としている。
    piは(gaussians_dim,)
    つまり、MDNの最終出力は、(2*latent_dim+1)*gaussians_dim
    """
    
    def __init__(self, latent_dim, actions_dim, hidden_dim, gaussians_dim):
        super().__init__()
        self.rnn = nn.LSTM(input_size=actions_dim+latent_dim,hidden_size=hidden_dim)
        self.mdn_layer = nn.Linear(hidden_dim, (2*latent_dim+1)*gaussians_dim)
        self.latent_dim = latent_dim
        self.actions_dim = actions_dim
        self.hidden_dim = hidden_dim
        self.gaussians_dim = gaussians_dim
    
    def forward(self, actions, zs):
        seq_len, batch_size = actions.shape[0], actions.shape[1]
        input = torch.cat([actions, zs], dim=2)
        out, _ = self.rnn(input)
        gmm_output = self.mdn_layer(out)
        
        base = self.latent_dim*self.gaussians_dim
        mus = gmm_output[:, :, :base]
        mus = mus.reshape(seq_len, batch_size, self.gaussians_dim, self.latent_dim)
        sigmas = gmm_output[:, :, base:2*base]
        sigmas = sigmas.reshape(seq_len, batch_size, self.gaussians_dim, self.latent_dim)
        sigmas = torch.exp(sigmas) # sigma>0処理
        logpi = torch.log_softmax(gmm_output[:, :, 2*base:], dim=-1)
        
        return mus, sigmas, logpi

def gmm_loss(mus, sigmas, logpi, zs):
    """
    mus shape:(seq_len, batch_size, gaussians_dim, latent_dim)
    sigmas shape:(seq_len, batch_size, gaussians_dim, latent_dim)
    zs shape:(seq_len, batch_size, latent_dim)
    
    Loss = - 1/(seq_len*batch_size) * (sigma_i sigma_j log p(z_(t+1)|z_t,a_t,h_t))
    """
    
    zs = zs.unsqueeze(dim=-2) # shape: (seq_len, batch_size, 1, latent_dim)
    normal_dist = Normal(mus, sigmas)
    zs_log_probs = normal_dist.log_prob(zs) # shape: (seq_len, batch_size, gaussians_dim, latent_dim)
    zs_log_probs = torch.sum(zs_log_probs, dim=-1) # shape: (seq_len, batch_size, gaussians_dim)
    zs_log_probs = logpi + zs_log_probs # shape: (seq_len, batch_size, gaussians_dim)
    max_log_probs = torch.max(zs_log_probs, dim=-1, keepdim=True).values # shape: (seq_len, batch_size, 1)
    zs_log_probs = zs_log_probs - max_log_probs
    zs_probs = torch.exp(zs_log_probs)
    probs = torch.sum(zs_probs, dim=-1) # shape: (seq_len, batch_size)
    log_prob = max_log_probs.squeeze(dim=-1) + torch.log(probs)
    return -torch.mean(log_prob)
