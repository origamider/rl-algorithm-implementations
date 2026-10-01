import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    VAE Encoder
    ->入力xからmu,sigmaを推測する。
    入力xの形状は64*64*3。
    conv1について
    out_channels=32,kernel_size=4,stride=2
    output shape:
    conv1
    OH = (H + 2P - FH) / S + 1 = (64 - 4) / 2 + 1 = 31
    conv2
    OH = (H + 2P - FH) / S + 1 = (31 - 4) / 2 + 1 = 14
    conv3
    OH = (H + 2P - FH) / S + 1 = (14 - 3) / 2 + 1 = 6
    conv4
    OH = (H + 2P - FH) / S + 1 = (6 - 3) / 2 + 1 = 2
    (NCHW形式で書きます)
    形状変化
    (N,3,64,64)
    -conv1->(N,32,31,31)
    -conv2->(N,64,14,14)
    -conv3->(N,128,6,6)
    -conv4->(N,256,2,2)
    -resize->(N,256*2*2)
    -mu,sigma->(N,32)
    """
    def __init__(self, img_channels=3, latent_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2)
        self.relu = nn.ReLU()
        self.fc_mu = nn.Linear(2*2*256, latent_dim)
        self.fc_logsigma = nn.Linear(2*2*256, latent_dim)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = x.reshape(x.shape[0],-1)
        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)
        sigma = torch.exp(0.5*logsigma)

        return mu, sigma

class Decoder(nn.Module):
    """
    VAE Decoder
    -> 入力zからx^を推測する。
    output shape:
    deconv1
    OH = (H - 1) * S + FH = (1 - 1) * 2 + 5 = 5
    deconv2
    OH = (H - 1) * S + FH = (5 - 1) * 2 + 5 = 13
    deconv3
    OH = (H - 1) * S + FH = (13 - 1) * 2 + 6 = 30
    deconv4
    OH = (H - 1) * S + FH = (30 - 1) * 2 + 6 = 64
    形状変化
    (N,32)
    -fc1->(N,1024)
    -reshape->(N,1024,1,1)
    -deconv1->(N,128,5,5)
    -deconv2->(N,64,13,13)
    -deconv3->(N,32,30,30)
    -deconv4->(N,3,64,64)
    """
    def __init__(self, img_channels, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 1024)
        self.deconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=6, stride=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, z):
        z = self.fc1(z)
        z = z.reshape(z.shape[0],z.shape[1],1,1)
        z = self.relu(self.deconv1(z))
        z = self.relu(self.deconv2(z))
        z = self.relu(self.deconv3(z))
        x_hat = self.sigmoid(self.deconv4(z))
        return x_hat

"""
変数変換トリック。
本来、N(z;mu,sigma^2 I)でzをサンプリングするが、これだと逆伝播ができない。
そこで、N(z;mu,sigma^2 I)を、
ε ~ N(ε;0,I)
z = mu + sigma * ε
とおくことで、mu,sigmaノードに勾配を流すことができる。
"""
def reparameterize(mu, sigma):
    epsilon = torch.randn_like(sigma)
    z = mu + epsilon * sigma
    return z

"""
World Modelの論文に書かれた構成で構築。
"""

class VAE(nn.Module):
    def __init__(self, img_channels, latent_dim):
        super().__init__()
        self.encoder = Encoder(img_channels, latent_dim)
        self.decoder = Decoder(img_channels, latent_dim)
    
    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = reparameterize(mu, sigma)
        x_hat = self.decoder(z)
        return x_hat
    
    def get_loss(self, x):
        """
        損失関数
        Loss = sigma_d=1ToD (x_d - x_hat_d)^2 - sigma_h=1ToH (1 + log sigma_h^2 - mu_h^2 - sigma_h^2)
        """
        
        mu, sigma = self.encoder(x)
        z = reparameterize(mu, sigma)
        x_hat = self.decoder(z)
        
        N = len(x)

        mse_loss = nn.MSELoss(reduction = 'sum')
        L1 = mse_loss(x, x_hat)
        L2 = - torch.sum(1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2)
        
        return (L1 + L2) / N

