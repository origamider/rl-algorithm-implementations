import torch
from torchvision import transforms,datasets
from vae_model import VAE
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib

def main():
    checkpoint = torch.load("vae_latent2.pt",weights_only=False)
    latent_dim = checkpoint["latent_dim"]
    hidden_dim = checkpoint["hidden_dim"]
    input_dim = 784
    model = VAE(input_dim,hidden_dim,latent_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(torch.flatten)
    ])
    test_dataset = datasets.MNIST(
        root="./data/",
        train=False,
        transform=transform
    )
    all_labels = []
    all_x = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]) #(10000,784)
    all_labels = np.array([test_dataset[i][1] for i in range(len(test_dataset))]) #(10000,1)
    
    
    # ==============================
    # 実験2-A: 潜在空間の2D散布図
    # ==============================
    with torch.no_grad():
        all_mu, _ = model.encoder(all_x)
    
    all_mu = all_mu.numpy() #(10000,2)
    plt.figure(figsize=(10,8))
    
    for num in range(10):
        mask = all_labels == num
        plt.scatter(all_mu[mask,0],all_mu[mask,1],label=str(num))
    
    plt.legend()
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("実験2-A : 各数字における2次元潜在変数の可視化")
    
    # ==============================
    # 実験2-B: 0->7のモーフィング
    # ==============================
    
    z_start = torch.tensor(all_mu[all_labels == 0][0])
    z_end = torch.tensor(all_mu[all_labels == 7][0])
    
    n_steps = 10
    fig, axes = plt.subplots(1,n_steps,figsize=(15,3))
    with torch.no_grad():
        for i in range(n_steps):
            t = i/(n_steps-1)
            z_pos = z_start*(1-t) + z_end*t
            x = model.decoder(z_pos.unsqueeze(dim=0))
            
            axes[i].imshow(x.squeeze().view(28,28).numpy(),cmap="gray")
    plt.suptitle("実験2-B : 0から7までのモーフィング")
    plt.show()
    
    

if __name__ == "__main__":
    main()