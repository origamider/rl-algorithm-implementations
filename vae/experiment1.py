import matplotlib.pyplot as plt
import argparse
import torch
from vae_model import VAE
from torchvision import datasets, transforms
import japanize_matplotlib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim',type=int,default=20)
    args = parser.parse_args()
    
    input_dim = 784
    checkpoint = torch.load(f"vae_latent{args.latent_dim}.pt",weights_only=False)
    latent_dim = checkpoint['latent_dim']
    hidden_dim = checkpoint['hidden_dim']
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
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False
    )
    
    with torch.no_grad():
        x_test, x_label = next(iter(test_loader))
        x_test10 = x_test[0:10]
        print(x_test10.shape)
        x_hat10, _, _ = model(x_test10)
    
    fig, axes = plt.subplots(2,10,figsize=(15,5))
    
    for i in range(10):
        axes[0,i].imshow(x_test10[i].view(28,28).numpy(),cmap="gray")
        axes[1,i].imshow(x_hat10[i].view(28,28).numpy(),cmap="gray")
        if i == 0:
            axes[0,i].set_title("元の画像")
            axes[1,i].set_title("復元後の画像")
    plt.suptitle("元の画像と復元後の画像の比較")
    plt.show()

if __name__ == "__main__":
    main()