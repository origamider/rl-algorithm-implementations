from vae_model import VAE
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torchvision import datasets, transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim",type=int,default=20)
    parser.add_argument("--hidden_dim",type=int,default=100)
    parser.add_argument("--num_epochs",type=int,default=30)
    parser.add_argument("--learning_rate",type=float,default=3e-4)
    parser.add_argument("--batch_size",type=int,default=32)
    args = parser.parse_args()
    
    input_dim = 784
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(torch.flatten)
    ])
    
    train_dataset = datasets.MNIST(root="./data/",train=True,download=True,transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    
    model = VAE(input_dim,args.hidden_dim,args.latent_dim)
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
    
    for epoch in range(args.num_epochs):
        ct = 0
        loss_sum = 0
        
        for x, x_label in train_loader:
            optimizer.zero_grad()
            loss = model.get_loss(x)
            loss.backward()
            optimizer.step()
            ct += 1
            loss_sum += loss.item()
        
        loss_avg = loss_sum / ct
        print(f"Epoch {epoch+1}/{args.num_epochs}, Loss:{loss_avg}")
    
    save_path = f"vae_latent{args.latent_dim}.pt"
    torch.save({
        "model_state_dict" : model.state_dict(),
        "latent_dim" : args.latent_dim,
        "hidden_dim" : args.hidden_dim,
    }, save_path)
    print(f"「{save_path}」として、モデルを保存しました。")

if __name__ == "__main__":
    main()