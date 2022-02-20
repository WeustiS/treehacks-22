import wandb
import torch
from vit_pytorch.efficient import ViT
from nystrom_attention import Nystromformer
from vit_pytorch import MAE
import torchvision 
from tqdm import tqdm
from torchvision import transforms
from torchsummary import summary
import wandb 
import numpy as np 
from torch.optim.lr_scheduler import LambdaLR
wandb.init(project="treehacks", entity="weustis")

efficient_transformer = Nystromformer(
    dim = 1024,
    depth = 10,
    heads = 10,
    num_landmarks = 256
)

v = ViT(
    dim = 1024,
    image_size = (576, 960),
    patch_size = 48,
    num_classes = 1,
    transformer = efficient_transformer
)
mae = MAE(
    encoder = v,
    masking_ratio = 0.7,   # the paper recommended 75% masked patches
    decoder_dim = 512,      # paper showed good results with just 512
    decoder_depth = 6       # anywhere from 1 to 8
)

print(sum(p.numel() for p in mae.parameters() if p.requires_grad), 'params')

mae = mae.cuda()
mae = torch.nn.DataParallel(mae, device_ids=[0,1,4,6]) 

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor(),
    transforms.Resize((576,960))
])
dataset = torchvision.datasets.ImageFolder('./data', transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=48,
    shuffle=True,
    pin_memory=True,
    num_workers=20,
    persistent_workers=False,
    prefetch_factor=2
)



total_epoch = 90

opt = torch.optim.Adam(mae.parameters(), lr=5e-3)
# wandb.init(project="pulse-mae-nystrom", entity="weustis")

def warmup_cooldown(epoch):
    lr = 5e-3# (0.95**epoch) / (40*4)i
    if epoch>45:
        lr=2e-3
    wandb.log({"lr": lr})
    return lr

lr_lambda = lambda epoch: warmup_cooldown(epoch)

scheduler = LambdaLR(opt, lr_lambda=[lr_lambda])

for epoch in tqdm(range(total_epoch)):
    loss = 0
    for batch, _ in tqdm(dataloader):
        
        loss = mae(batch)
#         wandb.log({
#             'loss': loss.sum().item()
#         })
        wandb.log({"loss": loss.sum().item()})
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
        
        
    if epoch%4==1:
        torch.save(mae.state_dict(), f'./models/treehacks-mae-mnn_{epoch}.pt')
        torch.save(v.state_dict(), f'./models/treehacks-vit-mnn_{epoch}.pt')
        torch.save(opt.state_dict(), f'./models/treehacks-opt-mnn_{epoch}.pt')
    scheduler.step()
