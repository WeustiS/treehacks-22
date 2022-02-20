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
    dim = 512,
    depth = 8,
    heads = 8,
    num_landmarks = 64
)

v = ViT(
    dim = 512,
    image_size = 512,
    patch_size = (32,64),
    num_classes = 1,
    transformer = efficient_transformer
)
mae = MAE(
    encoder = v,
    masking_ratio = 0.6,   # the paper recommended 75% masked patches
    decoder_dim = 256,      # paper showed good results with just 512
    decoder_depth = 4       # anywhere from 1 to 8
)

print(sum(p.numel() for p in mae.parameters() if p.requires_grad), 'params')

mae = mae.cuda()
mae = torch.nn.DataParallel(mae) 

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor(),
    transforms.RandomResizedCrop((512,512))
])
dataset = torchvision.datasets.ImageFolder('./data', transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    pin_memory=True,
    num_workers=16,
    persistent_workers=True,
    prefetch_factor=2
)

mem_params = sum([param.nelement()*param.element_size() for param in mae.parameters()])
mem_bufs = sum([buf.nelement()*buf.element_size() for buf in mae.buffers()])
mem = mem_params + mem_bufs
print("Mem!", mem)

summary(v, input_size=(3, 512, 512))

total_epoch = 120

opt = torch.optim.Adam(mae.parameters(), lr=5e-4)
# wandb.init(project="pulse-mae-nystrom", entity="weustis")

def warmup_cooldown(epoch):
    if epoch < 10:
        lr = (1.03**epoch - .95)/1000
    elif epoch < total_epoch-15:
        lr_adj = epoch%3  # 0,1,2
        lr = 5e-4 + lr_adj * 5e-5
    else:
        tmp = total_epoch-epoch  # 4,3,2,1,or 0
        lr = 1e-6 + 5e-4*tmp/15
    print(f"LR_{epoch}:", lr)
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
        
        
    if epoch%10 ==9:
        torch.save(mae.state_dict(), f'./models/treehacks-mae-ny_{epoch}.pt')
        torch.save(v.state_dict(), f'./models/treehacks-vit-ny_{epoch}.pt')
        torch.save(opt.state_dict(), f'./models/treehacks-opt-ny_{epoch}.pt')
    scheduler.step()