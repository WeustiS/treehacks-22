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
from n_people_head import CountPeopleHead

from LabelledDatasetBB import LabelledDatasetBB as LabelledDataset

wandb.init(project="treehacks_supr", entity="weustis")

efficient_transformer = Nystromformer(
    dim = 512,
    depth = 12,
    heads = 12,
    num_landmarks = 128
)

v = ViT(
    dim = 512,
    image_size = (576, 960),
    patch_size = 48,
    num_classes = 1,
    transformer = efficient_transformer
)
full = CountPeopleHead(encoder=v)

v.load_state_dict(torch.load("vit.pt"))
v.mlp_head = torch.nn.Identity()
v.eval()
print(v)
full = full.cuda()
full = torch.nn.DataParallel(full)

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.Resize((576,960))
])
dataset = LabelledDataset("tasks (4).json")# torchvision.datasets.ImageFolder('./data', transform=transform)
test_dataset = LabelledDataset("tasks (4).json", train=False)# torchvision.datasets.ImageFolder('./data', transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=2
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=2
)

mse = torch.nn.MSELoss()
total_epoch = 50

opt = torch.optim.Adam(v.parameters(), lr=1e-4)
# wandb.init(project="pulse-mae-nystrom", entity="weustis")

for epoch in tqdm(range(total_epoch)):
    loss = 0
    for batch, label in tqdm(dataloader):
        
        pred, enc = full(batch)
        
        #print(pred.shape, enc.shape)
        
        loss = mse(pred, label.cuda())
        wandb.log({
            "loss": loss.item(),
            "gt": label.squeeze(),
            "pred": pred.squeeze()
        })
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    for batch, label in tqdm(test_dataloader):
        
        pred, enc = full(batch)
        
        #print(pred.shape, enc.shape)
        
        loss = mse(pred, label.cuda())
        wandb.log({
            "test_loss": loss.item(),
            "test_gt": label.squeeze(),
            "test_pred": pred.squeeze()
        })
    opt.zero_grad()

        
    if epoch%10 ==9:
        torch.save(full.state_dict(), f'./models/treehacks-full-time_{epoch}.pt')