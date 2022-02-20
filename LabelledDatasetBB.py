import os
import pandas as pd
from torchvision.io import read_image
from torchvision import transforms
import json 
import datetime
import torch


class LabelledDatasetBB(torch.utils.data.Dataset):
    def __init__(self, annotations_file, train=True, transform=None):
        self.file = annotations_file
        self.transform = transforms.Compose([
            # you can add other transformations in this list
          #   transforms.ToTensor(),
          #  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize((576,960))
        ])
        f = open(self.file)
        self.data = json.load(f)
        
        self.fnames = []
        for item in self.data:
            try:
                file_name = item['metadata']['fname']
                print(f"appending {file_name}")
                
                n_ppl = len(item['response']['annotations'])
                print(f"appending {file_name} \\ {n_ppl}")
                self.fnames.append((file_name, n_ppl))
            except:
                print(item)
                continue
        cutoff = int(len(self.fnames)*.8)
        if train:
            self.fnames = self.fnames[:cutoff]
        else:
            self.fnames = self.fnames[cutoff:]
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        
        file_name, ppl = self.fnames[idx]

        img_data = read_image(f"data/imgs/{file_name}.jpg").float()
        
        img_data = self.transform(img_data)
        
        time = int(file_name.split("_")[1])
        dt = datetime.datetime.fromtimestamp(time, datetime.timezone(datetime.timedelta(hours=-6)))
        
        time_of_day = dt.time().hour + dt.time().minute/60
        
        label = torch.tensor(ppl).float()

        
        return img_data, label