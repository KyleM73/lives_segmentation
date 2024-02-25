import numpy as np
import pandas as pd
import torch
import os

LIVES_DATA_PATH = os.path.dirname(os.path.realpath(__file__))

def make_dataset(k: int = 5) -> int:
    csv_dir = LIVES_DATA_PATH + "/csv"
    raw_dir = LIVES_DATA_PATH + "/raw"
    
    counter = 0
    for dir in os.listdir(csv_dir):
        if dir == ".DS_Store": continue
        for run in os.listdir(os.path.join(csv_dir,dir)):
            if run == ".DS_Store": continue
            for file in os.listdir(os.path.join(os.path.join(csv_dir,dir),run)):
                path = os.path.join(os.path.join(csv_dir,dir),run) + "/" + file
                if "pose" in file:
                    pose = pd.read_csv(path,usecols=[1,2,3],header=None).to_numpy()
                elif "laser" in file:
                    raw_label = pd.read_csv(path,skiprows=lambda i:(i+1)%3,header=None).to_numpy() 
                    label = np.where(raw_label<1,-1.,1.) #two class
                    lidar = pd.read_csv(path,skiprows=lambda i:(i+2)%3,header=None).to_numpy() 
                #elif "expected" in file:
                #    expected_label = pd.read_csv(path,skiprows=lambda i:(i+1)%3,header=None).to_numpy() 
                #    expected_lidar = pd.read_csv(path,skiprows=lambda i:(i+2)%3,header=None).to_numpy()
            for i in range(k,pose.shape[0]):
                p = pose[i-k:i,:]
                x = lidar[i-k:i,:]
                y = label[i-k:i,:]
                os.makedirs(raw_dir+"/{}".format(counter))
                np.save(raw_dir+"/{}/p.npy".format(counter),p)
                np.save(raw_dir+"/{}/x.npy".format(counter),x)
                np.save(raw_dir+"/{}/y.npy".format(counter),y)
                counter += 1
    return counter

def clear_dataset(dir: str) -> None:
    for sub_dir in os.listdir(dir):
        for file in os.listdir(os.path.join(dir,sub_dir)):
            os.remove(os.path.join(os.path.join(dir,sub_dir),file))
        os.rmdir(os.path.join(dir,sub_dir))

class LidarDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_dir: str,
                 count: int = None,
                 device: str = "cpu") -> None:
        self.data_dir = data_dir
        if isinstance(count,int):
            self.count = count
        else:
            self.count = len(os.listdir(self.data_dir))
        self.device = device
                
    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = torch.from_numpy(np.load(self.data_dir+"/{}/p.npy".format(idx), allow_pickle=False)).to(self.device,dtype=torch.float32)
        x = torch.from_numpy(np.load(self.data_dir+"/{}/x.npy".format(idx), allow_pickle=False)).to(self.device,dtype=torch.float32)
        y = torch.from_numpy(np.load(self.data_dir+"/{}/y.npy".format(idx), allow_pickle=False)).to(self.device,dtype=torch.float32)
        return p,x,y

def make_batched_dataset(
        batch_size: int = 4096,
        k: int = 5,
        clear: bool = True
        ) -> None:
    raw_dir = LIVES_DATA_PATH + "/raw"
    batched_dir = LIVES_DATA_PATH + "/batched"
    if clear:
        print("[Clearing Existing Dataset]")
        try:
            clear_dataset(raw_dir)
            clear_dataset(batched_dir)
            print("[Finished Clearing Existing Dataset]")
        except FileNotFoundError:
            print("Warning: previous dataset creation may have failed")
    print("[Make Unbatched Dataset]")
    make_dataset(k)
    print("[Finished Unbatched Dataset]")
    dataset = LidarDataset(raw_dir,device="cpu")
    loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size)
    print("[Make Batched Dataset]")
    for i, data in enumerate(loader):
        p,x,y = data
        os.makedirs(batched_dir+"/{}".format(i))
        np.save(batched_dir+"/{}/p.npy".format(i),p.cpu().numpy())
        np.save(batched_dir+"/{}/x.npy".format(i),x.cpu().numpy())
        np.save(batched_dir+"/{}/y.npy".format(i),y.cpu().numpy())
    print("[Finished Batched Dataset]")
