# import pandas as pd
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# import torch

# class ApneaDataset(Dataset):
#     def __init__(self, data_file, labels_file, seq_len):
#         self.data = pd.read_csv(data_file)
#         self.labels = pd.read_csv(labels_file)
#         self.seq_len = seq_len
#         self.num_sequences = len(self.data) // seq_len
#         data_file = '/home/ubuntu22/Time-Series-Library-main/datasets/Apnea/Apnea_23/01/PSG/psgresp20.csv'
#         labels_file = '/home/ubuntu22/Time-Series-Library-main/datasets/Apnea/Apnea_23/01/PSG/psgevents.csv'
#         seq_len = 600
#         Dataset = ApneaDataset(data_file, labels_file, seq_len, shuffle=True)
#         assert self.num_sequences == len(self.labels),"Number of sequences does not match number of labels"
    
#     def __len__(self):
#         return self.num_sequences
    
#     def __getitem__(self, idx):
#         start_idx = idx * self.seq_len
#         end_idx = start_idx + self.seq_len
#         sequence = self.data.iloc[start_idx:end_idx].values
#         label = self.labels.iloc[idx].value[0]
#         return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

import os
import pandas as pd
import torch 
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np

class Apnea_dataset(Dataset):
    def __init__(self, root_dir:str, seq_len:int):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.data_files = sorted([i for i in glob.glob(os.path.join(root_dir,'**','*.csv'), recursive=True)  if not 'events' in i])
        self.label_files = sorted([i for i in glob.glob(os.path.join(root_dir,'**','*.csv'), recursive=True)  if 'events' in i])
        self.data = [pd.read_csv(f) for f in self.data_files]
        self.data = np.vstack(self.data)[:,1:]
        self.label_raw = [pd.read_csv(f) for f in self.label_files]
        self.label_raw = np.vstack(self.label_raw)
        
        """
        process label , np.zeros((len(self.label,1)))
        if self.label_raw[:,1][i] == none, label[i]=0, else label[i] =1
        output self.label [n_label, 2]
        """
        self.label = np.zeros((len(self.label_raw), 2))
        for i in range(len(self.label)):
            if type(self.label_raw[:,1][i])==str:
                # if type ==str, label = 1, change to (n,2)
                self.label[i] = np.array([0,1])
            #     self.label[i] = 1
            else:
                self.label[i] = np.array([1,0])
            
        
    
        # for folder_name in sorted(os.listdir(root_dir)):
        #     folder_path = os.path.join(root_dir, folder_name)
        #     if os.path.isdir(folder_path):
        #         data_file = os.path.join(folder_path, 'psgresp20.csv')
        #         label_file = os.path.join(folder_path, 'psgevents.csv')
        #         if os.path.exists(data_file) and os.path.exists(label_file):
        #             self.data_files.append(data_file)
        #             self.label_files.append(label_file)
        
        self.num_seq = 0
        self.total_labels = 0
        # assert len(self.data_files) != len(self.label_files)
        # for i in range(len(self.data_files)):
        #     data_len = len(pd.read_csv(self.data_files[i])) // seq_len
        #     label_len = len(pd.read_csv(self.label_files[i]))
        #     self.num_seq += data_len
        #     self.total_labels += label_len
            
        #     assert data_len == label_len, f"Number of sequences ({data_length}) does not match number of labels ({label_len}) in file {data_file}"
        # assert self.num_seq == self.total_labels, "Total number of sequences does not match total number of labels"
        
    def __len__(self):
        return len(pd.read_csv(self.label_files[0]))
    
    def __getitem__(self, idx):
       # read from  data
        data = self.data[idx*self.seq_len:idx*self.seq_len +self.seq_len,]
        label = self.label[idx]
        return data, label
            

if __name__ == "__main__":
    # init dataset class
    # define the input
    root_dir = '/home/ubuntu22/Time-Series-Library-main/Time-series-self/datasets/Apnea/Apnea_23'
    seq_len = 600
    dataset = Apnea_dataset(root_dir, seq_len)
    dataloader = DataLoader(dataset, batch_size=8)
    x,y = next(iter(dataloader))
    pass