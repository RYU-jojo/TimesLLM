import os
import torch
import numpy as np
import random
import argparse
import sys
import torch.fft
from sklearn.metrics import f1_score

from torch.utils.data import DataLoader
from models.timesllm import Model
import torch.nn.functional as F
from data_loader.Apnea_dataloader import Apnea_dataset

class Trainer():
    def __init__(self, model, trainloader, testloader, args, device):
        self.model = model
        self.dataloader = trainloader
        self.testloader = testloader
        self.device = device
        self.args = args
        # loss
        self.loss_mse = torch.nn.MSELoss()
        self.loss_bce = torch.nn.BCEWithLogitsLoss()
        self.loss_ce = torch.nn.CrossEntropyLoss()
        # optimizer
        trained_parameters = []
        for p in self.model.parameters():
            if p.requires_grad is True:
                trained_parameters.append(p)
        self.optimizer = torch.optim.Adam(params=trained_parameters,lr =args.learning_rate)
        self.output_path = './output'

    def train(self):
        for epoch in range(self.args.train_epochs):
            loss_sum = 0
            iter = 0
            for i, (data, label) in enumerate(self.dataloader):
                data, label = data.to(device), label.to(device)
                iter +=1
                self.optimizer.zero_grad()
                pred = self.model(data.float())
                if self.args.task_name == 'classification':
                    loss = self.loss_ce(pred, label)
                else: 
        
                    loss = self.loss_mse(pred, label)
                loss.backward()
                loss_sum += loss
                self.optimizer.step()
            print('epoch {}: {}'.format(epoch,loss_sum/iter))
            # self.test()
    
    def test(self):
        f1_sum = 0  
        iter = 0   
        for  i, (data,label) in enumerate(self.testloader):   
            with torch.no_grad():
                iter +=1
                data = data.to(self.device)
                pred = self.model(data.float())                
                pred = pred.softmax(dim=1).cpu().numpy()
                
                pred = np.around(pred, 0).astype(int)
                # f1 = f1_score(pred, label)
                # f1_sum+=f1
        print('f1 score: {}'.format(f1_sum/iter))
                
    def load_checkpoint():
        pass
        
    def save_checkpoint():
        pass
    
    def adjust_lr():
        pass
            
if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str,  default='classification',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int,  default=1, help='status')
    parser.add_argument('--model_id', type=str,  default='test', help='model id')
    parser.add_argument('--model_comment', type=str, default='none', help='prefix when saving test results')
    parser.add_argument('--model', type=str,  default='Autoformer',
                        help='model name, options: [Autoformer, DLinear]')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--use_gpu', type=int, default=1, help='use gpu or not')
    parser.add_argument('--use_multi_gpu', type=int, default=0, help='use multi-gpu or not')
    # data loader
    parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; '
                            'M:multivariate predict multivariate, S: univariate predict univariate, '
                            'MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--loader', type=str, default='modal', help='dataset type')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, '
                            'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                            'you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--enc_in', type=int, default=2, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=2, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=2, help='output size')
    parser.add_argument('--d_model', type=int, default=8, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--prompt_domain', type=int, default=0, help='')
    parser.add_argument('--llm_model', type=str, default='BERT-base', help='LLM model') # LLAMA, GPT2, BERT
    parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768


    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--percent', type=int, default=100)

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)
    
    device = 'cuda:0'
    # load data
    # trainset = ApenaDataset(root_dir=args.root_path, mode='train') 
    # train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=False)
    # testset = ApenaDataset(root_dir=args.root_path, mode='test')
    # test_loader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False)
    root_dir = 'dataset/Apnea_23'
    seq_len = 600
    dataset = Apnea_dataset(root_dir, seq_len)
    train_size =int(0.8*len(dataset))
    test_size = len(dataset)-train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)
    # load model
    timesllm = Model(configs=args)
    timesllm.to(device)
    # train model
    trainer = Trainer(model=timesllm, trainloader=train_loader, testloader=test_loader,args=args, device=device)
    trainer.train()
    
    