import torch
import time
import ast
import numpy as np
from datetime import datetime


class training():
    def __init__(self, keys, last_auth,url,refresh, meshnet,comp, runid, dice, loader, modelAE, dbfile, dist,l_r=0.003125, classes=3, epochs = 10, cubes=1, label = 'GWlabels' ):
        self.keys = keys
        self.refresh = refresh
        self.last_auth = last_auth
        self.url = url
        self.dice = dice
        self.runid = runid
        self.dist = dist
        self.comp = comp
        self.cubes = cubes
        self.classes = classes
        self.shape = 256 // self.cubes
        self.epochs = epochs
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.lr = l_r
        self.train, self.valid, self.test = loader.Scanloader(dbfile, label_type=label, num_cubes=self.cubes).get_loaders()
        self.model = meshnet.enMesh_checkpoint(1, self.classes, 1, modelAE).to(self.device, dtype=torch.float32)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=l_r)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
          self.optimizer, 
          max_lr=self.lr, 
          div_factor=100,
          pct_start=0.2,
          steps_per_epoch=len(self.train),
          epochs=self.epochs)

    def train_f(self):
        self.dist.activate_user(self.url, self.runid ,self.keys)
        epoch = 0
        while epoch != self.epochs :
            for image, label in self.train:
                ref = self.dist.refresh(self.url, self.refresh, self.last_auth)
                if ref:
                    self.keys = ref['body']['AuthenticationResult']['IdToken']
                    self.last_auth = datetime.now()
                    self.refresh = ref['body']['AuthenticationResult']['RefreshToken']
                    
                self.optimizer.zero_grad()
                output = self.model(image.reshape(-1, 1, self.shape, self.shape, self.shape))
                loss = self.criterion(output, label.reshape(-1, self.shape, self.shape, self.shape).long() * 2)
                train_metrics = {'Train_loss':float(loss.item())}
                if self.cubes == 1:
                    dice_loss = self.dice.faster_dice(torch.argmax(torch.squeeze(output), 0), label.reshape(self.shape, self.shape, self.shape) * 2, labels=[i for i in range(self.classes)])
                else:
                    dice_loss = self.dice.faster_dice(torch.argmax(torch.squeeze(output), 1), label.reshape(-1, self.shape, self.shape, self.shape) * 2, labels=[i for i in range(self.classes)])
                loss.backward()
                for cls in range(self.classes):
                    train_metrics.update({f'Train_dice_{cls}':float(dice_loss[cls])})
                train_metrics.update({'LR':self.scheduler.get_last_lr()[0]})
                self.comp.insert_simulation_data(self.runid,train_metrics,'train')
                local_gradients = [param.grad.clone() for param in self.model.parameters()]
                numpy_arrays = [tensor.cpu().numpy() for tensor in local_gradients]
                nested_lists = [array.tolist() for array in numpy_arrays]
                self.dist.upload_gradients(self.url, self.runid, self.keys, str(nested_lists))

                result =  self.dist.get_gradients(self.url, self.runid, self.keys)
                while result['result'] =='None':
                    time.sleep(5)
                    result =  self.dist.get_gradients(self.url, self.runid, self.keys)

                agg_grad = [np.array(array) for array in ast.literal_eval(result['result'])]

                with torch.no_grad():
                    for param, avg_grad in zip(self.model.parameters(), agg_grad):
                        if param.requires_grad:
                            avg_grad = torch.tensor(avg_grad, dtype=param.grad.dtype, device=param.grad.device)
                            param.grad = avg_grad.clone().detach().to(param.grad.device)
                print(self.scheduler.get_last_lr()[0])
                self.optimizer.step()
                self.scheduler.step()

            with torch.no_grad():
                for image, label in self.valid:
                    output = self.model(image.reshape(-1,1,self.shape,self.shape,self.shape))
                    loss = self.criterion(output, label.reshape(-1, self.shape, self.shape, self.shape).long() * 2)
                    valid_metrics = {'Valid_loss':float(loss.item())}
                    if self.cubes == 1:
                        dice_loss = self.dice.faster_dice(torch.argmax(torch.squeeze(output), 0), label.reshape(self.shape, self.shape, self.shape) * 2, labels=[i for i in range(self.classes)])
                    else:
                        dice_loss = self.dice.faster_dice(torch.argmax(torch.squeeze(output), 1), label.reshape(-1, self.shape, self.shape, self.shape) * 2, labels=[i for i in range(self.classes)])
                    for cls in range(self.classes):
                        valid_metrics.update({f'Valid_dice_{cls}':float(dice_loss[cls])})
                    self.comp.insert_simulation_data(self.runid,valid_metrics,'valid')
            
            epoch+=1
        self.dist.deactivate_user(self.url, self.runid ,self.keys)
        self.comp.end_simulation(self.runid)

                


                
                






