#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Copyright (C) 2022  Gabriele Cazzato

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.
'''

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, Subset

from utils import inference


class Client(object):
    def __init__(self, args, datasets, idxs):
        self.args = args

        # Create dataloaders
        self.train_bs = self.args.train_bs if self.args.train_bs > 0 else len(idxs['train'])
        self.loaders = {}
        self.loaders['train'] = DataLoader(Subset(datasets['train'], idxs['train']), batch_size=self.train_bs, shuffle=True) if len(idxs['train']) > 0 else None
        self.loaders['valid'] = DataLoader(Subset(datasets['valid'], idxs['valid']), batch_size=args.test_bs, shuffle=False) if idxs['valid'] is not None and len(idxs['valid']) > 0 else None
        self.loaders['test'] = DataLoader(Subset(datasets['test'], idxs['test']), batch_size=args.test_bs, shuffle=False) if len(idxs['test']) > 0 else None

        # Set criterion
        if args.fedir:
            # Importance Reweighting (FedIR)
            labels = set(datasets['train'].targets)
            p = torch.tensor([(torch.tensor(datasets['train'].targets) == label).sum() for label in labels]) / len(datasets['train'].targets)
            q = torch.tensor([(torch.tensor(datasets['train'].targets)[idxs['train']] == label).sum() for label in labels]) / len(torch.tensor(datasets['train'].targets)[idxs['train']])
            weight = p/q
        else:
            # No Importance Reweighting
            weight = None
        self.criterion = CrossEntropyLoss(weight=weight)

    def train(self, model, optim, device):
    #def train(self, model, optim, device, comp_divergence):
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
        torch.use_deterministic_algorithms(False)
        _ = torch.set_grad_enabled(True)
        # Drop client if train set is empty
        if self.loaders['train'] is None:
            if not self.args.quiet: print(f'            No data!')
            return None, 0, 0, None

        # Determine if client is a straggler and drop it if required
        straggler = np.random.binomial(1, self.args.hetero)
        if straggler and self.args.drop_stragglers:
            if not self.args.quiet: print(f'            Dropped straggler!')
            return None, 0, 0, None
        epochs = np.random.randint(1, self.args.epochs) if straggler else self.args.epochs
        #epochs = np.random.randint(1, self.args.epochs) if straggler else (comp_divergence * self.args.epochs)
        
        

        # Create training loader
        if self.args.vc_size is not None:
            # Virtual Client (FedVC)
            if len(self.loaders['train'].dataset) >= self.args.vc_size:
                train_idxs_vc = torch.randperm(len(self.loaders['train'].dataset))[:self.args.vc_size]
            else:
                train_idxs_vc = torch.randint(len(self.loaders['train'].dataset), (self.args.vc_size,))
            train_loader = DataLoader(Subset(self.loaders['train'].dataset, train_idxs_vc), batch_size=self.train_bs, shuffle=True)
        else:
            # No Virtual Client
            train_loader = self.loaders['train']

        client_stats_every = self.args.client_stats_every if self.args.client_stats_every > 0 and self.args.client_stats_every < len(train_loader) else len(train_loader)

        # Train new model
        model.to(device)
        self.criterion.to(device)
        model.train()
        model_server = deepcopy(model)
        iter = 0
        for epoch in range(epochs):
            loss_sum, loss_num_images, num_images = 0., 0, 0
            for batch, (examples, labels) in enumerate(train_loader):
                examples, labels = examples.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(examples)
                loss = self.criterion(log_probs, labels)

                if self.args.mu > 0 and epoch > 0:
                    # Add proximal term to loss (FedProx)
                    w_diff = torch.tensor(0., device=device)
                    for w, w_t in zip(model.parameters(), model_server.parameters()):
                        w_diff += torch.pow(torch.norm(w.data - w_t.data), 2)
                        #w.grad.data += self.args.mu * (w.data - w_t.data)
                        w.grad.data += self.args.mu * (w_t.data - w.data)
                    loss += self.args.mu / 2. * w_diff

                loss_sum += loss.item() * len(labels)
                loss_num_images += len(labels)
                num_images += len(labels)

                #loss.register_hook(lambda grad: print(grad)) 
                loss.retain_grad()
                loss.backward()
                #print(loss.is_leaf)
                #print(loss.grad)
                optim.step()
                
                #for w in model.parameters():
                    #print('TESTE2 :', w.grad.data)
                #    print('TESTE2 :', w.grad)
                
                #print('parametros :', model.parameters())

                # After client_stats_every batches...
                if (batch + 1) % client_stats_every == 0:
                    # ...Compute average loss
                    loss_running = loss_sum / loss_num_images

                    # ...Print stats
                    if not self.args.quiet:
                        print('            ' + f'Epoch: {epoch+1}/{epochs}, '\
                                               f'Batch: {batch+1}/{len(train_loader)} (Image: {num_images}/{len(train_loader.dataset)}), '\
                                               f'Loss: {loss.item():.6f}, ' \
                                               f'Running loss: {loss_running:.6f}')

                    loss_sum, loss_num_images = 0., 0

                iter += 1

        # Compute model update
        model_update = {}
        for key in model.state_dict():
            model_update[key] = torch.sub(model_server.state_dict()[key], model.state_dict()[key])
        '''
        # mostrar gradientes AQUIII
        #pesos = np.array([])
        x = 0
        for w in model.parameters():
            if x == 0:
                pesos = w.grad.data.cpu().detach().numpy()
                pesos = pesos.flatten()
            else:
                aux = w.grad.data.cpu().detach().numpy()
                aux = aux.flatten()
                np.concatenate((pesos, aux), axis=0)
                
            #aux = w.grad.data.cpu().detach().numpy()
            #aux = deepcopy(w.grad.data)
            #print('tipo :', type(aux))
            #aux = aux.flatten()
            #np.concatenate((pesos, aux), axis=0)
            x = x + 1
            
        desvio_padrao2 = np.std(pesos,ddof=1) # 
        media2 =  np.mean(pesos) #  

        #print('media gradientes : ',media2)
        print(media2)
        '''

        return model_update, len(train_loader.dataset), iter, loss_running

    def inference(self, model, type, device):
        return inference(model, self.loaders[type], device)

