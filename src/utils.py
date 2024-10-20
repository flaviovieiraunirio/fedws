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

import numpy as np
import io, re
from copy import deepcopy
from contextlib import redirect_stdout

import torch
from torch.nn import CrossEntropyLoss
from torchinfo import summary

import math
import optimizers, schedulers

from sklearn.metrics import classification_report


y_pred_list = [] # para classification report
labels_list= [] # para classification report

types_pretty = {'train': 'training', 'valid': 'validation', 'test': 'test'}


class Scheduler():
    def __str__(self):
        sched_str = '%s (\n' % self.name
        for key in vars(self).keys():
            if key != 'name':
                value = vars(self)[key]
                if key == 'optimizer': value = str(value).replace('\n', '\n        ').replace('    )', ')')
                sched_str +=  '    %s: %s\n' % (key, value)
        sched_str += ')'
        return sched_str

def client_dist(client_id, w, n_k, m):
    w_avg = deepcopy(w) # pesos medios dos clientes
    w_avg2 = deepcopy(w) # pesos medios dos clientes
    w_std = deepcopy(w) # Desvio padrao dos pesos dos clientes
    w_ret = deepcopy(w) # retorno 
    
    print("++++++++++++")
    print("client_id:", client_id) 
    print("++++++++++++") 
    
    conta = 0
    for key in w_avg.keys(): # media
        conta = conta + 1
        w_avg[key] = torch.mul(w_avg[key], n_k)
        for i in range(1, len(w)):
            w_avg[key] = torch.add(w_avg[key], w[i][key], alpha=n_k[i])
        w_avg[key] = torch.div(w_avg[key], sum(n_k))
    print('conta: ',conta)   

    for key in w_avg2.keys(): # media
        conta = conta + 1
        w_avg2[key] = torch.mul(w_avg2[key], n_k)
        for i in range(1, len(w)):
            w_avg2[key] = torch.add(w_avg2[key], w[i][key], alpha=n_k[i])
        w_avg2[key] = torch.div(w_avg2[key], m)    
        
    for key in w_std.keys():  #  desvio padrão
        w_std[key] = torch.mul(w_std[key], n_k)
        for i in range(1, len(w)):
            w_std[key] = torch.add(w_std[key], w[i][key], alpha=n_k[i])
        
        #w_std[key] = torch.sqrt( torch.div(torch.pow(w_std[key]-w_avg[key],2),sum(n_k) ) )
        w_std[key] = torch.sqrt( torch.div(torch.pow(w_std[key]-w_avg2[key],2),20 ) )
        #print("w_std :",w_std[key])
        #print(w_std[key])
        #print("key :")
        #print(key)

    dentro = 0
    fora = 0
    
    # chaves são as classes retornar somente as classes ok

    for key in w_ret.keys(): # retorno
        w_ret[key] = torch.mul(w_ret[key], n_k[0])
        
        #for i in range(1, len(w)):
        #    w_ret[key] = torch.add(w_ret[key], w[i][key], alpha=n_k[i])    
            
        aux1 = torch.sub(w_avg[key],w_std[key])
        aux2 = torch.add(w_avg[key],w_std[key])
        print("aux1 :",aux1)
        print("aux2 :",aux2)       
        comp1 = torch.le(aux1,w_ret[key]) 
        comp2 = torch.le(w_ret[key],aux2)

        #comp3 = torch.lt(w_ret[key],aux1)
        #comp4 = torch.gt(w_ret[key],aux2)

        if (torch.all(comp1).item() == bool('True')) and (torch.all(comp2).item() == bool('True')):
            dentro = dentro + 1
        else:
            fora = fora + 1
        
        print('comp1: ',comp1)
        print('comp2: ',comp2)
        print('comp1: ',torch.all(comp1).item())
        print('comp2: ',torch.all(comp2).item())
        #print('comp3: ',torch.any(comp3).item())
        #print('comp4: ',torch.any(comp4).item())
    print('dentro:')
    print(dentro)
    print('fora:')
    print(fora)
        #print(torch.any(comp1))
        #print(torch.any(comp2))
        #print(torch.any(comp3))
        #print(torch.any(comp4))
           

        
    return w_avg

def average_updates(w, n_k):
    w_avg = deepcopy(w[0])
    conta1 = 0
    conta2 = 0
    for key in w_avg.keys():
        conta1 = conta1 + 1
        w_avg[key] = torch.mul(w_avg[key], n_k[0])
        for i in range(1, len(w)):
            conta2 = conta2 + 1
            w_avg[key] = torch.add(w_avg[key], w[i][key], alpha=n_k[i])
        #print('conta2x: ',conta2)
        w_avg[key] = torch.div(w_avg[key], sum(n_k))
    #print('conta1x: ',conta1) 
    #print('conta2x: ',conta2) 
    return w_avg
    
def average_updates_new2(w, n_k):
    
    w_avg = deepcopy(w[0]) # pesos medios dos clientes
    w_avg2 = deepcopy(w[0]) # pesos medios dos clientes
    w_std = deepcopy(w[0]) # Desvio padrao dos pesos dos clientes
    w_ret = deepcopy(w[0]) # retorno 
    
    print("shape de w:", np.asarray(w).shape)
    print("shape de w_avg:", np.asarray(w_avg).shape)
    
    #print("===========")
    #print("n_k") 
    #print("===========") 
    
    conta = 0
    # Calcular a média total w_avg2
    
    soma_clientes = 0/.255
    
    for i in range(0, len(w)):
        for key in w[i].keys():
            if (key.find('weight')> - 1):
                soma_clientes = soma_clientes + torch.sum(w[i][key])
            
    media_total = soma_clientes / len(w)+1
    
    desvio_padrao = 0/.255
    
    soma_ao_quadrado = 0/.255
    
    for i in range(0, len(w)):
        for key in w[i].keys():
        
            if (key.find('weight')> - 1):
        
                if (i == 0):
                    valor = w[i][key]
                    valor = torch.flatten(valor)
                else:
                    valor = torch.cat((valor,torch.flatten(w[i][key])),0)
        
            #diferencas = torch.sub(w[i][key],media_total)
            
            #print('pesos:',w[i][key])
            #print('diferencas:',diferencas)
            
            #soma_ao_quadrado = soma_ao_quadrado + pow(torch.sub(w[i][key] - media_total),2)
            #soma_ao_quadrado = soma_ao_quadrado + pow(torch.sum(diferencas),2)
       
    soma_clientes2 = torch.sum(valor)       
    media_total = torch.mean(valor)
    desv_pad = torch.std(valor)
    
    #desvio_padrao = math.sqrt(soma_ao_quadrado / len(w)+1)
    
    #print('soma_ao_quadrado:',soma_ao_quadrado)
    print('soma_clientes:',soma_clientes)
    print('soma_clientes2:',soma_clientes2)
    print('media_total:',media_total)
    #print('desvio_padrao:',desvio_padrao)
    print('desv_pad:', desv_pad)
    
    # Construir a lista de clientes ignorados 
    dentro = 0
    fora = 0
    clientes_fora = []
    for key in w_ret.keys(): # retorno

        if (key.find('weight')> - 1):
            aux1 = torch.sub(media_total,desv_pad)
            aux2 = torch.add(media_total,desv_pad)

        #w_ret[key] = torch.mul(w_ret[key], n_k[0])
        
        #print("aux1 :",aux1)
        #print("aux2 :",aux2)       
        #comp1 = torch.le(aux1,w_ret[key]) 
        #comp2 = torch.le(w_ret[key],aux2)
        
            for i in range(1, len(w)):
        
                if (i not in clientes_fora):
                    atual1  = w_ret[key]
                    #print('comp2: ',comp2)
                    #pesos_cliente_atual = torch.add(atual1, w[i][key])
                    pesos_cliente_atual = w[i][key]
                    t_aux1 = torch.tensor(aux1)
                    t_aux2 = torch.tensor(aux2)
                    #print('pesos_cliente_atual: ',pesos_cliente_atual)
                    comp1 = torch.le(t_aux1,pesos_cliente_atual) 
                    comp2 = torch.le(pesos_cliente_atual,t_aux2)
                    print('comp1: ',comp1)
                    print('comp2: ',comp2)
                    #print('comp1: ',torch.all(comp1).item())
                    #print('comp2: ',torch.all(comp2).item())
            
                    if (torch.all(comp1).item() == bool('True')) and (torch.all(comp2).item() == bool('True')):
                        dentro = dentro + 1
                        w_ret[key] = torch.add(w_ret[key], w[i][key])
                        print('nao zeros true',torch.count_nonzero(comp1))
                        print('nao zeros true',torch.count_nonzero(comp2))
                    else:
                        #print('comp1: ',comp1)
                        #print('comp2: ',comp2)
                        fora = fora + 1
                        print('nao zeros false',torch.count_nonzero(comp1))
                        print('nao zeros false',torch.count_nonzero(comp2))
                        clientes_fora.append(i) 
       


        
        #print('comp1: ',comp1)
        #print('comp2: ',comp2)
        #print('comp1: ',torch.all(comp1).item())
        #print('comp2: ',torch.all(comp2).item())
        #print('comp3: ',torch.any(comp3).item())
        #print('comp4: ',torch.any(comp4).item())
    #print('dentro:', dentro)
    #print('fora:', fora)
    print('clientes fora:', clientes_fora)
        #print(torch.any(comp1))
        #print(torch.any(comp2))
        #print(torch.any(comp3))
        #print(torch.any(comp4))
    
    # Adicionar o cliente somente quando puder
    # Fazer a média    
    
    # chaves são as classes retornar somente as classes ok

    
           
    for key in w_avg.keys(): # media
        #conta = conta + 1
        #contaw = 0
        w_avg[key] = torch.mul(w_avg[key], n_k[0])
        for i in range(1, len(w)):
            w_avg[key] = torch.add(w_avg[key], w[i][key], alpha=n_k[i])
            #contaw = contaw + 1
        #print('contaw: ',contaw)
        w_avg[key] = torch.div(w_avg[key], sum(n_k))
    #print('conta: ',conta)  
        
    return w_avg    
    
def average_updates_new(w, n_k):
    
    w_avg = deepcopy(w[0]) # pesos medios dos clientes
    w_avg2 = deepcopy(w[0]) # pesos medios dos clientes
    w_std = deepcopy(w[0]) # Desvio padrao dos pesos dos clientes
    w_ret = deepcopy(w[0]) # retorno 
    
    #print("shape de w:", np.asarray(w).shape)
    #print("shape de w_avg:", np.asarray(w_avg).shape)
    
    #print("===========")
    #print("n_k") 
    #print("===========") 
    
    conta = 0
    # Calcular a média total w_avg2
    
    #soma_clientes = []
    
    for key in w_avg2.keys(): # media
        if (key.find('weight')> - 1):
            conta = conta + 1
            w_avg2[key] = torch.mul(w_avg2[key], n_k[0])
            #print(n_k[0])
            #soma_clientes.append(torch.sum(w_avg2[key]))
            for i in range(1, len(w)):
                w_avg2[key] = torch.add(w_avg2[key], w[i][key], alpha=n_k[i])
                #print(n_k[i])
                #soma_clientes.append(torch.sum(w_avg2[key]))
            w_avg2[key] = torch.div(w_avg2[key], len(w)+1)    
       
    #media_total = mean(soma_clientes)
    
    #for i in range()
    # outra_forma

    
    
       
    # Calcular o desvio padrão 
    
    for key in w_std.keys():  #  desvio padrão
        if (key.find('weight')> - 1):
            w_std[key] = torch.mul(w_std[key], n_k[0])
            for i in range(1, len(w)):
                w_std[key] = torch.add(w_std[key], w[i][key], alpha=n_k[i])
        
        #w_std[key] = torch.sqrt( torch.div(torch.pow(w_std[key]-w_avg[key],2),sum(n_k) ) )
            w_std[key] = torch.sqrt( torch.div(torch.pow(w_std[key]-w_avg2[key],2),len(w)+1 ) )
        #print("w_std :",w_std[key])
        #print(w_std[key])
        #print("key :")
        #print(key)
        
    # Construir a lista de clientes ignorados 
    dentro = 0
    fora = 0
    clientes_fora = []
    for key in w_ret.keys(): # retorno
        
        
        aux1 = torch.sub(w_avg[key],w_std[key])
        aux2 = torch.add(w_avg[key],w_std[key])

        #w_ret[key] = torch.mul(w_ret[key], n_k[0])
        
        #print("aux1 :",aux1)
        #print("aux2 :",aux2)       
        #comp1 = torch.le(aux1,w_ret[key]) 
        #comp2 = torch.le(w_ret[key],aux2)
        
        for i in range(1, len(w)):
            if (key.find('weight')> - 1):
                print("key :", key)
                if (i not in clientes_fora):
                    atual1  = w_ret[key]
                    #print('comp2: ',comp2)
                    #pesos_cliente_atual = torch.add(atual1, w[i][key])
                    pesos_cliente_atual = w[i][key]
                    #print('pesos_cliente_atual: ',pesos_cliente_atual)
                    comp1 = torch.le(aux1,pesos_cliente_atual) 
                    comp2 = torch.le(pesos_cliente_atual,aux2)
                    #print('comp1: ',comp1)
                    #print('comp2: ',comp2)
                    print('comp1: ',torch.all(comp1).item())
                    print('comp2: ',torch.all(comp2).item())
            
                    if (torch.all(comp1).item() == bool('True')) and (torch.all(comp2).item() == bool('True')):
                        dentro = dentro + 1
                    #w_ret[key] = torch.add(w_ret[key], w[i][key])
                    #print('nao zeros true',torch.count_nonzero(comp1))
                    #print('nao zeros true',torch.count_nonzero(comp2))
                        w_ret[key] = torch.add(w_ret[key], w[i][key], alpha=n_k[i])
                    else:
                    #print('comp1: ',comp1)
                    #print('comp2: ',comp2)
                        fora = fora + 1
                    #print('nao zeros false',torch.count_nonzero(comp1))
                    #print('nao zeros false',torch.count_nonzero(comp2))
                        # clientes_fora.append(i) 
            else:
                w_ret[key] = torch.add(w_avg[key], w[i][key], alpha=n_k[i])
                
        w_ret[key] = torch.div(w_ret[key], sum(n_k)-fora)


        
        #print('comp1: ',comp1)
        #print('comp2: ',comp2)
        #print('comp1: ',torch.all(comp1).item())
        #print('comp2: ',torch.all(comp2).item())
        #print('comp3: ',torch.any(comp3).item())
        #print('comp4: ',torch.any(comp4).item())
    #print('dentro:', dentro)
    #print('fora:', fora)
    print('clientes fora:', clientes_fora)
        #print(torch.any(comp1))
        #print(torch.any(comp2))
        #print(torch.any(comp3))
        #print(torch.any(comp4))
    
    # Adicionar o cliente somente quando puder
    # Fazer a média    
    
    # chaves são as classes retornar somente as classes ok

    
           
    for key in w_avg.keys(): # media
        #conta = conta + 1
        #contaw = 0
        w_avg[key] = torch.mul(w_avg[key], n_k[0])
        for i in range(1, len(w)):
            w_avg[key] = torch.add(w_avg[key], w[i][key], alpha=n_k[i])
            #contaw = contaw + 1
        #print('contaw: ',contaw)
        w_avg[key] = torch.div(w_avg[key], sum(n_k))
    #print('conta: ',conta)  
    
    return w_avg
    
def inference(model, loader, device):
    if loader is None:
        return None, None


    criterion = CrossEntropyLoss().to(device)
    loss, total, correct = 0., 0, 0
    model.eval()
    with torch.no_grad():
        for batch, (examples, labels) in enumerate(loader):
            examples, labels = examples.to(device), labels.to(device)
            log_probs = model(examples)
            loss += criterion(log_probs, labels).item() * len(labels)
            _, pred_labels = torch.max(log_probs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            
            y_pred_labels = pred_labels.cpu().numpy().tolist()
            y_label = labels.cpu().numpy().tolist()
            
            y_pred_list.extend(y_pred_labels)
            labels_list.extend(y_label)

    accuracy = correct/total
    loss /= total   
    

    
    return accuracy, loss 
    
''' Original
def inference(model, loader, device):
    if loader is None:
        return None, None

    criterion = CrossEntropyLoss().to(device)
    loss, total, correct = 0., 0, 0
    model.eval()
    with torch.no_grad():
        for batch, (examples, labels) in enumerate(loader):
            examples, labels = examples.to(device), labels.to(device)
            log_probs = model(examples)
            loss += criterion(log_probs, labels).item() * len(labels)
            _, pred_labels = torch.max(log_probs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

    accuracy = correct/total
    loss /= total   
    
    return accuracy, loss 
'''

'''    
def inference_ext(model, loader, device):
    if loader is None:
        return None, None

    criterion = CrossEntropyLoss().to(device)
    loss, total, correct = 0., 0, 0
    model.eval()
    
    from sklearn.metrics import recall_score, f1_score, precision
    
    with torch.no_grad():
        for batch, (examples, labels) in enumerate(loader):
            examples, labels = examples.to(device), labels.to(device)
            log_probs = model(examples)
            loss += criterion(log_probs, labels).item() * len(labels)
            _, pred_labels = torch.max(log_probs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

    accuracy = correct/total
    loss /= total
    
    # Calc average precision 
    
    from sklearn.metrics import recall_score, f1_score, precision
    
    f1 = f1_score(pred_labels, y_pred)
    
    precision = precision_score(pred_labels, y_pred)
    
    recall = recall_score(pred_labels, y_pred)
    
    
    return accuracy, f1, precision, recall, loss 
'''

def get_acc_avg(acc_types, clients, model, device):


    acc_avg = {}
    for type in acc_types:
        acc_avg[type] = 0.
        num_examples = 0
        for client_id in range(len(clients)):
            acc_client, _ = clients[client_id].inference(model, type=type, device=device)
            if acc_client is not None:
                acc_avg[type] += acc_client * len(clients[client_id].loaders[type].dataset)
                num_examples += len(clients[client_id].loaders[type].dataset)
        acc_avg[type] = acc_avg[type] / num_examples if num_examples != 0 else None
        
    #print(labels_list)
    #print(y_pred_list)

    print(classification_report(labels_list, y_pred_list))

    y_pred_list.clear() # para classification report
    labels_list.clear() # para classification report

    return acc_avg

'''    
def get_ext_avg(acc_types, clients, model, device):
    acc_avg = {}
    for type in acc_types:
        if type <>  'train':
            acc_avg[type] = 0.
            num_examples = 0
            for client_id in range(len(clients)):
                acc_client, _ = clients[client_id].inference(model, type=type, device=device)
                if acc_client is not None:
                    acc_avg[type] += acc_client * len(clients[client_id].loaders[type].dataset)
                    num_examples += len(clients[client_id].loaders[type].dataset)
            acc_avg[type] = acc_avg[type] / num_examples if num_examples != 0 else None

    return acc_avg
    
#acc_types = ['train', 'test'] if datasets_actual['valid'] is None else ['train', 'valid']    

def get_acc_avg_orig(acc_types, clients, model, device):
    acc_avg = {}
    for type in acc_types:
        acc_avg[type] = 0.
        num_examples = 0
        for client_id in range(len(clients)):
            acc_client, _ = clients[client_id].inference(model, type=type, device=device)
            if acc_client is not None:
                acc_avg[type] += acc_client * len(clients[client_id].loaders[type].dataset)
                num_examples += len(clients[client_id].loaders[type].dataset)
        acc_avg[type] = acc_avg[type] / num_examples if num_examples != 0 else None

    return acc_avg
    
#    'train'
'''

def printlog_stats(quiet, logger, loss_avg, acc_avg, acc_types, lr, round, iter, iters):
    if not quiet:
        print(f'        Iteration: {iter}', end='')
        if iters is not None: print(f'/{iters}', end='')
        print()
        print(f'        Learning rate: {lr}')
        print(f'        Average running loss: {loss_avg:.6f}')
        for type in acc_types:
            print(f'        Average {types_pretty[type]} accuracy: {acc_avg[type]:.3%}')

    if logger is not None:
        logger.add_scalar('Learning rate (Round)', lr, round)
        logger.add_scalar('Learning rate (Iteration)', lr, iter)
        logger.add_scalar('Average running loss (Round)', loss_avg, round)
        logger.add_scalar('Average running loss (Iteration)', loss_avg, iter)
        for type in acc_types:
            logger.add_scalars('Average accuracy (Round)', {types_pretty[type].capitalize(): acc_avg[type]}, round)
            logger.add_scalars('Average accuracy (Iteration)', {types_pretty[type].capitalize(): acc_avg[type]}, iter)
        logger.flush()

def exp_details(args, model, datasets, splits):
    if args.device == 'cpu':
        device = 'CPU'
    else:
        device = str(torch.cuda.get_device_properties(args.device))
        device = (', ' + re.sub('_CudaDeviceProperties\(|\)', '', device)).replace(', ', '\n            ')

    input_size = (args.train_bs,) + tuple(datasets['train'][0][0].shape)
    
    print('args.fedsgd:',args.fedsgd)
    print("input_size:",input_size)
    summ = str(summary(model, input_size, depth=10, verbose=0, col_names=['output_size','kernel_size','num_params','mult_adds'], device=args.device))
    summ = '        ' + summ.replace('\n', '\n        ')

    optimizer = getattr(optimizers, args.optim)(model.parameters(), args.optim_args)
    scheduler = getattr(schedulers, args.sched)(optimizer, args.sched_args)

    if args.centralized:
        algo = 'Centralized'
    else:
        if args.fedsgd:
            algo = 'FedSGD'
        else:
            algo = 'FedAvg'
        if args.server_momentum:
            algo += 'M'
        if args.fedir:
            algo += ' + FedIR'
        if args.vc_size is not None:
            algo += ' + FedVC'
        if args.mu:
            algo += ' + FedProx'
        if args.drop_stragglers:
            algo += ' (Drop Stragglers)'

    f = io.StringIO()
    with redirect_stdout(f):
        print('Experiment summary:')
        print(f'    Algorithm:')
        print(f'        Algorithm: {algo}')
        print(f'        ' + (f'Rounds: {args.rounds}' if args.iters is None else f'Iterations: {args.iters}'))
        print(f'        Clients: {args.num_clients}')
        print(f'        Fraction of clients: {args.frac_clients}')
        print(f'        Client epochs: {args.epochs}')
        print(f'        Training batch size: {args.train_bs}')
        print(f'        System heterogeneity: {args.hetero}')
        print(f'        Server learning rate: {args.server_lr}')
        print(f'        Server momentum (FedAvgM): {args.server_momentum}')
        print(f'        Virtual client size (FedVC): {args.vc_size}')
        print(f'        Mu (FedProx): {args.mu}')
        print()

        print('    Dataset and split:')
        print('        Training set:')
        print('            ' + str(datasets['train']).replace('\n','\n            '))
        if datasets['valid'] is not None:
            print('        Validation set:')
            print('            ' + str(datasets['valid']).replace('\n','\n            '))
        print('        Test set:')
        print('            ' + str(datasets['test']).replace('\n','\n            '))
        print(f'        Identicalness: {args.iid} (EMD = {splits["train"].emd["class"]})')
        print(f'        Balance: {args.balance} (EMD = {splits["train"].emd["client"]})')
        print()

        print('    Scheduler: %s' % (str(scheduler).replace('\n', '\n    ')))
        print()

        print('    Model:')
        print(summ)
        print()

        print('    Other:')
        print(f'        Test batch size: {args.test_bs}')
        print(f'        Random seed: {args.seed}')
        print(f'        Device: {device}')

    return f.getvalue()
