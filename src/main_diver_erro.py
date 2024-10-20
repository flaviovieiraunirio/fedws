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


import random, re, os
from copy import deepcopy
from os import environ
from time import time
from datetime import timedelta
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import datasets, models, optimizers, schedulers
from options import args_parser
from utils import average_updates, exp_details, get_acc_avg, printlog_stats, client_dist, average_updates_new, average_updates_new2
from datasets_utils import Subset, get_datasets_fig
from sampling import get_splits, get_splits_fig
from client import Client


if __name__ == '__main__':
    # Start timer
    start_time = time()

    # Parse arguments and create/load checkpoint
    args = args_parser()
    if not args.resume:
        checkpoint = {}
        checkpoint['args'] = args
    else:
        checkpoint = torch.load(f'save/{args.name}')
        rounds = args.rounds
        iters = args.iters
        device =args.device
        args = checkpoint['args']
        args.resume = True
        args.rounds = rounds
        args.iters = iters
        args.device = device

    ## Initialize RNGs and ensure reproducibility
    if args.seed is not None:
        environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        if not args.resume:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
        else:
            torch.set_rng_state(checkpoint['torch_rng_state'])
            np.random.set_state(checkpoint['numpy_rng_state'])
            random.setstate(checkpoint['python_rng_state'])

    # Load datasets and splits
    print('args',args.dataset)
    print('args',args.client_failures)
    if not args.resume:
        datasets = getattr(datasets, args.dataset)(args, args.dataset_args)
        splits = get_splits(datasets, args.num_clients, args.iid, args.balance)
        datasets_actual = {}
        for dataset_type in splits:
            if splits[dataset_type] is not None:
                idxs = []
                for client_id in splits[dataset_type].idxs:
                    idxs += splits[dataset_type].idxs[client_id]
                datasets_actual[dataset_type] = Subset(datasets[dataset_type], idxs)
            else:
                datasets_actual[dataset_type] = None
        checkpoint['splits'] = splits
        checkpoint['datasets_actual'] = datasets_actual
    else:
        splits = checkpoint['splits']
        datasets_actual = checkpoint['datasets_actual']
    acc_types = ['train', 'test'] if datasets_actual['valid'] is None else ['train', 'valid']

    # Load model
    num_classes = len(datasets_actual['train'].classes)
    num_channels = datasets_actual['train'][0][0].shape[0]
    model = getattr(models, args.model)(num_classes, num_channels, args.model_args).to(args.device)
    if args.resume:
        model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer and scheduler
    optim = getattr(optimizers, args.optim)(model.parameters(), args.optim_args)
    sched = getattr(schedulers, args.sched)(optim, args.sched_args)
    if args.resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])
        sched.load_state_dict(checkpoint['sched_state_dict'])

    # Create clients
    if not args.resume:
        clients = []
        for client_id in range(args.num_clients):
            client_idxs = {dataset_type: splits[dataset_type].idxs[client_id] if splits[dataset_type] is not None else None for dataset_type in splits}
            clients.append(Client(args=args, datasets=datasets, idxs=client_idxs))
        checkpoint['clients'] = clients
    else:
        clients = checkpoint['clients']

    # Set client sampling probabilities
    if args.vc_size is not None:
        # Proportional to the number of examples (FedVC)
        p_clients = np.array([len(client.loaders['train'].dataset) for client in clients])
        p_clients = p_clients / p_clients.sum()
    else:
        # Uniform
        p_clients = None

    # Determine number of clients to sample per round
    m = max(int(args.frac_clients * args.num_clients), 1)

    # Print experiment summary
    summary = exp_details(args, model, datasets_actual, splits)
    print('\n' + summary)

    # Log experiment summary, client distributions, example images
    if not args.no_log:
        logger = SummaryWriter(f'runs/{args.name}')
        if not args.resume:
            logger.add_text('Experiment summary', re.sub('^', '    ', re.sub('\n', '\n    ', summary)))

            splits_fig = get_splits_fig(splits, args.iid, args.balance)
            logger.add_figure('Splits', splits_fig)

            datasets_fig = get_datasets_fig(datasets_actual, args.train_bs)
            logger.add_figure('Datasets', datasets_fig)

            input_size = (1,) + tuple(datasets_actual['train'][0][0].shape)
            fake_input = torch.zeros(input_size).to(args.device)
            logger.add_graph(model, fake_input)
    else:
        logger = None

    if not args.resume:
        # Compute initial average accuracies
        acc_avg = get_acc_avg(acc_types, clients, model, args.device)
        acc_avg_best = acc_avg[acc_types[1]]

        # Print and log initial stats
        if not args.quiet:
            print('Training:')
            print('    Round: 0' + (f'/{args.rounds}' if args.iters is None else ''))
        loss_avg, lr = torch.nan, torch.nan
        printlog_stats(args.quiet, logger, loss_avg, acc_avg, acc_types, lr, 0, 0, args.iters)
    else:
        acc_avg_best = checkpoint['acc_avg_best']

    init_end_time = time()

    # Train server model
    if not args.resume:
        last_round = 0 #-1
        iter = 0
        v = None
    else:
        last_round = checkpoint['last_round']
        iter = checkpoint['iter']
        v = checkpoint['v']
        
    # SGD Train TESTE
    if (args.compute_weight_divergence == True):
        print("SGD Trainig:")
        
        args2 = deepcopy(args)
        
        args2.num_clients = 1
        args2.frac_clients = 1
        args2.epochs = 1
        args2.hetero = 0
        args2.iid = float('inf')
        args2.balance = float('inf')
        args2.vc_size = None
        args2.fedir = False
        args2.mu = 0
        args2.fedsgd = False
        args2.server_lr = 1
        args2.server_momentum = 0
        args2.train_bs = args2.train_bs * 10
        args2.dataset = 'cifar10'
        
        #args2 = args.model_args
        #args2.model_args["norm"] = None
        args2.model_args['norm'] = None
        print ('args2[norm]: ',args2.model_args['norm'])
        
        # Dataset 
        
        #datasets2 = getattr(datasets, args2.dataset)(args2, args2.dataset_args)
        #datasets2 = getattr(datasets, args2.dataset)(args2, args2.dataset_args)
        splits2 = get_splits(datasets, args2.num_clients, args2.iid, args2.balance)
        datasets_actual2 = {}
        for dataset_type in splits2:
            if splits2[dataset_type] is not None:
                idxs2 = []
                for client_id in splits2[dataset_type].idxs:
                    idxs += splits2[dataset_type].idxs[client_id]
                datasets_actual2[dataset_type] = Subset(datasets[dataset_type], idxs)
            else:
                datasets_actual2[dataset_type] = None       
        # Load model
        num_classes2 = len(datasets_actual2['train'].classes)
        num_channels2 = datasets_actual2['train'][0][0].shape[0]
        client_model = getattr(models, args2.model)(num_classes2, num_channels2, args2.model_args).to(args2.device) 
        
        
        # Clients
        
        clients2 = []
        for client_id2 in range(args2.num_clients):
            client_idxs2 = {dataset_type: splits2[dataset_type].idxs[client_id2] if splits2[dataset_type] is not None else None for dataset_type in splits2}
            clients2.append(Client(args=args, datasets=datasets, idxs=client_idxs2))
        
        #client_model = getattr(models, args2.model)(num_classes, num_channels, args2.model_args).to(args2.device)
        #client_model = deepcopy(model)
        client_update2, client_num_examples2, client_num_iters2, client_loss2 = clients[1].train(model=client_model, optim=optim, device=args2.device)
        #client_update2, client_num_examples2, client_num_iters2, client_loss2 = clients[1].train(model=client_model, optim=optim, device=args.device, comp_divergence=int(10))
        summary2 = exp_details(args, client_model, datasets_actual, splits)
        print('\n' + summary2)
        
        

    # Federated Train
    print("Federated Trainig:")
    print(args.num_clients)
    for round in range(last_round + 1, args.rounds):
        if not args.quiet:
            # print(f'    Round: {round+1}' + (f'/{args.rounds}' if args.iters is None else ''))
            print(f'    Round: {round}' + (f'/{args.rounds}' if args.iters is None else ''))

        # Sample clients
        client_ids = np.random.choice(range(args.num_clients), m, replace=False, p=p_clients)

        conta_updates = 0
        # Train client models
        updates, num_examples, max_iters, loss_tot = [], [], 0, 0
        
        falhas = args.client_failures
        
        for i, client_id in enumerate(client_ids):
        
            if (falhas == 0):
                if not args.quiet: print(f'        Client: {client_id} ({i+1}/{m})')
            else:
                if not args.quiet: print(f'        Client: {client_id} ({i+1}/{m}) FALHA')
                falhas = falhas - 1
                continue

            
            
            client_model = deepcopy(model)
            optim.__setstate__({'state': defaultdict(dict)})
            optim.param_groups[0]['params'] = list(client_model.parameters())

            client_update, client_num_examples, client_num_iters, client_loss = clients[client_id].train(model=client_model, optim=optim, device=args.device)
            #client_update, client_num_examples, client_num_iters, client_loss = clients[client_id].train(model=client_model, optim=optim, device=args.device, comp_divergence=1)
            
            #print('client_num_iters: ', client_num_iters)
            #teste = client_dist(client_id, client_update, client_num_examples, m)

            if client_num_iters > max_iters: max_iters = client_num_iters

            if client_update is not None:
                conta_updates = conta_updates + 1
                updates.append(deepcopy(client_update))
                loss_tot += client_loss * client_num_examples
                num_examples.append(client_num_examples)

        #print('conta_updates: ',conta_updates)
        iter += max_iters
        lr = optim.param_groups[0]['lr']

        if len(updates) > 0:
        

        
            # Update server model
            #update_avg = average_updates_new(updates, num_examples)
            #update_avg = average_updates_new2(updates, num_examples)
            update_avg = average_updates(updates, num_examples)
            
            
            if (args.compute_weight_divergence == True):
            #if (args.compute_weight_divergence == True and round == 10):
                round = args.rounds
                break
                
            #if (round == 5): #AQUI
            #    break
            
            if v is None:
                v = deepcopy(update_avg)
            else:
                for key in v.keys():
                    v[key] = update_avg[key] + v[key] * args.server_momentum
            #new_weights = deepcopy(model.state_dict())
            #for key in new_weights.keys():
            #    new_weights[key] = new_weights[key] - v[key] * args.server_lr
            #model.load_state_dict(new_weights)
            
            #for param in model.parameters():
            #    print('gradientes : ', param.grad)
            #for w in model.parameters():
            #     print('TESTE2 gradientes:', w.grad.data)
            #    print('TESTE2 :', w.grad)     
            
            #grad_dict = {k:v.grad for k, v in zip(model.state_dict(), model.parameters())}
            #print('graddict:', grad_dict)
            
            #print('TESTE ',model.feature_extractor[3][0].weight.grad.data) 
            #print('TESTE ',model.feature_extractor[3][0].weight) 
            
            #print('TESTE ',model.feature_extractor[0][0].weight.grad) 
            #print('TESTE ',model.feature_extractor[3][0].weight.grad) 
            #print('TESTE ',model.classifier[1].weight.grad) 
            #print('TESTE ',model.classifier[3].weight.grad) 
            #print('TESTE ',model.classifier[5].weight.grad) 
            
            #for name, param in model.named_parameters():
            #    if param.grad is not None:
            #        print(name, param.grad.sum())
            #    else:
            #        print(name, param.grad)
            
            for key in model.state_dict():
                #print('key:', key)
                #print('v[key]: ', v[key])
                #print('model.state_dict()[key]:', model.state_dict()[key])
                if (model.state_dict()[key].type() == v[key].type()):
                    #print('iguais')
                    model.state_dict()[key] -= v[key] * args.server_lr
                else:
                    #print('diferentes')
                    aux = v[key] * args.server_lr
                    aux = aux.long()
                    model.state_dict()[key] -= aux
                
            #print('BBBBBBBBB: ',round)
            # Compute round average loss and accuracies
            if (round % args.server_stats_every) == 0: # or (round == args.rounds):
                loss_avg = loss_tot / sum(num_examples)
                acc_avg = get_acc_avg(acc_types, clients, model, args.device)

                if acc_avg[acc_types[1]] > acc_avg_best:
                    acc_avg_best = acc_avg[acc_types[1]]

        # Save checkpoint
        checkpoint['model_state_dict'] = model.state_dict()
        checkpoint['optim_state_dict'] = optim.state_dict()
        checkpoint['sched_state_dict'] = sched.state_dict()
        checkpoint['last_round'] = round
        checkpoint['iter'] = iter
        checkpoint['v'] = v
        checkpoint['acc_avg_best'] = acc_avg_best
        checkpoint['torch_rng_state'] = torch.get_rng_state()
        checkpoint['numpy_rng_state'] = np.random.get_state()
        checkpoint['python_rng_state'] = random.getstate()
        
        torch.save(checkpoint, f'save/{args.name}')

        #print('BBBBBBBBB: ',round)
        # Print and log round stats
        if (round % args.server_stats_every == 0) or (round == args.rounds):
            #printlog_stats(args.quiet, logger, loss_avg, acc_avg, acc_types, lr, round+1, iter, args.iters)
            printlog_stats(args.quiet, logger, loss_avg, acc_avg, acc_types, lr, round, iter, args.iters)


        # Stop training if the desired number of iterations has been reached
        if args.iters is not None and iter >= args.iters: break

        # Step scheduler
        if type(sched) == schedulers.plateau_loss:
            sched.step(loss_avg)
        else:
            sched.step()

    train_end_time = time()

    # Compute final average test accuracy
    acc_avg = get_acc_avg(['test'], clients, model, args.device)

    test_end_time = time()

    # Print and log test results
    print('\nResults:')
    print(f'    Average test accuracy: {acc_avg["test"]:.3%}')
    print(f'    Train time: {timedelta(seconds=int(train_end_time-init_end_time))}')
    print(f'    Total time: {timedelta(seconds=int(time()-start_time))}')

    
    if (args.compute_weight_divergence == True):
        
        # Calculo das divergencias dos pesos das camadas Densas
        
       

        
        np_arr_d0 = client_update2['classifier.1.weight'].cpu().detach().numpy()
        np_arr_d1 = client_update2['classifier.3.weight'].cpu().detach().numpy()
        np_arr_d2 = client_update2['classifier.5.weight'].cpu().detach().numpy()
        
        np_arr_d3 = update_avg['classifier.1.weight'].cpu().detach().numpy()
        np_arr_d4 = update_avg['classifier.3.weight'].cpu().detach().numpy()
        np_arr_d5 = update_avg['classifier.5.weight'].cpu().detach().numpy()
        
        
        np_arr_d0 = np_arr_d0.flatten()
        np_arr_d1 = np_arr_d1.flatten()
        np_arr_d2 = np_arr_d2.flatten()
        np_arr_d3 = np_arr_d3.flatten()
        np_arr_d4 = np_arr_d4.flatten()
        np_arr_d5 = np_arr_d5.flatten()
        
        np_arr_dsgd  = np.concatenate((np_arr_d0, np_arr_d1, np_arr_d2), axis=0)
        np_arr_dfed  = np.concatenate((np_arr_d3, np_arr_d4, np_arr_d5), axis=0)        
        
        div2 = np.zeros_like(np_arr_dsgd)
        
        for x in range(0, np_arr_dsgd.size):
            if (np_arr_dsgd[x] == 0):
                div2[x] = abs(np_arr_dfed[x])
            else:
                div2[x] = abs(np_arr_dfed[x] - np_arr_dsgd[x]) / np_arr_dsgd[x]
        
        desvio_padrao2 = np.std(div2,ddof=1) # 
        media2 =  np.mean(div2) # 
        
        
        # Calculo das divergencias
        
        np_arr0 = client_update2['feature_extractor.0.0.weight'].cpu().detach().numpy()
        np_arr1 = client_update2['feature_extractor.3.0.weight'].cpu().detach().numpy()
        
        np_arr2 = update_avg['feature_extractor.0.0.weight'].cpu().detach().numpy()
        np_arr3 = update_avg['feature_extractor.3.0.weight'].cpu().detach().numpy()
        
        np_arr0 = np_arr0.flatten()
        np_arr1 = np_arr1.flatten()
        np_arr2 = np_arr2.flatten()
        np_arr3 = np_arr3.flatten()
        
        np_arr_sgd  = np.concatenate((np_arr0, np_arr1), axis=0)
        np_arr_fed  = np.concatenate((np_arr2, np_arr3), axis=0)
        
        div1 = np.zeros_like(np_arr_sgd)
        #div2 = np.zeros_like(np_arr5)
        
        for x in range(0, np_arr_sgd.size):
            if (np_arr_sgd[x] == 0):
                div1[x] = abs(np_arr_fed[x])
            else:
                div1[x] = abs(np_arr_fed[x] - np_arr_sgd[x]) / np_arr_sgd[x]
                
        #for x in range(0, np_arr1.size):
        #    if (np_arr1[x] == 0):
        #        div2[x] = abs(np_arr3[x])
        #    else:
        #        div2[x] = abs(np_arr3[x] - np_arr1[x]) / np_arr1[x]
        
        #div1 = abs(np_arr2 - np_arr0) / np_arr0
        #div2 = abs(np_arr3 - np_arr1) / np_arr1
        
        desvio_padrao1 = np.std(div1,ddof=1) # 
        media1 =  np.mean(div1) # 
        
        #desvio_padrao2 = np.std(div2) # 
        #media2 =  np.mean(div2) # 
        
        print('Media div feature_extractor.0.0.weight: ',media1)
        print('DP div feature_extractor.0.0.weight: ',desvio_padrao1)
        
        print('Media div classifier.0.0.weight: ',media2)
        print('DP div classifier.0.0.weight: ',desvio_padrao2)
        
        #print('Media div feature_extractor.3.0.weight: ',media2)    
        #print('DP div feature_extractor.3.0.weight: ',desvio_padrao2)
    

    # Controle
    print('File name:',  args.name)
    
    print('File name:',  args.model_args["norm"])
    
    print('File name:',  args.dataset)
    
    print('iid:', args.iid)
    
    resumo = args.name + ' ' + str(args.model_args["norm"]) + '_' + str(args.iid)
    arquivo = 'Controle_' + str(args.dataset) + '.txt'
    
    print('resumo:', resumo)
    print('arquivo:', arquivo)
    
    cmd = "echo Filename Folder '{}' >> {}".format(resumo,arquivo)
    os.system(cmd)

    if logger is not None: logger.close()
