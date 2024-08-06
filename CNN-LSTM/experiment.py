from __future__ import print_function
import os
import torch
from torch.utils.data import DataLoader,TensorDataset
from torch.optim.lr_scheduler import StepLR
from datasetTorch_Melek import structureData
from trainModel import train_model
from logs import logfile
from TCN_torch import TCN, TransferTCN, EncoderTCN, DecoderTCN
from inferenceExperiment import inference_experiment
from datasetLoader_Melek import datasetCreator
from CNNLSTM import CNNLSTM
# from model_select import model_select

import argparse
#filter warnings
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime, timedelta
from Plots import PlotExperiment
from utils import init_weights, debug_point, determine_logfile_columns
import time
import pickle
import numpy as np
import wandb
# test = True
test = False
working_directory = r"D:\Baris\codes\baris_LSTM - Copy"

parser = argparse.ArgumentParser(
    prog='deepquake',
    description='Vs30 Prediction',
    epilog='Interface: brsylmz23@hotmail.com')

parser.add_argument('--wandb', default=True,action="store_false")
parser.add_argument('--test', default=False,action="store_true")
parser.add_argument('--pc', default=True,action="store_false")
parser.add_argument('--fno', default=1, type=int)
parser.add_argument('--fsiz', default=4, type=int)
parser.add_argument('--batchsize', default=64, type=int)
parser.add_argument('--n_epochs', default=100, type=int)
parser.add_argument('--step_size', default=20, type=int)
parser.add_argument('--gamma', default=0.9, type=float)
parser.add_argument('--Transfer_model', default=False,action="store_true")
parser.add_argument('--Transfer_encoder', default=False,action="store_true")
parser.add_argument('--add_stat_info', default=True,action="store_false")
parser.add_argument('--add_station_altitude', default=True,action="store_false")
parser.add_argument('--gtnorm', default=False,action="store_true")
parser.add_argument('--gt_select', nargs='+', help='<Required> Set flag', default=["Distance"])
parser.add_argument('--FC_size', default=256, type=int)
parser.add_argument('--SP', default=False,action="store_true")
parser.add_argument('--statID', default=None)
parser.add_argument('--radius', default=300, type=int)
parser.add_argument('--magnitude', default=3.5, type=float)
parser.add_argument('--depth', default=10000, type=int)
parser.add_argument('--stat_dist', default=120, type=int)
parser.add_argument('--augmentation_flag', default=True,action="store_false")
parser.add_argument('--train_percentage', default=80, type=int)
parser.add_argument('--augmentation_parameter', default=1)
parser.add_argument('--dataset', default="STEAD")
parser.add_argument('--fs', default=100, type=int)
parser.add_argument('--signal_aug_rate', default=0.3)
parser.add_argument('--window_size', default=1, type=int)
parser.add_argument('--crossvalidation_type', default="Chronological")
parser.add_argument('--loss_function', default="MAE")

parser.add_argument('--lat', default=36.77)
parser.add_argument('--lon', default=-119.41)
parser.add_argument('--signaltime', default=60)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--freqtime', default=False, action="store_true")
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--network', default="TCN")

args = parser.parse_args()


    

wandb.login()
##############################################################################   
def main(lat,lon,signaltime,transferpath,hidden, lr,batch_size, LSTM_split):

    
    torch.manual_seed(42)
    if (torch.cuda.is_available()):
        seed_value = 42
        torch.cuda.manual_seed_all(seed_value)
        print(torch.cuda.get_device_name(0))
    else:   
        print("no GPU found")        
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
    device = torch.device(dev)
############################################################################## 
    # 1. ExpName & Parameters
    hyperparameters = {"fno" : 1,          
                    "fsiz" : 4,
                    "dropout_rate" : 0.1, #transfer ederken 0.5e değiştirilsin. 
                    "batchsize" : batch_size,
                    "n_epochs": 200,
                    "lr":lr, #0.00005      0.0001
                    "step_size": 20,
                    "gamma": 0.9
                    }
    
    parameters = {"Transfer_model": False,
                  "Transfer_encoder":False,
                      "transfer_path":transferpath,
                      "add_stat_info": True,
                      "add_station_altitude": True,
                      "hidden": hidden,
                      "gtnorm": True,
                      "LSTM_split":LSTM_split ,         # Size of the parts of CNNLSTM input
                      "gt_select": ["Vs30"],  #epiLAT,epiLON,Depth,Distance
                      "model_select":["ResNet"],           #"ResNet","TCN"
                      "FC_size" : 256,
                      "SP": False,
                      "statID" : None,                  # Station ID
                      "radius" : 2000000000,               # The radius at which the experiment will be carried out.
                      "latitude" : lat,                # In which Latitude the experiment will be performed.
                      "longitude" : lon,             # In which Longitude the experiment will be performed.
                      "signaltime" : signaltime,              # Time length of EQ signals
                      "magnitude": 1,               # The magnitude of the EQ signals (If it is less than this value, wont take it.)
                      "depth":10000000,                     # Depth of EQ signal (If it is greater than this value, will not use )(km)
                      "stat_dist": 120000000000000,               #(unit?) Distance between the station recording the EQ event and the epicenter. (If it is greater than this value, will not use )
                      "freq_flag": freqtime,              # Will I use a frequency signal? Or is it a time signal? (T for Freq, F for Time)
                      "augmentation_flag": True,      # Will I augment the signal?
                      "Train percentage": 80,         # Train-val-test percentage. (80 means %80 = train + val)
                      "Augmentation parameter": 1,     # How much will I augment the signal (1 means 2 pieces of 1 seconds, so at total 2 seconds.)
                      "Crossvalidation_type": "Station-based",         # Crossvalidation_type (Chronological, Station-based)
                      "dataset": "KANDILLI_AFAD"                # Dataset to use (AFAD, STEAD)
                    }
    
    constants = {"km2meter" : 1000,                 
                    "fs" : 100,                     
                    "signal_aug_rate" : 0.3,        # I augment the signal differently when the tpga is below this percentage value. (or over 1-signal_aug_rate)
                    "window_size" : 1,              
                    "channel_depth" : 3,
                    "AFAD_Path": r"D:\Baris\matfiles_Vs30addedv2",
                    "STEAD_Path": r"D:\Users\melek\codes\stead_Trivial",
                    "KANDILLI_Path": r"D:\Baris\krdeaList",
                    "working_directory": working_directory,
                    "wandb" : args.wandb                                 
                    }    

    os.chdir(os.path.realpath(Path(working_directory)))
    
    kwargs = {**parameters,**constants,**hyperparameters}    
    kwargs = logfile(hyperparameters, parameters, constants, **kwargs)
    run = wandb.init(
            # Set the project where this run will be logged

            project="test",
            entity="brsylmz23",
            tags=["baseline"],
            save_code=True,
            reinit=True,
            # Track hyperparameters and run metadata
            config={
                "signaltime": kwargs.get('signaltime'),
                "lr": kwargs.get('lr'),
                "fno": kwargs.get('fno'),
                "fsiz": kwargs.get('fsiz'),
                "batchsize": kwargs.get('batchsize'),
                "n_epochs": kwargs.get('n_epochs'),
                "step_size": kwargs.get('step_size'),
                "gamma": kwargs.get('gamma'),
                "Transfer_model": kwargs.get('Transfer_model'),
                "Transfer_encoder": kwargs.get('Transfer_encoder'),
                "transfer_path": kwargs.get('transferpath'),
                "add_stat_info": kwargs.get('add_stat_info'),
                "add_station_altitude": kwargs.get('add_station_altitude'),
                "gtnorm": kwargs.get('gtnorm'),
                "gt_select": kwargs.get('gt_select'),
                "FC_size": kwargs.get('FC_size'),
                "SP": kwargs.get('SP'),
                "statID": kwargs.get('statID'),
                "radius": kwargs.get('radius'),
                "latitude": kwargs.get('lat'),
                "longitude": kwargs.get('lon'),
                "magnitude": kwargs.get('magnitude'),
                "depth": kwargs.get('depth'),
                "stat_dist": kwargs.get('stat_dist'),
                "freq_flag": kwargs.get('freq_flag'),
                "augmentation_flag": kwargs.get('augmentation_flag'),
                "Train percentage": kwargs.get('Train_percentage'),
                "Augmentation parameter": kwargs.get('Augmentation_percentage'),
                "Crossvalidation_type": kwargs.get('Crossvalidation_type'),
                "dataset": kwargs.get('dataset'),
                "fs": kwargs.get('fs'),
                "signal_aug_rate": kwargs.get('signal_aug_rate'),
                "window_size": kwargs.get('window_size'),
                "AFAD_Path": kwargs.get('AFAD_Path')
            })
    ##add comments
    signal_width = kwargs.get('fs')*kwargs.get('window_size')/2+1 #overlap constant
    signal_height = 2*kwargs.get("signaltime") - 1

    # model, decoder_model = model_select(signal_height,signal_width,device,**kwargs)
    model = CNNLSTM(3,hidden,2,1).to(device)
    model.apply(init_weights) 
    if not test:
        attributes = datasetCreator(**kwargs)
    else:
        # if it is test
        # TEST PATHS for KANILLI_AFAD
        kwargs["AFAD_Path"] = r"D:\Baris\matfiles_small"
        kwargs["STEAD_Path"] =  r"D:\Baris\STEAD_small"
        kwargs["KANDILLI_Path"] = r"D:\Users\melek\codes\kandilli_test"   
        kwargs['n_epochs'] = 1  
        attributes = datasetCreator(**kwargs)
        
    
    params = {'batch_size': kwargs.get('batchsize'), 'shuffle': False}   
    
    training_set = structureData(attributes, phase = "training", **kwargs)
    trainingLoader = DataLoader(training_set, **params)  
    validation_set = structureData(attributes,phase = "validation" , **kwargs)
    validationLoader = DataLoader(validation_set, **params)    
    test_set = structureData(attributes, phase = "test", **kwargs)
    testLoader = DataLoader(test_set, **params)
                     
    # if kwargs.get("Transfer_model") and kwargs.get("Transfer_encoder"):
    #     optim = torch.optim.Adam(decoder_model.parameters(),kwargs.get('lr'))
    # else:
    optim = torch.optim.Adam(model.parameters(),kwargs.get('lr'))        
    
    scheduler = StepLR(optim, kwargs.get('step_size'), kwargs.get('gamma'))   

    # training_losses, validation_losses, plot_dict = train_model(model, scheduler, trainingLoader, optim, validationLoader, attributes, **kwargs)
    training_losses, validation_losses = train_model(model, scheduler, trainingLoader, optim, validationLoader, attributes, **kwargs)
 
    ##
    trn_dict = inference_experiment(trainingLoader, attributes,signal_height, signal_width, model, phase = "training", **kwargs)     
    val_dict = inference_experiment(validationLoader, attributes,signal_height, signal_width, model, phase = "validation", **kwargs) 
    test_dict = inference_experiment(testLoader, attributes,signal_height, signal_width, model, phase = "test", **kwargs)  
    
    plotargs =  {**trn_dict, **val_dict,**test_dict, **kwargs}
    plotargs["training_losses"] = training_losses
    plotargs["validation_losses"] = validation_losses
    
    PlotExperiment(attributes, plotargs)
    
    # Write the results into a text file
    exps_directory = os.path.join(working_directory, "exps")
    file_name = "batch_results_" + datetime.now().strftime("%m_%d") + ".txt"
    file_path = os.path.join(exps_directory, file_name)
    
    # check if the file already exists - append or write
    if os.path.exists(file_path):
        mode = "a" 
    else:
        mode = "w" 
        
        
    with open(file_path, mode) as file:
        last_columns, metrics = determine_logfile_columns(plotargs)
        if mode == "w":
            file.write("Model\t\tDomain\t\tLR\t\tDropout\t\t{}\n".format(last_columns))                
        if 'Vs30' in kwargs['gt_select']:    
            file.write(f"{kwargs.get('model_select')}\t\t{kwargs.get('freq_flag')}\t\t\t{kwargs.get('lr')}\t\t{kwargs.get('dropout_rate')}\t\t\t{np.round(np.mean(np.abs(plotargs.get('Vs30_diff'))),2)}\n")
        else:
            file.write(f"{kwargs.get('model_select')}\t\t{kwargs.get('freq_flag')}\t\t{kwargs.get('lr')}\t\t{kwargs.get('dropout_rate')}\t\t\t{np.round(np.mean(plotargs.get('metric_dist')),2)}\n")
    run.finish()
        
lat = [37]
lon = [34.5]
signaltime = [60]
lr = [0.0001]
freqtime = False #^time
batch_size = [16]
hidden = [256]
LSTM_split = [1000]
transferpath = r"D:\Baris\codes\transfer\EXPbaris2023_08_11_02_01_Freq_True_duration_60_Ep_100_lr_1e-06_dropout_01_ResNet/"

for l in range(len(LSTM_split)):
    for j in range(len(batch_size)):
        for i in range(len(lr)):
            for k in range(len(hidden)):
                      
                print("\n>>>>>>>>>>>>>>>>>>>>>> lr", lr[i],"hidden", hidden[k],"LSTM_split", LSTM_split[l])
                start_time = time.time()
                # main(lat[0],lon[0],signaltime[0],transferpath[0],freqtime[k], lr[i], dropout[l])
                main(lat[0],lon[0],signaltime[0],transferpath,hidden[k], lr[i],batch_size[j], LSTM_split[l])
                end_time = time.time()
                elapsed_time_secs = end_time - start_time
                formatted_time = str(timedelta(seconds=int(elapsed_time_secs)))
                print(f">>>>>>>>>>>>>>>>>>>>>> Execution time--> {formatted_time}")
