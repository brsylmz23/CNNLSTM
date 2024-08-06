from __future__ import print_function
import os
import torch
import numpy as np
import scipy.io as sio
from math import radians, cos, sin, asin, sqrt
from numpy.random import default_rng
import math
import scipy
from utils import haversine, loader_helper, CrossValidation, debug_point
import pdb
import math


def datasetCreator(**kwargs):
    """Create a dataset based on the specified kwargs
    
    Parameters:
    **kwargs: Keyword arguments specifying the dataset details
    
    Returns:
    list: List of attributes for the dataset
    """
    if kwargs["dataset"] == "AFAD":
        attributes = datasetCreator_AFAD(**kwargs)
        return attributes
    elif kwargs["dataset"] == "STEAD":
        attributes = datasetCreator_STEAD(**kwargs)
        return attributes
    elif kwargs["dataset"] == "KANDILLI":
        attributes = datasetCreator_KANDILLI(**kwargs)
        return attributes
    elif kwargs["dataset"] == "KANDILLI_AFAD":
        attributes = datasetCreator_KANDILLI_AFAD(**kwargs)
        return attributes

def datasetCreator_AFAD(**kwargs):
    
    """Create a dataset for the AFAD dataset based on the specified kwargs
    
    Parameters:
    **kwargs: Keyword arguments specifying the dataset details
    
    Returns:
    list: List of attributes for the dataset
    """
    input_signal = []
    groundTruth = []
    gettingfiles = []
    magnitude_val = []
    stat2epi = []
    signal_list = []
    epicenter_depth = []
    lats = []
    longs = []
    stationco = []
    epicentraldist = []
    altitudes = []
    stat_id = []
    sP_3_channel = []
    tpga_ind = []  # 0 = left augmentation, 1 = middle augmentation, 2 = right augmentation
    stat_info = []
    alt = []
    Vs30 = []
    #skipped_file=[[] for ii in range(2)]
    skipped_file = {'afad': {'epi2loc is greater than the radius': [],
                                'station_km is greater than the stat_dist': [],
                                'eliminated epiDist values' : [],
                                'depth is greater than the epicenterDepth': [],
                                'duration is smaller than desired': [],
                                'Epicenter is empty': [],
                                'Depth is empty': [],
                                'Not skipped, Magnitude is empty': [],
                                'Not skipped, Altitude is empty': [],
                                'Vs30': [],
                                'Misc: out of the try block': []
                                },
                    'kandilli': {'epi2loc is greater than the radius': [],
                                    'station_km is greater than the stat_dist': [],
                                    'eliminated epiDist values' : [],
                                    'depth is greater than the epicenterDepth': [],
                                    'duration is smaller than desired': [],
                                    'Epicenter is empty': [],
                                    'Depth is empty': [],
                                    'Not skipped, Magnitude is empty': [],
                                    'Not skipped, Altitude is empty': [],
                                    'Vs30': [],
                                    'Misc: out of the try block': []
                                    },
                    'stead': {'epi2loc is greater than the radius': [],
                                    'station_km is greater than the stat_dist': [],
                                    'eliminated epiDist values' : [],
                                    'depth is greater than the epicenterDepth': [],
                                    'duration is smaller than desired': [],
                                    'Epicenter is empty': [],
                                    'Depth is empty': [],
                                    'Not skipped, Magnitude is empty': [],
                                    'Not skipped, Altitude is empty': [],
                                    'Vs30': [],
                                    'Misc: out of the try block': []
                                    },
                    'except_block':[]}
     
    fname = kwargs.get('AFAD_Path')
    for b in os.listdir(fname)[0:len(os.listdir(fname))]: 
        dictname = 'afad'
        dataset=sio.loadmat('{}/{}'.format(fname, b))
        # Extract various data points from the loaded file, such as the latitude and longitude of the station, the number of data points,
        # the epicenter location, and the altitude.
        
        stationLat = dataset['EQ'][0][0]['anEQ']['statco'][0][0][0][0]
        stationLon = dataset['EQ'][0][0]['anEQ']['statco'][0][0][0][1]
        stationcoor = dataset['EQ'][0][0]['anEQ']['statco'][0][0][0]
                
        numofData = dataset['EQ'][0][0]['anEQ']['numofData'][0][0][0][0]
        
        epicenterLat = dataset['EQ'][0][0]['anEQ']['epicenter'][0][0][0][0]
        epicenterLon = dataset['EQ'][0][0]['anEQ']['epicenter'][0][0][0][1]
        
        
        # Ptime= dataset['EQ'][0][0]['anEQ']["Ptime"][0][0][0][0]
        # Stime =dataset['EQ'][0][0]['anEQ']["Stime"][0][0][0][0]
        
        Ptime= 0
        Stime =0
            
        epicenterDepth = float(dataset['EQ'][0][0]['anEQ']['depth'][0][0][0][0])/kwargs.get('km2meter')
        Vs30 = dataset['EQ'][0][0]['anEQ']["Vs30"][0][0][0][0]  
        # Pass if Vs30 is required
        if 'Vs30' in kwargs['gt_select']:
            if isinstance(Vs30, np.uint16):
                Vs30 = Vs30 
            # else:
            #     skipped_file[dictname]['Vs30'].append(b)
            #     continue
        # Check if station altitude information is gonna be fed to FC layers or not.
        if kwargs['add_station_altitude']:
            if 'alt' in dataset['EQ'][0][0]['anEQ'].dtype.fields:
                if len(dataset['EQ'][0][0]['anEQ']['alt'][0][0])!=0:                
                    alt = dataset['EQ'][0][0]['anEQ']['alt'][0][0][0][0]
                else:
                    alt = 0            
        
        # If available, extract the magnitude value of the earthquake from the file.
        if 'magnitudeval' in dataset['EQ'][0][0]['anEQ'].dtype.fields:
            if len(dataset['EQ'][0][0]['anEQ']['magnitudeval'][0][0])!=0:                
                mag = dataset['EQ'][0][0]['anEQ']['magnitudeval'][0][0][0][0]
            else:
                mag = 0    
                skipped_file[dictname]["Not skipped, Magnitude is empty"].append(b)
        
        # Extract the signal data from the file and find the time of the maximum acceleration.                        
        signal = dataset['EQ'][0][0]['anEQ']['Accel'][0][0] 
        
        if kwargs['SP']:
            signal_acc = dataset['EQ'][0][0]['anEQ']['Accel'][0][0] 
            sp_signal = np.zeros(len(signal_acc))
            for i in range(len(signal_acc)):
                if int(dataset['EQ'][0][0]['anEQ']["Ptime"][0][0][0][0]*kwargs['fs']) <= i <= int(dataset['EQ'][0][0]['anEQ']["Stime"][0][0][0][0]*kwargs['fs']):
                    sp_signal[i] = 1
            signal = np.hstack((signal_acc, sp_signal.reshape(len(signal_acc), 1)))
        else:
            signal = dataset['EQ'][0][0]['anEQ']['Accel'][0][0] 
            #######################################################
            
        gettingfiles, groundTruth, input_signal,  magnitude_val, stat2epi, signal_list, epicenter_depth, lats, longs, stationco , epicentraldist ,altitudes , stat_id , sP_3_channel, tpga_ind ,stat_info, skipped_file, Vs30, Ptime, Stime = loader_helper(
            b, signal, alt, mag, numofData, epicenterLat,epicenterLon, epicenterDepth, 
            stationLat,stationLon, stationcoor, gettingfiles, groundTruth, input_signal, 
            magnitude_val, stat2epi, signal_list, epicenter_depth, lats, longs, stationco, 
            epicentraldist, altitudes, stat_id , sP_3_channel , tpga_ind ,stat_info, 
            skipped_file, Vs30, Ptime,Stime, **kwargs)
        
        
    trind, vlind, tsind = CrossValidation(gettingfiles, stat_id, **kwargs)
    # SalihPS.txt dosyasındaki isimleri oku
    # with open(r"D:\Baris\codes\baris_LSTM - Copy\SalihPS.txt", "r") as dosya:
    #     salih_isimleri = set(dosya.read().splitlines())
    
    # # Eşleşen ve eşleşmeyen isimleri depolamak için boş listeler oluştur
    # eslesen_isimler = []
    # farkli_isimler = []
    
    # # Listenin elemanlarına bak
    # for isim in gettingfiles:
    #     if isim in salih_isimleri:
    #         eslesen_isimler.append(isim)
    #     else:
    #         farkli_isimler.append(isim)
            
    # # trind, vlind, tsind = CrossValidation(gettingfiles,eslesen_isimler,farkli_isimler, stat_id, **kwargs)
    # train_val = [index for index, isim in enumerate(gettingfiles) if isim not in salih_isimleri]
    # trind = train_val[:int(len(train_val)*0.95)]
    # vlind = train_val[int(len(train_val)*0.95):]
    # tsind = [index for index, isim in enumerate(gettingfiles) if isim in salih_isimleri]
    
         
    print('training data size is:' + str(len(trind)))
    print('validation set size is:' + str(len(vlind)))
    print('test set size is:' + str(len(tsind)))
    
    for i in range(len(groundTruth)):
        try:
            # İlk değerin dtype'ını kontrol et
            if not np.issubdtype(groundTruth[i][0].dtype, np.number) or np.isnan(groundTruth[i][0]):
                groundTruth[i] = np.array([0], dtype=np.float64)
        except Exception as e:
            groundTruth[i] = np.array([0], dtype=np.float64)
        
    # Making them arrays
    groundTruth = np.asarray(groundTruth, dtype=np.float32)
    input_signal = np.asarray(input_signal, dtype=np.float32)
    stat_info = np.asarray(stat_info, dtype=np.float32)

    #Calculate training set's mean values for each channel
    trMeanR = input_signal[trind,:,:,0].mean()
    trMeanG = input_signal[trind,:,:,1].mean()
    trMeanB = input_signal[trind,:,:,2].mean()
    
    #Calculate station info's mean values
    MeanStatLat = stat_info[trind,0].mean()
    MeanStatLon = stat_info[trind,1].mean()
    #MeanStatAlt= stat_info_temp[trind,2].mean()
    
    
    # SUbtract the station coordinates from epicenters' to provide a directional sense.
    for order,element in enumerate(kwargs.get("gt_select")):
        # To ensure that nothing is affected unless epiLAT or epiLON is present, trMeans=0 and trSTD=1 are set.
        if element == 'epiLAT':
            groundTruth[:, order] -= stationLat
        elif element == 'epiLON':
            groundTruth[:, order] -= stationLon
    # Calculate mean and std for each output column
    num_output = len(groundTruth[0])  
    trMeans = []  
    trStds = []
    
    for element in range(num_output):
        trMean = groundTruth[trind, element].mean()
        trStd = groundTruth[trind, element].std()    
        trMeans.append(trMean)
        trStds.append(trStd)

    groundTruth=torch.from_numpy(groundTruth)
    stat_info = torch.from_numpy(stat_info)
    
    #Save all of them in a dictionary
    attributes = {'Matfiles': gettingfiles, 'Signal': signal_list, 'station_km':stat2epi, 'Latitudes': lats,'trStds':trStds,'trMeans':trMeans,'trMeanR': trMeanR,'trMeanG':trMeanG,'trMeanB':trMeanB, 
                  'Longitudes': longs, 'Depths': epicenter_depth, 'Station_Coordinates': stationco, 'Epicentral_distance(epi2loc)': epicentraldist, 
                  'Magnitudes': magnitude_val, 'Altitudes': altitudes, 'Station ID': stat_id, 'trind':trind, 'vlind':vlind, 'tsind': tsind, 
                  'groundTruth': groundTruth, 'stat_info': stat_info, "tpga_ind": tpga_ind, "MeanStatLat":MeanStatLat, "MeanStatLon":MeanStatLon, 'skipped_file':skipped_file}                                
 
    return attributes  
def datasetCreator_STEAD(**kwargs):
    """Create a dataset for the STEAD dataset based on the specified kwargs
    
    Parameters:
    **kwargs: Keyword arguments specifying the dataset details
    
    Returns:
    list: List of attributes for the dataset
    """
    
    input_signal = []
    groundTruth = []
    gettingfiles = []
    magnitude_val = []
    stat2epi = []
    signal_list = []
    epicenter_depth = []
    lats = []
    longs = []
    stationco = []
    epicentraldist = []
    altitudes = []
    stat_id = []
    sP_3_channel = []
    tpga_ind = []  # 0 = left augmentation, 1 = middle augmentation, 2 = right augmentation
    stat_info = []
    alt = []
    Vs30 = []
    #skipped_file=[[] for ii in range(2)]
    skipped_file = {'afad': {'epi2loc is greater than the radius': [],
                                'station_km is greater than the stat_dist': [],
                                'eliminated epiDist values' : [],
                                'depth is greater than the epicenterDepth': [],
                                'duration is smaller than desired': [],
                                'Epicenter is empty': [],
                                'Depth is empty': [],
                                'Not skipped, Magnitude is empty': [],
                                'Not skipped, Altitude is empty': [],
                                'Vs30': [],
                                'Misc: out of the try block': []
                                },
                    'kandilli': {'epi2loc is greater than the radius': [],
                                    'station_km is greater than the stat_dist': [],
                                    'eliminated epiDist values' : [],
                                    'depth is greater than the epicenterDepth': [],
                                    'duration is smaller than desired': [],
                                    'Epicenter is empty': [],
                                    'Depth is empty': [],
                                    'Not skipped, Magnitude is empty': [],
                                    'Not skipped, Altitude is empty': [],
                                    'Vs30': [],
                                    'Misc: out of the try block': []
                                    },
                    'stead': {'epi2loc is greater than the radius': [],
                                    'station_km is greater than the stat_dist': [],
                                    'eliminated epiDist values' : [],
                                    'depth is greater than the epicenterDepth': [],
                                    'duration is smaller than desired': [],
                                    'Epicenter is empty': [],
                                    'Depth is empty': [],
                                    'Not skipped, Magnitude is empty': [],
                                    'Not skipped, Altitude is empty': [],
                                    'Vs30': [],
                                    'Misc: out of the try block': []
                                    },
                    'except_block':[]}
    
    fname = kwargs.get('STEAD_Path')
    for b in os.listdir(fname)[0:len(os.listdir(fname))]: 
        if not b.endswith(".mat"):
            continue
        try:
            dataset=sio.loadmat('{}/{}'.format(fname, b))
            dictname = 'stead'
            
                
            if 'epiLAT' in kwargs['gt_select'] or 'epiLON' in kwargs['gt_select']:
                if len(dataset["anEQ"]["AccWaveform"][0][0]) == 0:
                    skipped_file[dictname]['Epicenter is empty'].append(b)
                    #breakpoint()
                    continue
                
            # if 'Depth' in kwargs['gt_select']:
            #     if len(dataset["anEQ"]["depth"][0][0][0]) == 0 or math.isnan(dataset["anEQ"]['depth']): 
            #         skipped_file[dictname]['Depth is empty'].append(b)
            #         #debug_point()
            #         #breakpoint()
            #         continue
                
            stationLat = dataset["anEQ"]["receiver_latitude"][0][0][0][0]
            stationLon = dataset["anEQ"]["receiver_longitude"][0][0][0][0]
            stationcoor = [stationLat,stationLon]
            station_name = dataset['anEQ']['network_code'][0][0][0][0]
            stat_id.append(station_name)
            numofData = 6000
            
            epicenterLat = dataset["anEQ"]["source_latitude"][0][0][0][0]
            epicenterLon = dataset["anEQ"]["source_longitude"][0][0][0][0]
            epicenterDepth = dataset["anEQ"]["source_depth_km"][0][0][0][0] /kwargs.get('km2meter')
            
            # Check if station altitude information is gonna be fed to FC layers or not.
            if kwargs['add_station_altitude']:
                alt = dataset["anEQ"]["receiver_elevation_m"][0][0][0][0]
            mag = dataset["anEQ"]["source_magnitude"][0][0][0][0]
            
            spt = int(dataset["anEQ"]["p_arrival_sample"][0][0][0][0])
            sst = int(dataset["anEQ"]["s_arrival_sample"][0][0][0][0])
            P_t = float(dataset["anEQ"]["p_travel_sec"][0][0][0][0])
            
            dis = round(float(dataset["anEQ"]["source_distance_km"][0][0][0][0]))
 
            #######################################################
        except: 
            print(b)
            skipped_file[dictname]["Misc: out of the try block"].append(b)
            #breakpoint()
            continue
           
            # Extract the signal data from the file and find the time of the maximum acceleration.                        
        signal = dataset["anEQ"]["AccWaveform"][0][0] 
        gettingfiles, groundTruth, input_signal,  magnitude_val, stat2epi, signal_list, epicenter_depth, lats, longs, stationco , epicentraldist ,altitudes , stat_id , sP_3_channel, tpga_ind ,stat_info, skipped_file, Vs30 = loader_helper(
            b, signal, alt, mag, numofData, epicenterLat,epicenterLon, epicenterDepth, 
            stationLat,stationLon, stationcoor, gettingfiles, groundTruth, input_signal, 
            magnitude_val, stat2epi, signal_list, epicenter_depth, lats, longs, stationco, 
            epicentraldist, altitudes, stat_id , sP_3_channel , tpga_ind ,stat_info, 
            skipped_file, Vs30, **kwargs)
            
    trind, vlind, tsind = CrossValidation(gettingfiles, stat_id, **kwargs)

    
    # Convert to np array
    groundTruth = np.asarray(groundTruth, dtype=np.float32)
    input_signal = np.asarray(input_signal, dtype=np.float32)
    stat_info = np.asarray(stat_info, dtype=np.float32)

    #Calculate training set's mean values for each channel
    trMeanR = input_signal[trind,:,:,0].mean()
    trMeanG = input_signal[trind,:,:,1].mean()
    trMeanB = input_signal[trind,:,:,2].mean()
    
    #Calculate station info's mean values
    MeanStatLat = stat_info[trind,0].mean()
    MeanStatLon = stat_info[trind,1].mean()
    # MeanStatAlt= stat_info_temp[trind,2].mean()

    # SUbtract the station coordinates from epicenters' to provide a directional sense.
    for order,element in enumerate(kwargs.get("gt_select")):
        # To ensure that nothing is affected unless epiLAT or epiLON is present, trMeans=0 and trSTD=1 are set.
        if element == 'epiLAT':
            groundTruth[:, order] -= stationLat
        elif element == 'epiLON':
            groundTruth[:, order] -= stationLon
    # Calculate mean and std for each output column
    num_output = len(groundTruth[0])  
    trMeans = []  
    trStds = []
    
    for element in range(num_output):
        trMean = groundTruth[trind, element].mean()
        trStd = groundTruth[trind, element].std()
        trMeans.append(trMean)
        trStds.append(trStd)

      
    groundTruth=torch.from_numpy(groundTruth)
    stat_info = torch.from_numpy(stat_info)
    
    #Save all of them in a dictionary
    attributes = {'Matfiles': gettingfiles, 'Signal': signal_list, 'station_km':stat2epi, 'Latitudes': lats,'trStds':trStds,'trMeans':trMeans,'trMeanR': trMeanR,'trMeanG':trMeanG,'trMeanB':trMeanB, 
                  'Longitudes': longs, 'Depths': epicenter_depth, 'Station_Coordinates': stationco, 'Epicentral_distance(epi2loc)': epicentraldist, 
                  'Magnitudes': magnitude_val, 'Altitudes': altitudes, 'Station ID': stat_id, 'trind':trind, 'vlind':vlind, 'tsind': tsind, 
                  'groundTruth': groundTruth, 'stat_info': stat_info, "tpga_ind": tpga_ind, "MeanStatLat":MeanStatLat, "MeanStatLon":MeanStatLon, "skipped_files":skipped_file }                                
                                                     
    return attributes
def datasetCreator_KANDILLI(**kwargs):
    """

    """
    input_signal = []
    groundTruth = []
    gettingfiles = []
    magnitude_val = []
    stat2epi = []
    signal_list = []
    epicenter_depth = []
    lats = []
    longs = []
    stationco = []
    epicentraldist = []
    altitudes = []
    stat_id = []
    sP_3_channel = []
    tpga_ind = []  # 0 = left augmentation, 1 = middle augmentation, 2 = right augmentation
    stat_info = []
    alt = []
    Vs30 = []
    #skipped_file=[[] for ii in range(2)]
    skipped_file = {'afad': {'epi2loc is greater than the radius': [],
                                'station_km is greater than the stat_dist': [],
                                'eliminated epiDist values' : [],
                                'depth is greater than the epicenterDepth': [],
                                'duration is smaller than desired': [],
                                'Epicenter is empty': [],
                                'Depth is empty': [],
                                'Not skipped, Magnitude is empty': [],
                                'Not skipped, Altitude is empty': [],
                                'Vs30': [],
                                'Misc: out of the try block': []
                                },
                    'kandilli': {'epi2loc is greater than the radius': [],
                                    'station_km is greater than the stat_dist': [],
                                    'eliminated epiDist values' : [],
                                    'depth is greater than the epicenterDepth': [],
                                    'duration is smaller than desired': [],
                                    'Epicenter is empty': [],
                                    'Depth is empty': [],
                                    'Not skipped, Magnitude is empty': [],
                                    'Not skipped, Altitude is empty': [],
                                    'Vs30': [],
                                    'Misc: out of the try block': []
                                    },
                    'stead': {'epi2loc is greater than the radius': [],
                                    'station_km is greater than the stat_dist': [],
                                    'eliminated epiDist values' : [],
                                    'depth is greater than the epicenterDepth': [],
                                    'duration is smaller than desired': [],
                                    'Epicenter is empty': [],
                                    'Depth is empty': [],
                                    'Not skipped, Magnitude is empty': [],
                                    'Not skipped, Altitude is empty': [],
                                    'Vs30': [],
                                    'Misc: out of the try block': []
                                    },
                    'except_block':[]}

    fname = kwargs.get('KANDILLI_Path')
    for b in os.listdir(fname)[0:len(os.listdir(fname))]: 
        
        dataset=sio.loadmat('{}/{}'.format(fname, b))
        dictname = 'kandilli'
        
       

        stationLat = dataset["anEQ"]["statco"][0][0][0][0]
        stationLon = dataset["anEQ"]["statco"][0][0][0][1]
        stationcoor = [stationLat,stationLon]
        station_name = dataset["anEQ"]["statID"][0][0][0]
        stat_id.append(station_name)
        
        numofData = dataset["anEQ"]["numofData"][0][0][0][0]
        
        try:
            epicenterLat = dataset["anEQ"]["epicenter"][0][0][0][0]
        except Exception as e:
            skipped_file[dictname]["Epicenter is empty"].append(b)
            gettingfiles.remove(b)
            #breakpoint()
            continue
        
        epicenterLon = dataset["anEQ"]["epicenter"][0][0][0][1]
        epicenterDepth_meters = dataset["anEQ"]["depth"][0][0][0][0]
        epicenterDepth =  epicenterDepth_meters/1000
       
        Ptime= 0
        Stime =0
           
        epicenterDepth = float(dataset['anEQ']['depth'][0][0][0][0])/kwargs.get('km2meter')
        Vs30 = dataset['anEQ']["Vs30"][0][0][0][0]  
        # Pass if Vs30 is required
        if 'Vs30' in kwargs['gt_select']:
            if isinstance(Vs30, np.uint16):
                Vs30 = Vs30 
            # else:
            #     skipped_file[dictname]['Vs30'].append(b)
            #     continue
        # Check if station altitude information is gonna be fed to FC layers or not.
        if kwargs['add_station_altitude']:
            if 'alt' in dataset['anEQ'].dtype.fields:
                if len(dataset['anEQ']['alt'][0][0])!=0:                
                    alt = dataset['anEQ']['alt'][0][0][0][0]
                else:
                    alt = 0            
       
        # If available, extract the magnitude value of the earthquake from the file.
        if 'magnitudeval' in dataset['anEQ'].dtype.fields:
            if len(dataset['anEQ']['magnitudeval'][0][0])!=0:                
                mag = dataset['anEQ']['magnitudeval'][0][0][0][0]
            else:
                mag = 0    
                skipped_file[dictname]["Not skipped, Magnitude is empty"].append(b)
       
        # Extract the signal data from the file and find the time of the maximum acceleration.                        
        signal = dataset['anEQ']['Accel'][0][0] 
       
        if kwargs['SP']:
            signal_acc = dataset['anEQ']['Accel'][0][0] 
            sp_signal = np.zeros(len(signal_acc))
            for i in range(len(signal_acc)):
                if int(dataset['anEQ']["Ptime"][0][0][0][0]*kwargs['fs']) <= i <= int(dataset['EQ'][0][0]['anEQ']["Stime"][0][0][0][0]*kwargs['fs']):
                    sp_signal[i] = 1
            signal = np.hstack((signal_acc, sp_signal.reshape(len(signal_acc), 1)))
        else:
            signal = dataset['anEQ']['Accel'][0][0] 
            #######################################################
           
        gettingfiles, groundTruth, input_signal,  magnitude_val, stat2epi, signal_list, epicenter_depth, lats, longs, stationco , epicentraldist ,altitudes , stat_id , sP_3_channel, tpga_ind ,stat_info, skipped_file, Vs30, Ptime, Stime = loader_helper(
            b, signal, alt, mag, numofData, epicenterLat,epicenterLon, epicenterDepth, 
            stationLat,stationLon, stationcoor, gettingfiles, groundTruth, input_signal, 
            magnitude_val, stat2epi, signal_list, epicenter_depth, lats, longs, stationco, 
            epicentraldist, altitudes, stat_id , sP_3_channel , tpga_ind ,stat_info, 
            skipped_file, Vs30, Ptime,Stime, **kwargs)
           
           
    trind, vlind, tsind = CrossValidation(gettingfiles, stat_id, **kwargs)
    # SalihPS.txt dosyasındaki isimleri oku
    # with open(r"D:\Baris\codes\baris_LSTM - Copy\SalihPS.txt", "r") as dosya:
    #     salih_isimleri = set(dosya.read().splitlines())
    
    # # Eşleşen ve eşleşmeyen isimleri depolamak için boş listeler oluştur
    # eslesen_isimler = []
    # farkli_isimler = []
    
    # # Listenin elemanlarına bak
    # for isim in gettingfiles:
    #     if isim in salih_isimleri:
    #         eslesen_isimler.append(isim)
    #     else:
    #         farkli_isimler.append(isim)
            
    # # trind, vlind, tsind = CrossValidation(gettingfiles,eslesen_isimler,farkli_isimler, stat_id, **kwargs)
    # train_val = [index for index, isim in enumerate(gettingfiles) if isim not in salih_isimleri]
    # trind = train_val[:int(len(train_val)*0.95)]
    # vlind = train_val[int(len(train_val)*0.95):]
    # tsind = [index for index, isim in enumerate(gettingfiles) if isim in salih_isimleri]
    
         
    print('training data size is:' + str(len(trind)))
    print('validation set size is:' + str(len(vlind)))
    print('test set size is:' + str(len(tsind)))
    
    for i in range(len(groundTruth)):
        try:
            # İlk değerin dtype'ını kontrol et
            if not np.issubdtype(groundTruth[i][0].dtype, np.number) or np.isnan(groundTruth[i][0]):
                groundTruth[i] = np.array([0], dtype=np.float64)
        except Exception as e:
            groundTruth[i] = np.array([0], dtype=np.float64)
        
    # Making them arrays
    groundTruth = np.asarray(groundTruth, dtype=np.float32)
    input_signal = np.asarray(input_signal, dtype=np.float32)
    stat_info = np.asarray(stat_info, dtype=np.float32)

    #Calculate training set's mean values for each channel
    trMeanR = input_signal[trind,:,:,0].mean()
    trMeanG = input_signal[trind,:,:,1].mean()
    trMeanB = input_signal[trind,:,:,2].mean()
    
    #Calculate station info's mean values
    MeanStatLat = stat_info[trind,0].mean()
    MeanStatLon = stat_info[trind,1].mean()
    #MeanStatAlt= stat_info_temp[trind,2].mean()
    
    
    # SUbtract the station coordinates from epicenters' to provide a directional sense.
    for order,element in enumerate(kwargs.get("gt_select")):
        # To ensure that nothing is affected unless epiLAT or epiLON is present, trMeans=0 and trSTD=1 are set.
        if element == 'epiLAT':
            groundTruth[:, order] -= stationLat
        elif element == 'epiLON':
            groundTruth[:, order] -= stationLon
    # Calculate mean and std for each output column
    num_output = len(groundTruth[0])  
    trMeans = []  
    trStds = []
    
    for element in range(num_output):
        trMean = groundTruth[trind, element].mean()
        trStd = groundTruth[trind, element].std()    
        trMeans.append(trMean)
        trStds.append(trStd)

    groundTruth=torch.from_numpy(groundTruth)
    stat_info = torch.from_numpy(stat_info)
    
    #Save all of them in a dictionary
    attributes = {'Matfiles': gettingfiles, 'Signal': signal_list, 'station_km':stat2epi, 'Latitudes': lats,'trStds':trStds,'trMeans':trMeans,'trMeanR': trMeanR,'trMeanG':trMeanG,'trMeanB':trMeanB, 
                  'Longitudes': longs, 'Depths': epicenter_depth, 'Station_Coordinates': stationco, 'Epicentral_distance(epi2loc)': epicentraldist, 
                  'Magnitudes': magnitude_val, 'Altitudes': altitudes, 'Station ID': stat_id, 'trind':trind, 'vlind':vlind, 'tsind': tsind, 
                  'groundTruth': groundTruth, 'stat_info': stat_info, "tpga_ind": tpga_ind, "MeanStatLat":MeanStatLat, "MeanStatLon":MeanStatLon, 'skipped_file':skipped_file}                                
 
    return attributes  

def datasetCreator_KANDILLI_AFAD(**kwargs):
    
    input_signal = []
    groundTruth = []
    gettingfiles = []
    magnitude_val = []
    stat2epi = []
    signal_list = []
    epicenter_depth = []
    lats = []
    longs = []
    stationco = []
    epicentraldist = []
    altitudes = []
    stat_id = []
    sP_3_channel = []
    tpga_ind = []  # 0 = left augmentation, 1 = middle augmentation, 2 = right augmentation
    stat_info = []
    alt = []
    Vs30 = []
    num_afad = 0
    stat2epi_afad = []
    num_kandilli = 0
    stat2epi_kandilli = []
    #skipped_file=[[] for ii in range(2)]
    skipped_file = {'afad': {'epi2loc is greater than the radius': [],
                                'station_km is greater than the stat_dist': [],
                                'eliminated epiDist values' : [],
                                'depth is greater than the epicenterDepth': [],
                                'duration is smaller than desired': [],
                                'Epicenter is empty': [],
                                'Depth is empty': [],
                                'Not skipped, Magnitude is empty': [],
                                'Not skipped, Altitude is empty': [],
                                'Vs30': [],
                                'Misc: out of the try block': []
                                },
                    'kandilli': {'epi2loc is greater than the radius': [],
                                    'station_km is greater than the stat_dist': [],
                                    'eliminated epiDist values' : [],
                                    'depth is greater than the epicenterDepth': [],
                                    'duration is smaller than desired': [],
                                    'Epicenter is empty': [],
                                    'Depth is empty': [],
                                    'Not skipped, Magnitude is empty': [],
                                    'Not skipped, Altitude is empty': [],
                                    'Vs30': [],
                                    'Misc: out of the try block': []
                                    },
                    'stead': {'epi2loc is greater than the radius': [],
                                    'station_km is greater than the stat_dist': [],
                                    'eliminated epiDist values' : [],
                                    'depth is greater than the epicenterDepth': [],
                                    'duration is smaller than desired': [],
                                    'Epicenter is empty': [],
                                    'Depth is empty': [],
                                    'Not skipped, Magnitude is empty': [],
                                    'Not skipped, Altitude is empty': [],
                                    'Vs30': [],
                                    'Misc: out of the try block': []
                                    },
                    'except_block':[]}
    
    #############################
    #Combining the afad &kandilli files and sorting them acc to event dates.
    afad_path = kwargs.get('AFAD_Path')
    kandilli_path = kwargs.get('KANDILLI_Path')
    
    afad_fnames = os.listdir(afad_path)
    afad_event_dates = [fname[:14] for fname in afad_fnames]

    kandilli_fnames = os.listdir(kandilli_path)
    kandilli_event_dates = [fname[:14] for fname in kandilli_fnames]
    
    afad_files = list(zip(afad_event_dates, afad_fnames))
    kandilli_files = list(zip(kandilli_event_dates, kandilli_fnames))
    
    all_files = afad_files + kandilli_files
    all_files.sort(key=lambda x: x[0])
    
    for b_ in all_files[0:len(all_files)]: 
        b = b_[1]
        
        # Rest of the fcn is directed to AFAD or KANDILLI fcns according to the last 4 digits of fnames, i.e., whether they are numeric or string valued
        last_four = b[len(b)-8:len(b)-4]
        
        if last_four.isdigit():
            dictname = 'afad'
            dataset=sio.loadmat('{}/{}'.format(afad_path, b))
            # Extract various data points from the loaded file, such as the latitude and longitude of the station, the number of data points,
            # the epicenter location, and the altitude.
            
            stationLat = dataset['EQ'][0][0]['anEQ']['statco'][0][0][0][0]
            stationLon = dataset['EQ'][0][0]['anEQ']['statco'][0][0][0][1]
            stationcoor = dataset['EQ'][0][0]['anEQ']['statco'][0][0][0]
                    
            numofData = dataset['EQ'][0][0]['anEQ']['numofData'][0][0][0][0]
            
            epicenterLat = dataset['EQ'][0][0]['anEQ']['epicenter'][0][0][0][0]
            epicenterLon = dataset['EQ'][0][0]['anEQ']['epicenter'][0][0][0][1]
            
            
            # Ptime= dataset['EQ'][0][0]['anEQ']["Ptime"][0][0][0][0]
            # Stime =dataset['EQ'][0][0]['anEQ']["Stime"][0][0][0][0]
            
            Ptime= 0
            Stime =0
                
            epicenterDepth = float(dataset['EQ'][0][0]['anEQ']['depth'][0][0][0][0])/kwargs.get('km2meter')
            Vs30 = dataset['EQ'][0][0]['anEQ']["Vs30"][0][0][0][0]  
            # Pass if Vs30 is required
            if 'Vs30' in kwargs['gt_select']:
                if isinstance(Vs30, np.uint16):
                    Vs30 = Vs30 
                # else:
                #     skipped_file[dictname]['Vs30'].append(b)
                #     continue
            # Check if station altitude information is gonna be fed to FC layers or not.
            if kwargs['add_station_altitude']:
                if 'alt' in dataset['EQ'][0][0]['anEQ'].dtype.fields:
                    if len(dataset['EQ'][0][0]['anEQ']['alt'][0][0])!=0:                
                        alt = dataset['EQ'][0][0]['anEQ']['alt'][0][0][0][0]
                    else:
                        alt = 0            
            
            # If available, extract the magnitude value of the earthquake from the file.
            if 'magnitudeval' in dataset['EQ'][0][0]['anEQ'].dtype.fields:
                if len(dataset['EQ'][0][0]['anEQ']['magnitudeval'][0][0])!=0:                
                    mag = dataset['EQ'][0][0]['anEQ']['magnitudeval'][0][0][0][0]
                else:
                    mag = 0    
                    skipped_file[dictname]["Not skipped, Magnitude is empty"].append(b)
            
            # Extract the signal data from the file and find the time of the maximum acceleration.                        
            signal = dataset['EQ'][0][0]['anEQ']['Accel'][0][0] 
            
            if kwargs['SP']:
                signal_acc = dataset['EQ'][0][0]['anEQ']['Accel'][0][0] 
                sp_signal = np.zeros(len(signal_acc))
                for i in range(len(signal_acc)):
                    if int(dataset['EQ'][0][0]['anEQ']["Ptime"][0][0][0][0]*kwargs['fs']) <= i <= int(dataset['EQ'][0][0]['anEQ']["Stime"][0][0][0][0]*kwargs['fs']):
                        sp_signal[i] = 1
                signal = np.hstack((signal_acc, sp_signal.reshape(len(signal_acc), 1)))
            else:
                signal = dataset['EQ'][0][0]['anEQ']['Accel'][0][0] 
                #######################################################
                
            gettingfiles, groundTruth, input_signal,  magnitude_val, stat2epi, signal_list, epicenter_depth, lats, longs, stationco , epicentraldist ,altitudes , stat_id , sP_3_channel, tpga_ind ,stat_info, skipped_file, Vs30, Ptime, Stime = loader_helper(
                b, signal, alt, mag, numofData, epicenterLat,epicenterLon, epicenterDepth, 
                stationLat,stationLon, stationcoor, gettingfiles, groundTruth, input_signal, 
                magnitude_val, stat2epi, signal_list, epicenter_depth, lats, longs, stationco, 
                epicentraldist, altitudes, stat_id , sP_3_channel , tpga_ind ,stat_info, 
                skipped_file, Vs30, Ptime,Stime, **kwargs)
        else:
            dataset=sio.loadmat('{}/{}'.format(kandilli_path, b))
            dictname = 'kandilli'
            
           

            stationLat = dataset["anEQ"]["statco"][0][0][0][0]
            stationLon = dataset["anEQ"]["statco"][0][0][0][1]
            stationcoor = [stationLat,stationLon]
            station_name = dataset["anEQ"]["statID"][0][0][0]
            stat_id.append(station_name)
            
            numofData = dataset["anEQ"]["numofData"][0][0][0][0]
            
            try:
                epicenterLat = dataset["anEQ"]["epicenter"][0][0][0][0]
            except Exception as e:
                skipped_file[dictname]["Epicenter is empty"].append(b)
                # gettingfiles.remove(b)
                #breakpoint()
                continue
            
            epicenterLon = dataset["anEQ"]["epicenter"][0][0][0][1]
            epicenterDepth_meters = dataset["anEQ"]["depth"][0][0][0][0]
            epicenterDepth =  epicenterDepth_meters/1000
           
            Ptime= 0
            Stime =0
               
            epicenterDepth = float(dataset['anEQ']['depth'][0][0][0][0])/kwargs.get('km2meter')
            Vs30 = dataset['anEQ']["Vs30"][0][0][0][0]  
            # Pass if Vs30 is required
            if 'Vs30' in kwargs['gt_select']:
                if isinstance(Vs30, np.uint16):
                    Vs30 = Vs30 
                # else:
                #     skipped_file[dictname]['Vs30'].append(b)
                #     continue
            # Check if station altitude information is gonna be fed to FC layers or not.
            if kwargs['add_station_altitude']:
                if 'alt' in dataset['anEQ'].dtype.fields:
                    if len(dataset['anEQ']['alt'][0][0])!=0:                
                        alt = dataset['anEQ']['alt'][0][0][0][0]
                    else:
                        alt = 0            
           
            # If available, extract the magnitude value of the earthquake from the file.
            if 'magnitudeval' in dataset['anEQ'].dtype.fields:
                if len(dataset['anEQ']['magnitudeval'][0][0])!=0:                
                    mag = dataset['anEQ']['magnitudeval'][0][0][0][0]
                else:
                    mag = 0    
                    skipped_file[dictname]["Not skipped, Magnitude is empty"].append(b)
           
            # Extract the signal data from the file and find the time of the maximum acceleration.                        
            signal = dataset['anEQ']['Accel'][0][0] 
           
            if kwargs['SP']:
                signal_acc = dataset['anEQ']['Accel'][0][0] 
                sp_signal = np.zeros(len(signal_acc))
                for i in range(len(signal_acc)):
                    if int(dataset['anEQ']["Ptime"][0][0][0][0]*kwargs['fs']) <= i <= int(dataset['EQ'][0][0]['anEQ']["Stime"][0][0][0][0]*kwargs['fs']):
                        sp_signal[i] = 1
                signal = np.hstack((signal_acc, sp_signal.reshape(len(signal_acc), 1)))
            else:
                signal = dataset['anEQ']['Accel'][0][0] 
                #######################################################
               
            gettingfiles, groundTruth, input_signal,  magnitude_val, stat2epi, signal_list, epicenter_depth, lats, longs, stationco , epicentraldist ,altitudes , stat_id , sP_3_channel, tpga_ind ,stat_info, skipped_file, Vs30, Ptime, Stime = loader_helper(
                b, signal, alt, mag, numofData, epicenterLat,epicenterLon, epicenterDepth, 
                stationLat,stationLon, stationcoor, gettingfiles, groundTruth, input_signal, 
                magnitude_val, stat2epi, signal_list, epicenter_depth, lats, longs, stationco, 
                epicentraldist, altitudes, stat_id , sP_3_channel , tpga_ind ,stat_info, 
                skipped_file, Vs30, Ptime,Stime, **kwargs)
            
            
    trind, vlind, tsind = CrossValidation(gettingfiles, stat_id, **kwargs)
         
    print('training data size is:' + str(len(trind)))
    print('validation set size is:' + str(len(vlind)))
    print('test set size is:' + str(len(tsind)))   
    
    for i in range(len(groundTruth)):
        try:
            # İlk değerin dtype'ını kontrol et
            if not np.issubdtype(groundTruth[i][0].dtype, np.number) or np.isnan(groundTruth[i][0]):
                groundTruth[i] = np.array([0], dtype=np.float64)
        except Exception as e:
            groundTruth[i] = np.array([0], dtype=np.float64)
            
    # Making them arrays
    groundTruth = np.asarray(groundTruth, dtype=np.float32)
    input_signal = np.asarray(input_signal, dtype=np.float32)
    stat_info = np.asarray(stat_info, dtype=np.float32)

    #Calculate training set's mean values for each channel
    trMeanR = input_signal[trind,:,:,0].mean()
    trMeanG = input_signal[trind,:,:,1].mean()
    trMeanB = input_signal[trind,:,:,2].mean()

    #Calculate station info's mean values
    MeanStatLat = stat_info[trind,0].mean()
    MeanStatLon = stat_info[trind,1].mean()
   # MeanStatAlt= stat_info_temp[trind,2].mean()
    
    
    #Calculate ground truth means and stds
    for order,element in enumerate(kwargs.get("gt_select")):
        # To ensure that nothing is affected unless epiLAT or epiLON is present, trMeans=0 and trSTD=1 are set.
        if element == 'epiLAT':
            groundTruth[:, order] -= stat_info[:,0]
        elif element == 'epiLON':
            groundTruth[:, order] -= stat_info[:,1]

    num_output = len(groundTruth[0])  
    trMeans = []  
    trStds = []
    
    for element in range(num_output):
        trMean = groundTruth[trind, element].mean()
        trStd = groundTruth[trind, element].std()
        trMeans.append(trMean)
        trStds.append(trStd)

      
    groundTruth=torch.from_numpy(groundTruth)
    stat_info = torch.from_numpy(stat_info)
    
    #Save all of them in a dictionary
    attributes = {'Matfiles': gettingfiles, 'Signal': signal_list, 'station_km':stat2epi, 'Latitudes': lats,'trStds':trStds,'trMeans':trMeans,'trMeanR': trMeanR,'trMeanG':trMeanG,'trMeanB':trMeanB, 
                  'Longitudes': longs, 'Depths': epicenter_depth, 'Station_Coordinates': stationco, 'Epicentral_distance(epi2loc)': epicentraldist, 
                  'Magnitudes': magnitude_val, 'Altitudes': altitudes, 'Station ID': stat_id, 'trind':trind, 'vlind':vlind, 'tsind': tsind, 
                  'groundTruth': groundTruth, 'stat_info': stat_info, "tpga_ind": tpga_ind, "MeanStatLat":MeanStatLat, "MeanStatLon":MeanStatLon, "skipped_files":skipped_file }                                
    
    # # Record data stats
    # if not os.path.exists(kwargs['save_path']):
    #     os.makedirs(kwargs['save_path'])
    # # Create and save the txt file
    # filename = "kandilli_afad_stats_epiDist_" + str(kwargs.get('stat_dist')) + "_duration_" + str(kwargs.get('signaltime')) + ".txt"
    # fname = os.path.join(kwargs['save_path'], filename)
    # total_eliminated_afad = 0
    # total_eliminated_kandilli = 0
    # for idx,b in enumerate(gettingfiles):
    #     last_four = b[len(b)-8:len(b)-4]
    #     if last_four.isdigit():
    #         num_afad +=1
    #         stat2epi_afad.append(stat2epi[idx])
    #     else:
    #         num_kandilli +=1
    #         stat2epi_kandilli.append(stat2epi[idx])
    
    # with open(fname, "w") as f:
    #     f.write("\t\t\tNUMBER OF FILES \tMEAN epiDist \tSTD epiDist \n") 
    #     f.write("-------------------------------------------------------------\n") 
        
    #     f.write('REMAINING FILES' + "\n")
    #     f.write('Afad:\t\t\t\t' + str(num_afad) + '\t\t' + str(np.round(np.mean(stat2epi_afad), 2)) + '\t\t' + str(np.round(np.std(stat2epi_afad), 2)) + "\n")
    #     f.write('Kandilli:\t\t\t' + str(num_kandilli) + '\t\t' + str(np.round(np.mean(stat2epi_kandilli), 2)) + '\t\t' + str(np.round(np.std(stat2epi_kandilli), 2)) + "\n\n")
    #     f.write("-------------------------------------------------------------\n")
    #     f.write('ELIMINATED FILES' + "\n")
    #     f.write('epiDist limit:\t\t' + str(kwargs['stat_dist']) + "\n")
    #     f.write('Kandilli-epiDist:\t' + str(len(skipped_file['kandilli']['station_km is greater than the stat_dist'])) + '\t\t' + 
    #             str(np.round(np.mean(skipped_file['kandilli']['eliminated epiDist values']), 2)) + '\t\t' + 
    #             str(np.round(np.std(skipped_file['kandilli']['eliminated epiDist values']), 2)) + "\n")
    #     f.write('Afad-epiDist:\t\t' + str(len(skipped_file['afad']['station_km is greater than the stat_dist'])) + '\t\t' + 
    #             str(np.round(np.mean(skipped_file['afad']['eliminated epiDist values']), 2)) + '\t\t' + 
    #             str(np.round(np.std(skipped_file['afad']['eliminated epiDist values']), 2)) + "\n\n")
        
    #     f.write("-------------------------------------------------------------\n\n")
    #     f.write('AFAD ELIMINTATED FILES' + "\n")
    #     # align the numbers in txt file
    #     max_length = max(len(key) for key in skipped_file['kandilli'].keys())
    #     for key in skipped_file['afad']:
    #         spaces = max_length - len(key) + 2
    #         num_elements = len(skipped_file['afad'][key])
    #         total_eliminated_afad += num_elements
    #         line = f"{key}{spaces * ' '}{num_elements}\n"
    #         f.write(line)
    #     f.write('Total\t' + str(total_eliminated_afad) + "\n")            
    #     f.write("-------------------------------------------------------------\n")
    #     f.write('KANDILLI ELIMINTATED FILES' + "\n")
        
    #     for key in skipped_file['kandilli']:
    #         spaces = max_length - len(key) + 2
    #         num_elements = len(skipped_file['kandilli'][key])
    #         total_eliminated_afad += num_elements
    #         line = f"{key}{spaces * ' '}{num_elements}\n"
    #         f.write(line)
    #     f.write('Total\t' + str(total_eliminated_afad) + "\n") 
    return attributes
