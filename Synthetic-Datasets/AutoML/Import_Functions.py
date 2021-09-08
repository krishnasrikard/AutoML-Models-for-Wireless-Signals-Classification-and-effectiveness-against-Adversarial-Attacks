import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import seaborn as sns
import scipy.io
import os

# Importing Data for all SNR Ratio's
def ImportData(Path):
	ModulationSchemes = os.listdir(Path)

	Dataset = {}
	for scheme in ModulationSchemes:
		DataPath = Path + "/" + scheme
		files = os.listdir(DataPath)
		DataofScheme = {}
		for f in files:
			Data = scipy.io.loadmat(DataPath + "/" + f)
			snr = Data['snr'][0][0]
			rx = Data['rx']
			InPhase, QuadPhase = rx.real, rx.imag
			DataofScheme[snr] = np.append(InPhase, QuadPhase, axis=1)
		Dataset[scheme] = DataofScheme

	return Dataset


# Train and Validation Datasets
"""
SNRs = [-15,-10,-5,0,5,10,15,20,25,30]dB
Training:
Model is trained on all SNR Ratios

Validation:
Model is evaluated on all SNR Ratios
"""
def ImportDatasets(Channel,L=None,test_size=0.2):
	if Channel == "AWGN":
		Path =  "../Data/" + Channel
	elif Channel == "Rayleigh":
		Path = "../Data/" + Channel + "/" + str(L)
	Data = ImportData(Path)
    
	Dataset = {}
	Dataset['Classes'] = list(Data.keys())
	OneHotClasses = np.eye(len(Dataset['Classes']))

	Classes = {}
	for i in range(len(Dataset['Classes'])):
		Classes[Dataset['Classes'][i]] = OneHotClasses[i]

	Valid_SNRs = [-15,-10,-5,0,5,10,15,20,25,30]

	if Channel == "AWGN":
		X_Train, y_Train = [],[]
		X_Valid, y_Valid = {}, {}

		for snr in Valid_SNRs:
			X_Valid[snr] = []
			y_Valid[snr] = []
			for modType in Classes.keys():
				data = Data[modType][snr]
				N = int(test_size*data.shape[0])
				
				train = data[N:]
				valid = data[:N]
				
				X_Train.append(train)
				X_Valid[snr].append(valid)
				
				y_Train.append(np.repeat(np.expand_dims(Classes[modType],axis=0),train.shape[0],axis=0))
				y_Valid[snr].append(np.repeat(np.expand_dims(Classes[modType],axis=0),valid.shape[0],axis=0))
			
			X_Valid[snr] = np.array(X_Valid[snr]).reshape(-1,2)
			y_Valid[snr] = np.array(y_Valid[snr]).reshape(-1,3)
		
		X_Train = np.array(X_Train).reshape(-1,2)
		y_Train = np.array(y_Train).reshape(-1,3)
		
		
	if Channel == "Rayleigh":
		X_Train, y_Train = [],[]
		X_Valid, y_Valid = {}, {}

		for snr in Valid_SNRs:
			X_Valid[snr] = []
			y_Valid[snr] = []
			for modType in Classes.keys():
				data = Data[modType][snr]
				N = int(test_size*data.shape[0])
				
				train = data[N:].reshape(-1,100,2)
				valid = data[:N].reshape(-1,100,2)
				
				X_Train.append(train)
				X_Valid[snr].append(valid)
				
				y_Train.append(np.repeat(np.expand_dims(Classes[modType],axis=0),train.shape[0],axis=0))
				y_Valid[snr].append(np.repeat(np.expand_dims(Classes[modType],axis=0),valid.shape[0],axis=0))
			
			X_Valid[snr] = np.array(X_Valid[snr]).reshape(-1,100,2)
			y_Valid[snr] = np.array(y_Valid[snr]).reshape(-1,3)
		
		X_Train = np.array(X_Train).reshape(-1,100,2)
		y_Train = np.array(y_Train).reshape(-1,3)	
		
	return X_Train, y_Train, X_Valid, y_Valid
