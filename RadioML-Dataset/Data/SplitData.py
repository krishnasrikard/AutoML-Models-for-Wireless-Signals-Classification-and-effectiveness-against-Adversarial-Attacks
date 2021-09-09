import numpy as np
import pickle

# Importing Data for all SNR Ratio's
def ImportData(Path):
    # Extracting Data
    with open(Path,'rb') as f:
        Data = pickle.load(f,encoding='latin')
    SNRs, ModulationSchemes = map(lambda j: sorted(list(set(map(lambda x: x[j], Data.keys())))), [1,0])
    
    Dataset = {}
    for modType in ModulationSchemes:
        Dataset[modType] = {}
        for snr in SNRs:
            data = Data[(modType,snr)]
            Dataset[modType][snr] = data 
    return Dataset
    

# Train and Validation Datasets
"""
Training:
Model is trained on all SNR Ratios

Validation:
Model is evaluated on all SNR Ratios
"""
def SplitData(Path, SavePath, test_size=0.2):
	Dataset = ImportData(Path)
	ModulationSchemes = list(Dataset.keys())
	SNRs = list(Dataset[ModulationSchemes[0]].keys())
	
	# Saving Modulation Map
	ModulationMap = dict(zip(list(range(0,len(ModulationSchemes))), ModulationSchemes))
	f = open(SavePath + "/ModulationMap.pkl", "wb")
	pickle.dump(ModulationMap,f)
	f.close()
	
	TrainDataset = {}
	ValidDataset = {}
	
	for modType in ModulationSchemes:
		TrainDataset[modType] = {}
		ValidDataset[modType] = {}
		for snr in SNRs:
			Data = Dataset[modType][snr]
			
			N = Data.shape[0]
			ValidInd = np.random.choice(N, size = int(N*test_size), replace=False)
			TrainInd = list(set(np.arange(N)) - set(ValidInd))
			
			TrainDataset[modType][snr] = Data[TrainInd]
			ValidDataset[modType][snr] = Data[ValidInd]
	
	# Saving Validation Data
	f = open(SavePath + "/TrainDataset.pkl","wb")
	pickle.dump(TrainDataset,f)
	f.close()

	f = open(SavePath + "/ValidDataset.pkl","wb")
	pickle.dump(ValidDataset,f)
	f.close()

SplitData("Dataset/RML2016.10a_dict.pkl","Dataset",test_size=0.2)
