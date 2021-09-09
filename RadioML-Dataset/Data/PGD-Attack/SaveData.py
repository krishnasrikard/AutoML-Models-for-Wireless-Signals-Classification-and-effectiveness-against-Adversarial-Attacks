import numpy as np
import pickle

def SaveTrainData(Dataset, SavePath):
    ModulationSchemes = list(Dataset.keys())
    SNRs = list(Dataset[ModulationSchemes[0]].keys())
    
    OneHotClasses = np.eye(len(ModulationSchemes))
    Classes = {}
    for i in range(len(ModulationSchemes)):
        Classes[ModulationSchemes[i]] = OneHotClasses[i]

    X_Train, y_Train = [],[]

    for modType in ModulationSchemes:
        for snr in SNRs:
            Data = Dataset[modType][snr]
                        
            X_Train.append(Data)
            y_Train.append(np.repeat(np.expand_dims(Classes[modType],axis=0),Data.shape[0],axis=0))
    	
    X_Train = np.array(X_Train).reshape(-1,2,128)
    y_Train = np.array(y_Train).reshape(-1,len(ModulationSchemes))
    
    np.save(SavePath + "/Adversarial_X_Train.npy", X_Train)
    np.save(SavePath + "/Adversarial_y_Train.npy", y_Train)
	
	
def SaveValidData(Dataset, SavePath):
	ModulationSchemes = list(Dataset.keys())
	SNRs = list(Dataset[ModulationSchemes[0]].keys())
	OneHotClasses = np.eye(len(ModulationSchemes))

	Classes = {}
	for i in range(len(ModulationSchemes)):
		Classes[ModulationSchemes[i]] = OneHotClasses[i]

	X_Valid, y_Valid = {},{}

	for snr in SNRs:
		X_Valid[snr] = []
		y_Valid[snr] = []
		for modType in ModulationSchemes:
			Data = Dataset[modType][snr]

			X_Valid[snr].append(Data)
			y_Valid[snr].append(np.repeat(np.expand_dims(Classes[modType],axis=0),Data.shape[0],axis=0))

		X_Valid[snr] = np.array(X_Valid[snr]).reshape(-1,2,128)
		y_Valid[snr] = np.array(y_Valid[snr]).reshape(-1,len(ModulationSchemes))
	
	f = open(SavePath + "/Adversarial_X_Valid.pkl","wb")
	pickle.dump(X_Valid,f)
	f.close()

	f = open(SavePath + "/Adversarial_y_Valid.pkl","wb")
	pickle.dump(y_Valid,f)
	f.close()
