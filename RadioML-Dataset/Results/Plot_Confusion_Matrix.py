import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix

def Plot_Confusion_Matrices(ModelDescriptions, DataDescriptions, Path, ModulationMap, CMap):
	"""
	- ModelDescriptions:
		- ModelType: Describes whether the Model depending on Dataset it is Trained
		- ModelName: Name of Model
		- Model: TensorFlow Model
	- DataDescriptions:
		- DataType: Describes Type of Dataset
		- X_Valid
		- y_Valid
	"""
	
	for ModelType,ModelName,Model in ModelDescriptions:
		for DataType,X_Valid,y_Valid in DataDescriptions:
			Valid_SNR = list(X_Valid.keys())
			Accuracy = []
			for snr in Valid_SNR:
				SavePath = Path + "/" + str(ModelName) + ", SNR = " + str(snr) + "dB"
				y_true = np.argmax(y_Valid[snr],axis=1)
				if (Model.layers[0].input_shape[0][1:] == (2,128)):
					y_pred = np.argmax(Model.predict(X_Valid[snr]),axis=1)
				elif (Model.layers[0].input_shape[0][1:] == (128,2,1)):
					y_pred = np.argmax(Model.predict(np.expand_dims(np.transpose(X_Valid[snr], (0,2,1)), axis=-1)),axis=1)
				
				CM = confusion_matrix(y_true,y_pred)
			
				CM_ColumnSum = np.expand_dims(np.sum(CM,axis=1),axis=-1)
				CM = np.round(np.divide(CM,CM_ColumnSum),decimals=3)
				
				CM_ColumnSum = np.expand_dims(np.sum(CM,axis=1),axis=-1)
				CM = np.round(np.divide(CM,CM_ColumnSum),decimals=3)
			
				df_CM = pd.DataFrame(CM, index = list(ModulationMap.values()), columns = list(ModulationMap.values()))
				plt.figure(figsize=(10,10))
				sns.heatmap(df_CM,annot=True,cmap=CMap,fmt='.3f')
				plt.title(str(ModelName) + ", SNR = " + str(snr) + "dB")
				plt.savefig(SavePath)
				#plt.show()
				#print ()
			
			print ('Confusion Matrices of SNR = ' + str(snr) + "dB Completed")
