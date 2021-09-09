import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Plot to estimate success of the Adversarial Attack
def AttackSuccess(ModelDescriptions, DataDescriptions, SavePath, ylimits, ValidBatchSize=32):
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
	print ("Evaluating Models on Datasets")	

	plt.figure(figsize=(9,9))
	plt.title("Attack Success Accuracy vs SNR")
	plt.xlabel("SNR (dB)")
	plt.ylabel("Attack Success Accuracy")
	plt.ylim(ylimits[0] - 1e-2, ylimits[1] + 1e-2)
	plt.grid()
	
	for ModelType,ModelName,Model in ModelDescriptions:
		for DataType,X_Valid,y_Valid in DataDescriptions:
			Valid_SNR = list(X_Valid.keys())
			Accuracy = []
			for snr in Valid_SNR:
				if (Model.layers[0].input_shape[0][1:] == (2,128)):
					y_true = np.argmax(y_Valid[snr], axis=1)
					y_pred = np.argmax(Model.predict(X_Valid[snr]), axis=1)
					Acc = 1 - accuracy_score(y_true, y_pred)
				elif (Model.layers[0].input_shape[0][1:] == (128,2,1)):
					y_true = np.argmax(y_Valid[snr], axis=1)
					y_pred = np.argmax(Model.predict(np.expand_dims(np.transpose(X_Valid[snr], (0,2,1)), axis=-1)), axis=1)
					Acc = 1 - accuracy_score(y_true, y_pred)
				Accuracy.append(Acc)
			Accuracy = np.array(Accuracy)
			plt.plot(Valid_SNR,Accuracy, label=ModelType + " " + str(ModelName) + ", " + DataType + " Dataset")
			plt.scatter(Valid_SNR,Accuracy)

	plt.legend()
	plt.savefig(SavePath)
	plt.savefig(SavePath[:-3] + "eps", bbox_inches='tight', pad_inches=0.05, dpi=125)
	plt.show()
