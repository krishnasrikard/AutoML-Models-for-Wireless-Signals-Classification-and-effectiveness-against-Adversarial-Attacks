import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import seaborn as sns
import scipy.io
import os

# Evaluating Data
def EvaluateData(Model, X_Valid, y_Valid, SavePath, ValidBatchSize=32):
    Valid_SNR = np.array([-15,-10,-5,0,5,10,15,20,25,30])
    Accuracy = []

    print ("Evaluating Model")
    for snr in Valid_SNR:
        Loss, Acc = Model.evaluate(X_Valid[snr], y_Valid[snr], batch_size=ValidBatchSize, verbose=0)
        print ("SNR:", snr, "Accuracy:", Acc)
        Accuracy.append(Acc)

    Accuracy = np.array(Accuracy)

    plt.figure()
    plt.plot(Valid_SNR,Accuracy, color='blue')
    plt.scatter(Valid_SNR,Accuracy, color='red')
    plt.title("Accuracy vs SNR")
    plt.xlabel("SNR")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig(SavePath)
    plt.savefig(SavePath[:-3] + "eps")
    plt.show()
    
    

# Plot Results
def PlotResults(Models, X_Valid, y_Valid, SavePath, ValidBatchSize=32):
	Model_Accuracies = []
	Model_Names = []
	Valid_SNR = np.array([-15,-10,-5,0,5,10,15,20,25,30])
	
	for ModelName,Model in Models.items():
		Model_Names.append(ModelName)
		Accuracy = []
		for snr in Valid_SNR:
			try:
				Loss, Acc = Model.evaluate(X_Valid[snr], y_Valid[snr], batch_size=ValidBatchSize, verbose=0)
			except:
				X_Valid[snr] = X_Valid[snr].reshape(-1,100,2)
				y_Valid[snr] = y_Valid[snr][::100]
				Loss, Acc = Model.evaluate(X_Valid[snr], y_Valid[snr], batch_size=ValidBatchSize, verbose=0)
			
			Accuracy.append(Acc)

		Model_Accuracies.append(Accuracy)
	
	print ()
	plt.figure()
	plt.title("Accuracy for various Models for all SNRs")
	plt.xlabel("SNR")
	plt.ylabel("Accuracy")
	plt.grid()
	for i in range(len(Model_Accuracies)):
		Accuracy = Model_Accuracies[i]
		plt.plot(Valid_SNR,Accuracy, label = Model_Names[i])
		plt.scatter(Valid_SNR,Accuracy, label = Model_Names[i])
	plt.legend()
	plt.savefig(SavePath)
	plt.savefig(SavePath[:-3] + "eps")
	plt.show()	
