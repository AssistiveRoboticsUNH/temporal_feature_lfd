import pandas as pd 
import numpy as np 

import matplotlib
import matplotlib.pyplot as plt

import argparse

from sklearn.metrics import confusion_matrix, accuracy_score

def get_accuracy(df):
	expected = df["expected_action"]
	observed = df["observed_action"]

	return accuracy_score(y_true = actual_label, y_pred = pred_label )

def viz_confusion_matrix(label_list, predictions):

	target_names = range(num_classes)

	def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(target_names))
		plt.xticks(tick_marks, target_names, rotation=45)
		plt.yticks(tick_marks, target_names)
		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')

	plt.figure(figsize=(20,10))

	cm = confusion_matrix(label_list, predictions)
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	# plot and save the confusion matrix
	plot_confusion_matrix(cm)
	#plt.show()
	plt.savefig('cm.png')

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate IADs from input files')
	parser.add_argument('input_file', help='the checkpoint file to use with the model')
	parser.add_argument('--output_dir', default="../csv_output",help='the checkpoint file to use with the model')
	args = parser.parse_args()

	filename = os.path.join(args.output_dir, args.input_file)

	df = pd.read_csv(filename)

	acc = get_accuracy(df)
	viz_confusion_matrix(df)

	print("filename: ", filename)
	print("accuracy: ", acc)
