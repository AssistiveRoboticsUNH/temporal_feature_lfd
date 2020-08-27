import pandas as pd 
import numpy as np 

import matplotlib
import matplotlib.pyplot as plt

import argparse, os

from sklearn.metrics import confusion_matrix, accuracy_score

def get_accuracy(df):
	expected = df["expected_action"]
	observed = df["observed_action"]

	return accuracy_score(y_true = actual_label, y_pred = pred_label )

def viz_confusion_matrix(df, output_filename):
	expected = df["expected_action"]
	observed = df["observed_action"]

	num_classes = expected.unique()

	print("num_classes:", num_classes)

	target_names = range(num_classes)

	plt.figure(figsize=(20,10))

	cm = confusion_matrix(expected, observed)
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	title='Confusion matrix'
	cmap=plt.cm.Blues

	# plot and save the confusion matrix
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(target_names))
	plt.xticks(tick_marks, target_names, rotation=45)
	plt.yticks(tick_marks, target_names)
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

	# save confusion matrix to file
	plt.savefig(output_filename)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate IADs from input files')
	parser.add_argument('input_file', help='the checkpoint file to use with the model')
	parser.add_argument('--output_dir', default="../csv_output",help='the checkpoint file to use with the model')
	parser.add_argument('--fig_dir', default="fig",help='the checkpoint file to use with the model')
	args = parser.parse_args()

	src_filename = os.path.join(args.output_dir, args.input_file)

	fig_dir = os.path.join(args.fig_dir, args.input_file[:-4])
	if (not os.path.exists(fig_dir)):
		os.makedirs(fig_dir)

	df = pd.read_csv(src_filename)

	acc = get_accuracy(df)
	viz_confusion_matrix(df, os.path.join(fig_dir, "cm.png"))

	print("filename: ", src_filename)
	print("accuracy: ", acc)
