import pandas as pd
import numpy as np

def entropy(P):
	e = 0.0
	for p in P:
		if p < 1e-6: # p = 0
			continue
		p = np.float(p)
		e -= p*np.log2(p)
	return e

def category_entropy(train_data, column_number):
	'''
	c0 c1 c2 c3 ... C
	where ci is the non-category attributes, C is the category.
	'''
	label = [ a[-1] for a in train_data]
	nb_samples = len(label)
	if column_number == None:
		p = np.float(np.sum(label)) / len(label)
		return entropy([p, 1.0-p])
	else:
		for j, c in enumerate(column_number):
			feature_c = train_data[:, c]
			features = set(feature_c)
			sub_entropy = np.zeros((len(features)))
			for f in features:
				label_subset = [label[i] for i, value in enumerate(feature_c) if value == f]
				nb_subset_samples = np.float(len(label_subset))
				p = np.float(np.sum(label_subset)) / nb_subset_samples
				sub_entropy[j] += nb_subset_samples / nb_samples * entropy([p, 1.0 - p])
		return sub_entropy

train_data = np.array([[1,1,1,0], \
						[1,2,1,0],\
						[1,2,2,0],\
						[2,1,1,0],\
						[2,1,2,0],\
						[2,2,2,1],\
						[2,2,1,1],\
						[3,1,1,1],\
						[3,2,2,1],\
						[3,2,1,1]])
print category_entropy(train_data, [0])