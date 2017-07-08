import pandas as pd
import numpy as np

ZERO=1e-6

def entropy(P):
	e = 0.0
	for p in P:
		if p < ZERO: # p = 0
			continue
		p = np.float(p)
		e -= p*np.log2(p)
	return e


def category_entropy(train_data, label, column_number):
	'''
	c0 c1 c2 c3 ... C
	where ci is the non-category attributes, C is the category.
	'''
	nb_samples = len(label)
	if column_number == None:
		p = np.float(np.sum(label)) / len(label)
		return entropy([p, 1.0-p])
	else:
		sub_entropy = np.zeros(len(column_number))
		for j, c in enumerate(column_number):
			feature_c = train_data[:, c] 
			features = set(feature_c) 
			for f in features:
				label_subset = [label[i] for i, value in enumerate(feature_c) if value == f]
				nb_subset_samples = np.float(len(label_subset))
				p = np.float(np.sum(label_subset)) / nb_subset_samples
				sub_entropy[j] += nb_subset_samples / nb_samples * entropy([p, 1.0 - p])
		return np.array(sub_entropy)


class Node(object):
	def __init__:(self, \
					category_name='node', \
					childs=[], \
					entropy=0.0,
					arc = []):
		# Name of category
		self.category_name = category_name
		# Children is a list which contains judge or child
		self.childs = childs
		# Entropy of the attribute
		self.entropy = entropy
		# Attribute value
		self.arc = arc

def generateTree(parent=None, arc = -1.0, \
				train_data, label, \
				attr_col_number, \
				attribute_names):

	if !parent:
		node = Node()
		# Don't apply any attributes.
		root_entropy = category_entropy(train_data, label, None)
		# Gain == 0 for root
		if root_entropy < ZERO:
			node.category_name = 'Random_Guess'
			node.childs.append(label[0])
			return
	else:
		root_entropy = parent.entropy
		# Gain == 0
		if root_entropy < ZERO:
			parent.childs.append(label[0])
			return 
	sub_entropy = category_entropy(train_data, label, attr_col_number)
	# Gain < 0
	if (sub_entropy >= root_entropy).all() 
		if !parent:
			# Make prediction, just make it.
			node.category_name = 'Random_Guess'
			node.entropy = root_entropy
			node.childs = [np.int(sum(label) > (0.5 * len(label)))] 
			return
		else:
			parent.childs.append(np.int(sum(label) > (0.5 * len(label))))
			return
	else:
		node = Node()
		for ncol, col in enumerate(train_data.T):
			picked_attr = attribute_names[ncol]
			node.category_name = attribute_names[picked_attr]
			attribute_names.remove(picked_attr)
			attr_value = set(col)
			for arc in attr_value:
				sub_train = train_data[col==attr_value,:]
				sub_label = label[col==attr_value]
				attr_col_number.remove(ncol)
				generateTree(parent=node, arc = arc \
							sub_train, sub_label, \
							attr_col_number, \
							attribute_names)
				attr_col_number.append(ncol)
			attribute_names.append(picked_attr)
	return 


	




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
print (category_entropy(train_data, [0,1,2]))