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
	if column_number == 0:
		p = np.float(np.sum(label)) / len(label)
		return entropy([p, 1.0-p])
	else:
		sub_entropy = np.zeros(column_number)
		for j in range(column_number):
			feature_col_j = train_data[:, j] 
			features = set(feature_col_j) 
			for f in features:
				label_subset = [label[i] for i, value in enumerate(feature_col_j) if value == f]
				nb_subset_samples = np.float(len(label_subset))
				p = np.float(np.sum(label_subset)) / nb_subset_samples
				sub_entropy[j] += nb_subset_samples / nb_samples * entropy([p, 1.0 - p])
		return np.array(sub_entropy)



class Node(object):
	def __init__(self, \
					category_name='', \
					childs=[], \
					entropy=0.0, \
					arc = [], \
					node_id = -1):
		# Name of category
		self.category_name = category_name
		# Children is a list which contains judge or child
		self.childs = childs
		# Entropy of the attribute
		self.entropy = entropy
		# Attribute value
		self.arc = arc
		self.node_id = node_id

def helper(train_data, label, attribute_names, parent):
	root_entropy = category_entropy(train_data, label, 0)
	# 100% SURE
	if root_entropy < ZERO:
		parent.childs.append(label[0])
		return
	sub_entropy = category_entropy(train_data, label, train_data.shape[1])
	# Gain < 0 for all nodes
	if (sub_entropy > root_entropy).all() :
		parent.childs.append(np.int(sum(label) > (0.5 * len(label))))
		return 

	# Gain > 0 for at least one node, find the node with largest gain
	else:
		node = Node(node_id=parent.node_id+1)
		min_entropy_idx = np.argmin(sub_entropy)
		node.category_name = attribute_names[min_entropy_idx]
		cols = np.asarray([i for i in range(len(train_data[0]))])
		feature_col = train_data[:, min_entropy_idx]
		features = set(feature_col)
		attribute_names.remove(node.category_name)
		for ac in features:
			node.arc.append(ac)
			sub_train = train_data[:, cols!=min_entropy_idx]
			sub_train = sub_train[feature_col==ac, :]
			sub_label = label[feature_col==ac]
			helper(sub_train, sub_label, \
						attribute_names, \
						node)
		print(node.childs)	
		print(parent.childs)
		parent.childs.append(node)
		attribute_names.append(node.category_name)
		return 
			
def generateTree(train_data, label, \
				attribute_names):
	
	
	node = Node(node_id=0)
	# 100% SURE
	root_entropy = category_entropy(train_data, label, 0)
	if root_entropy < ZERO:
		node.category_name='Random_Guess'
		node.childs.append(label[0])
		return node
	sub_entropy = category_entropy(train_data, label, train_data.shape[1])
	# Gain < 0 for all nodes
	if (sub_entropy > root_entropy).all() :
		# Make prediction, just make it.
		node.category_name = 'Random_Guess'
		node.childs = [np.int(sum(label) > (0.5 * len(label)))] 
		return node
		
	# Gain > 0 for at least one node, find the node with largest gain
	else:
		min_entropy_idx = np.argmin(sub_entropy)
		node.category_name = attribute_names[min_entropy_idx]
		cols = np.asarray([i for i in range(len(train_data[0]))])
		feature_col = train_data[:, min_entropy_idx]
		features = set(feature_col)
		attribute_names.remove(node.category_name)
		for ac in features:
			node.arc.append(ac)
			sub_train = train_data[:, cols!=min_entropy_idx]
			sub_train = sub_train[feature_col==ac, :]
			sub_label = label[feature_col==ac]
			helper(sub_train, sub_label, \
						attribute_names, \
						node)
		attribute_names.append(node.category_name)
		return node


data = np.array([[1,1,1,0], \
				[1,2,1,0],\
				[1,2,2,0],\
				[2,1,1,0],\
				[2,1,2,0],\
				[2,2,2,1],\
				[2,2,1,1],\
				[3,1,1,1],\
				[3,2,2,1],\
				[3,2,1,1]])
train_data = np.asarray(data[:, :3])
label = data[:, -1]
attribute_names=['Age', 'Competition', 'Type']
root = generateTree(train_data, label, attribute_names)
# print(root.childs)






