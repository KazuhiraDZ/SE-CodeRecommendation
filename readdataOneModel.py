import tensorflow as tf
from tree import Tree
from vocab import Vocab
import pickle
import os

#global_path = '/Users/zfy/test'
finalTotal = 0
total = 0

def read_classname_tree(parents,label,word,node,trace,holeSize,block,variables,originalStatement,noise_label,vocab2):
	if parents != None:
		size = len(parents)
		nodes = [None for i in xrange(size)]
		parentNode = None
		for i in xrange(size):
			parents[i] = parents[i] - 1
			if parentNode == None:
				#print 'succ'
				rootIndex = parents[i]
				nodes[parents[i]] = Tree()
				nodes[parents[i]].index = parents[i]
				#print nodes[parents[i]].index
				parentNode = nodes[parents[i]]
			elif nodes[parents[i]]!= None:
				#print nodes[parents[i]].index
				parentNode = nodes[parents[i]]
			else:
				nodes[parents[i]] = Tree()
				nodes[parents[i]].index = parents[i]
				parentNode.add_child(nodes[parents[i]])
			nodes[parents[i]].gold_label = None
			nodes[parents[i]].word = word[parents[i]]
			nodes[parents[i]].node = None

	root = nodes[rootIndex]
	if label != None:
		label = label.replace('\r','')
		label = label.replace(' ','')
		label = label.replace('\n','')
		if vocab2.contains(label) == False:
			root.gold_label = None
		else:
			root.gold_label = vocab2.index(label)
		root.word = word[rootIndex]
		root.node = node
		root.trace = trace
		root.holeSize = holeSize
		root.block = block
		root.variables = variables
		root.originalStatement = originalStatement
		root.noise_label = noise_label
		return root
	else:
		return None

def read_apicall_tree(parents, label, word, node, class_label, trace, holeSize, block, variables, originalStatement, noise_label, vocab2):
	root = read_classname_tree(parents,label,word,node,trace,holeSize,block,variables,originalStatement,noise_label,vocab2)
	if root == None:
		return None
	else:
		return root

def read_apicall_trees(vocab2,trees,parent_path,label_path,node_path,word_path,class_label_path,trace_path,holeSize_path,block_path,variables_path,originalStatement_path,noise_path,file_path,file_path2,class_prediction_path,method_prediction_path):
	if parent_path != None:
		parent_file = open(parent_path,'r')
	if label_path != None:
		label_file = open(label_path,'r')
	if node_path != None:
		node_file = open(node_path,'r')
	if word_path != None:
		word_file = open(word_path,'r')
	if class_label_path != None:
		class_label_file = open(class_label_path,'r')
	if trace_path != None:
		trace_file = open(trace_path,'r')
	if holeSize_path != None:
		holeSize_file = open(holeSize_path,'r')
	if block_path != None:
		block_file = open(block_path,'r')
	if variables_path != None:
		variables_file = open(variables_path,'r')
	if originalStatement_path != None:
		originalStatement_file = open(originalStatement_path,'r')
	if noise_path != None:
		noise_file = open(noise_path,'r')
	total = 0
	count = 0
	global finalTotal
	writer = tf.python_io.TFRecordWriter(file_path)
	f2 = open(file_path2,'w')
	classPrediction = open(class_prediction_path,'w')
	methodPrediction = open(method_prediction_path,'w')
	while True:
		line = parent_file.readline()
		labelLine = label_file.readline()
		nodeLine = node_file.readline()
		sentenceLine = word_file.readline()
		classLine = class_label_file.readline()
		trace = trace_file.readline()
		holeSize = holeSize_file.readline()
		block = block_file.readline()
		variables = variables_file.readline()
		originalStatement = originalStatement_file.readline()
		noise = noise_file.readline()
		parentsTable = []
		wordTable = []
		if not line:
			break
		elif line == "":
			continue
		else:
			count = count + 1
			line = line.replace('\r','')
			line = line.replace('\n','')
			tokens = line.split()
			for parent in tokens:
				parentsTable.append(int(parent))
			labelLine = labelLine.replace('\r','')
			labelLine = labelLine.replace('\n','')
			nodeLine = nodeLine.replace('\r','')
			nodeLine = nodeLine.replace('\n','')
			tokens2 = nodeLine.split()
			nodeIndex = int(tokens2[0]) - 1
			sentenceLine = sentenceLine.replace('\r','')
			sentenceLine = sentenceLine.replace('\n','')
			tokens3 = sentenceLine.split()
			for word in tokens3:
				wordTable.append(word)
			classLine = classLine.replace('\r','')
			classLine = classLine.replace('\n','')
			trace = trace.replace('\r','')
			trace = trace.replace('\n','')
			holeSize = holeSize.replace('\r','')
			holeSize = holeSize.replace('\n','')
			block = block.replace('\r','')
			block = block.replace('\n','')
			variables = variables.replace('\r','')
			variables = variables.replace('\n','')
			originalStatement = originalStatement.replace('\r','')
			originalStatement = originalStatement.replace('\n','')
			noise = noise.replace('\r','')
			noise = noise.replace('\n','')
			if vocab2 != None:
				tree = read_apicall_tree(parentsTable, labelLine, wordTable, nodeIndex, classLine, trace, holeSize, block, variables, originalStatement, noise, vocab2)
				if tree != None and tree.gold_label != None:
					total = total + 1
					finalTotal = finalTotal + 1
					if total % 10000 == 0:
						print(total,'/',count)
					tree = pickle.dumps(tree,2)
					data = tf.train.Example(features=tf.train.Features(feature={'tree': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tree]))}))
					writer.write(data.SerializeToString())
					classPrediction.write('%s' %classLine)
					classPrediction.write('\n')
					methodPrediction.write('%s' %labelLine)
					methodPrediction.write('\n')
	#print total,'/',count
	f2.write('%d' %total)
	writer.close()
	f2.close()
	classPrediction.close()
	methodPrediction.close()
	parent_file.close()
	label_file.close()
	node_file.close()
	word_file.close()
	class_label_file.close()
	trace_file.close()
	holeSize_file.close()
	block_file.close()
	variables_file.close()
	originalStatement_file.close()
	noise_file.close()
	return trees

def read_apicall_dataset(data_path, vocab, file_path, file_path2, class_prediction_path, method_prediction_path):
	trees = []
	for j in xrange(1):
		#print 'read ', data_path[j] , ' file'
		read_apicall_trees(vocab,trees,data_path[j] + 'trainingTree.txt', data_path[j] + 'trainingPrediction.txt',data_path[j] + 'generationNode.txt', data_path[j] + 'treeSentence.txt', data_path[j] + 'trainingClassPrediction.txt',data_path[j] + 'trace.txt',data_path[j] + 'holesize.txt',data_path[j] + 'blockPredictions.txt',data_path[j] + 'trainVariableNames.txt',data_path[j] + 'trainOriginalStatements.txt',data_path[j] + 'noise.txt',file_path, file_path2,class_prediction_path,method_prediction_path)
	return trees


#data_path = []
#data_path.append('/Users/lingxiaoxia/IdeaProjects/CodeRecommendation/Extractor/src/main/java/constructdata/data2/batch1/batch1.10/')
#file_path = '/Volumes/zzz/trainingdataset/dataset10/tree.tfrecords'
#file_path2 = '/Volumes/zzz/trainingdataset/dataset10/total.txt'
#start = time.time()
#read_classname_dataset(data_path,vocab,vocab2,file_path,file_path2)
#print time.time()-start
#data_path2 = []
#data_path2.append('/Users/lingxiaoxia/IdeaProjects/CodeRecommendation/Extractor/src/main/java/constructdata/data2/batch1/batch1.2/')
#file_path = '/Users/lingxiaoxia/Desktop/py/dataset2/tree.tfrecords'
#file_path2 = '/Users/lingxiaoxia/Desktop/py/dataset2/total.txt'
#start = time.time()
#read_classname_dataset(data_path2,vocab,vocab2,file_path,file_path2)
#print time.time()-start
#data_path3 = []
#data_path3.append('/Users/lingxiaoxia/IdeaProjects/CodeRecommendation/Extractor/src/main/java/constructdata/data2/batch1/batch1.3/')
#file_path = '/Users/lingxiaoxia/Desktop/py/dataset3/tree.tfrecords'
#file_path2 = '/Users/lingxiaoxia/Desktop/py/dataset3/total.txt'
#start = time.time()
#read_classname_dataset(data_path3,vocab,vocab2,file_path,file_path2)
#print time.time()-start


#tree.depth_first(tree,vocab)
#print tree.gold_label
#print tree.node



