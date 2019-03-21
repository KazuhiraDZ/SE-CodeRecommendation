import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import tensorflow as tf
import tensorflow_fold as td
import numpy as np
import pickle

import codecs
import functools

import zerorpc

import time
import readdataOneModel as rd
from dbOneModel import select
from sklearn.utils import shuffle
from vocab import Vocab
from tree import Tree
from collections import Counter

class ChildSumTreeLSTM(tf.contrib.rnn.BasicLSTMCell):
	
	def __init__(self,mem_dim,keep_prob = 1.0):
		super(ChildSumTreeLSTM,self).__init__(mem_dim)
		#self.in_dim = in_dim
		self.mem_dim = mem_dim
		#self.out_dim = out_dim
		self._keep_prob = keep_prob
	
	def __call__(self,inputs,state,scope=None):
		with tf.variable_scope(scope or type(self).__name__):
			child_state = state

			com = [hi for ci,hi in child_state]
			com.insert(0,inputs)

			
			concat = tf.contrib.layers.linear(tf.concat(com,1)
											  ,(len(child_state)+3)*self._num_units)
			splits = tf.split(value=concat,num_or_size_splits=(len(child_state)+3),axis=1)
			i = splits[0]
			j = splits[1]
			o = splits[-1]
			f = splits[2:-1]
			j = self._activation(j)
			if not isinstance(self._keep_prob, float) or self._keep_prob < 1:
				j = tf.nn.dropout(j,self._keep_prob)
			#print("f0: ",f[0])
			one = child_state[0][0]*tf.sigmoid(f[0]+self._forget_bias)
			#print "f-len: ",len(f)
			for k in range(len(f)):

				one += child_state[k][0]*tf.sigmoid(f[k]+self._forget_bias)
			
			new_c = one+tf.sigmoid(i)*j
			
			new_h = self._activation(new_c)*tf.sigmoid(o)
			
			new_state = tf.contrib.rnn.LSTMStateTuple(new_c,new_h)
			
		return new_h, new_state

def create_embedding(weight_matrix):
	return td.Embedding(*weight_matrix.shape, initializer=weight_matrix, name='word_embedding')

def create_model(word_embedding,NUM_CLASS,vocab,lstm_num_units=300,keep_prob=1):
	
	tree_lstm1 = td.ScopedLayer(
			tf.contrib.rnn.DropoutWrapper(
				ChildSumTreeLSTM(lstm_num_units, keep_prob=keep_prob),
				input_keep_prob=keep_prob,output_keep_prob=keep_prob),
			name_or_scope='tree_lstm1')

	tree_lstm2 = td.ScopedLayer(
			tf.contrib.rnn.DropoutWrapper(
				ChildSumTreeLSTM(lstm_num_units, keep_prob=keep_prob),
				input_keep_prob=keep_prob,output_keep_prob=keep_prob),
			name_or_scope='tree_lstm2')

	tree_lstm3 = td.ScopedLayer(
			tf.contrib.rnn.DropoutWrapper(
				ChildSumTreeLSTM(lstm_num_units, keep_prob=keep_prob),
				input_keep_prob=keep_prob,output_keep_prob=keep_prob),
			name_or_scope='tree_lstm3')

	tree_lstm4 = td.ScopedLayer(
			tf.contrib.rnn.DropoutWrapper(
				ChildSumTreeLSTM(lstm_num_units, keep_prob=keep_prob),
				input_keep_prob=keep_prob,output_keep_prob=keep_prob),
			name_or_scope='tree_lstm4')

	tree_lstm5 = td.ScopedLayer(
			tf.contrib.rnn.DropoutWrapper(
				ChildSumTreeLSTM(lstm_num_units, keep_prob=keep_prob),
				input_keep_prob=keep_prob,output_keep_prob=keep_prob),
			name_or_scope='tree_lstm5')
	tree_lstm6 = td.ScopedLayer(
			tf.contrib.rnn.DropoutWrapper(
				ChildSumTreeLSTM(lstm_num_units, keep_prob=keep_prob),
				input_keep_prob=keep_prob,output_keep_prob=keep_prob),
			name_or_scope='tree_lstm6')

	output_layer = td.FC(NUM_CLASS, activation=None, name='output_layer')   
	
	sub_cut = td.ForwardDeclaration(name="sub_cut")
	
	def lookup_word(word):
		return vocab.index(word)

	def logit_and_state():

		def get_length(root):
			length = len(root)
			if length > 1:
				length = 1 + len(root[1])

			return length

		word2vec = td.GetItem(0) >> td.InputTransform(lookup_word) >> td.Scalar('int32') >> word_embedding
		zeros_state = td.Zeros((tree_lstm1.state_size,)*2,name='zeros_state')        
		word_case = td.AllOf(word2vec,zeros_state,name='word_case') >> tree_lstm2

		pair2vec1 = (sub_cut(),) 
		pair2vec2 = (sub_cut(),sub_cut())
		pair2vec3 = (sub_cut(),sub_cut(),sub_cut())
		pair2vec4 = (sub_cut(),sub_cut(),sub_cut(),sub_cut())
		pair2vec5 = (sub_cut(),sub_cut(),sub_cut(),sub_cut(),sub_cut())
		pair2vec6 = (sub_cut(),sub_cut(),sub_cut(),sub_cut(),sub_cut(),sub_cut())

		pair_case1 = td.AllOf(td.GetItem(0) >> td.InputTransform(lookup_word) >> td.Scalar('int32') >> word_embedding
							  ,td.GetItem(1)>>pair2vec1,name='pair_case1') >> tree_lstm1
		pair_case2 = td.AllOf(td.GetItem(0) >> td.InputTransform(lookup_word) >> td.Scalar('int32') >> word_embedding
							  ,td.GetItem(1)>>pair2vec2,name='pair_case2') >> tree_lstm2  
		pair_case3 = td.AllOf(td.GetItem(0) >> td.InputTransform(lookup_word) >> td.Scalar('int32') >> word_embedding
							  ,td.GetItem(1)>>pair2vec3,name='pair_case3') >> tree_lstm3
		pair_case4 = td.AllOf(td.GetItem(0) >> td.InputTransform(lookup_word) >> td.Scalar('int32') >> word_embedding
							  ,td.GetItem(1)>>pair2vec4,name='pair_case4') >> tree_lstm4
		pair_case5 = td.AllOf(td.GetItem(0) >> td.InputTransform(lookup_word) >> td.Scalar('int32') >> word_embedding
							  ,td.GetItem(1)>>pair2vec5,name='pair_case5') >> tree_lstm5
		pair_case6 = td.AllOf(td.GetItem(0) >> td.InputTransform(lookup_word) >> td.Scalar('int32') >> word_embedding
							  ,td.GetItem(1)>>pair2vec6,name='pair_case6') >> tree_lstm6

		ans = td.OneOf(get_length,[(1,word_case),(2,pair_case1),(3,pair_case2),
								   (4,pair_case3),(5,pair_case4),(6,pair_case5),
								   (7,pair_case6)],name='ans')

		last = ans >> (output_layer,td.Identity())
		return last

	model = emb_tree(logit_and_state(),is_root=True)
	sub_cut.resolve_to(emb_tree(logit_and_state(),is_root=False))

	print("Create completed!")
	
	
	compiler = td.Compiler.create(model)

	
	metrics = {k: tf.reduce_mean(v) for k,v in compiler.metric_tensors.items()}
	metrics
	
	return compiler,metrics

def emb_tree(logit_and_state,is_root):

	if is_root == True:
		return td.InputTransform(cut_tree_and_get_label) >> (td.Scalar('int32'),logit_and_state) >> addmetric()
	else:
		return td.InputTransform(cut_tree) >> logit_and_state >> addmetric2()

def cut_tree(tree):
	current = [tree.word]
	if (tree.num_children) > 0:
		current.append(tree.children)
	#print "current: ",current
	return current

def cut_tree_and_get_label(label_and_tree):
	#label = label_and_tree[0]
	tree = label_and_tree
	label = tree.gold_label
	current = [tree.word]
	if(tree.num_children) > 0:
		current.append(tree.children)
	#print 'The label: %d, The current: %s' % (label,current)
	return label,current

def addmetric():
	c = td.Composition()
	with c.scope():

		labels = c.input[0]
		logits = td.GetItem(0).reads(c.input[1])
		state = td.GetItem(1).reads(c.input[1])
		
		loss = td.Function(tf_node_loss)
		td.Metric('root_loss').reads(loss.reads(logits,labels))
		
		hits = td.Function(tf_fine_grained_hits)
		td.Metric('root_hits').reads(hits.reads(logits,labels))

		c.output.reads(loss,hits,logits,state)
	return c       

def addmetric2():
	c = td.Composition()
	with c.scope():
		#logit = td.GetItem(0).reads(c.input)
		state = td.GetItem(1).reads(c.input)
		c.output.reads(state)
	return c

def tf_node_loss(logits, labels):
	#labels_one_hot = tf.reshape(labels,logits.get_shape())
	return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels)

def tf_fine_grained_hits(logits, labels):
 
	prediction = tf.cast(tf.argmax(logits,1),tf.int32)

	return tf.cast(tf.equal(prediction, labels),tf.float64)

def getbatch(dataset,batchsize): 
	totalbatch = len(dataset)/batchsize
	gold_label = [tree.gold_label for tree in dataset]
	if len(dataset)%batchsize > 0:
		totalbatch += 1
	for i in range(totalbatch):
		yield dataset[i*batchsize:(i+1)*batchsize],gold_label[i*batchsize:(i+1)*batchsize]

def getTestdataAns(dataset,batchsize):
	correct = {'top-1':0,'top-2':0,'top-5':0,'top-10':0}
	rate = {'top-1':0,'top-2':0,'top-5':0,'top-10':0}
	distribution = {'top-1':Counter(),'top-2':Counter(),'top-5':Counter(),'top-10':Counter()}
	totalbatch = len(dataset)/batchsize
	if len(dataset)%batchsize > 0:
		totalbatch += 1
	for i,(batchTree,batchLabel) in enumerate(getbatch(dataset,batchsize),1):
		logit = sess.run([logits],compiler.build_feed_dict(batchTree))
		
		class_top_10 = tf.nn.top_k(logit[0],20)
		class_top_10_indices = class_top_10.indices
		class_top_10_values = tf.nn.softmax(class_top_10.values)
		
		top10_idx,top10_prob = sess.run([class_top_10_indices,class_top_10_values])
		for idx,tree in enumerate(batchTree):
			if tree.gold_label in top10_idx[idx][:1]:
				correct['top-1'] += 1
				distribution['top-1'][APIname.token(tree.gold_label)] += 1
			if tree.gold_label in top10_idx[idx][:2]:
				correct['top-2'] += 1
				distribution['top-2'][APIname.token(tree.gold_label)] += 1
			if tree.gold_label in top10_idx[idx][:5]:
				correct['top-5'] += 1
				distribution['top-5'][APIname.token(tree.gold_label)] += 1
			if tree.gold_label in top10_idx[idx][:10]:
				correct['top-10'] += 1
				distribution['top-10'][APIname.token(tree.gold_label)] += 1
		print('{}/{}'.format(i,totalbatch))
	
	rate['top-1'] = correct['top-1']/float(len(dataset))*100
	rate['top-2'] = correct['top-2']/float(len(dataset))*100
	rate['top-5'] = correct['top-5']/float(len(dataset))*100
	rate['top-10'] = correct['top-10']/float(len(dataset))*100
	return correct,rate,distribution

def read_weight_matrix():
	weight_matrix = np.random.uniform(-0.05,0.05,(21344,50)).astype(np.float32)
	return weight_matrix

class Treelstm(object):
	
	def __init__(self):
		self.vocab = Vocab('/home/x/mydisk/newestVocabulary/vocabulary.txt')
		self.APIname = Vocab('/home/x/mydisk/newestVocabulary/APIClassName.txt')
		self.weight_matrix = read_weight_matrix()
		self.compiler, self.top_10 = self.getSession()
		self.sess = self.restore_graph()
		print 'Now serving...'

	def create_tree(self,treeNumber,serialNumberString,treeSentence):
		contents = []

		contents.append(treeNumber)
		contents.append(serialNumberString)
		contents.append(treeSentence)
		parentsTable = []
		wordTable = []
		# read the concrete index in tree node
		tokens = contents[0].split()
		for parent in tokens:
			parentsTable.append(int(parent))
		# read the slice of code
		tokens2 = contents[2].split()
		for word in tokens2:
			wordTable.append(word)
		# get the number of node to add new node(predicted class)
		nodeIndex = 0

		# transform to lstmtree
		labelLine = 'termination'
		new_tree = rd.read_classname_tree(parentsTable,labelLine,wordTable,nodeIndex,None,None,None,None,None,None,self.APIname)
		
		#new_tree.depth_first(new_tree,self.vocab)
		return new_tree

	def getSession(self):
		print 'prepare model...'
		tf.reset_default_graph()
		word_embedding = create_embedding(self.weight_matrix)
		print 'embedding layer completed!'
		learning_rate = 0.01
		compiler, metrics = create_model(word_embedding, 21342, vocab = self.vocab)
		print 'compiler completed!'

		logits = compiler.output_tensors[2]
		top_10 = tf.nn.top_k(logits,20)
		loss = tf.reduce_mean(compiler.metric_tensors['root_loss'])
		opt = tf.train.AdagradOptimizer(learning_rate=learning_rate)
		grads_and_vars = opt.compute_gradients(loss)
		train = opt.apply_gradients(grads_and_vars)
		self.saver = tf.train.Saver()
		print 'model completed!'
		return compiler, top_10

	def restore_graph(self):
		checkpoint = '/home/x/mydisk/Zhao.treelstm/data/model49/Zhao.class_model-49'
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		sess = tf.InteractiveSession(config=config)
		print 'restoring params...'
		self.saver.restore(sess,checkpoint)
		print 'get session completed!'
		return sess

	def prediction(self,treeNumber,serialNumberString,treeSentence):
		#print treeNumber
		#print serialNumberString
		#print treeSentence
		inputTree = self.create_tree(treeNumber,serialNumberString,treeSentence)
		#logtis1 = self.sess.run(self.compiler.output_tensors[2],self.compiler.build_feed_dict([inputTree]))
		top_10 = self.sess.run(self.top_10,self.compiler.build_feed_dict([inputTree]))
		class_top_10_indices = top_10.indices
		class_top_10_values = self.sess.run(tf.nn.softmax(top_10.values.tolist()))
		
		idx,prob = [class_top_10_indices,class_top_10_values]
		words = [self.APIname.token(index) for index in idx[0]]
		remote_ans = ''
		for i,(pred, value) in enumerate(zip(words,prob[0]),1):
			#print pred,value
			temp = pred + ' ' + str(value)
			if i < len(words):
				temp += '\n'
			remote_ans += temp
		#print remote_ans
		return remote_ans

	def printinfo(self,strings):
		return str(strings+' '+strings)

	def readtestdata(self,path):

		#path = '../conditionTesting/spring.tfrecords'
		data_set = []
		count = 0
		total = 0
		predictions = []
		for serialized_example in tf.python_io.tf_record_iterator(path):
			total += 1
			if total > 0:
				example = tf.train.Example()
				example.ParseFromString(serialized_example)
				tree = example.features.feature['tree'].bytes_list.value
				data = pickle.loads(tree[0])
				#if data.size() > 50 or data.get_child_num(data) != 0:
				#if data.size() > 50 or data.get_child_num(data) != 0 or APIname.token(data.gold_label) == 'end' or APIname.token(data.gold_label) == 'termination'or APIname.token(data.gold_label) == 'conditionEnd':
				#    continue
				
				data_set.append(data)
				count+=1
				if count%100 == 0:
					print(count)
				#if count%30000 == 0:
				#    break
		return data_set

	def returnTestdata(self,dataset,batchsize):
		totalbatch = len(dataset)/batchsize
		if len(dataset)%batchsize > 0:
			totalbatch += 1
		f = open('batchreturn.txt','w')
		for i,(batchTree,batchLabel) in enumerate(getbatch(dataset,batchsize),1):
			class_top_10 = self.sess.run(self.top_10,self.compiler.build_feed_dict(batchTree))
			
			class_top_10_indices = class_top_10.indices
			class_top_10_values = self.sess.run(tf.nn.softmax(class_top_10.values.tolist()))
			for num in range(len(batchTree)):
				current_top_10_indices = class_top_10_indices[num]
				current_top_10_values = class_top_10_values[num]
				temp = ''
				for idx,(predidx,value) in enumerate(zip(current_top_10_indices,current_top_10_values),1):
					temp += self.APIname.token(predidx) + ' '+ str(value)
					if idx < len(current_top_10_indices):
						temp += ';'
					else:
						temp += '\n'
				print(temp)
				f.write(temp)
		f.close()
		print('End...')

	def testtest(self,path):
		dataset = self.readtestdata(path)
		self.returnTestdata(dataset,256)




s = zerorpc.Server(Treelstm())
s.bind("tcp://0.0.0.0:4242")
s.run()
