import MySQLdb as mdb
import pickle
import sys 
import tensorflow as tf 
from vocab import Vocab
import time
import gc
from time import sleep
#reload(sys)  
#sys.setdefaultencoding("utf-8")
sys.setrecursionlimit(100000000)

def insert(path,class_prediction_path,method_prediction_path):
	print(path)
	start = time.time()
	conn = mdb.connect(host='127.0.0.1', port=3306, user='root', passwd='fdse', db='dataset2', charset='utf8')
	cursor = conn.cursor()
	count = 0
	values = []
	classPrediction = open(class_prediction_path,'r')
	methodPrediction = open(method_prediction_path,'r')
	for serialized_example in tf.python_io.tf_record_iterator(path):
		classLine = classPrediction.readline()
		methodLine = methodPrediction.readline()
		classLine = classLine.replace('\n','')
		methodLine = methodLine.replace('\n','')
		example = tf.train.Example()
		example.ParseFromString(serialized_example)
		tree = example.features.feature['tree'].bytes_list.value
		data = pickle.loads(tree[0])
		trace = data.trace
		block = data.block
		holeSize = data.holeSize
		variables = data.variables
		originalStatement = data.originalStatement
		noise_label = data.noise_label
		if data.size() <= 50 and data.get_child_num(data) == 0:
			db_data = pickle.dumps(data,0)
			count = count + 1
			values.append([db_data,classLine,methodLine,trace,block,holeSize,variables,originalStatement,noise_label])
		if count % 1000 == 0:
			try:
				cursor.executemany('INSERT INTO newdataset(data,classPrediction,methodPrediction,trace,block,holeSize,variables,originalStatement,noise_label) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)',values)
				conn.commit()
			except:
			    import traceback
			    traceback.print_exc()
			    conn.rollback()
			    for value in values:
			    	try:
			    		cursor.execute('INSERT INTO newdataset(data,classPrediction,methodPrediction,trace,block,holeSize,variables,originalStatement,noise_label) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)',value)
			    		conn.commit()
			    	except:
			    		conn.rollback()
			    		print (value[7])
			    		value[7] = "error encode"
			    		cursor.execute('INSERT INTO newdataset(data,classPrediction,methodPrediction,trace,block,holeSize,variables,originalStatement,noise_label) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)',value)
			    		conn.commit()
			values = []
			print (count)
	if len(values) != 0:
			try:
				cursor.executemany('INSERT INTO newdataset(data,classPrediction,methodPrediction,trace,block,holeSize,variables,originalStatement,noise_label) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)',values)
				conn.commit()
			except:
			    import traceback
			    traceback.print_exc()
			    conn.rollback()
			    for value in values:
			    	try:
			    		cursor.execute('INSERT INTO newdataset(data,classPrediction,methodPrediction,trace,block,holeSize,variables,originalStatement,noise_label) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)',value)
			    		conn.commit()
			    	except:
			    		conn.rollback()
			    		print(value[7])
			    		value[7] = "error encode"
			    		cursor.execute('INSERT INTO newdataset(data,classPrediction,methodPrediction,trace,block,holeSize,variables,originalStatement,noise_label) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)',value)
			    		conn.commit()
			finally:
				cursor.close()
				conn.close()
	values = []
	classPrediction.close()
	methodPrediction.close()
	print(time.time()-start)

def select(start,end):
	conn = mdb.connect(host='127.0.0.1', port=3306, user='root', passwd='fdse', db='dataset2', charset='utf8')
	cursor = conn.cursor()
	data = []
	#gold_labels = []
	#variables = []
	try:
		cursor.execute("select data,variables from newdataset where `index` > " + str(start) + " limit " + str(end))
		#cursor.scroll(0,mode='relative')
		numrows = int(cursor.rowcount)
		#results = cursor.fetchall()
		for i in xrange(numrows):
			row = cursor.fetchone()
			#variable = row[1]
			#variable = variable.replace('\r','')
			#variable = variable.replace('\n','')
			#variable = variable.encode('ascii')
			#variable = variable.split(' ')
			#if len(variable) > 10:
			#	variable = list(reversed(variable))[:10]
			tree = pickle.loads(row[0])
			data.append(tree)
			#gold_labels.append(tree.gold_label)
			#variables.append(variable)
			#tree.depth_first(tree,vocab)
			#print tree.gold_label
			#print tree.node
	except:
		import traceback
		traceback.print_exc()
		cursor.close()
		conn.close()
		data = []
		#gold_labels = []
		#variables = []
		return data
		#conn.rollback()
	finally:
		cursor.close()
		conn.close()
	return data

def select_for_sequence(start,end,vocab):
	conn = mdb.connect(host='127.0.0.1', port=3306, user='root', passwd='fdse', db='dataset2', charset='utf8')
	cursor = conn.cursor()
	data = []
	gold_labels = []
	#gold_labels = []
	#variables = []
	try:
		cursor.execute("select data,variables from newdataset where `index` > " + str(start) + " limit " + str(end))
		#cursor.scroll(0,mode='relative')
		numrows = int(cursor.rowcount)
		#results = cursor.fetchall()
		for i in xrange(numrows):
			row = cursor.fetchone()
			#variable = row[1]
			#variable = variable.replace('\r','')
			#variable = variable.replace('\n','')
			#variable = variable.encode('ascii')
			#variable = variable.split(' ')
			#if len(variable) > 10:
			#	variable = list(reversed(variable))[:10]
			block = []
			tree = pickle.loads(row[0])
			if tree.gold_label != 0 and tree.gold_label != 2 and tree.gold_label != 20562:
				tree.depth_first_index(tree,block,vocab)
				if vocab.index('hole') in block:
					index = block.index(vocab.index('hole'))
					block = block[:index]
					data.append(block)
					gold_labels.append(tree.gold_label)
			#gold_labels.append(tree.gold_label)
			#variables.append(variable)
			#tree.depth_first(tree,vocab)
			#print tree.gold_label
			#print tree.node
	except:
		import traceback
		traceback.print_exc()
		cursor.close()
		conn.close()
		data = []
		#gold_labels = []
		#variables = []
		return data
		#conn.rollback()
	finally:
		cursor.close()
		conn.close()
	return data,gold_labels


def count():
	conn = mdb.connect(host='127.0.0.1', port=3306, user='root', passwd='fdse', db='dataset2', charset='utf8')
	cursor = conn.cursor()
	try:
		cursor.execute("select data from classdataset")
		numrows = int(cursor.rowcount)
		print(numrows)
	except:
		import traceback
		traceback.print_exc()
	finally:
		cursor.close()
		conn.close()


#for i in  xrange(10):
#	for j in xrange(5):
#		file_path = '/home/x/mydisk/Zhao.treelstm/newestrecords/tree'+str(i+1)+'_'+str(j+1)+'.tfrecords'
#		class_prediction_path = '/home/x/mydisk/Zhao.treelstm/newestrecords/classPrediction'+str(i+1)+'_'+str(j+1)+'.txt'
#		method_prediction_path = '/home/x/mydisk/Zhao.treelstm/newestrecords/methodPrediction'+str(i+1)+'_'+str(j+1)+'.txt'
#		start = time.time()
#		insert(file_path,class_prediction_path,method_prediction_path)
#		print time.time()-start
#		if(i == 9 and j == 2):
#			break

