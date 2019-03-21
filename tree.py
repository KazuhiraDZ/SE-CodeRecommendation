class Tree:
	def __init__(self):
		self.parent = None
		self.num_children = 0
		self.children = list()
		self.gold_label = None
		self.word = None
		self._size = None
		self.index = None
		self._depth = None
		self.trace = None
		self.holeSize = None
		self.block = None
		self.variables = None
		self.originalStatement = None
		self.noise_label = None

	def add_child(self,child):
		child.parent = self
		self.num_children += 1
		self.children.append(child)

	def size(self):
		if getattr(self,'_size'):
			return self._size
		count = 1
		for i in xrange(self.num_children):
			count += self.children[i].size()
		self._size = count
		return self._size

	def depth(self):
		count = 0
		if self.num_children > 0:
			for i in xrange(self.num_children):
				child_depth = self.children[i].depth()
				if child_depth > count:
					count = child_depth
		count += 1
		return count

	def depth_first(self,tree, vocab):
		print ('prarent: ', tree.index ,' ' , tree.word, ' ', vocab.index(tree.word))
		for i in xrange(tree.num_children):
			print ('child: ' , tree.children[i].index , ' ' , tree.children[i].word, vocab.index(tree.children[i].word))
		for i in xrange(tree.num_children):
			self.depth_first(tree.children[i],vocab)	
            
	def depth_first_index(self,tree,block,vocab):
		if tree.word != "conditionEnd" and tree.word != "end":
			block.append(vocab.index(tree.word))
		for i in range(tree.num_children):
			self.depth_first_index(tree.children[i],block,vocab)
        
	
	def test(self,tree,vocab):
		if vocab.index(tree.word) > 2277786:
			print (tree.word,': ',vocab.index(tree.word))
		for i in xrange(tree.num_children):
			self.test(tree.children[i],vocab)

	def get_child_num(self,tree):
		num = 0
		if tree.num_children > 6:
			num = tree.num_children
			return num
		else:
			for i in xrange(tree.num_children):
				num = self.get_child_num(tree.children[i])
				if num > 6:
					return num
				else:
					num = 0
					continue
			return num




		#nodes.append(tree)
		#print 'succ'
		#for i in xrange(tree.num_children):
			#iterate(tree.children[i],nodes)

