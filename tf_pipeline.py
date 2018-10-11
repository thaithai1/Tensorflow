import tensorflow as tf

#basic function
# input_value = tf.constant(1.0)


def monitor_operation():
    graph = tf.get_default_graph()  # The nodes of the TensorFlow graph are called “operations,” or “ops.” 
    operations=graph.get_operations()
    print("Operations\n\n",operations,"\n======================")
    print("Operations Details :\n")
    for op in operations: 
        print(op.name)      # We can see what operations are in the graph
        print("Input:")
        for op_input in op.inputs:print(op_input)
        print("---------------")


def one_hot_encode(labels,C):
    """
    Arguments:
    labels -- vector containing the labels -- shape=(n,)
    C -- int. number of different classes

    Return:
    one-hot Matrix -- shape(n,C)
    """
    with tf.Session() as sess:
        res=tf.one_hot(labels,C)
        res=sess.run(res)
    return res




def initialize_parameters(arch,n_x,n_y):
    """
    Arguments : 
    arch -- archiecture :  [(10,"reLu"),(10,"reLu"),(None, "softmax")]  
    n_x -- input dimension
    n_y -- output dimension
    """
    parameters={}
    layers=[n_x] + [l[0] for l in arch if l[0] is not None] + [n_y]
    for i in range(1,len(layers)):
        parameters["W"+str(i)]= tf.get_variable("W"+str(i),[layers[i],layers[i-1]],initializer = tf.contrib.layers.xavier_initializer())
                parameters["b"+str(i)]= tf.get_variable("b"+str(i),[layers[i],1],initializer = tf.zeros_initializer)
    return parameters


def forward_prop(X,parameters,arch):
	activation={"A0": X}
	activations_func=[act[1] for act in arch]
	for i in range(1,len(activations_func)+1):
		if activations_func[i-1]=="reLu":
			activation["A"+str(i)]=tf.nn.relu(tf.matmul(parameters["W"+str(i)], activation["A"+str(i-1)]) + parameters["b"+str(i)])
		elif activations_func[i-1]=="softmax":
			activation["A"+str(i)]=tf.nn.softmax(tf.matmul(parameters["W"+str(i)], activation["A"+str(i-1)]) + parameters["b"+str(i)])
	return activation["A"+str(len(activations_func))]



def model(arch, X_train, X_test, Y_train, Y_test, lr=0.001, epochs=200, bs=32, verbose=True):

	#dimension
    n_x=X_test.shape[0]
    n_y=Y_train.shape[0]
    

    #Create Placeholders
    X = tf.Placeholders(tf.float32,shape=[n_x,None])
    Y = tf.Placeholders(tf.float32,shape=[n_y,None])








    
print("o")



