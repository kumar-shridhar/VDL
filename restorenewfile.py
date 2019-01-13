import tensorflow as tf
import numpy as np
import operator
import collections
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    datasets = data_helpers.get_datasets_political_parties()
    x_text, y = data_helpers.load_data_labels(datasets)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    print("vocab_processor",vocab_processor)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    #print(x)

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(0.2 * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    
    print('shape',x_train.shape)
    #print("Vocabulary". vocab_processor.vocabulary_)
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev




with tf.Session() as sess:
        saver = tf.train.import_meta_graph("./pruned_data/data-all.meta")
        saver.restore(sess, tf.train.latest_checkpoint('./pruned_data'))
        graph = tf.get_default_graph()
        #w1 = graph.get_tensor_by_name("conv-maxpool-3/W:0")
        val = sess.run("conv-maxpool-4/W:0")
        print('value', val[:,:,:,9])

        variables_names = [v.name for v in tf.trainable_variables()]
        print('variables_names',variables_names)
        # optimizer_scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # print(optimizer_scope)
        # variables_list = [n.name for n in tf.get_default_graph().as_graph_def().node]
        # print('variables_list',variables_list)
        dir = os.path.dirname(os.path.realpath(__file__))
        new_dir = os.path.join(dir,"summarypruneddata")
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        dev_summary_writer = tf.summary.FileWriter(new_dir, sess.graph)
        x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
        trainable_collection = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
       
        variables_to_remove = list()
        for var in trainable_collection:
        	if var.name=="conv-maxpool-3/W:0" or var.name=="conv-maxpool-4/W:0" or var.name=="conv-maxpool-5/W:0":
        		variables_to_remove.append(var)
        for rem in variables_to_remove:
        	trainable_collection.remove(rem)

        
        			



        print('trainable_collection',trainable_collection)
        predictions = tf.get_default_graph().get_tensor_by_name('output/predictions:0')
        
        loss = tf.get_collection("loss")[0]
        print('lossfromcollection',loss)
        global_step = tf.get_default_graph().get_tensor_by_name('global_step:0')
        print('Global_step', global_step.eval())

		#print('Global Step',global_step.eval())

        #print(sess.run(loss))
        accuracy = tf.get_collection("accuracy")[0]
        print('accuracyfromcollection',accuracy)
        #optimizer = tf.get_collection("train_op")[0]
        #train_op = graph.get_operation_by_name("train_op")
        #print('train_op',train_op)
        # optimizer_scope = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if (v.name.find('Adam')!=-1)] 

        # #optimizer_scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # print('optimizer_scope',optimizer_scope)
        # pruned_filter_weights = pruned_weights('conv-maxpool-3/W:0')
        # val = sess.run(pruned_filter_weights)

        #sess.run(tf.initialize_variables(optimizer_scope))
        
        # optimizer = tf.get_collection("train_op")[0]
        # print('optimizer',optimizer)
        # #sess.run(tf.initialize_variables(optimizer))
        # reset_optimizer_op = tf.variables_initializer(optimizer)
        # sess.run(reset_optimizer_op)
		
        optimizer = tf.train.GradientDescentOptimizer(0.005)
        train_op=optimizer.minimize(loss, var_list=trainable_collection)
        sess.run(tf.variables_initializer(optimizer.variables()))
        print('trainable_variables',tf.trainable_variables())
        #adam_initializers = [var for var in tf.global_variables() if 'Adam' in var.name]
        #print('adam_initializers',adam_initializers)

        
        #sess.run(tf.global_variables_initializer())
        #print('embedding_Adam',sess.run('embedding/W/Adam:0'))	
        #val = sess.run('conv-maxpool-3/W:0')
        #print(val[:,:,:,31])


        #input_x = tf.get_default_graph().get_tensor_by_name('input_x:0')
        #input_y = tf.get_default_graph().get_tensor_by_name('input_y:0')
        #dropout_keep_prob = tf.get_default_graph().get_tensor_by_name('dropout_keep_prob:0')
        #accuracy = tf.get_default_graph().get_tensor_by_name('accuracy/accuracy:0')
        #loss = tf.get_default_graph().get_tensor_by_name('loss/Mean:0')
        step = 0
        batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), 64,10)
        #print('batches',batches)
        for batch in batches:
        	print('Batch shape',batch.shape)
        	x_batch, y_batch = zip(*batch)
        	_, loss_val, acc = sess.run([train_op, loss, accuracy], feed_dict={"input_x:0" : x_batch, "input_y:0" : y_batch, "dropout_keep_prob:0": 0.5})
        	step = step+1
        	time_str = datetime.datetime.now().isoformat()
        	#current_step = tf.train.global_step(sess, global_step)
        	print("{}:, loss {:g}, acc {:g}".format(time_str, loss_val, acc))
        	
        		#print('i',i%100)
        	if step % 100 == 0:

        		loss_validation, accuracy_validation = sess.run([loss, accuracy], feed_dict={"input_x:0" : x_dev, "input_y:0" : y_dev, "dropout_keep_prob:0": 1.0})
        		time_str = datetime.datetime.now().isoformat()
        		print("{}:, loss_validation {:g}, accuracy_validation {:g}".format(time_str, loss_validation, accuracy_validation))




        #print(sess.run(w1[:,:,:,20]))



