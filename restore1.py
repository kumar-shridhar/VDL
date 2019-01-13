import tensorflow as tf
import numpy as np
import operator
import collections
import os
tf.reset_default_graph()
imported_meta =tf.train.import_meta_graph("./runs/1546990973/checkpoints/model-12100.meta")


with tf.Session() as sess:

    imported_meta.restore(sess, tf.train.latest_checkpoint('./runs/1546990973/checkpoints'))
    #print("Model restore finished, current globle step: %d" % global_step.eval())
    global_step = tf.get_default_graph().get_tensor_by_name('global_step:0')
    print('Global Step',global_step.eval())
    variables_names = [v.name for v in tf.trainable_variables() if (v.name.endswith('W:0') & (v.name.find('conv-maxpool')!=-1))]
    print('Trainable_variable',tf.trainable_variables())
    #print(variables_names)
    optimizer_scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    #print('optimizer_scope',optimizer_scope)
    optimizer = tf.get_collection("train_op")[0]
    #print('Optimizer',optimizer)
    #new_node = tf.get_default_graph().get_tensor_by_name('conv-maxpool-4/W/Adam:0')
    #weights = tf.get_default_graph().get_tensor_by_name('embedding/W:0')
    #val = sess.run(weights)
    #print('Weights',val[0].shape)
    #print('Weights',val[0])
    W1 = tf.get_default_graph().get_tensor_by_name('conv-maxpool-4/W:0')
    print('W1',sess.run(W1))
    B = tf.get_default_graph().get_tensor_by_name('conv-maxpool-3/b:0')
    print('Bias', sess.run(B))


    #weights = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('W:0')]
    #print('weights',weights)
    
    def pruned_weights(variable_name):

            indices_List = []
            var = sess.run(variable_name)
            #print('Original Value',var[:,:,:,124])
            c = np.var(var)
            weight_variance_list = []
            weight_var = []
    #print(var[0])
            for i in range(128):

                weight_variance = np.var(var[:,:,:,i])
                #print(weight_variance)
                weight_var.append(weight_variance)
        
            #print('weight_var',weight_var)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
            weight_var_numpy = np.asarray(weight_var)
            #print(weight_var_numpy)
            weight_var_sorted = weight_var_numpy.argsort()
    #print(weight_var_numpy.argsort())
            for x in range(20):

                indices = weight_var_sorted[x]
                indices_List.append(indices)
                #print(indices)
                var[:,:,:,indices] = 0


                 

                #weight_var_numpy[indices] = 0

            print('Indices List',indices_List)
            return var      


    
    for variable_name in variables_names:
        pruned_filter_weights = pruned_weights(variable_name)
        

        #pruned_filter_weights = pruned_weights('conv-maxpool-3/W:0')
    
        #print(type(pruned_filter_weights))

    # with tf.variable_scope("conv-maxpool-3",tf.AUTO_REUSE):
    #     bar2 = tf.get_variable("W", [1])
    #     print('bar',bar2.name)



    

        new_node = tf.get_default_graph().get_tensor_by_name(variable_name)
        assign_node = tf.assign(new_node,pruned_filter_weights)   
        sess.run(assign_node)
        saver = tf.train.Saver(tf.global_variables())
        print('variable list saver',saver._var_list)
        dir = os.path.dirname(os.path.realpath(__file__))
        new_dir = os.path.join(dir,"pruned_data")
        if not os.path.exists(new_dir):
                os.makedirs(new_dir)
        saver.save(sess, new_dir+ '/data-all')

        
        


    













        


