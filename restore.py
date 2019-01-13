import tensorflow as tf
import numpy as np
import operator
import collections
tf.reset_default_graph()
imported_meta =tf.train.import_meta_graph("./runs/1546023334/checkpoints/model-200.meta")
with tf.Session() as sess:

    imported_meta.restore(sess, tf.train.latest_checkpoint('./runs/1546023729/checkpoints'))
    variables_names = [v.name for v in tf.trainable_variables() if (v.name.endswith('W:0') & (v.name.find('conv-maxpool')!=-1))]
    print(variables_names)
    #weights = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('W:0')]
    #print('weights',weights)

    def pruned_weights(variable_name):
            var = sess.run(variable_name)
            print(type(var))
            c = np.var(var)
            weight_variance_list = []
            weight_var = []
    #print(var[0])
            for i in range(128):

                weight_variance = np.var(var[:,:,:,i])
                print(weight_variance)
                weight_var.append(weight_variance)
        
            print('weight_var',weight_var)
            weight_var_numpy = np.asarray(weight_var)
            print(weight_var_numpy)
            weight_var_sorted = weight_var_numpy.argsort()
    #print(weight_var_numpy.argsort())
            for x in range(20):
                indices = weight_var_sorted[x]
                print(indices)
                var[:,:,:,indices] = 0


            assign_node =  tf.assign(conv-maxpool-3/W:0,var)      

                #weight_var_numpy[indices] = 0

            print(var)
            return var      


    for variable_name in variables_names:
        pruned_filter_weights = pruned_weights(variable_name)













        


