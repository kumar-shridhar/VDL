import tensorflow as tf
import numpy as np
import operator
import collections
tf.reset_default_graph()
imported_meta =tf.train.import_meta_graph("./runs/1546457124/checkpoints/model-200.meta")


with tf.Session() as sess:

    imported_meta.restore(sess, tf.train.latest_checkpoint('./runs/1546457124/checkpoints'))
    
    #pruned_filter_weights1 = tf.convert_to_tensor(pruned_filter_weights)
    
    #for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv-maxpool-3'):
        #print(i)
    #current_scope = tf.get_variable_scope()
    #print(current_scope)    
        #with tf.variable_scope('conv-maxpool-3',reuse=tf.AUTO_REUSE):
            #v = tf.get_variable("W", [1])

    

    var_weights = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == 'conv-maxpool-3/W:0']
    x = sess.run(var_weights)
    print(x)
    var_weights_updated = var_weights.assign(var_weights-100)
    y = sess.run(var_weights_updated)
    print(y)
    saver.save(sess,tf.train.latest_checkpoint('./runs/1546457124/checkpoints'))
    




       
        
    #conv = tf.get_variable('conv-maxpool-3/W:0',[3,128,1,128])
    #print(current_scope.name )
    
    #print(v.name)

    #tf.cast('conv-maxpool-3/W:0', tf.float32)
    #assign_node =  tf.assign(conv,pruned_filter_weights)
    #sess.run(assign_node)
















        


