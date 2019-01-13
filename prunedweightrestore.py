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



tf.flags.DEFINE_float("dev_sample_percentage", 0.2, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters  
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS





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
    print(x)

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    
    print('shape',x_train.shape)
    #print("Vocabulary". vocab_processor.vocabulary_)
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev

tf.reset_default_graph()

with tf.Session() as sess:
    imported_meta =tf.train.import_meta_graph("./pruned_data/data-all.meta")
    imported_meta.restore(sess, tf.train.latest_checkpoint("./pruned_data"))
    variables_list = [n.name for n in tf.get_default_graph().as_graph_def().node]
    print('variables_list',variables_list)

    dir = os.path.dirname(os.path.realpath(__file__))
    new_dir = os.path.join(dir,"summarypruneddata")
    if not os.path.exists(new_dir):
            os.makedirs(new_dir)

    dev_summary_writer = tf.summary.FileWriter(new_dir, sess.graph)
    variables_names = [v.name for v in tf.trainable_variables() if (v.name.endswith('W:0') & (v.name.find('conv-maxpool')!=-1))]
    print(tf.trainable_variables())
    #print(variables_names)
    conv_val = sess.run('conv-maxpool-5/W:0')
    #print(conv_val[:,:,:,38])
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()

    cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)


    # def train_step(x_batch, y_batch):

    #     feed_dict = {
    #     cnn.input_x: x_batch,
    #     cnn.input_y: y_batch,
    #     cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
    #     }
    #     loss, accuracy = sess.run([cnn.loss, cnn.accuracy], feed_dict)
    #     time_str = datetime.datetime.now().isoformat()
    #     print("loss {:g}, acc {:g}".format(loss, accuracy))
                
    

    # def dev_step(x_batch, y_batch, writer=None):

                
    #     feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 1.0}
    #     loss, accuracy = sess.run([cnn.loss, cnn.accuracy],feed_dict)
    #     time_str = datetime.datetime.now().isoformat()
    #     print("loss {:g}, acc {:g}".format(loss, accuracy))
    # #sequence_length = x_train.shape[1]
    #num_classes = y_train.shape[1]
    #input_x= tf.placeholder(tf.int32, [None, sequence_length])
    #input_y = tf.placeholder(tf.float32, [None, num_classes])
    #dropout_keep_prob = tf.placeholder(tf.float32)
        #weights = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
        #print('weights',weights)
    #weights = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('W:0')]
    #print('weights',weights)

    sess.run(tf.global_variables_initializer())

    batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        output = sess.run(cnn.predictions, feed_dict={cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 0.5})
        print('Output',output)
        #train_step(x_batch, y_batch)

        
        #dev_step(x_dev, y_dev, writer=dev_summary_writer)   
            
                    

    

        
        


    













        


