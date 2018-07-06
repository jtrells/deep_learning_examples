from base.base_model import BaseModel
import tensorflow as tf

class MnistModel(BaseModel):
    def __init__(self, config):
        super(MnistModel, self).__init__(config)
        self.set_config_vars()
        self.build_model()
        self.init_saver()

    def set_config_vars(self):
        self.BATCH_SIZE = self.config['batch_size']
        self.IMAGE_SIZE = self.config['image_size']
        self.NUM_CHANNELS = self.config['num_channels']
        self.SEED = self.config['seed']
        self.NUM_LABELS = self.config['num_labels']
        self.EVAL_BATCH_SIZE = self.config['eval_batch_size']
        self.MAX_TO_KEEP = self.config['max_to_keep']
        self.SEED = self.config['seed']

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.MAX_TO_KEEP)
    
    def build_model(self):
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.train_size = tf.placeholder(tf.int32, name='train_size')
        
        self.x = tf.placeholder(tf.float32, name='x',
            shape=(self.BATCH_SIZE, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CHANNELS))
        self.y = tf.placeholder(tf.int64, name='y', shape=(self.BATCH_SIZE,))
        self.eval_data = tf.placeholder(tf.float32, name='eval_x', 
            shape=(self.EVAL_BATCH_SIZE, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CHANNELS))
        
        self.set_model_weights()
        logits = self.model(self.x)
        
        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=logits))
            self.regularizers = (tf.nn.l2_loss(self.fc1_weights)
                + tf.nn.l2_loss(self.fc1_biases)
                + tf.nn.l2_loss(self.fc2_weights)
                + tf.nn.l2_loss(self.fc2_biases))
            self.cross_entropy += 5.e-4 * self.regularizers
            
            # Optimizer: set up a variable that's incremented once per batch and
            # controls the learning rate decay.
            self.batch = tf.Variable(0, dtype=tf.float32)
            
            # Decay once per epoch, using an exponential schedule starting at 0.01.
            self.learning_rate = tf.train.exponential_decay(
                0.01,                           # Base learning rate.
                self.batch * self.BATCH_SIZE,   # Current index into the dataset.
                self.train_size,                # Decay step.
                0.95,                           # Decay rate.
                staircase=True)

            # Use simple momentum for the optimization.
            self.train_step = tf.train.MomentumOptimizer(self.learning_rate,
                                       0.9).minimize(self.cross_entropy, global_step=self.batch)

            # Predictions for the current training minibatch.
            self.train_prediction = tf.nn.softmax(logits)
            
            # Predictions for the test and validation, which we'll compute less often.
            self.eval_prediction = tf.nn.softmax(self.model(self.eval_data))

        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.cross_entropy)
            self.merged = tf.summary.merge_all()

    def model(self, data):
        # network architecture
        with tf.name_scope("conv1"):
            conv = tf.nn.conv2d(data, self.conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv1_biases))
            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        with tf.name_scope("conv2"):
            conv = tf.nn.conv2d(pool, self.conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv2_biases))
            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        with tf.name_scope("fully_connected_1"):
            pool_shape = pool.get_shape().as_list()
            reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)        
        
            hidden = tf.cond(
                tf.equal(self.is_training, tf.constant(True)), 
                lambda: tf.nn.dropout(hidden, 0.5, seed=self.SEED), 
                lambda: hidden)
            
        with tf.name_scope("fully_connected_2"):
            logits = tf.matmul(hidden, self.fc2_weights) + self.fc2_biases
        
        return logits

    def set_model_weights(self):
        # Conv1 weights: 5x5 filters, 32 filters (depth)
        self.conv1_weights = tf.Variable(
            tf.truncated_normal([5, 5, self.NUM_CHANNELS, 32], stddev=0.1, seed=self.SEED, dtype=tf.float32))
        self.conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float32))

        # Conv2 weights: 5x5 filters, 64 filters. Remember the 32 is the number of channels
        # from Conv1
        self.conv2_weights = tf.Variable(
            tf.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=self.SEED, dtype=tf.float32))
        self.conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))

        # image size halved twice by max pooling layers, and the last conv layer has 64 filters
        reduced_image_size = self.IMAGE_SIZE // 4 * self.IMAGE_SIZE // 4 * 64
        # fully connected layer with 512 nodes
        self.fc1_weights = tf.Variable(
            tf.truncated_normal([reduced_image_size, 512], stddev=0.1, seed=self.SEED, dtype=tf.float32))
        self.fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32))

        # output for the last 10 digits (num_labels)
        self.fc2_weights = tf.Variable(
            tf.truncated_normal([512, self.NUM_LABELS], stddev=0.1, seed=self.SEED, dtype=tf.float32))
        self.fc2_biases = tf.Variable(tf.constant(0.1, shape=[self.NUM_LABELS], dtype=tf.float32))        
        