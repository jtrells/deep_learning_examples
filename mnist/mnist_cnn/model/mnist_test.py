import tensorflow as tf
import numpy as np

class MnistTest():
    def __init__(self, sess, model, config):
        self.model = model
        self.sess = sess
        self.config = config
        sess.run(tf.global_variables_initializer())
        model.load(sess)

    def eval_in_batches(self, data, sess):
        """Get predictions for a dataset by running it in small batches."""
        size = data.shape[0]
        if size < self.config['eval_batch_size']:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = np.ndarray(shape=(size, self.config['num_labels']), dtype=np.float32)
        
        for begin in range(0, size, self.config['eval_batch_size']):
            end = begin + self.config['eval_batch_size']
            if end <= size:
                predictions[begin:end, :] = sess.run(self.model.eval_prediction, 
                        feed_dict={self.model.eval_data: data[begin:end, ...], self.model.is_training: False})
            else:
                batch_predictions = sess.run(self.model.eval_prediction, 
                    feed_dict={self.model.eval_data: data[-self.config['eval_batch_size']:, ...], self.model.is_training: False})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions
        
    def eval(self, test_data):
        predictions = self.eval_in_batches(test_data, self.sess)
        return predictions