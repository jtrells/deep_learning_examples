from base.base_train import BaseTrain
import tensorflow as tf
import numpy as np
import os

class MnistTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(MnistTrainer, self).__init__(sess, model, data, config, logger)
        self.num_epochs = self.config['num_epochs']
        self.batch_size = self.config['batch_size']
        self.train_size = data['train_y'].shape[0]
    
    def train_epoch(self):
        num_it_per_epoch = int(self.train_size) // self.batch_size
        for step in range(num_it_per_epoch):
            self.step = step
            self.offset = (step * self.batch_size) % (self.train_size - self.batch_size)
            loss = self.train_step()
            
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        summaries_dict = {'loss': loss,}
        self.logger.summarize(cur_epoch, summaries_dict=summaries_dict, scope="summaries")
        self.model.save(self.sess)
            
        
    def train_step(self):
        batch_x = self.data['train_x'][self.offset:(self.offset + self.batch_size), ...]
        batch_y = self.data['train_y'][self.offset:(self.offset + self.batch_size)]
        
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True, self.model.train_size: self.train_size}
        # run the optimizer
        _, loss = self.sess.run([self.model.train_step, self.model.cross_entropy], feed_dict=feed_dict)
        
        if self.step % self.config['eval_frequency'] == 0:
            lr, predictions = self.sess.run([self.model.learning_rate, self.model.train_prediction], feed_dict=feed_dict)
            print('Step %d (epoch %.2f),' % (self.step, float(self.step) * self.config['batch_size'] / self.train_size))
            print('Minibatch loss: %.3f, learning rate: %.6f' % (loss, lr))
            print('Minibatch error: %.1f%%' % self.error_rate(predictions, batch_y))
            print('Validation error: %.1f%%' % self.error_rate(
              self.eval_in_batches(self.data['val_x'], self.sess, False), self.data['val_y']))
        return loss

    def eval_in_batches(self, data, sess, is_training):
        """Get predictions for a dataset by running it in small batches."""
        size = data.shape[0]
        if size < self.config['eval_batch_size']:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = np.ndarray(shape=(size, self.config['num_labels']), dtype=np.float32)
        
        for begin in range(0, size, self.config['eval_batch_size']):
            end = begin + self.config['eval_batch_size']
            if end <= size:
                predictions[begin:end, :] = sess.run(self.model.eval_prediction, 
                        feed_dict={self.model.eval_data: data[begin:end, ...], self.model.is_training: is_training})
            else:
                batch_predictions = sess.run(self.model.eval_prediction, 
                    feed_dict={self.model.eval_data: data[-self.config['eval_batch_size']:, ...], self.model.is_training: is_training})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions

    def error_rate(self, predictions, labels):
        """Return the error rate based on dense predictions and sparse labels."""
        return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])

    def eval_test(self):
        test_error = self.error_rate(self.eval_in_batches(self.data['test_x'], self.sess, False), self.data['test_y'])
        print('Test error: %.1f%%' % test_error)