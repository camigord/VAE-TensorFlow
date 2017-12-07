'''
Code based on the implementation from Jan Hendrik Metzen
presented in this blog: https://jmetzen.github.io/2015-11-27/vae.html
'''

import numpy as np
import tensorflow as tf
from VAE import *
import argparse

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
n_samples = mnist.train.num_examples

def train(network_architecture, restore_model, save_path, learning_rate=0.001, batch_size=100, training_epochs=10, display_step=5):

    with tf.Session() as sess:
        vae = VAE(sess, network_architecture, save_path, learning_rate=learning_rate, batch_size=batch_size)

        if restore_model:
            vae._restore_model()
            print('Model restored...')

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_samples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, _ = mnist.train.next_batch(batch_size)

                # Fit training using batch data
                cost = vae.partial_fit(batch_xs)
                # Compute average loss
                avg_cost += cost / n_samples * batch_size

            # Display logs per epoch step
            if epoch % display_step == 0:
                vae._save_model()
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print('Training finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--latent_dim', type=int, default=20)
    parser.add_argument('--save_path', type=str, default='./model/')
    conf = parser.parse_args()

    np.random.seed(0)
    tf.set_random_seed(0)

    network_architecture = dict(n_hidden_recog_1=500, # 1st layer encoder neurons
                                n_hidden_recog_2=500, # 2nd layer encoder neurons
                                n_hidden_gener_1=500, # 1st layer decoder neurons
                                n_hidden_gener_2=500, # 2nd layer decoder neurons
                                n_input=784, # MNIST data input (img shape: 28*28)
                                n_z=conf.latent_dim)  # dimensionality of latent space

    print('Starting training...')

    train(network_architecture, conf.restore, conf.save_path, training_epochs=75)
