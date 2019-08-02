"""
reference

GAN:
Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley,
Sherjil Ozair, Aaron Courville, Yoshua Bengio 2014
Generative Adversarial Networks

https://github.com/mchablani/deep-learning/blob/master/gan_mnist/Intro_to_GANs_Exercises.ipynb?source=post_page---------------------------

Tensorpack:
Wu, Yuxin et al. 2016
https://github.com/tensorpack/

"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorpack as tp

mode="gen" # "train" or "gen"

z_dim = 100
alpha = 0.01 # for leaky ReLU

batch_size = 64
NEPOCHS = 60
lr_d = 2e-3
lr_g = 1e-4

def main(mode=mode):
    """The main function"""
    if mode == "train":
        # Create dataset and iterator
        training_iterator = tp.dataset.Mnist('train')
        training_iterator = tp.BatchData(training_iterator, batch_size)
        
        # Build computation graph
        inputs_img, inputs_z, d_loss, g_loss = make_graph()
        d_train_op, g_train_op = train_op(d_loss, g_loss, lr_d=lr_d, lr_g=lr_g)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Run training
            samples, d_losses, g_losses = run_training(
                    sess, training_iterator, d_train_op, g_train_op,
                    inputs_img, inputs_z, d_loss, g_loss, n_epochs=NEPOCHS
                    )
            print('Done training!')
        _ = plot_training_curves(d_losses, g_losses, "losses.png")
        #_ = plot_one_set_of_samples(samples, -1, "samples_training_epoch%d.png"%NEPOCHS)
        _ = plot_training_samples_improvement(samples, "samples_training_progress.png")
    
    elif mode == "gen":
        _, inputs_z, _, _ = make_graph()
        
        z = np.random.uniform(-1, 1, size=(9, z_dim))
        
        with tf.Session() as sess:
            samples = generate_samples(sess, inputs_z, z)
        
        _ = plot_one_set_of_samples([samples], 0, "samples_gen.png")
    
    return True




def generator(z, reuse=False):
    """the generator (network)
    
    Args:
        z (tf.placeholder): Placeholder for variables z
        reuse: 
    Returns:
        out (tf.Tensor): generated imgs from z
    """
    with tf.variable_scope('generator', reuse=reuse):
        w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
        b_init = tf.constant_initializer(0.)
        # hidden layers
        w0 = tf.get_variable('g_w0', [100, 256], initializer=w_init)
        b0 = tf.get_variable('g_b0', [256], initializer=b_init)
        net = tf.nn.relu(tf.matmul(z, w0) + b0)
        
        w1 = tf.get_variable('g_w1', [256, 512], initializer=w_init)
        b1 = tf.get_variable('g_b1', [512], initializer=b_init)
        net = tf.nn.relu(tf.matmul(net, w1) + b1)
        
        w2 = tf.get_variable('g_w2', [512, 1024], initializer=w_init)
        b2 = tf.get_variable('g_b2', [1024], initializer=b_init)
        net = tf.nn.relu(tf.matmul(net, w2) + b2)
        # output layer
        w3 = tf.get_variable('g_w3', [1024, 784], initializer=w_init)
        b3 = tf.get_variable('g_b3', [784], initializer=b_init)
        return tf.nn.tanh(tf.matmul(net, w3) + b3)

def discriminator(x, reuse=False, alpha=alpha):
    """the discriminator (network)
    Args:
        x (tf.placeholder): Placeholder for images
        reuse: 
        alpha:
    Returns:
        out:
        logits (tf.Tensor): discriminating results
    """
    with tf.variable_scope('discriminator', reuse=reuse):
        w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
        b_init = tf.constant_initializer(0.)
        # hidden layers
        w0 = tf.get_variable('d_w0', [784, 1024], initializer=w_init)
        b0 = tf.get_variable('d_b0', [1024], initializer=b_init)
        net = tf.nn.leaky_relu(tf.matmul(x, w0) + b0, alpha=alpha)
        
        w1 = tf.get_variable('d_w1', [1024, 512], initializer=w_init)
        b1 = tf.get_variable('d_b1', [512], initializer=b_init)
        net = tf.nn.leaky_relu(tf.matmul(net, w1) + b1, alpha=alpha)
        
        w2 = tf.get_variable('d_w2', [512, 256], initializer=w_init)
        b2 = tf.get_variable('d_b2', [256], initializer=b_init)
        net = tf.nn.leaky_relu(tf.matmul(net, w2) + b2, alpha=alpha)
        # output layer
        w3 = tf.get_variable('d_w3', [256, 1], initializer=w_init)
        b3 = tf.get_variable('d_b3', [1], initializer=b_init)
        return tf.matmul(net, w3) + b3

def loss_discriminator(logits_real, logits_fake):
    """compute the loss of discriminator (-original objective, so minimize later)
    Args:
        logits_real (tf.Tensor): discriminating results of the real imgs
        logits_fake (tf.Tensor): discriminating results of the generating imgs
    Return:
        loss (tf.Tensor): the loss for each example in the mini-batch.
    """
    labels_real = tf.ones_like(logits_real)
    labels_fake = tf.zeros_like(logits_fake)
    
    loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels_real,
            logits=logits_real
            )
    loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels_fake,
            logits=logits_fake
            )
    return tf.reduce_mean(loss_real + loss_fake)

def loss_generator(logits_fake):
    """compute the loss of generator (-original objective, so minimize later)
    Args:
        logits_fake (tf.Tensor): discriminating results of the generating imgs
    Return:
        loss (tf.Tensor): the loss for each example in the mini-batch.
    """
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(logits_fake),
            logits=logits_fake
            )
    loss = tf.reduce_mean(loss)
    return loss

def make_graph():
    """Creates computation graph for model.

    Arguments:
        
    Returns:
        d_loss,
        g_loss: tf.Tensor holding the resulting loss of discriminator, generator
    """
    # the inputs can be real images and latent variables z
    inputs_img = tf.placeholder(tf.float32, (None, 784), name="inputs_img")
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name="inputs_z")
    
    genert_img = generator(inputs_z, reuse=False)
    
    d_logits_real = discriminator(inputs_img, reuse=False)
    d_logits_fake = discriminator(genert_img, reuse=True)
    
    d_loss = loss_discriminator(d_logits_real, d_logits_fake)
    g_loss = loss_generator(d_logits_fake)
    
    return inputs_img, inputs_z, d_loss, g_loss

def train_op(d_loss, g_loss, lr_d=lr_d, lr_g=lr_g):
    """
    Arguments:
        d_loss: tf.Tensor, loss of the discriminator
        g_loss: tf.Tensor, loss of the generator
        lr_d,
        lr_g: float, learning rate for discriminator, generator
    Returns:
        d_train_op,
        g_train_op: training operation for the discriminator, generator
    """
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if var.name.startswith("generator")]
    d_vars = [var for var in t_vars if var.name.startswith("discriminator")]
    
    d_train_op = tf.train.AdamOptimizer(lr_d).minimize(d_loss, var_list=d_vars)
    g_train_op = tf.train.AdamOptimizer(lr_g).minimize(g_loss, var_list=g_vars)
    return d_train_op, g_train_op
    
def run_training(sess, training_iterator, d_train_op, g_train_op,
                 inputs_img, inputs_z, d_loss, g_loss, n_epochs):
    """
    Args:
        sess (tf.Session): tensorflow Session object
        train_iterator (iterable): iterator over training data
        d_train_opt,
        g_train_opt: Training operation for discriminator, generator
        inputs_img (tf.placeholder): Placeholder for images
        inputs_z (tf.placeholder): Placeholder for variables z
        d_loss,
        g_loss (tf.Tensor): loss tensor
        nepochs (int): number of epochs to run training
    Returns:
        samples: list of [9,28*28] np arrays, imgs generated before training 
                 and at the end of each epoch
        d_losses,
        g_losses: lists of losses of discriminator, generator
    """
    samples = [sess.run(
                generator(inputs_z, reuse=True),
                feed_dict={inputs_z: np.random.uniform(-1, 1, size=(9, z_dim))}
                )]
    d_losses = []
    g_losses = []
    saver = tf.train.Saver(var_list =
        [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        )
    it=1
    for i_epoch in range(n_epochs):
        # Train one epoch
        training_iterator.reset_state()
        for batch_x, _ in training_iterator:
            # Reshape to feed it into the network
            batch_x = batch_x.reshape(-1, 784)
            batch_x = batch_x*2 - 1 # because we use tanh for output of generator
            
            batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
            # Run
            train_loss_d, _ = sess.run([
                    d_loss, d_train_op
                    ], feed_dict={inputs_img: batch_x, inputs_z: batch_z})
            train_loss_g, _ = sess.run([
                    g_loss, g_train_op
                    ], feed_dict={inputs_z: batch_z})
            if (it % 100) == 0:
                print('[epoch=%3d, it=%6d]\nLoss: discrimin # generator\n%15.4f # %4.4f'
                      % (i_epoch+1, it, train_loss_d, train_loss_g))  
                
                d_losses.append(train_loss_d)
                g_losses.append(train_loss_g)
            
            it=it+1
        
        # Sample from generator
        z = np.random.uniform(-1, 1, size=(9, z_dim))
        gen_samples = sess.run(
                       generator(inputs_z, reuse=True),
                       feed_dict={inputs_z: z}
                       )
        samples.append(gen_samples)
        saver.save(sess, './checkpoints/generator.ckpt')
    return samples, d_losses, g_losses

def generate_samples(sess, inputs_z, samples_z):
    """load saved parameters and generate images
    Args:
        sess (tf.Session): tensorflow Session object
        z (tf.placeholder): Placeholder for variables z
        samples_z (np.array): input z, to be fed into the generator
    """
    saver = tf.train.Saver(var_list =
        [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        )
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    return sess.run(
                   generator(inputs_z, reuse=True),
                   feed_dict={inputs_z: samples_z})





def plot_training_curves(d_losses, g_losses, filename="losses.png"):
    fig, ax = plt.subplots()
    plt.plot(d_losses, label='Loss_discriminator')
    plt.plot(g_losses, label='Loss_generator')
    plt.xlabel("hundred iterations")
    plt.legend()
    plt.savefig(filename, bbox_inches='tight')
    return fig

def plot_one_set_of_samples(samples, epoch, filename="samples_training_.png"):
    fig = plt.figure(figsize=(7,7))
    for i, img in enumerate(samples[epoch]):
        ax=plt.subplot(3,3,i+1)
        ax.imshow(img.reshape((28,28)), cmap='gray')
        ax.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    return fig

def plot_training_samples_improvement(samples, filename="samples_training_progress.png"):
    nepoch = len(samples) - 1 # the first one is set of images before training
    fig = plt.figure( figsize=(18,2*((nepoch-1)//5)+2) )
    epoch_idx = 0
    while True:
        # plot
        for j, img in enumerate(samples[epoch_idx]):
            ax=plt.subplot((nepoch-1)//5 + 2, 9, epoch_idx*9/5 + j+1)
            ax.imshow(img.reshape((28,28)), cmap='gray')
            ax.axis('off')
            if j==0:
                if epoch_idx == 0:
                    ax.set_title("Before training")
                else:
                    ax.set_title("epoch %d"%(epoch_idx))
        # the epoch we want to plot next
        if epoch_idx == nepoch:
            break
        elif epoch_idx <= nepoch-5:
            epoch_idx = epoch_idx+5
        else:
            epoch_idx = nepoch
    plt.savefig(filename, bbox_inches='tight')
    return fig

if __name__ == '__main__':
    main()
