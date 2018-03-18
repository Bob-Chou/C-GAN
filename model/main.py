from c_gan import *
from tensorflow.examples.tutorials.mnist import input_data

def main():
	'''
	A quick startup for this model
	'''
	batch_size = 128
	epoch = 5
	mnist = input_data.read_data_sets('./MNIST_data', one_hot=False)
	train_iter = mnist.train.num_examples * epoch // batch_size
	print_every = 100
	plot_every = 200
	log_every = 200
	# use CNN to construct the c-GAN
	startup = CGAN(28, 'cnn', 2, d_hidden=[32, 64], g_hidden=[1024, 128])
	# use DNN to construct the c-GAN
	# startup = CGAN(28, 'dnn', 2, d_hidden=[256, 256], g_hidden=[1024, 1024])

	# train our model
	with tf.Session(graph=startup.graph) as sess:
		sess.run(tf.global_variables_initializer())
		for t in range(train_iter):
			# get data and condition
			x, labels = mnist.train.next_batch(batch_size)
			conditions = get_mnist_condition(batch_size, labels)
			# train one step
			d_loss, g_loss = startup.train(sess, batch_size, x, conditions)
			if print_every > 0 and t % print_every == 0:
				print('training: {}/{}, discriminator: {:.4}, generator: {:.4}'.format(t, train_iter, d_loss, g_loss))
			if plot_every > 0 and  t % plot_every == 0:
				sample_conditions = get_mnist_condition(100, np.arange(100) % 10)
				show_images(image_denorm(startup.sample(sess, 100, sample_conditions, denorm=True)))
			if log_every > 0 and t % log_every == 0:
				startup.log(sess, batch_size, x, conditions, t)
		# show  the final results
		sample_conditions = get_mnist_condition(100, np.arange(100) % 10)
		show_images(image_denorm(startup.sample(sess, 100, sample_conditions, denorm=True)))

if __name__ == '__main__':
	main()

