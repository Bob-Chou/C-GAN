from utils import *

class CGAN(object):
	"""docstring for CGAN"""
	def __init__(self, image_size, model, layers_num, d_hidden, g_hidden, noise_dim=100, condition_dim=10, learning_rate=1e-3, beta1=0.5, use_summary=True, use_save=True, use_model=False):
		super(CGAN, self).__init__()
		self.image_dim = image_size * image_size
		self.image_size = image_size
		self.model = model
		self.layers_num = layers_num
		if len(g_hidden) != layers_num or len(d_hidden) != layers_num:
			raise ValueError('input g_hidden and d_hidden should match the layers_num!')
		self.hidden_dim={'d': d_hidden, 'g': g_hidden}
		self.noise_dim = noise_dim
		self.condition_dim = condition_dim
		self.solver = {
			'd': tf.train.AdamOptimizer(learning_rate, beta1),
			'g': tf.train.AdamOptimizer(learning_rate, beta1)
		}
		self.graph = tf.Graph()
		self.use_summary = use_summary
		self.use_save = use_save
		self.use_model = use_model
		with self.graph.as_default():	
			with tf.variable_scope('c_gan', reuse=tf.AUTO_REUSE):
				# placeholder for images from the training dataset
				self.inputs = tf.placeholder(tf.float32, (None, self.image_dim), name='real')
				self.noise_batch = tf.placeholder(tf.int32, (None), name='noise_batch')
				self.condition = tf.placeholder(tf.float32, (None, self.image_size, self.image_size, self.condition_dim), name='condition')
				# random noise fed into our generator
				z = tf.random_uniform((self.noise_batch, self.noise_dim), minval=-1, maxval=1, name='noise')
				# combine the noise and the condition
				z = tf.concat([z, self.condition[:, 0, 0, :]], axis=1)
				# now define the gan
				self.products = generator(z, self.image_size, self.model, self.layers_num, self.hidden_dim['g'])
				# sythesize the condition for discriminator
				real = discriminator(image_normal(self.inputs), self.noise_batch, self.image_size, self.model, self.layers_num, self.hidden_dim['d'], condition=self.condition)
				fake = discriminator(self.products, self.noise_batch, self.image_size, self.model, self.layers_num, self.hidden_dim['d'], condition=self.condition)
				# compute the loss
				d_loss, g_loss = gan_loss(real, fake)
				# record some operations
				self.loss = {
					'd': d_loss,
					'g': g_loss
				}
				# Get the list of parameters of the discriminator and generator for training
				self.params = {
					'd': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'c_gan/discriminator'),
					'g': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'c_gan/generator')
				}
				# Define the training operation
				self.train_step = {
					'd': self.solver['d'].minimize(self.loss['d'], var_list = self.params['d']),
					'g': self.solver['g'].minimize(self.loss['g'], var_list = self.params['g'])		
				}
				# (if) summary the training process
				if self.use_summary:
					self.log_path = '../log'
					self.summary_writer = tf.summary.FileWriter(self.log_path, self.graph)
					tf.summary.scalar('sum_d_loss', self.loss['d'])
					tf.summary.scalar('sum_g_loss', self.loss['g'])
					tf.summary.image('sum_products', tf.reshape(self.products[:1], (1, self.image_size, self.image_size, -1)))
					self.summary = tf.summary.merge_all()


	def log(self, session, minibatch, data, condition, iteration):
		if session.graph is not self.graph:
			raise ValueError('Please feed in a session running on the object\'s graph (self.graph)')
		self.summary_writer.add_summary(session.run(self.summary, feed_dict={self.inputs: data, self.noise_batch: minibatch, self.condition: condition}), iteration)
	
	def train(self, session, minibatch, data, condition):
		if session.graph is not self.graph:
			raise ValueError('Please feed in a session running on the object\'s graph (self.graph)')
		# train the discriminator
		d_loss, _ = session.run([self.loss['d'], self.train_step['d']], feed_dict={self.inputs: data, self.noise_batch: minibatch, self.condition: condition})
		# train the generator
		g_loss, _ = session.run([self.loss['g'], self.train_step['g']], feed_dict={self.noise_batch: minibatch, self.condition: condition})
		return d_loss, g_loss

	def sample(self, session, batch, condition, denorm=False):
		if session.graph is not self.graph:
			raise ValueError('Please feed in a session running on the object\'s graph (self.graph)')
		sample = session.run(self.products, feed_dict={self.noise_batch: 100, self.condition: condition})
		if denorm:
			return image_denorm(sample)
		else:
			return sample



