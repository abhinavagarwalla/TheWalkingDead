import tensorflow as tf
import numpy as np

def pullBandit(bandit):
    if np.random.randn(1) > bandit:
        return 1
    return -1

def marmband():
	bandits = [0.2,0,-0.2,-5]
	num_bandits = len(bandits)
	tf.reset_default_graph()
	W = tf.Variable(tf.ones([num_bandits]))
	action = tf.argmax(W, 0)

	action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
	reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
	responsible_weight = tf.slice(W, action_holder, [1])
	loss = -tf.log(responsible_weight)*reward_holder

	trainer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
	updateModel = trainer.minimize(loss)

	init = tf.global_variables_initializer()
	e = 0.1
	num_episodes = 2000
	total_reward = np.zeros(num_bandits)
	with tf.Session() as sess:
		sess.run(init)
		for i in range(num_episodes):
			if np.random.rand(1) < e:
				a = np.random.randint(num_bandits)
			else:
				a = sess.run(action)
			r = pullBandit(bandits[a])
			total_reward[a] += r

			_, W1 = sess.run([updateModel, W], feed_dict={reward_holder:[r],
				action_holder:[a]})
			if i % 50 == 0:
				print "Running reward for the " + str(num_bandits) + " bandits: " + str(total_reward)
	print "The agent thinks bandit " + str(np.argmax(W1)+1) + " is the most promising...."

class ContextualBandit:
	def __init__(self):
		self.bandits = np.array([[0.2,0,-0.0,-5],[0.1,-5,1,0.25],[-5,5,5,5]])
		self.state = 0
		self.num_bandits = self.bandits.shape[0]
		self.num_actions = self.bandits.shape[1]

	def getBandit(self):
		self.state = np.random.randint(0, self.num_bandits)
		x = np.zeros((1,self.num_bandits))
		x[0, self.state] = 1.
		return x

	def pullArm(self, a):
		if np.random.randn(1) > self.bandits[self.state,a]:
			return 1
		return -1

def multibandit():
	cb = ContextualBandit()
	tf.reset_default_graph()
	states = tf.placeholder(shape=[1, cb.num_bandits], dtype=tf.float32)
	W = tf.Variable(tf.random_uniform([cb.num_bandits, cb.num_actions], 0, 0.02))
	Wout = tf.reshape(tf.matmul(states, W), [-1])
	action = tf.argmax(Wout, 0)

	action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
	reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
	responsible_weight = tf.slice(Wout, action_holder, [1])
	loss = -tf.log(responsible_weight)*reward_holder
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
	updateModel = optimizer.minimize(loss)

	init = tf.global_variables_initializer()
	e = 0.1
	num_episodes = 10000
	total_reward = np.zeros((cb.num_bandits, cb.num_actions))
	with tf.Session() as sess:
		sess.run(init)
		for i in range(num_episodes):
			s = cb.getBandit()
			if np.random.rand(1) < e:
				a = np.random.randint(cb.num_actions)
			else:
				a = sess.run(action, feed_dict={states:s})
			r = cb.pullArm(a)
			total_reward[np.where(s[0]==1.)[0],a] += r

			_, W1 = sess.run([updateModel, W], feed_dict={reward_holder:[r],
				action_holder:[a], states: s})
			if i % 50 == 0:
				e = 1./((i/50) + 10)
				print "Mean reward for each of the " + str(cb.num_bandits) + " bandits: " + str(np.mean(total_reward,axis=1))
	for a in range(cb.num_bandits):
	    print "The agent thinks action " + str(np.argmax(W1[a])+1) + " for bandit " + str(a+1) + " is the most promising...."
# marmband()
multibandit()