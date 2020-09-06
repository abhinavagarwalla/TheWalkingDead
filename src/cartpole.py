import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import cPickle as pickle
import gym

env = gym.make('CartPole-v0')
env.reset()
print env.observation_space
print env.action_space.n

H = 10	#number of hidden units
batch_size = 10 #Update network after #episodes
lr = 1e-2
gamma = 0.99	#discount factor
D = 4	#state space

def discount_rewards(r):
	""" take 1D float array of rewards and compute discounted reward """
	discounted_r = np.array([val * (gamma ** i) for i, val in enumerate(r)])
	discounted_r -= np.mean(discounted_r)
	discounted_r /= np.std(discounted_r)
	return discounted_r

def learn_agent():
	tf.reset_default_graph()

	obs = tf.placeholder(shape=[None, D], dtype=tf.float32, name="state")
	W1 = tf.get_variable("W1", shape=[D, H], initializer=tf.contrib.layers.xavier_initializer())
	l1 = tf.nn.relu(tf.matmul(obs, W1))
	W2 = tf.get_variable("W2", shape=[H, 1], initializer=tf.contrib.layers.xavier_initializer())
	l2 = tf.matmul(l1, W2)
	probs = tf.nn.sigmoid(l2)

	tvars = tf.trainable_variables()
	input_y = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='targets')
	advantages = tf.placeholder(dtype=tf.float32, name='rewards')

	loglik = tf.log(input_y*(probs) + (1-input_y)*(1-probs))
	loss = -tf.reduce_mean(loglik*advantages)
	newGrads = tf.gradients(loss, tvars)

	adam = tf.train.AdamOptimizer(learning_rate=lr)
	W1Grad = tf.placeholder(dtype=tf.float32, name="batch_grad1")
	W2Grad = tf.placeholder(dtype=tf.float32, name="batch_grad2")
	batchGrad = [W1Grad, W2Grad]
	updateGrads = adam.apply_gradients(zip(batchGrad, tvars))

	xs, drs, ys = [], [], []
	running_reward = None
	reward_sum = 0
	episode_number = 1
	total_episodes = 10000
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		rendering = False
		sess.run(init)
		observation = env.reset()
		gradBuffer = np.array([np.zeros(var.get_shape()) for var in tvars])

		while episode_number <= total_episodes:
			# if reward_sum/batch_size > 100 or rendering==True:
			# 	env.render()
			# 	rendering = True

			x = np.reshape(observation, [1,D])
			tfprob = sess.run(probs, feed_dict={obs:x})
			action = 1 if tfprob > np.random.uniform() else 0

			y = 1 if action==0 else 0
			observation, reward, done, info = env.step(action)
			reward_sum += reward

			xs.append(x)
			ys.append(y)
			drs.append(reward)

			if done:
				episode_number += 1
				epx = np.vstack(xs)
				epy = np.vstack(ys)
				epr = np.vstack(drs)
				xs, drs, ys = [], [], []
				discounted_epr = discount_rewards(epr)

				gradBuffer += sess.run(newGrads, feed_dict={advantages:discounted_epr,
					input_y:epy, obs:epx})

				if episode_number%batch_size==0:
					sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0],
						W2Grad: gradBuffer[1]})
					
					gradBuffer *= 0

					print("Average reward for episode {}: {}".format(episode_number, reward_sum/batch_size))
					# running_reward = reward_sum if running_reward is None \
					# else running_reward*0.99 + 0.01*reward_sum

					if reward_sum/batch_size >= 200.:
						print "Task solved in ", episode_number, " episodes!~"
						break
					reward_sum = 0
				observation = env.reset()
		print episode_number,'Episodes completed.'

learn_agent()