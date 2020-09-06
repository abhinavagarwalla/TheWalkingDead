import numpy as np
import tensorflow as tf
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
mH = 256

def discount_rewards(r):
	""" take 1D float array of rewards and compute discounted reward """
	discounted_r = np.array([val * (gamma ** i) for i, val in enumerate(r)])
	discounted_r -= np.mean(discounted_r)
	discounted_r /= np.std(discounted_r)
	return discounted_r

def stepModel(sess, xs, action):
	feed = np.reshape(np.hstack([xs[-1][0], np.array(action)]), [1,5])
	predS = sess.run(predS, feed_dict={previous_state:feed})
	reward = predS[:,4]
	state = predS[:, 0:4]
	state[:,0] = np.clip(state[:,0], -2.4, +2.4)
	state[:,2] = np.clip(state[:,2], -0.4, +0.4)
	doneP = np.clip(state[:,5], 0, 1)
	if doneP > 0.1 or len(xs)>= 300:
		done = True
	else:
		done = False
	return state, reward, done

def learn_agent():
	tf.reset_default_graph()

	## Policy Network
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


	## World Model Network
	# input_data = tf.placeholder(shape=[None, 5], dtype=tf.float32)
	# with tf.variable_scope('rnnlm'):
	# 	softmax_w = tf.get_variable(shape=[mH, 50], name="softmax_w")
	# 	softmax_b = tf.get_variable(shape=[50], name="softmax_b")

	previous_state = tf.placeholder(shape=[None, 5], dtype=tf.float32, name="previous_state")
	W1M = tf.get_variable(shape=[5,mH], name="W1M", initializer=tf.contrib.layers.xavier_initializer())
	B1M = tf.get_variable(tf.zeros([mH]), name="B1M")
	layer1M = tf.nn.relu(tf.matmul(previous_state,W1M) + B1M)

	W2M = tf.get_variable(shape=[mH, mH], name="W2M", initializer=tf.contrib.layers.xavier_initializer())
	B2M = tf.get_variable(tf.zeros([mH]), name="B2M")
	layer2M = tf.nn.relu(tf.matmul(layers1M,W2M) + B2M)

	wO = tf.get_variable(shape=[mH, 4], name="wO", initializer=tf.contrib.xavier_initializer())
	wR = tf.get_variable(shape=[mH, 1], name="wR", initializer=tf.contrib.xavier_initializer())
	wD = tf.get_variable(shape=[mH, 1], name="wD", initializer=tf.contrib.xavier_initializer())

	bO = tf.get_variable(tf.zeros([4]), name="bO")
	bR = tf.get_variable(tf.zeros([1]), name="bR")
	bD = tf.get_variable(tf.zeros([1]), name="bD")

	predO = tf.matmul(layer2M, wO, name="predO")+bO
	predR = tf.matmul(layer2M, wR, name="predO")+bR
	predD = tf.nn.sigmoid(tf.matmul(layer2M, wD, name="predO")+bD)

	trueO = tf.placeholder(shape=[None, 4], dtype=tf.float32, name="trueO")
	trueR = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="trueR")
	trueD = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="trueD")

	predS = tf.concat(1, [predO, predR, predD])
	lossO = tf.square(trueO - predO)
	lossR = tf.square(trueR - predR)
	lossD = -tf.log(tf.mul(trueD, predD) + tf.mul(1-trueD, 1-predD))
	model_loss = tf.reduce_mean(lossO + lossR + lossD)

	optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(model_loss)


	xs, drs, ys, ds = [], [], [], []
	running_reward = None
	reward_sum = 0
	episode_number = 1
	real_episodes = 1
	total_episodes = 10000
	
	drawFromModel = False
	trainModel = True
	trainPolicy = False
	switch_point = 1
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		rendering = False
		sess.run(init)
		observation = env.reset()
		gradBuffer = np.array([np.zeros(var.get_shape()) for var in tvars])

		while episode_number <= total_episodes:
			# if reward_sum/batch_size > 150 or rendering==True:
			# 	env.render()
			# 	rendering = True

			x = np.reshape(observation, [1,D])
			tfprob = sess.run(probs, feed_dict={obs:x})
			action = 1 if tfprob > np.random.uniform() else 0

			y = 1 if action==0 else 0
			xs.append(x)
			ys.append(y)
			
			if drawFromModel:
				observation, reward, done = stepModel(sess, xs, action)
			else:
				observation, reward, done, info = env.step(action)
			reward_sum += reward
			ds.append(done)
			drs.append(reward)

			if done:
				if drawFromModel == False: 
					real_episodes += 1
				episode_number += 1

				epx = np.vstack(xs)
				epy = np.vstack(ys)
				epr = np.vstack(drs)
				epd = np.vstack(ds)
				xs, drs, ys, ds = [], [], [], []

				if trainModel:
					sess.run([optimizer], feed_dict={previous_state:, trueO:, trueR:, trueD:})
				if trainPolicy:
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
