import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def simple_table():
	env = gym.make('FrozenLake-v0')

	Q = np.zeros([env.observation_space.n, env.action_space.n])
	lr = 0.85
	y = 0.99
	num_episodes = 2000
	rList = []
	for i in range(num_episodes):
		s = env.reset()
		rAll = 0
		d = False
		j = 0
		while j < 99:
			j += 1
			# env.render()
			a = np.argmax(Q[s,:]+np.random.randn(1, env.action_space.n)/(i+1.))
			s1, r, d, _ = env.step(a)
			Q[s,a] = Q[s,a] + lr*(r + y*max(Q[s1,:])-Q[s,a])
			rAll += r
			s = s1
			if d == True:
				break
		rList.append(rAll)

	print "Score over time: " +  str(sum(rList)/num_episodes)
	print "Final Q-Table Values"
	print Q

def simple_network():
	env = gym.make('FrozenLake-v0')

	##networks
	tf.reset_default_graph()
	states = tf.placeholder(shape=[1,16], dtype=tf.float32)
	W = tf.Variable(tf.random_uniform([16,4],0, 0.02))
	Qout = tf.matmul(states, W)
	predict = tf.argmax(Qout, 1)

	nextQ = tf.placeholder(shape=[1,4], dtype=tf.float32)
	loss = tf.reduce_sum(tf.square(nextQ - Qout))
	trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
	updateModel = trainer.minimize(loss)

	init = tf.global_variables_initializer()
	y = 0.99
	e = 0.1
	num_episodes = 2000
	rList = []
	with tf.Session() as sess:
		sess.run(init)
		for i in range(num_episodes):
			s = env.reset()
			rAll = 0
			d = False
			j = 0
			while j < 99:
				j += 1
				a, allQ = sess.run([predict, Qout], feed_dict={states:np.identity(16)[s:s+1]})
				if np.random.rand(1) < e:
					a[0] = env.action_space.sample()
				s1, r, d, _ = env.step(a[0])
				Q1 = sess.run(Qout, feed_dict={states:np.identity(16)[s1:s1+1]})
				maxQ1 = np.max(Q1)
				targetQ = allQ
				targetQ[0, a[0]] = r + y*maxQ1

				_, W1 = sess.run([updateModel, W], feed_dict={states:np.identity(16)[s:s+1],
					nextQ:targetQ})
				rAll += r
				s = s1
				if d == True:
					e = 1./((i/50) + 10)
					break
			rList.append(rAll)
	print "Score over time: " +  str(sum(rList)/num_episodes)
	plt.plot(rList)

simple_network()