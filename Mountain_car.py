"""
Mountain_car with actor-critic
REFERENCES :
-https://www.youtube.com/watch?v=KHZVXao4qXs&index=7&list=PL7-jPKtc4r78-wCZcQn5IqyuWhBZ8fOxT) david silver lecture on actor critic
-actor critic framework presented here is inspired by morvanZhou github page
  https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/8_Actor_Critic_Advantage/AC_CartPole.py
-preprocessing the samples is by BAILOOL github page
  https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/Continuous%20MountainCar%20Actor%20Critic%20Solution.ipynb
"""
import numpy as np
import tensorflow as tf
import gym
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
import matplotlib.pyplot as plt


env = gym.envs.make("MountainCar-v0")
#some global variables
N_S = env.observation_space.shape[0]
N_A = env.action_space.n
EPISODES = 1000
TEST_ITER = 200
FEATURIZED_DIM = (1, 200)

# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
# basically were taking 10k samples ,compute their mean,compute their std(standard_deviation) and preform (samples - mean) / variance
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to convert a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
# RBFSampler is a transformer object which preforms the transformation y = exp(-gamma * x^2) the n_comp' arg is the number of monte_carlo samples?
featurizer = sklearn.pipeline.FeatureUnion([
    ("rbf1", RBFSampler(gamma=5.0, n_components=50)),
    ("rbf2", RBFSampler(gamma=2.0, n_components=50)),
    ("rbf3", RBFSampler(gamma=1.0, n_components=50)),
    ("rbf4", RBFSampler(gamma=0.5, n_components=50))
])
# transform preforms the scaler's duty, which is to preform (samples - mean) / std for each sample (the sample has 2 dim so we have mean\std for each dim)
featurizer.fit(scaler.transform(observation_examples))
# the fit method applies each transformer and combines their output


def process_state(state):
    scaled = scaler.transform([state])  # (state-mean)/std = scaled
    featurized = featurizer.transform(scaled)
    # preform the feature union to the state,each transformer goes over the state and preforms its own transformation
    return np.reshape(featurized[0], FEATURIZED_DIM)


class Actor:
    """
    Actor class,holds actor which determines actions upon a certain environment
    Public Methods:
    -learn(state, a, td_error) --> this method accept state,action and the td_error (reward + gamma*next_value - value) and preforms backprop
                                    to the NN in the direction of the gradient(gradient ascent to maximize the expeceted value).
    -choose_action(state) --> this method accepts the current state, and chooses an action to preform by passing the state as input to the NN
                              the output of the neural network is a probability distribution over the action space via a softmax activation function
    Private Methods:
    _build_actor() --> this method constructs the brain of the actor,
                        1.constructs the NN which is incharge to receive a state and outputs a probability distribution over the action space
                        2.constructs the expected value of our current policy(where policy is the NN ) multiply by the Advantage function
                          the advantage function is the td_error (from david silver course at youtube)
    """

    def __init__(self, sess, n_actions, n_states, input_dim, lr=0.01):
        self.lr = lr
        self.action_size = n_actions
        self.state_size = n_states
        self.sess = sess
        self.input_dim = input_dim
        self._build_actor()

    def _build_actor(self):
        self.s = tf.placeholder(dtype=tf.float32, shape=self.input_dim , name="state")
        self.a = tf.placeholder(dtype=tf.int32, shape=None, name="action")
        self.td_error = tf.placeholder(dtype=tf.float32, shape=None, name="td_error")
        l1 = tf.layers.dense(
                     inputs=self.s,
                     units=50,  # number of hidden units
                     activation=tf.nn.leaky_relu,
                     kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                     bias_initializer=tf.constant_initializer(0.1),  # biases
                     name='l1'
                     )
        self.acts_prob = tf.layers.dense(
                     inputs=l1,
                     units=self.action_size,  # output units
                     activation=tf.nn.softmax,  # get action probabilities
                     kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                     bias_initializer=tf.constant_initializer(0.1),  # biases
                     name='acts_prob'
                     )

        log_prob = tf.log(self.acts_prob[0, self.a])
        self.expected_value = tf.reduce_mean(log_prob * self.td_error)
        # equation derived from David silver course, the td error is the baseline reduction
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(-self.expected_value)
        # negative because we want gradient ascent

    def learn(self, state, a, td_error):
        feed_dict = {self.s: state, self.td_error: td_error, self.a: a}
        _, expected_value = self.sess.run([self.train_op, self.expected_value], feed_dict=feed_dict)
        return expected_value

    def choose_action(self, state):
        feed_dict = {self.s: state}
        action_distribution = self.sess.run(self.acts_prob, feed_dict=feed_dict)
        return np.random.choice(np.arange(self.action_size), p=action_distribution.ravel())


class Critic:
    """
    Critic class,holds actor which determines the value function for  <action,state>, the critic job is to observe the actor and
    "point" him in the right direction.
    Public Methods:
    -learn(state, reward, next_state) --> this method insert the state to the NN ,which outputs a value of the state, next it computes the next_Valu
                                          for the next_state, with both values the critic can compute the td_error which is the basis for the loss
                                          function that we want to minimize using backprop through the NN
    Private Methods:
    _build_actor() --> this method constructs the brain of the critic,
                        1.constructs the NN which is incharge to receive a state and outputs a value of the state V(s)
                        2.constructs the loss function that we want to minimize
    """

    def __init__(self, sess, n_actions, n_states, lr=0.01, gamma=0.99):
        self.lr = lr
        self.action_size = n_actions
        self.state_size = n_states
        self.sess = sess
        self.gamma = gamma
        self._build_critic()

    def _build_critic(self):
        self.s = tf.placeholder(dtype=tf.float32, shape=FEATURIZED_DIM, name="state")
        self.r = tf.placeholder(dtype=tf.float32, shape=None, name="reward")
        self.next_value = tf.placeholder(dtype=tf.float32, shape=None, name="next_value")
        critic_l1 = tf.layers.dense(inputs=self.s,
                                    units=50,
                                    activation=tf.nn.leaky_relu,
                                    kernel_initializer=tf.random_normal_initializer(0., .1),
                                    bias_initializer=tf.constant_initializer(0.1),
                                    name="critic_l1")
        self.value = tf.layers.dense(inputs=critic_l1,
                                     units=1,
                                     activation=None,
                                     kernel_initializer=tf.random_normal_initializer(0., .1),
                                     bias_initializer=tf.constant_initializer(0.1),
                                     name="value")

        self.td_error = self.r + self.gamma * self.next_value - self.value
        self.loss = tf.square(self.td_error)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def learn(self, s, r, s_):
        next_value = self.sess.run(self.value, feed_dict={self.s: s_})
        td_error, train_op = self.sess.run([self.td_error, self.train_op],
                                           feed_dict={self.r: r, self.next_value: next_value, self.s: s})
        return td_error


def main():
    global env
    sess = tf.Session()
    actor = Actor(sess, N_A, N_S, lr=0.0001, input_dim=FEATURIZED_DIM)
    critic = Critic(sess, N_A, N_S, lr=0.001)
    sess.run(tf.global_variables_initializer())  # important

    def train():
        print("begin training...")
        highest = -10  # init value
        train_data = np.zeros((EPISODES, 1))
        for e in range(EPISODES):
            best_per_episode = -10
            done = False
            sum_score = 0
            state = env.reset()
            hight = state[0]
            state = process_state(state)
            t = 0
            while not done:  # while not in terminal state
                action = actor.choose_action(state)
                next_state, reward, done, info = env.step(action)
                t += 1
                if best_per_episode < hight:
                    best_per_episode = hight
                if hight > 0.48 :  # encourage the agent to get to the flag
                   reward += (200 - t)
                hight = next_state[0]
                next_state = process_state(next_state)
                td_error = critic.learn(state, reward, next_state)
                actor.learn(state, action, td_error)
                state = next_state
                if done:
                    train_data[e] = t
                    if highest < best_per_episode:
                        highest = best_per_episode
                    print("episode {}/{} , highest_in_episode {}, best_location {}, time {}".format(e, EPISODES, best_per_episode, highest, t))
                    break
        return train_data
    train_data = train()

    def evaluate():
        print("begin evaluating..")
        highest = -10
        test_data = np.zeros((TEST_ITER, 1))
        for e in range(TEST_ITER):
            done = False
            t = 0
            state = env.reset()
            state = process_state(state)
            best_per_episode = -10
            while not done:  # while not in terminal
                env.render()
                action = actor.choose_action(state)
                next_state, reward, done, info = env.step(action)
                hight = next_state[0]
                next_state = process_state(next_state)
                t += 1
                state = next_state
                if best_per_episode < hight:
                    best_per_episode = hight
                if done:
                    test_data[e] = t
                    if highest < best_per_episode:
                        highest = best_per_episode
                    print("TEST: episode {}/{} , best_location {}, time {}".format(e, EPISODES, best_per_episode, highest, t))
        return test_data
    test_data = evaluate()
    #  Plotting the data
    fig = plt.figure(1)
    plt.subplot(211)
    plt.xlabel('epoches')
    plt.ylabel('scores')
    plt.title('Training phase')
    plt.plot(np.arange(EPISODES), train_data, 'r', label="train_data")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.subplot(212)
    plt.xlabel('epoches')
    plt.ylabel('scores')
    plt.title('Test phase')
    plt.plot(np.arange(TEST_ITER), test_data, label="test_data")
    plt.show()
    fig.savefig('Actor_critic_Mountain_Car_reward_modify_5.png')


    try:
        del env
    except ImportError:
        pass



if __name__ == '__main__':
    main()
