from datetime import datetime
import random
import numpy as np
from collections import deque
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model, load_model
import environment_NoEarlyTerm as environment
import pandas as pd
import matplotlib.pyplot as plt
import time
import math


def OurModel(input_shape, action_space):
    X_input = Input(input_shape)
    X = Dense(60, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform', use_bias=True)(X_input)
    #X = BatchNormalization()(X)
    X = Dense(60, activation="relu", kernel_initializer='he_uniform', use_bias=True)(X)
    #X = BatchNormalization()(X)
    #X = Dense(60, activation="relu", kernel_initializer='he_uniform', use_bias=True)(X)
    #X = BatchNormalization()(X)
    # X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs = X_input, outputs = X, name='DQN_model')
    model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    model.summary()
    return model


class DQNAgent:
    def __init__(self):
        self.env = environment.VehicleRouterEnvironment()
        self.state_size = len(self.env.state())
        self.action_size = len(self.env.action_space)
        self.EPISODES = 6000
        self.memory = deque(maxlen=10000)
        
        self.gamma = 0.99   # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.9999
        self.batch_size = 512
        self.train_start = 1000 # how many items in memory before training
        self.eps_decay_start = 900 # how many episodes before beginning epsilon decay

        self.itercount = 0
        # create main model
        self.model = OurModel(input_shape=(self.state_size,), action_space = self.action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min and self.eps_decay_start < self.itercount:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        if len(state.shape) == 1: # catch for dimensionality issues
            state = np.array(state).reshape([1, len(state)])

        if np.random.uniform(0,1) < self.epsilon:
            return np.random.choice(self.env.action_space)
        else:
            q_out = self.model.predict(state)
            action = np.argmax(q_out)
            return action

    def replay(self):
        if len(self.memory) < self.train_start:
            return

        # Randomly sample minibatch from the memory
        t0 = time.time()
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        #print('1: ', time.time() - t0)

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []


        # assign data into state, next_state, action, reward and done from minibatch
        t0 = time.time()
        for i in range(self.batch_size):
            ministate, miniaction, minireward, mininextstate, minidone = minibatch[i]
            state[i] = ministate.reshape([1, self.state_size])
            next_state[i] = mininextstate
            action.append(self.act(state[i]))
            reward.append(minireward)
            done.append(minidone)
        action = np.array(action)
        reward = np.array(reward)
        #print('2: ', time.time() - t0)

        #print('action / state / next state / reward')
        #print(action[0])
        #print(state[0])
        #print(next_state[0])
        #print(reward[0])
        #print('')

        t0 = time.time()
        flat_target = self.model.predict(state).flatten()
        Q_target_next = np.max(self.model.predict(next_state), axis=1)
        action_indices = action + np.arange(self.batch_size) * self.action_size # provides index in flattened array of where this action maps to
        flat_target[action_indices] = reward + self.gamma * Q_target_next.flatten()
        target = flat_target.reshape([self.batch_size, self.action_size]) # pull back into shape
        #print('3: ', time.time() - t0)

        #target = np.zeros((self.batch_size, self.action_size))

        ## compute value function of current(call it target) and value function of next state(call it target_next)
        #for i in range(self.batch_size):
        #    a = action[i]
        #    if done[i]:
        #        target[i] = self.model.predict(state[i].reshape([1, len(state[i])]))
        #        target[i, a] = reward[i]
        #    else:
        #        Q_target_next = np.max(self.model.predict(next_state[i].reshape([1, len(next_state[i])])))
        #        target[i] = self.model.predict(state[i].reshape([1, len(state[i])]))
        #        target[i, a] = reward[i] + self.gamma * Q_target_next

        t0 = time.time()
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0, epochs=1)
        #print('4: ', time.time() - t0)



    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)
            
    def training(self):
        scores = []
        stops = []

        for e in range(self.EPISODES):
            self.itercount += 1
            self.env.reset(e)
            state = self.env.state()

            done = False
            i = 0

            actions = []
            max_iterations = 100
            total_reward = 0
            while not done and i < max_iterations:
                t0 = time.time()

                action = self.act(state)
                #print('1: ', time.time()-t0)
                actions.append(action)
                t0 = time.time()
                next_state, reward, done, _ = self.env.step(action)
                #print('2: ', time.time()-t0)
                total_reward += reward
                t0 = time.time()
                self.remember(state, action, reward, next_state, done)
                #print('3: ', time.time()-t0)

                state = next_state
                i += 1
                if done:
                    # dateTimeObj = datetime.now()
                    # timestampStr = dateTimeObj.strftime("%H:%M:%S")
                    time_taken, n_serviced = self.env.get_stats()
                    print("episode: {}/{}, stops_made: {}, epsilon: {:.2}, r: {}, t: {}, serviced: {}".format(e+1, self.EPISODES, i, self.epsilon, total_reward, time_taken, n_serviced))
                    scores.append(total_reward)
                    stops.append(i)
                    #print('--> actions {}'.format(actions))

                #if math.log(i, 10) % 1 > math.log(i+1, 10) % 1: # only replay when the num episodes has crossed an order of magnitude
                    # this speeds things up
                    # self.replay()

                t0 = time.time()
                self.replay()
                #print('4: ', time.time() - t0)

        return scores, stops


if __name__ == "__main__":
    agent = DQNAgent()
    scores, stops = agent.training()

    episodes = np.arange(len(scores))
    plt.plot(episodes, scores)
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.show()

    plt.plot(episodes, stops)
    plt.xlabel('Episodes')
    plt.ylabel('Stops')

    plt.savefig('NoEarlyTerm.png')
    plt.show()