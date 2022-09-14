import os
import pandas as pd
import numpy as np


class VehicleRouterEnvironment:

    def __init__(self, path='./data/'):
        self.path = path
        files = os.listdir(self.path)
        alldata = pd.DataFrame(index=None, columns=range(6))
        for filename in files:
            with open(os.path.join(self.path, filename)) as f:
                data = f.readlines()
                numdata = data[9:]
                file_data = pd.DataFrame(index=[int(line.split()[0]) for line in numdata],
                                         columns=range(6),
                                         data=[line.split()[1:] for line in numdata])  # start time at 0
                alldata = pd.concat([alldata, file_data], axis=0)

        alldata = alldata.astype("float")
        dist_scale = alldata.loc[:, [0, 1]].max().max() / 10
        time_scale = alldata.loc[:, [3, 4]].max().max() / 10
        self._timescale = time_scale
        self._distscale = dist_scale
        self._speedscaler = 0.5 # bigger = vehicle moves slower

        self.scaler = np.array([dist_scale, dist_scale, 1, time_scale, time_scale])
        self._setup(problem_file_index=0, scaler=self.scaler)

    def _setup(self, problem_file_index=0, scaler=None):

        problem_file_index = 0
        if scaler is None:
            scaler = np.array([1, 1, 1, 1, 1])

        files = os.listdir(self.path)
        self.problem_file = files[problem_file_index % len(files)]

        # read the txt file and load data into a dataframe
        with open(os.path.join(self.path, self.problem_file)) as datafile:
            data = datafile.readlines()
            headers = data[7]
            headers = headers.replace("CUST NO.", "CUST_NO.")
            headers = headers.replace("READY TIME", "READY_TIME")
            headers = headers.replace("DUE DATE", "DUE_DATE")
            numdata = data[9:]
            self.data = pd.DataFrame(index=[int(line.split()[0]) for line in numdata],
                                     columns=headers.split()[1:-1],
                                     data=[line.split()[1:] for line in numdata])  # start time at 0

            self.data = self.data.sort_values('READY_TIME')
            #self.data = self.data.sample(10).dropna() # reduce the size of the problem to speed up
            self.data = self.data.dropna().iloc[::10]
            self.data = self.data.sample(frac=1) # shuffle order
            self.data.index = range(len(self.data))

            self.data = self.data.astype("float")
            self.service_time = self.data["SERVICE"][5]/self._timescale # 5 is random, 0 is always depot
            self.data.drop('SERVICE', axis=1, inplace=True)
            self.data = self.data.divide(scaler)
            self.data['COMPLETED?'] = 0  # keep track of who has been serviced
            self.data.loc[0, 'COMPLETED?'] = 1  # by default we say the depot has been serviced

        #self.vehiclenumber = 25 # this many vehicles at play

        # define the state space and action space
        # state is the self.data dataframe + current time + current X coord + current Y coord
        self.time = self.data['READY_TIME'].min()  # start clock at 0
        self.max_time = self.data["DUE_DATE"].max() * 2

        self.XCOORD = self.data["XCOORD."][0] # depot is different at each file
        self.YCOORD = self.data["YCOORD."][0]

        self.action_space = self.data.index.tolist()  # action space is the customer numbers (ie: which customer to visit next)

        self._max_episode_steps = 1  # ? figure this is useful + is also in the framework for assignment 4

    def reset(self, episode_num):
        # refreshes the system to use a new problem file
        self._setup(problem_file_index=episode_num, scaler=self.scaler)
        return None

    def state(self):
        # state is 1) dist to customer 0, 2) customer times minus current time
        state = self.data.copy()
        state.drop('XCOORD.', axis=1, inplace=True)
        state.rename(columns={"YCOORD.": "DISTANCE"}, inplace=True)
        state.loc[:, ["READY_TIME", "DUE_DATE"]] = state.loc[:, ["READY_TIME", "DUE_DATE"]] - self.time
        state.loc[:, ["DISTANCE"]] = np.sqrt((self.data.loc[:, "XCOORD."] - self.XCOORD).pow(2) + \
                                             (self.data.loc[:, "YCOORD."] - self.YCOORD).pow(2))
        return state.values.flatten()

    def step(self, action):
        # returns next_state, reward, done, _

        # update the self.data (the state)
        prior_time, prior_Xcoord, prior_Ycoord = self.time, self.XCOORD, self.YCOORD
        next_Xcoord, next_Ycoord = self.data.loc[action, ['XCOORD.', 'YCOORD.']]  # next coords are defined by the next customer
        move_time = np.sqrt((prior_Xcoord - next_Xcoord) ** 2 + (prior_Ycoord - next_Ycoord) ** 2)
        move_time = move_time * self._speedscaler * self._distscale / self._timescale
        previously_completed = int(self.data.loc[action, 'COMPLETED?'])
        ready_time = self.data.loc[action, 'READY_TIME']
        due_date = self.data.loc[action, 'DUE_DATE']

        if move_time == 0.:
            move_time = 1 / self._timescale

        # only include service time if we haven't already serviced this customer and if we visited after the ready time

        #print(action, self.time, ready_time, prior_time + move_time + self.service_time)
        #print(self.data)
        #print('--')

        if previously_completed == 0:
            next_time = max(prior_time + move_time + self.service_time, ready_time)
            self.data.loc[action, "COMPLETED?"] = 1
        else:
        #    print('-100 1')
            return self.state(), -1, True, None
        #print(self.data)

        # # work out rewards
        #reward = + int((previously_completed == 0) & (prior_time + move_time >= ready_time) * (next_time < due_date)) \
        #         + int((previously_completed == 0) & (prior_time + move_time >= ready_time) ) \
        #         - np.sum((self.data.loc[:, "COMPLETED?"] == 0) & (self.data.loc[:, "READY_TIME"] <= next_time) & (self.data.loc[:, "READY_TIME"] >= prior_time)) \
        #        - np.sum((self.data.loc[:, "COMPLETED?"] == 0) & (self.data.loc[:, "DUE_DATE"] <= next_time) & (self.data.loc[:, "DUE_DATE"] >= prior_time))

        if np.sum((self.data.loc[:, "COMPLETED?"] == 0) & (self.data.loc[:, "READY_TIME"] <= next_time) & (self.data.loc[:, "READY_TIME"] >= prior_time)) > 0:
            #print('-100 2')
            return self.state(), -1, True, None

        #print('reward: ', 1)
        #print('')
        #print('')

        reward = 10

        reward = float(reward)#/100 # normalise reward
        self.time = next_time # move time forward
        self.XCOORD = next_Xcoord
        self.YCOORD = next_Ycoord

        ## now fast forward time to when the next action is required
        #next_readytime = self.data.loc[self.data.loc[:, "COMPLETED?"]==0, "READY_TIME"].min()
        ##print('next readytime: ', next_readytime)
        #if not np.isnan(next_readytime):
        #    next_cust = self.data.loc[(self.data.loc[:, "READY_TIME"] == next_readytime)&(self.data.loc[:, "COMPLETED?"]==0)].iloc[0]
        #    next_dist = np.sqrt((self.XCOORD - next_cust["XCOORD."]) ** 2 + (self.YCOORD - next_cust["YCOORD."]) ** 2)
        #    next_move_time = next_dist * self._speedscaler * self._distscale / self._timescale
        #    #print('--> ', self.time, next_move_time, next_readytime, self.time + next_move_time)
        #    if self.time + next_move_time < next_readytime:
        #        self.time = next_readytime - next_move_time # fast forward the clock if there's null time in the system

        # now set the newly serviced customer's COMPLETED? to 1 (True)
        done = 0 not in self.data.loc[:, 'COMPLETED?'].tolist()  # done is True if everyone has been serviced

        if next_time > self.max_time:   # done also True if max time reached
            done = True if next_time > self.max_time else done # done also True if max time reached

        # print('new state following action: \n{}'.format(self.data))
        # print('action: {}'.format(action))
        # print('time: {}'.format(self.time))
        # print('reward: {}'.format(reward))
        # print('done? {}'.format(done))

        return self.state(), reward, done, None

if __name__ == "__main__":
    trial = VehicleRouterEnvironment()


