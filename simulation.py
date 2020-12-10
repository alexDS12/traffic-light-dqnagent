import numpy as np
import traci

"""
Codes for light phases (actions)
0 => from TOP EAST to TOP WEST and to TOP and from BOTTOM WEST to BOTTOM EAST

1 => from BOTTOM to TOP WEST, to TOP and to BOTTOM EAST
"""
PHASES_DICT = {
    'PHASE_1_GREEN' : 0,
    'PHASE_1_YELLOW': 1,
    'PHASE_2_GREEN' : 2,
    'PHASE_2_YELLOW': 3
}
#roads before reaching traffic light => Bottom Left, Top Right, Bottom Center
ROADS = ['BL', 'TR', 'BC']
#lanes within each road
LANES = [['BL_0', 'BL_1'], 'TR_0', ['TR_1', 'TR_2'], 'BC_0', 'BC_1', 'BC_2']

class Simulation(object):
    def __init__(self, agent, sumo, max_steps, is_training=None):
        self.agent = agent
        self.sumo = sumo
        self.max_steps = max_steps
        self.is_training = is_training
        #storing data from every epoch
        self.total_rewards = []
        self.waiting_lengths = []
        self.total_waiting_times = []

    #Refer to Andrea Vidali's thesis if any question about this method
    def get_state(self):
        #mapping all vehicles into an array
        state = np.zeros(self.agent.nb_states)
        #get all vehicles to define current state        
        for veh_id in traci.vehicle.getIDList():
            lane_id = traci.vehicle.getLaneID(veh_id)
            #get distance from traffic light
            veh_pos = traci.lane.getLength(lane_id) - traci.vehicle.getLanePosition(veh_id)    

            #get correct lane
            lane = -1 # discard if vehicle crossed light
            for i, l in enumerate(LANES):
              if lane_id in l:
                lane = i
            
            """
            Dividing lane in 10 groups where each vehicle = 7 (size 5 + gap 2)
            1st group  => 1 vehicle
            2nd group  => 1 vehicle
            3rd group  => 2 vehicles
            4th group  => 2 vehicles
            ...        => ...
            10th group => 5 vehicles
            """
            for index, lane_group in enumerate([7, 14, 28, 42, 63, 84, 112, 140, 175, 211]):
                #mapping using distance's index, if vehicle is in 1st group (7 meters from traffic light),
                #it's lane group will be 0 and so on.
                if veh_pos < lane_group:
                    veh_lane_group = index
            
            #add only vehicles that did not cross the light
            if lane >= 0:
                #since state is size N, we create the valid vehicle's position as a number between 1-N
                #e.g. if current vehicle's lane is 1 and lane_group is 5,
                #his index inside state array is gonna be 15.
                veh_pos = str(lane) + str(veh_lane_group) 
                state[int(veh_pos)] = 1
        return state

    def get_waitinglanes_length(self):
        waiting_lengths = 0
        for road in ROADS:
            waiting_lengths += traci.edge.getLastStepHaltingNumber(road)
        return waiting_lengths

    def get_waiting_times(self):
        waiting_times = 0
        for road in ROADS:
            waiting_times += traci.edge.getWaitingTime(road)
        return waiting_times

    def execute_step(self, action, step):
        self.set_light_phase(action)
        steps_done = 0
        sum_waiting_time = 0
        sum_waiting_length = 0
        #check if simulation is not ending prematurely
        tlight_steps = int(traci.trafficlight.getPhaseDuration('center'))
        if (step + tlight_steps) >= self.max_steps:
            tlight_steps = self.max_steps - step

        #simulate one step at a time and gather info
        while tlight_steps > 0:
            traci.simulationStep()   
            #get waiting times in all outcoming lanes
            sum_waiting_time += self.get_waiting_times()
            sum_waiting_length += self.get_waitinglanes_length()
            steps_done += 1
            tlight_steps -= 1
        return self.get_state(), steps_done, self.get_waiting_times(), [sum_waiting_time, sum_waiting_length]

    def get_next_phase(self, action):
        for k, v in PHASES_DICT.items():
          if v == action:
            key = k
        phases_list = list(PHASES_DICT)
        try:
          next_key = phases_list[phases_list.index(key) + 1]
        except IndexError: #if it's the last phase, next key is the first
          next_key = phases_list[0]
        return PHASES_DICT[next_key]

    def set_light_phase(self, action):
        next_action = self.get_next_phase(action)
        traci.trafficlight.setPhase('center', next_action)

    def run(self, epsilon):
        traci.start(self.sumo)

        total_reward = 0
        total_waiting_time = 0
        total_waiting_length = 0
        step = 0
        last_state = self.get_state()
        last_total_waiting = 0

        while step < self.max_steps:
            action = self.agent.select_action(last_state, epsilon)
            
            next_state, steps_done, total_waiting, observation = self.execute_step(action, step)

            reward = last_total_waiting - total_waiting

            #save data in agent's memory only if training
            if not self.is_training == None:            
                self.agent.memory.add_sample((last_state, next_state, action, reward))

            step += steps_done
            total_waiting_time += observation[0]
            total_waiting_length += observation[1]
            last_state = next_state
            last_total_waiting = total_waiting

            if reward < 0:
              total_reward += reward

        print('Total reward: {}'.format(total_reward))
        #save stats for later plotting
        self.save_stats(total_reward, total_waiting_time, total_waiting_length)
        traci.close()

        #train NN only in between epochs only if training
        if not self.is_training == None:        
            self.agent.experience_replay()

    def save_stats(self, total_reward, total_waiting_time, total_waiting_length):
        self.total_rewards.append(total_reward)
        self.waiting_lengths.append(total_waiting_length)
        self.total_waiting_times.append(total_waiting_time)

    def get_stats(self):        
        return {
                'Reward' : self.total_rewards,
                'Mean Waiting Length (m)' : np.divide(self.waiting_lengths, self.max_steps),
                'Mean Waiting Time (s)' : np.divide(self.total_waiting_times, self.max_steps)
                }