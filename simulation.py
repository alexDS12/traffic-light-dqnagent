import numpy as np
import traci

"""
Codes for light phases (actions)
1 => from TOP EAST to TOP WEST and to TOP and from BOTTOM WEST to BOTTOM EAST

2 => from BOTTOM to TOP WEST, to TOP and to BOTTOM EAST
"""
PHASES_DICT = {
    'PHASE_1_GREEN' : 0,
    'PHASE_1_YELLOW': 1,
    'PHASE_2_GREEN' : 2,
    'PHASE_2_YELLOW': 3
}

class Simulation(object):
    """
    This class handles things related to Deep Q-Learning and traci. Executing
    steps in simulation, getting env's state, rewarding agent and finally gathering
    statistics about agent's decision-making.
    
    Attributes
    ----------
    max_steps : int
        How long simulation will last.
    sumo : string
        SUMO config to establish TCP connection to traci.
    agent : DQNAgent
        Agent to be tested with its memory and NN.
    
    Methods
    -------
    get_state():
        Maps environment's states while simulation is running dividing lanes in
        10 groups * 6 -> number of lanes (60 states).
    get_waitinglanes_length():
        Gets all number of halt vehicles in all outcoming lanes.
    get_waiting_times():
        Sums all vehicles' waiting time in all outcoming lanes.
    execute_step(action):
        Executes x steps while gathering waiting lengths, waiting time, ...
    get_next_phase(phase_key):
        Gets next phase in dict for the traffic light (if it's the last phase, gets the first).
    set_light_phase(action):
        Gets the next light phase and sets in traci.
    run(epsilon):
        Runs an epoch for the simulation. Agent takes decisions, observes new state,
        saves data to memory and trains NN at the end of epoch.
    save_stats(total_reward):
        Saves relevant data about epoch that ran and later on used for plotting.
    """
    def __init__(self, agent, sumo, max_steps):
        self.max_steps = max_steps
        self.sumo = sumo
        self.agent = agent
        #storing data from every epoch
        self.total_rewards = []
        self.waiting_lengths = []
        self.total_waiting_time = []
    
    #Please refer to Andrea Vidali's thesis if any question about this method.
    def get_state(self):
        #mapping all vehicles into an array
        state = np.zeros(self.agent.nn.nb_states)
        #get all vehicles to define current state        
        for veh_id in traci.vehicle.getIDList():
            lane_id = traci.vehicle.getLaneID(veh_id)
            #get distance from traffic light
            veh_pos = traci.lane.getLength(lane_id) - traci.vehicle.getLanePosition(veh_id)            
            #get correct lane          
            if lane_id == 'BL_0' or lane_id == 'BL_1':
                lane = 0
            elif lane_id == 'TR_0':
                lane = 1
            elif lane_id == 'TR_1' or lane_id == 'TR_2':
                lane = 2
            elif lane_id == 'BC_0':
                lane = 3
            elif lane_id == 'BC_1':
                lane = 4
            elif lane_id == 'BC_2':
                lane = 5
            else:
                lane = -1 # discard if vehicle crossed light
            
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
        #sum_waiting_length = 0
        #for lane_id in enumerate(['BL', 'BC', 'TR']):
        #sum_waiting_length += traci.edge.getLastStepHaltingNumber(lane_id)
        nb_vwest = traci.edge.getLastStepHaltingNumber('BL')
        nb_vsouth = traci.edge.getLastStepHaltingNumber('BC')
        nb_vtopeast = traci.edge.getLastStepHaltingNumber('TR')
        return (nb_vwest + nb_vsouth + nb_vtopeast)
    
    def get_waiting_times(self):
        waiting_vwest = traci.edge.getWaitingTime('BL')
        waiting_vsouth = traci.edge.getWaitingTime('BC')
        waiting_vtopeast = traci.edge.getWaitingTime('TR')
        return (waiting_vwest + waiting_vsouth + waiting_vtopeast)
    
    def execute_step(self, step, action):
        steps_done = 0
        sum_waiting_length = 0
        sum_waiting_time = 0
        #check if simulation is not ending prematurely
        tlight_steps = int(traci.trafficlight.getPhaseDuration('center'))
        if (step + tlight_steps) >= self.max_steps:
            tlight_steps = self.max_steps - step
        
        #simulate one step at a time and gather info
        while tlight_steps > 0:
            traci.simulationStep()
            steps_done += 1
            #get length of vehicles waiting in all outcoming lanes
            sum_waiting_length += self.get_waitinglanes_length()
            #get waiting times in all outcoming lanes
            sum_waiting_time += self.get_waiting_times()
            tlight_steps -= 1
        
        next_state = self.get_state()
        return next_state, steps_done, [sum_waiting_length, sum_waiting_time]
           
    def get_next_phase(phase_key):
        phases_list = list(PHASES_DICT)
        try:
            next_key = phases_list[phases_list.index(phase_key) + 1]
        except IndexError: #if it's the last phase, next key is the first
            next_key = phases_list[0]
        return next_key
    
    def set_light_phase(self, action):
        next_action = self.get_next_phase(action)
        traci.trafficlight.setPhase('center', PHASES_DICT[next_action])
    
    def run(self, epsilon):
        traci.start(self.sumo)
        last_state = self.get_state()
        total_waiting_length = 0 #length of vehicles with speed <= 0.1
        total_waiting_time = 0 #sum of waiting time of all vehicles
        total_reward = 0 #total reward in each epoch
        total_waiting = 0
        
        for step in range(self.max_steps):            
            #choose action (light phase)
            action = self.agent.select_action(last_state, epsilon)
                        
            self.set_light_phase(action)
            #observation contains [sum_waiting_length, sum_waiting_time] after executing step
            next_state, steps_done, observation = self.execute_step(step, action)
            
            reward = total_waiting - observation[1]
            #save data in agent's memory
            self.agent.memory.add_sample((last_state, next_state, action, reward))
            
            last_state = next_state
            total_waiting = reward
            total_reward += reward
            total_waiting_length += observation[0]
            total_waiting_time += observation[1]
            #need to add traffic light duration to current step
            step += steps_done
        
        print('Total reward: {}'.format(total_reward))
        #save stats for later plotting
        self.save_stats(total_reward, total_waiting_length, total_waiting_time)   
        traci.close()
        
        #train NN only in between epochs
        self.agent.experience_replay()
        
    def save_stats(self, total_reward, total_waiting_length, total_waiting_time):
        self.total_rewards.append(total_reward)
        self.waiting_lengths.append(total_waiting_length)
        self.total_waiting_time.append(total_waiting_time)
    
    def get_stats(self):        
        return {
                'Rewards' : self.total_rewards,
                'Waiting Lengths' : self.waiting_lengths,
                'Mean Waiting Time' : (self.total_waiting_time / self.max_steps)
                }