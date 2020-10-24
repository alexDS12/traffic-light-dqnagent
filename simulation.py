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
        Gets environment's states while simulation is running.
    execute_step(action):
    
    get_next_phase(phase_key):
        Gets next phase in dict for the traffic light
        (if it's the last phase, gets the first).
    set_light_phase(action):
        Gets the next light phase and sets on traci.
    run(epsilon):
        Runs an epoch for the simulation. Agent takes decisions, saves to memory
        and trains NN at the end of epoch.
    save_stats(total_reward):
        Saves relevant data about epoch that ran and later on used for plotting.
    """
    def __init__(self, agent, sumo, max_steps):
        self.max_steps = max_steps
        self.sumo = sumo
        self.agent = agent
        #storing data from every epoch
        self.total_rewards = []
        self.mean_waiting_time = []
        self.total_vehicles = []
    
    #Please refer to Andrea Vidali's thesis if any question about this method.
    def get_state(self):
        #mapping all vehicles into an array
        state = np.zeros(self.agent.nn.nb_states)
        #get all vehicles to define current state        
        for veh_id in traci.vehicle.getIDList():
            lane_id = traci.vehicle.getLaneID(veh_id)
            #get distance from traffic light
            veh_pos = traci.lane.getLength(lane_id) - traci.vehicle.getLanePosition(veh_id)
            
            """
            Dividing lane in 5 groups where each vehicle = 7 (size 5 + gap 2)
            1st group => 1 vehicle
            2nd group => 2 vehicles
            3rd group => 3 vehicles
            4th group => 4 vehicles
            5th group => 5 vehicles
            """
            for index, lane_group in enumerate([7, 14, 21, 28, 36]):
                if veh_pos < lane_group:
                    veh_lane_group = index
            
            #get which lane within group            
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
                lane = -1
                
            if 1 <= lane <= 5:
                #since state is size N, we create the valid vehicle's position as a number between 1-N
                #e.g. if vehicle is in lane_group 5 and lane 1 within that specific lane,
                #his index inside state array is gonna be 51.
                veh_pos = str(veh_lane_group) + str(lane)
                state[int(veh_pos)] = 1
                
        return state
    
    def execute_step(self, action):
        pass
           
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
        #last_total_waiting_vehicle = 0
        #waiting_times = {} 
        #waiting_time_vehicle = 0 #sum of waiting time in line
        total_reward = 0 #total reward in each epoch
        #total_vehicles = 0 #sum of vehicles generated
        
        for step in range(self.max_steps):            
            #choose action (light phase)
            action = self.agent.select_action(last_state, epsilon)
                        
            self.set_light_phase(action)
            next_state, reward = self.execute_step(action)
            
            #save data in agent's memory
            self.agent.memory.add_sample((last_state, next_state, action, reward))
            
            last_state = next_state
            total_reward += reward
            
        
        print('Total reward: {}'.format(total_reward))
        #save stats for later plotting
        self.save_stats(total_reward)        
        traci.close()  
        
        #train NN only in between epochs
        self.agent.experience_replay()
        
    def save_stats(self, total_reward):
        self.total_rewards.append(total_reward)
        self.mean_waiting_time.append(1/ self.max_steps)