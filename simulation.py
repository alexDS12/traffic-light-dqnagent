import numpy as np
import traci

"""
Codes for light phases (actions)
1 => from TOP EAST to TOP WEST and to TOP and from BOTTOM to BOTTOM EAST

2 => from BOTTOM to TOP WEST, to TOP and to BOTTOM EAST
 
3 => from BOTTOM WEST to BOTTOM EAST and from TOP EAST to TOP WEST and TOP
"""

PHASES_DICT = {
    'PHASE_1_GREEN' : 0,
    'PHASE_1_YELLOW': 1,
    'PHASE_2_GREEN' : 2,
    'PHASE_2_YELLOW': 3,
    'PHASE_3_GREEN' : 4,
    'PHASE_3_YELLOW': 5
}

class Simulation(object):
    def __init__(self, agent, sumo, max_steps):
        self.max_steps = max_steps
        self.sumo = sumo
        self.agent = agent
        #storing data from every epoch
        self.total_rewards = []
        self.mean_waiting_time = []
        self.total_vehicles = []
        
    def get_state(self):
        #mapping all vehicles into an array
        state = np.zeros(self.agent.nn.nb_states)
        #get all vehicles to define current state        
        for veh_id in traci.vehicle.getIDList():
            lane_id = traci.vehicle.getLaneID(veh_id)
            #get distance from traffic light
            lane_pos = 100 - traci.vehicle.getLanePosition(veh_id)
            
    
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
        last_action = -1
        #last_total_waiting_vehicle = 0
        #waiting_times = {} 
        #waiting_time_vehicle = 0 #sum of waiting time in line
        total_reward = 0 #total reward in each epoch
        #total_vehicles = 0 #sum of vehicles generated
        
        for step in range(self.max_steps):            
            #choose action (light phase)
            action = self.agent.select_action(last_state, epsilon)
            
            #check whether chosen action isn't the same as last_action
            if last_action != action:
                #change light to next phase (green -> yellow -> red -> ...)
                self.set_light_phase(action)
                self.execute_step()
            
            self.set_light_phase(action)
            next_state, reward = self.execute_step(action)
            
            #save data in agent's memory
            self.agent.memory.add_sample((last_state, next_state, action, reward))
            
            last_state = next_state
            last_action = action    
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