import traci

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
        pass
    
    def run(self, epsilon):
        traci.start(self.sumo)
        last_state = -1
        last_action = -1  
        last_reward = 0
        #last_total_waiting_vehicle = 0
        #waiting_times = {} 
        #waiting_time_vehicle = 0 #sum of waiting time in line
        #total_reward = 0 #total reward in each epoch
        #total_vehicles = 0 #sum of vehicles generated
        
        for step in range(1, self.max_steps+1):
            #get current state
            new_state = self.get_state() 
            
            #save current step in agent's memory
            self.agent.save_tomemory((last_state, new_state, last_action, last_reward))
            
            #choose action (light phase)
            action = self.agent.select_action(new_state, epsilon)
            
            
            
            #reward = 0
            
            
            
            
            
            #take next action only if it's not the same as the last action
            if last_action != action:
                pass