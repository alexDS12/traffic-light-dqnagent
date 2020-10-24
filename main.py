from ai import Network, DQNAgent
from traffic_generator import TrafficGenerator
from simulation import Simulation
from utils import read_config, sumo_config
#from datetime import datetime

def main():    
    config = read_config('training_config.txt')
    
    sumo_cfg = sumo_config(config['simulation']['gui'], 
                              config['simulation']['max_steps'])
    
    traffic_generator = TrafficGenerator(int(config['simulation']['nb_cars_generated']), 
                                         int(config['simulation']['max_steps']))    
    
    nn = Network(int(config['agent']['nb_states']), 
                 int(config['agent']['nb_actions']),
                 int(config['neural_network']['nb_hidden_layers']), 
                 int(config['neural_network']['width_layers']),   
                 float(config['neural_network']['learning_rate']))
    
    agent = DQNAgent(nn=nn,  
                     gamma       = float(config['agent']['discount_rate']),
                     memory_size = int(config['memory']['size']),                                       
                     epochs      = int(config['neural_network']['epochs']),
                     batch_size  = int(config['neural_network']['batch_size']))  
                
    simulation = Simulation(agent, sumo_cfg, int(config['simulation']['max_steps']))
    
    for epoch in range(int(config['simulation']['total_episodes'])):
        print('Starting simulation - Episode: {}/{}'.format(epoch+1, config['simulation']['total_episodes']))
        traffic_generator.generate_routefile() #seed?
        epsilon = 1 - (epoch / int(config['simulation']['total_episodes']))
        simulation.run(epsilon)
    
    


if __name__ == '__main__':
    main()