from ai import Network, DQNAgent
from traffic_generator import TrafficGenerator
from simulation import Simulation
from utils import read_config, sumo_config, create_folder, plot_data
from shutil import copyfile
from os import path

def save_data():
    model_path = create_folder()
    copyfile('training_config.txt', path.join(model_path, 'training_config.txt'))
    return model_path

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
                     discount_rate = float(config['agent']['discount_rate']),  
                     batch_size    = int(config['agent']['nb_states']),         
                     epochs        = int(config['agent']['epochs']),
                     memory_size   = int(config['memory']['size']))  
                
    simulation = Simulation(agent, sumo_cfg, int(config['simulation']['max_steps']))
    
    for epoch in range(int(config['simulation']['total_episodes'])):
        print('Starting simulation - Episode: {}/{}'.format(epoch+1, config['simulation']['total_episodes']))
        traffic_generator.generate_routefile() #seed?
        epsilon = 1 - (epoch / int(config['simulation']['total_episodes']))
        simulation.run(epsilon)
        
    model_path = save_data()
    agent._save_model(model_path)
    
    for key, value in simulation.get_stats().items():        
        plot_data(data=value, y_label=key, model_path=model_path)


if __name__ == '__main__':
    main()