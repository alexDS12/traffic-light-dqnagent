from ai import Network, Memory
from traffic_generator import TrafficGenerator
from utils import *


if __name__ == '__main__':
    config = read_config('training_config.txt')
    
    neural_network = Network(int(config['agent']['nb_states']),
                             int(config['agent']['nb_actions']),
                             int(config['neural_network']['nb_hidden_layers']),
                             int(config['neural_network']['width_layers']),
                             float(config['neural_network']['learning_rate']))
    
    memory = Memory(int(config['memory']['size']))
    
    traffic_generator = TrafficGenerator(int(config['simulation']['nb_cars_generated']), 
                                         int(config['simulation']['max_steps']))
    print(memory.size)
    print(neural_network.model.summary())
    traffic_generator.generate_routefile()