from ai import Network, DQNAgent
from traffic_generator import TrafficGenerator
from simulation import Simulation
from utils import *
import timeit

def initialize_config(config_file):
    config = read_config(config_file)

    sumo = sumo_config(config['simulation']['gui'], \
                       config['simulation']['max_steps'])
  
    traffic_generator = TrafficGenerator(int(config['simulation']['nb_cars_generated']), \
                                         int(config['simulation']['max_steps']))
    return config, sumo, traffic_generator

def main_testing():
    config, sumo, traffic_generator = initialize_config('testing_config.txt')

    nn = Network(int(config['agent']['nb_states']), \
                 nb_model = config['utils']['model'])
  
    agent = DQNAgent(nn = nn)

    simulation = Simulation(agent, sumo, int(config['simulation']['max_steps']))

    model_path = 'models/' + config['utils']['model']
    save_data('testing_config.txt', model_path)

    for epoch in range(int(config['simulation']['total_episodes'])):
        print('Testing agent - Episode: {}/{}'.format(epoch+1, config['simulation']['total_episodes']))
        traffic_generator.generate_routefile()
        start_time = timeit.default_timer()
        simulation.run(epsilon=-1) #now best action needs to be always chosen
        print('Execution time: {:.1f} s\n'.format(timeit.default_timer() - start_time))
  
    for key, value in simulation.get_stats().items():
        plot_data(value, key, model_path, 'testing')


def main_training():
    config, sumo, traffic_generator = initialize_config('training_config.txt')

    nn = Network(int(config['agent']['nb_states']), \
                 int(config['agent']['nb_actions']), \
                 int(config['neural_network']['nb_hidden_layers']), \
                 int(config['neural_network']['width_layers']), \
               float(config['neural_network']['learning_rate']))
  
    agent = DQNAgent(nn=nn, \
                     discount_rate = float(config['agent']['discount_rate']), \
                     batch_size    = int(config['neural_network']['batch_size']), \
                     epochs        = int(config['agent']['epochs']), \
                     memory_size   = int(config['memory']['size']))  
  
    simulation = Simulation(agent, sumo, int(config['simulation']['max_steps']), is_training=True)

    model_path = create_folder()
    save_data('training_config.txt', model_path)
  
    for epoch in range(int(config['simulation']['total_episodes'])):
        print('Starting simulation - Episode: {}/{}'.format(epoch+1, config['simulation']['total_episodes']))
        traffic_generator.generate_routefile()
        start_time = timeit.default_timer()
        simulation.run(agent.get_epsilon(epoch)) #get next epsilon and send to simulation
        print('Execution time: {:.1f} s\n'.format(timeit.default_timer() - start_time))
        if (epoch+1) % 100 == 0 or (epoch+1) == int(config['simulation']['total_episodes']): #backup
            agent.nn._save_model(model_path)
            for key, value in simulation.get_stats().items():
                plot_data(value, key, model_path, 'training')

if __name__ == '__main__':
    op = input('Select option: \n(1) Training (2) Testing ')
    if op == '1':
      main_training()
    elif op == '2':
      main_testing()
    