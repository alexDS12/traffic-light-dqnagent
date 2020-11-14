import numpy as np
import random
from os import path
import sys
from keras.layers import Dense
from keras.models import Sequential, save_model, load_model
from keras.optimizers import Adam

class Network(object):
    """
    This is related to topology of NN used to train agent.
    Usually only one hidden layer can solve most of the problems, in this case we're
    assuming all information about NN is given by user in config file.
    
    Attributes
    ----------
    nb_states : int
        The number of possible states agent can be at.
    nb_actions : int
        The number of possible actions agent can take.
    nb_hidden_layers : int
        Quantity of hidden layers NN will have.
    width_layers : int
        Quantity of nodes each layer will have.
    learning_rate : float
        Related to adjusting weights each NN's training session.    
    model : Network
        Model already structured and built.
        
    Methods
    -------
    build_model():
        Creates model's structure and compiles using Adam optimizer.
    predict_best(state):
        Gets the best action giving a state.
    predict_batch(states):
        Predicts N actions giving N states.
    train_batch(states, q_values):
        Trains NN fitting states as input and q_values as output.        
    """
    def __init__(self, nb_states, nb_actions, nb_hidden_layers, width_layers, learning_rate):
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.nb_hidden_layers = nb_hidden_layers
        self.width_layers = width_layers
        self.learning_rate = learning_rate
        self.model = self.build_model()
    
    def build_model(self):
        model = Sequential()
        model.add(Dense(units=self.width_layers, input_shape=(self.nb_states,), activation='relu', name='hidden0'))        
        for i in range(self.nb_hidden_layers-1):
            model.add(Dense(units=self.width_layers, activation='relu', name='hidden{}'.format(i+1)))
            
        model.add(Dense(units=self.nb_actions, activation='linear', name='output'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')
        return model
    
    def predict_best(self, state):
        return self.model.predict(np.reshape(state, (1, self.nb_states)))
    
    def predict_batch(self, states):
        return self.model.predict(states)
    
    def train_batch(self, states, q_values):
        self.model.fit(x=states, y=q_values, epochs=1, verbose=0)
        
class Memory(object):
    """
    Represents DQN's Memory.
    It's needed while training our NN, sending batches to adjust weights.
    
    Attributes
    ----------
    size : int
        The number of samples agent's memory can hold.
    memory : list
        A list of samples stored in memory.
        
    Methods
    -------
    add_sample(sample):
        Stores a new sample into memory, if it's full, oldest memory is removed.
    get_sample(batch_size):
        Gets N random samples from memory, grouping same variables in one tuple.
        E.g. (state0, state1, ...) (action0, action1, ...) and returns list ([states], [actions], ...)
    """
    def __init__(self, size):
        self.size = size
        self.memory = []
        
    def memory_size(self):
        return len(self.memory)
        
    def add_sample(self, sample):
        self.memory.append(sample)
        if self.memory_size() > self.size:
            del self.memory[0]
    
    def get_sample(self, batch_size):
        if batch_size > self.memory_size():
            batch_size = self.memory_size()
        sample = zip(*random.sample(self.memory, batch_size))
        return np.array([item for item in sample])
    
class DQNAgent(object):
    """
    Represents the agent which is the traffic light controller.
    
    Attributes
    ----------
    nn : Network
        Agent's NN.
    memory : Memory
        Agent's memory.
    discount_rate : float
        Discount rate for every action agent takes.
    batch_size : int
        Quantity of actions to train NN.
    epochs : int
        Quantity of times a training session will run.
    nb_actions : int
        The number of possible actions agent can take.
    nb_states : int
        The number of possible states agent can be at.
    
    Methods
    -------
    experience_replay():
        At the end of every epoch, NN is trained adjusting its weights to
        improve agent's decisions.
    select_action(state):
        Randomizes an int in [0, 1] to decide whether
        next action is gonna be exploration or exploitation.
    _save_model(model_path):
        Uses built function to save NN's model in an unique dir.
    _load_model(model_path):
        Checks whether model exists and loads it, otherwise an error is given.
    """
    def __init__(self, nn, memory_size, discount_rate, batch_size, epochs):
        self.nn = nn        
        self.memory = Memory(memory_size)
        self.discount_rate = discount_rate
        self.batch_size = batch_size
        self.epochs = epochs        
        self.nb_actions = self.nn.nb_actions
        self.nb_states = self.nn.nb_states
        
    def experience_replay(self):
        for _ in range(self.epochs):
            if self.memory.memory_size() > 0:
                last_states, next_states, actions, rewards = self.memory.get_sample(self.batch_size)
                
                inputs = np.zeros((self.batch_size, self.nb_states))
                outputs = np.zeros((self.batch_size, self.nb_actions))
                
                q_values = self.nn.predict_batch(last_states)
                next_q_values = self.nn.predict_batch(next_states)
                
                for i in range(self.batch_size):
                    inputs[i] = last_states[i]
                    q_value = q_values[i]
                    q_value[actions[i]] = rewards[i] + self.discount_rate * np.amax(next_q_values[i])   
                    outputs[i] = q_value
                   
                self.nn.train_batch(inputs, outputs)
        
    def select_action(self, state, epsilon):
        #exploitation
        if random.random() > epsilon:
            return np.argmax(self.nn.predict_best(state))
        #exploration
        else:
            return random.randint(0, self.nb_actions-1)
        
    def _save_model(self, model_path):
        save_model(self.nn.model, filepath=path.join(model_path, 'training_model.h5'))
        
    def _load_model(self, model_path):
        #Check if model exists
        model_path = path.join(model_path, 'training_model.h5')
        if path.isfile(model_path):
            print('Model found')
            return load_model(filepath=model_path)
        else:
            sys.exit('Model not found')
        
    