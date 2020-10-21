import numpy as np
from keras.layers import Dense
from keras.models import Sequential
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
        model.add(Dense(units=self.width_layers, input_dim=self.nb_states, activation='relu', name='hidden0'))        
        for i in range(self.nb_hidden_layers-1):
            model.add(Dense(units=self.width_layers, activation='relu', name='hidden{}'.format(i+1)))
            
        model.add(Dense(units=self.nb_actions, activation='linear', name='output'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')
        return model
        
class Memory(object):
    """
    Represents DQN's Memory.
    It's needed while training our NN, sending batches to adjust weights.
    
    Attributes
    ----------
    size : int
        The number of samples our agent's memory can hold.
    memory : list
        A list of samples stored in memory.
        
    Methods
    -------
    add_sample(sample):
        Stores a new sample into memory, if it's full, oldest memory is removed.
    get_sample(batch_size):
        Gets N random samples from memory, if batch size is bigger than actual size, 
        it returns all memory scrambled.
    """
    def __init__(self, size):
        self.size = size
        self.memory = []
        
    def add_sample(self, sample):
        self.memory.append(sample)
        if len(self.memory) > self.size:
            del self.memory[0]
    
    def get_sample(self, batch_size):
        if batch_size > self.size:
            return np.random.choice(self.memory, self.size)        
        return np.random.choice(self.memory, batch_size, replace=False)
    