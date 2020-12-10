# DQN Agent for Reinforcement Learning and SUMO

Deep Q-Learning interacts with Simulation of Urban MObility while learning how to maximize traffic efficiency while minimizing waiting times and lengths in all lanes.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

All you will need:

1. TensorFlow and TensorFlow-gpu v2.x.
2. Keras v2.3.1.
3. SUMO v1.7.0.
4. Python v3.7.
5. TraCI.

With Anaconda:
```
conda install tensorflow-gpu
conda install keras
```
If using pip:
```
pip install --upgrade pip
pip install tensorflow
pip install keras
```
Or even Colab:
```
%tensorflow_version 2.x
```

Then you can get SUMO [here](https://sumo.dlr.de/docs/Downloads.php) or if using Linux/Colab:

```
!sudo add-apt-repository ppa:sumo/stable -y
!sudo apt-get update -y
!sudo apt-get install sumo sumo-tools sumo-doc
!pip install traci
```

## Running 

After making sure you have everything installed and set to go, you can customize inputs for the algorithm in the **config** file following the example:

```
[simulation]
gui=False,max_steps=5000,nb_cars_generated=1000,total_episodes=1000

[neural_network]
nb_hidden_layers=1,width_layers=50,learning_rate=0.001,batch_size=100

[memory]
size=50000

[agent]
nb_states=60,nb_actions=4,discount_rate=0.6,epochs=500
```

```
gui = boolean - Whether you want to visually see what's happening.
max_steps = int - For how long simulation will last.
nb_cars_generated = int - How many vehicles will be generated for the simulation.
total_episodes = int - Number of episodes that will run in a simulation.
nb_hidden_layers = int - How many hidden layers there will be (NN's structure).
width_layers = int - How many nodes there will be in each hidden layer.
learning_rate = float - Parameter that determines the step size at each epoch during NN's training.
batch_size = int - Length of data that will be passed from memory to training.
size = int - Agent's memory size for experience replay between episodes.
nb_states = int - NN's inputs.
nb_actions = int - NN's outputs.
discount_rate = float - Basically to control agent's decision making.
epochs = int - How many epochs agent will train and fitting weights.
```

Then you can enter the following command inside root folder:

```
python main.py
```

If everything goes as planned, code will run and ask whether you want to (1) train a new agent or (2) test an agent.

Every piece of code has it's own docstring, therefore it might be helpful. However, if any questions, please open an issue and I'll gladly help you.

When total episodes are reached, 3 plots will be generated in the path shown in console.

## Built With

* [Tensorflow](https://www.tensorflow.org/) - Machine Learning platform
* [Keras](https://keras.io/) - Deep Learning API
* [SUMO](https://www.eclipse.org/sumo/) - Traffic simulation package
* [Python](https://www.python.org/)

## Authors

* **Alex Sievers** - *Initial work* - [alexDS12](https://github.com/alexDS12)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* I would like to thank everyone that was involved in this project, my family, my friends and my thesis advisor.
* Also want to acknowledge [Andrea Vidali](https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control)'s thesis about the same subject, which in result gave me lots of ideas.
* My main inspiration for this project was to give something in return to society after spending a lot of time learning how Science can help humankind.
