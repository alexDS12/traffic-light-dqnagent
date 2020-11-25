import sys
import os
from sumolib import checkBinary
from matplotlib import pyplot as plt
from shutil import copyfile

"""
Functions to split headers and data from file and add result to nested dict. 
Example of file is given below:
    
[header]
config1='sometext',config2='anothertext'

Afterwards, data must be split by ',' then '=' characters.

data[0] => config1
data[1] => 'sometext'

Arguments:
    text_file : file

Returns:
    Nested dict config following the model [key][key][value].
"""        
def read_config(text_file):
    try:
        with open(text_file, 'r') as file:
            headers, rows = config_split(file)
            config = {}
            for header, row in zip(headers, rows):
                config[header] = {}
                #splitting config1='sometext' from config2='anothertext'
                for r in row.split(','):
                    #splitting config1 as key and 'sometext' as value
                    key, value = r.split('=')
                    config[header][key] = value
            return config
    except FileNotFoundError:
        sys.exit('File not found')

def config_split(file):
    headers = []
    data = []
    for row in file:
        #ignore comments if there is any in file
        row = row.split('#')[0].strip()
        if row.startswith('[') and row.endswith(']'):
            headers.append(row[1:-1])
        else:
            data.append(row)
    #filter is used just to remove blank lines amongst list
    return headers, filter(None, data)

"""
Function to configure TraCI.
Firstly it's needed to check if SUMO_HOME is in system path.
Then, we set GUI according to config file.

Arguments:
    gui : boolean
        Whether to display SUMO or not.
    time_steps : int
        
Returns:
    TraCI config.
"""
def sumo_config(gui, time_steps):
    #check if var is set, otherwise application won't run
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit('Please declare environment variable "SUMO_HOME"')   
    #check if using GUI, not necessarily needed after SUMO 0.28.0
    if gui == True:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')        
        
    return [sumoBinary, '-c', 'data/sumo_config.sumocfg', '--no-step-log', 'true']

"""
Function to create a new folder for each trained model,
giving an unique int name after checking last int.

Returns:
    Model's folder created with training and testing subfolders.
"""
def create_folder():
    models_folder = 'models'
    models_path = os.path.join(os.getcwd(), models_folder)
    if not os.path.exists(models_path):
        os.mkdir(models_folder)

    dirs = sorted(os.listdir(models_path))
    if dirs:
        new_dir = int(dirs[-1]) + 1
    else:
        new_dir = 1

    model_path = os.path.join(models_path, str(new_dir))
    os.mkdir(model_path)
    return model_path

"""
Function to plot training data and save as a png file
"""
def plot_data(data, y_label, model_path, train_or_test, x_label='Episode'):
    plt.rcParams.update({'font.size': 15})
    plt.plot(data)
    plt.title(x_label + ' vs ' + y_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(min(data), max(data))
    fig_name = train_or_test + '_' + y_label + '.png'
    plt.savefig(os.path.join(model_path, fig_name), dpi=100, bbox_inches = 'tight')

def save_data(file_tosave, model_path):
    print('Saving data at: %s\n' % model_path)
    copyfile(file_tosave, os.path.join(model_path, file_tosave))