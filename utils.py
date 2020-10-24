import sys
import os
from os import path
from sumolib import checkBinary

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
        return 'File not found'

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
        tools = path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit('Please declare environment variable "SUMO_HOME"')
        
    #check if using GUI, not necessarily needed after SUMO 0.28.0
    if gui:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')        
        
    return [sumoBinary, '-c', 'data/sumo_config.sumocfg', '--no-step-log', 'true', 
            '--waiting-time-memory', time_steps]