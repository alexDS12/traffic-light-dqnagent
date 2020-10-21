"""
Functions to split headers and data from file and add result to nested dict. 
Example of file is given below:
    
[header]
config1='sometext',config2='anothertext'

Afterwards, data must be split by ',' then '=' characters.

data[0] => config1
data[1] => 'sometext'
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