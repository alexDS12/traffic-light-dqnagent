import numpy as np

class TrafficGenerator(object):
    """
    Handles all traffic generation within SUMO environment
    
    Attributes
    ----------
    nb_vehicles : int
        The number of vehicles that will be generated
    time_steps : int
        Represents how long our simulation will last
        
    Methods
    -------
    generate_routefile()
        Generates environment's xml file -> type of vehicles and routes.
    """
    def __init__(self, nb_vehicles, time_steps):
        self.nb_vehicles = nb_vehicles
        self.time_steps = time_steps
        
    def generate_routefile(self):
        np.random.seed()
        
        #Generating random departures
        departures = np.sort([np.random.randint(1, self.time_steps) for _ in range(self.nb_vehicles)])
        
        with open('data/environment.rou.xml', 'w') as routes_file:
            """
            Generating types of vehicles and routes
            """
            routes_file.write("""<routes>
                <vType id="car" accel="1.0" decel="4.5" sigma="0.5" length="5.0" minGap="2.5" maxSpeed="20.0" guiShape="passenger" />
                <vType id="bus/city" accel="0.9" decel="4.5" sigma="0.5" length="5.0" minGap="2.5" maxSpeed="20.0" guiShape="bus" color="1,0,0"/>
                <vType id="motorcycle" accel="1.0" decel="4.5" sigma="0.5" length="2.0" minGap="3.0" maxSpeed="20.0" guiShape="motorcycle" color="0.5,0.5,1"/>
        
                <route id="S_N" edges="BC TC" />
                <route id="S_W" edges="BC TL" />
                <route id="S_E" edges="BC BR" />
                <route id="W_E" edges="BL BR" />
                <route id="E_N" edges="TR TC" />
                <route id="E_W" edges="TR TL" />""")
                
            """
            Generating random vehicles in lanes
            """
            for i in range(self.nb_vehicles):
                """
                Randomize whether our vehicle is going straight or turning when crossing lanes
                
                We suppose 70% of generated vehicles will go straight
                
                We also need to choose route each vehicle will take (source and destination)
                
                Then finally we randomly choose type of vehicle, which are:
                    0 => car,
                    1 => bus/city,
                    2 => motorcycle
                """
                randomness = np.random.uniform(0, 1)
                #get first departure and then pop it from list
                depart = departures[0]
                departures = np.delete(departures, 0)                
                  
                veh_num = np.random.randint(0, 3)
                vehicle_type = "car" if veh_num == 0 else ("bus/city" if veh_num == 1 else "motorcycle")
                
                #Now we randomize which route vehicle will take                
                route = np.random.randint(0, 3)
                if randomness < 0.7:
                    if route == 0: #west to east (Bot_Left -> Bot_Right)
                        route_totake = 'W_E'
                        vehicle_id = 'BL_BR'
                    elif route == 1: #east to west (Top_Right -> Top_Left)
                        route_totake = 'E_W'
                        vehicle_id = 'TR_TL'
                    else: #south to north (Bot_Center -> Top_Center)
                        route_totake = 'S_N'
                        vehicle_id = 'BC_TC'
                else:
                    if route == 0: #south to west (Bot_Center -> Top_Left)
                        route_totake = 'S_W'
                        vehicle_id = 'BC_TL'
                    elif route == 1: #south to east (Bot_Center -> Bot_Right)
                        route_totake = 'S_E'
                        vehicle_id = 'BC_BR'
                    else: #east to north (Top_Right -> Top_Center)
                        route_totake = 'E_N'
                        vehicle_id = 'TR_TC'                
                routes_file.write('\n    <vehicle id="{}_{}" type="{}" route="{}" depart="{}" departLane="random" departSpeed="10.0" />'.format(vehicle_id, i, vehicle_type, route_totake, depart))       
            #close routes tag           
            routes_file.write("\n</routes>")
            routes_file.close()
            print('Traffic generation done')