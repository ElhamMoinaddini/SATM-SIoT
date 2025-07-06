import random
import math
import networkx as nx
import argparse
from pathlib import Path
import time
import numpy as np
import math
from core import Sim
from application import *
import csv
from population import *
from topology  import Topology
import os
from yafs.stats import Stats
from yafs.distribution import deterministic_distribution
from application import fractional_selectivity

from selection_multipleDeploys import BroadPath,CloudPath_RR
from placement_Cluster_Edge import CloudPlacement,FogPlacement
lists = [] 
 


ACCList1=[]
"""
AttType can be one of the Attack types:
--------------
1: 'ME'   Malicious for every one
2: 'DA'	Discriminatory Attack
3: 'OOA'	On-off Attack
4: 'WA'	Withewash Attack
5: 'SPA'	Self-Promoting Attack
6: 'Mix'	Mix Attack
"""
AttType='ME'
# set the percentage of trustor attacker devices
AttackPercentage=0.25
# set the percentage of trustee attack devices
trusteeAttackPercentage=0.2 
Device_Ser={
                 1: {1,2,4,5,6,7,9,10,11,13,15},
                 2: {1,2,4,5,7,8,9,10,12,14,15,16},
                 3:{1,2	,3,4,5,7,8,10,11,14},
                 4:{3,5,8,10,12,13,14,15,16},
                 5:{2,4,6,7,8,11,13,15},
                 6:{2,3,4,7,10,11,14,16},
                 7:{3,7,8,12,14,15,16},
                 8:{5,6,7,9,10,13,16},
                 9:{1,2,3,4,9,12,14},
                 10:{1,2,5,6,7,12,15},
                 11:{1,2,6,7,8,	9,10,11,12,16},
                 12:{1,2,3,5,7,8,11,13,16},
                 13:{1,2,7,9,10,12,13,15},
                 14:{1,2,3,5,6,7,8,9,11,13,14},
                 15:{1,2,3,5,6,7,11,13,14,15,16},
                 16:{1,2,4,5,6,7,8,9,12,14}
                 }
# each device aske for service wich is not in its services
def Payloud(Device_name,modules):
    device_type = None
    for module in modules:
        if Device_name in module.keys():
            for _, attributes in module.items():
                if "Device_Type" in attributes:
                    device_type = attributes["Device_Type"]
                    for item in Device_Ser:
                        if device_type==item:
                            f=True
                            while f:
                                a=random.randint(1, 16)
                                if a not in list(Device_Ser[item]):
                                    f=False
                                    
                    break
    return(a)            
    
RANDOM_SEED = 1
# Define an event that creates a link between the two nodes when triggered

def create_friendships(x):
    # Create a dictionary to hold user friendships
    friendships = {i: [] for i in range(1, x + 1)}

    for user in range(1, x + 1):
        # Determine the number of friends to assign (between 2 and x/4)
        max_friends = x // 6
        num_friends = random.randint(1, max_friends)
        
        # Get potential friends (excluding the user themselves)
        potential_friends = [friend for friend in range(1, x + 1) if friend != user]
        
        # Randomly select friends
        selected_friends = random.sample(potential_friends, min(num_friends, len(potential_friends)))
        
        # Update friendships
        for friend in selected_friends:
            if user not in friendships[friend]:  # Ensure mutual friendship
                friendships[friend].append(user)
            friendships[user].append(friend)

    return friendships    

def create_application(NumOfFog,numOfDevicePerFog):
    # APLICATION
    a = Application(name="SimpleCase")
    
    no_user=math.floor((sum(numOfDevicePerFog)/3)) 
    no_device=sum(numOfDevicePerFog)
    Tdevice=sum(numOfDevicePerFog)
    numbers = list(range(1, Tdevice + 1))
    #select attacker devices
    num_to_select = max(1, int(len(numbers) * AttackPercentage))
    selected_numbers = random.sample(numbers, num_to_select)
    print("malicious devices",selected_numbers)
    #select Recommender attacker devices
    num_to_select1 = max(1, int(len(numbers) * trusteeAttackPercentage))
    selected_numbers1 = random.sample(numbers, num_to_select1)
    print("False Recommender devices",selected_numbers1)
    
    modules = [{"Cloud": {"RAM": 10, "Type": Application.TYPE_MODULE}}]
    Dno=0
    #create_friendships dic
    friendships=create_friendships(no_user)
    for idx in range(NumOfFog):
        fog_name = f"Fog{idx+1}"
        modules.append({fog_name: {"RAM": 10, "Type": Application.TYPE_MODULE, "Device_Type": "FFF"}})
        for j in range(numOfDevicePerFog[idx]):
            x = random.randrange(100)
            y = random.randrange(100)
            position=[x,y]
            Device_Type = random.randint(1, 16)
            for item in Device_Ser:
                if Device_Type==item:
                    Device_Services=Device_Ser[item]
            
            Device_Brand=random.randint(1, 2)#12
            Device_Model=random.randint(1, 3)#24
            User_ID=random.randint(1, no_user)
            #there are 4 possible roles for users
            User_Role=random.randint(1, 4)
            if User_ID in friendships:
                User_Friend=friendships[User_ID]
            else:
                User_Friend=[]
            xx=random.random()
            if xx<=0.3:
                ComputatinalC=1
            elif 0.3<xx<0.6: 
                ComputatinalC=2
            elif xx>=0.6: 
                ComputatinalC=3    
            """
            Computatinal Capability:
                1: The Very Constrained Devices Class
                2: The Quite Constrained Devices Class
                3: The Not Constrained Devices Class
            """
            if   ComputatinalC==1: Energy_limitation=random.randint(0, 1) # 50%  availability 0.8-1ms responcetime
            elif ComputatinalC==2: Energy_limitation=random.randint(1, 2) # 70%  availability 0.4-0.6ms responcetime
            elif ComputatinalC==3: Energy_limitation=random.randint(2, 3) # 90% availability  0.001-0.2ms  responcetime
            """
            Energy_limitation:
                0: Devices with energy limited by events (i.e. event base harvesting)
                1: Devices with energy limited by period (i.e. replaceable battery)
                2: Devices with energy limited by lifetime (i.e. rechargable battery)
                3: Devices without energy limitation (i.e. mains-powered)
            
            """    
                  
            Dno+=1 
            #set attacker
            if Dno in selected_numbers:
                modules.append({f"Device{idx+1,j+1}": {"Name":f"Device{idx+1,j+1}",
                                                   "RAM": 10, 
                                                   "Type": Application.TYPE_MODULE,
                                                   "Device_Type": Device_Type,
                                                   "Device_Brand":Device_Brand,
                                                   "Device_Model":Device_Model,
                                                   "Device_Services":Device_Services,
                                                   "Position":position,
                                                   "User_ID":User_ID,
                                                   "User_Role":User_Role,
                                                   "User_Friend":User_Friend,
                                                   "ComputatinalC":ComputatinalC,
                                                   "Energy_limitation":Energy_limitation,
                                                   "Attack_Type":AttType}})
                print(f"Device{idx+1,j+1}")
            #set benign devices
            else:
                 modules.append({f"Device{idx+1,j+1}": {"Name":f"Device{idx+1,j+1}",
                                                    "RAM": 10, 
                                                    "Type": Application.TYPE_MODULE,
                                                    "Device_Type": Device_Type,
                                                    "Device_Brand":Device_Brand,
                                                    "Device_Model":Device_Model,
                                                    "Device_Services":Device_Services,
                                                    "Position":position,
                                                    "User_ID":User_ID,
                                                    "User_Role":User_Role,
                                                    "User_Friend":User_Friend,
                                                    "ComputatinalC":ComputatinalC,
                                                    "Energy_limitation":Energy_limitation,
                                                    "Attack_Type":0}})
            #set Recommender attacker
            if Dno in selected_numbers1:
                modules.append({f"Device{idx+1,j+1}": {"Name":f"Device{idx+1,j+1}",
                                                   "RAM": 10, 
                                                   "Type": Application.TYPE_MODULE,
                                                   "Device_Type": Device_Type,
                                                   "Device_Brand":Device_Brand,
                                                   "Device_Model":Device_Model,
                                                   "Device_Services":Device_Services,
                                                   "Position":position,
                                                   "User_ID":User_ID,
                                                   "User_Role":User_Role,
                                                   "User_Friend":User_Friend,
                                                   "ComputatinalC":ComputatinalC,
                                                   "Energy_limitation":Energy_limitation,
                                                   "Attack_Type":2}})
            
            modules.append({f"Sensor{idx+1,j+1}": {"Type": Application.TYPE_SOURCE, "Device_Type": "SSS1"}})
            modules.append({f"Actuator{idx+1,j+1}": {"Type": Application.TYPE_SINK, "Device_Type": "AAA1"}})
             
    a.set_modules(modules)
    
    random_number = random.randint(1, 16)
    S1 = "S" + str(random_number)
    
    for idx in range(NumOfFog):
        random_number = random.randint(1, 16)
        S1 = "S" + str(random_number)
        SFog = f"Fog{idx+1}" 
        Sm_SendGModel=f"m_SendGModel{idx+1}"
        Sm_SendLModel=f"m_SendLModel{idx+1}"
        Sm_SendGModel = Message(f"M_SendGModel{idx+1}", "Cloud", f"Fog{idx+1}", instructions=5000*10**6, bytes=1000,payload=0, broadcasting=True)
        Sm_SendLModel= Message(f"M_SendLModel{idx+1}", f"Fog{idx+1}", "Cloud", instructions=3500*10**6, bytes=1000,payload=[])
        a.add_service_module(f"Fog{idx+1}", Sm_SendGModel, Sm_SendLModel, fractional_selectivity, threshold=1.0)
        dDistribution = deterministic_distribution(name="Deterministic", time=490)
        a.add_service_source("Cloud", dDistribution, Sm_SendGModel)
        for idm in range(numOfDevicePerFog[idx]):
            Sm_Sensor = f"m_Sensor{idx+1,idm+1}"
            Sm_RequstServer = f"m_RequstServer{idx+1,idm+1}"
            Sm_SendServer = f"m_SendServer{idx+1,idm+1}"
            Sm_Actuator = f"m_Actuator{idx+1,idm+1}"
            Sm_Trans = f"m_Trans{idx+1,idm+1}"
            
            Sm_Sensor = Message(f"M_Sensor{idx+1,idm+1}", f"Sensor{idx+1,idm+1}", f"Device{idx+1,idm+1}", instructions=20*10**6, bytes=100,payload=S1)
            Sm_RequstServer = Message(f"M_RequstServer{idx+1,idm+1}", f"Device{idx+1,idm+1}", f"Fog{idx+1}", instructions=2000*10**6, bytes=500,payload=Payloud(f"Device{idx+1,idm+1}",modules))
            Sm_SendServer = Message(f"M_SendServer{idx+1,idm+1}", f"Fog{idx+1}", f"Device{idx+1,idm+1}", instructions=3500*10**6, bytes=500,payload=[])
            Sm_Actuator= Message(f"M_Actuator{idx+1,idm+1}", f"Device{idx+1,idm+1}", f"Actuator{idx+1,idm+1}", instructions=2000*10**6, bytes=100,payload=[])
            Sm_Trans= Message(f"M_Trans{idx+1,idm+1}", f"Device{idx+1,idm+1}", f"Fog{idx+1}", instructions=2000*10**6, bytes=500,payload=[])
            """
            Defining which messages will be dynamically generated # the generation is controlled by Population algorithm
            """
            a.add_source_messages(Sm_Sensor)
            
            # MODULE SERVICES
            a.add_service_module(f"Device{idx+1,idm+1}", Sm_Sensor, Sm_RequstServer, fractional_selectivity, threshold=1.0)
            a.add_service_module(f"Fog{idx+1}", Sm_RequstServer, Sm_SendServer, fractional_selectivity, threshold=1.0)
            a.add_service_module(f"Device{idx+1,idm+1}", Sm_SendServer, Sm_Actuator, fractional_selectivity, threshold=1.0)
            a.add_service_module(f"Device{idx+1,idm+1}", Sm_SendServer, Sm_Trans, fractional_selectivity, threshold=1.0)
            a.add_service_module(f"Fog{idx+1}", Sm_Trans)
            a.add_service_module("Cloud", Sm_SendLModel)
            
    
    return a
    
   
def create_json_topology(NumOfFog,numOfDevicePerFog):
    """
       TOPOLOGY DEFINITION
       """

    # CLOUD Abstraction
    
    topology_json = {}
    topology_json["entity"] = []
    topology_json["link"] = []
    id = 0
    # CLOUD Abstraction
    
    topology_json["entity"].append({"id": id, "model": "Cloud-", "IPT": 44800 * 10 ** 6, "RAM": 40000,"COST": 3,"WATT":20.0})
    id +=1
    # SENSOR
    topology_json["entity"].append(
        {"id": id, "model": "Sen_Cloud", "COST": 0,"WATT":0.0})
    topology_json["link"].append({"s": id - 1, "d": id, "BW": 100, "PR": 4})
    id += 1
    # ACTUATOR
    topology_json["entity"].append(
        {"id": id, "model": "Aut_Cloud", "COST": 0,"WATT":0.0})
    topology_json["link"].append({"s": id - 2, "d": id, "BW": 100, "PR": 1})
    id += 1
    i=0
        
    for idx in range(NumOfFog):
            #GATEWAY DEVICE
            S = f"fog{idx+1}"+"-"
            gw = id
            topology_json["entity"].append({"id": id, "model": S, "IPT": 2800 * 10 ** 6, "RAM": 4000, "COST": 3,"WATT":40.0})
            topology_json["link"].append({"s": 0, "d": id, "BW": 100, "PR": 10})
            id += 1
        
            for idm in range(numOfDevicePerFog[i]):
                S = f"Dev{idx+1,idm+1}"+"-"
                
                # DEVICE
                topology_json["entity"].append({"id": id, "model": S, "IPT": 1000 * 10 ** 6, "RAM": 1000, "COST": 0,"WATT": 40.0})
                topology_json["link"].append({"s": gw, "d": id, "BW": 100, "PR": 2})
                id += 1
                # SENSOR
                topology_json["entity"].append(
                    {"id": id, "model": f"Sen{idx+1,idm+1}", "COST": 0,"WATT":0.0})
                topology_json["link"].append({"s": id - 1, "d": id, "BW": 100, "PR": 4})
                id += 1
                # ACTUATOR
                topology_json["entity"].append(
                    {"id": id, "model": f"Aut{idx+1,idm+1}", "COST": 0,"WATT":0.0})
                topology_json["link"].append({"s": id - 2, "d": id, "BW": 100, "PR": 1})
                id += 1
            i+=1      
                        
    return topology_json
def main(simulated_time,depth,police):
       
        folder_path = "."  # Current folder path
        for filename in os.listdir(folder_path):
            if filename.startswith("global_model"):
                file_path = os.path.join(folder_path, filename)
                os.remove(file_path)
                
        
                
        folder_path = "."  # Current folder path
        for filename in os.listdir(folder_path):
            if filename.startswith("positions"):
                file_path = os.path.join(folder_path, filename)
                os.remove(file_path) 
        for filename in os.listdir(folder_path):
            if filename.startswith("model"):
                file_path = os.path.join(folder_path, filename)
                os.remove(file_path)
        
                
        folder_results = Path("results/")
        folder_results.mkdir(parents=True, exist_ok=True)
        folder_results = str(folder_results)+"/"

         
        NumOfFog =4
        NoRecrds = [[] for _ in range(NumOfFog)]
        
        Max_numOfDevicePerFog =5  
        # Thus, this variable is used in the population algorithm
        # In YAFS simulator, entities representing mobiles devices (sensors or actuactors) are not necessary because they are simple "abstract" links to the  access points
        # in any case, they can be implemented with node entities with no capacity to execute services.
        #
        numOfDevicePerFog=[]
        iii=2
        for idx in range(NumOfFog):
            iii+=1
            l=Max_numOfDevicePerFog
            #l = random.randint(1, Max_numOfDevicePerFog)
            numOfDevicePerFog.append(l)
            
        t = Topology()
        t_json = create_json_topology(NumOfFog,numOfDevicePerFog)
        t.load(t_json)
        nx.write_gexf(t.G,folder_results+"graph_main") 
        # you can export the Graph in multiples format to view in tools like Gephi, and so on.
        #y=nx.read_gexf("graph_main")  nx.draw(y, with_labels=True)


        """
        APPLICATION
        """
        app = create_application(NumOfFog,numOfDevicePerFog)


        """
        PLACEMENT algorithm
        """
        #In this case: it will deploy all app.modules in the cloud
        if police == "Cloud":
            placement = CloudPlacement("onCloud")
            
        else:
            placement = FogPlacement("onFogs")
            

        # placement = ClusterPlacement("onCluster", activation_dist=next_time_periodic, time_shift=600)
        """
        POPULATION algorithm
        """
        #In ifogsim, during the creation of the application, the Sensors are assigned to the topology, in this case no. As mentioned, YAFS differentiates the adaptive sensors and their topological assignment.
        #In their case, the use a statical assignment.
        pop = Statical("Statical")
        #For each type of sink modules we set a deployment on some type of devices
        #A control sink consists on:
        #  args:
        #     model (str): identifies the device or devices where the sink is linked
        #     number (int): quantity of sinks linked in each device
        #     module (str): identifies the module from the app who receives the messages
        
        
        #pop.set_sink_control({"model": "Aut_Cloud","number":1,"module":app.get_sink_modules()[0]})
        j=0
        for idx in range(NumOfFog):
            for idm in range(numOfDevicePerFog[idx]):
                pop.set_sink_control({"model": f"Aut{idx+1,idm+1}","number":1,"module":app.get_sink_modules()[j]})
                j+=1
           
        #In addition, a source includes a distribution function:
        dDistribution = deterministic_distribution(name="Deterministic", time=100)
        #pop.set_src_control({"model": "Sen_Cloud", "number":1,"message": app.get_message("M_SensorCloud"), "distribution": dDistribution})
        i=0
        for idx in range(NumOfFog):
            for idm in range(numOfDevicePerFog[i]):
                pop.set_src_control({"model": f"Sen{idx+1,idm+1}", "number":1,"message": app.get_message(f"M_Sensor{idx+1,idm+1}"), "distribution": dDistribution})
            i+=1    
                
        """--
        SELECTOR algorithm
        """
        #Their "selector" is actually the shortest way, there is not type of orchestration algorithm.
        #This implementation is already created in selector.class,called: First_ShortestPath
        if police == "Cloud":
            selectorPath = CloudPath_RR()
        else:
            selectorPath = BroadPath(Max_numOfDevicePerFog)
            
        """
        SIMULATION ENGINE
        """
        
        stop_time = simulated_time
        s = Sim(t, default_results_path=folder_results+"Results_%s_%i_%i" % (police, stop_time, depth))
        s.deploy_app2(app, placement, pop, selectorPath)

        """
        RUNNING - last step
        """
        
        s.run(lists,ACCList1,NoRecrds,NumOfFog,numOfDevicePerFog,AttType,stop_time,app.services,app.data,test_initial_deploy=False,show_progress_monitor=False)
        s.print_debug_assignaments()
        # s.draw_allocated_topology() # for debugging
        print("number of fogs=",len(lists))
        
        
        
          
        with open("ACCList.csv", "w") as f:
          writer = csv.writer(f, lineterminator='\n')
          writer.writerows(ACCList1)  
        
          
        
        result_list=[]
        for nested_list in lists:
            result_list.append( [[sublist[-2], sublist[-1]] for sublist in nested_list])
        
        

if __name__ == '__main__':
        import logging.config
        import os
        folder_results = Path("results/")
        folder_results.mkdir(parents=True, exist_ok=True)
        folder_results = str(folder_results)+"/"

        time_loops = [["Device", "Fog","Device"]]

        logging.config.fileConfig(os.getcwd()+'/logging.ini')

        parser = argparse.ArgumentParser()
        parser.add_argument("-t", "--time", help="Simulated time ")
        parser.add_argument("-d", "--depth", help="Depths ")
        parser.add_argument("-p", "--police", help="Cloud or edge ")
        args = parser.parse_args()

        if not args.time:
            stop_time =1100
        else:
            stop_time = int(args.time)

        start_time = time.time()
        if not args.depth:
            dep  = 2
        else:
            dep = int(args.depth)

        if not args.police:
            police = "edge"
        else:
            police = str(args.police)

        #police ="cloud"
        main(stop_time,dep,police)
        s = Stats(defaultPath=folder_results+"Results_%s_%s_%s" % (police, stop_time, dep))
        #print("%f," %(s.valueLoop(stop_time, time_loops=time_loops)))

        print("\n--- %s seconds ---" % (time.time() - start_time))
