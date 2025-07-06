"""
    This type of algorithm have two obligatory functions:

        *initial_allocation*: invoked at the start of the simulation

        *run* invoked according to the assigned temporal distribution.

"""

from yafs.placement import Placement

class CloudPlacement(Placement):
    """
    This implementation locates the services of the application in the cheapest cloud regardless of where the sources or sinks are located.

    It only runs once, in the initialization.

    """
    def initial_allocation(self, sim, app_name):
        #We find the ID-nodo/resource
        value = {"model": "Cluster"}
        id_cluster = sim.topology.find_IDs(value) #there is only ONE Cluster
        value = {"model": "m-"}
        id_mobiles = sim.topology.find_IDs(value)

        #Given an application we get its modules implemented
        app = sim.apps[app_name]
        
    #end function


class FogPlacement(Placement):
    """
    This implementation locates the services of the application in the fog-device regardless of where the sources or sinks are located.

    It only runs once, in the initialization.

    """
    def initial_allocation(self, sim, app_name,NumOfFog,numOfDevicePerFog):
        
        app = sim.apps[app_name]
        services = app.services
        #print("NumOfFog",NumOfFog)
        #We find the ID-nodo/resource
        value = {"model": "Cloud-"}
        #id_Cloud =  #there is only ONE Cluster
        id_Cloud =sim.topology.find_IDs(value)
        #print("id_Cloud",id_Cloud)
        id_Fog=[]
        id_Device={}
        i=0
        for idx in range(NumOfFog):
                #GATEWAY DEVICE
                S = f"fog{idx+1}"+"-"
                value = {"model": S}
                id_Fog.append(sim.topology.find_IDs(value)) 
                
                                
                for idm in range(numOfDevicePerFog[i]):
                    S= f"Dev{idx+1,idm+1}"+"-"
                    value = {"model": S}
                    id_Device[f"Device{idx+1,idm+1}"]=sim.topology.find_IDs(value) 
                    
                i+=1                    
        
        #Given an application we get its modules implemented
        #print("id_Fog =",id_Fog )
        #print("id_Device =",id_Device )
        #print("services.keys()",services.keys())
        i=0
        for module in services.keys():
            #print(module)
            if  module.startswith("Cloud"):
                #print("id_Cloud=============",module,id_Cloud)
                idDES = sim.deploy_module(app_name, module, services[module],id_Cloud)
            else:
                i=0
                for idx in range(NumOfFog):
                    #print("idx",idx)
                    if module==f"Fog{idx+1}" :
                            idDES = sim.deploy_module(app_name, module, services[module],id_Fog[idx])
                           # print("id_fog=============",module,id_Fog[idx])
                            break
                    for idm in range(numOfDevicePerFog[i]):
                         #print(idm)
                         if module==f"Device{idx+1,idm+1}" :
                             idDES = sim.deploy_module(app_name, module, services[module],id_Device[f"Device{idx+1,idm+1}"])
                             #print("id_device=============",module,id_Device[f"Device{idx+1,idm+1}"])
                             break
                    i+=1         
        
                   
           


    
