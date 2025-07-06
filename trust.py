from tensorflow.keras.models import Sequential, model_from_json
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Only show errors
import warnings
warnings.filterwarnings(
    "ignore", message="The load function will merged with load_all_node_attr function")
warnings.filterwarnings(
    "ignore", message="Your TensorFlow version is newer than 2.4.0 and so graph support has been removed in eager mode and some static graphs may not be supported.")
warnings.filterwarnings("ignore", message="`tf.keras.backend.set_learning_phase` is deprecated and will be removed after 2020-10-11. To update it, simply pass a True/False value to the `training` argument of the `__call__` method of your layer or model.")

import math
import random
import re
import logging
import csv
import time
from Kmeans import *


positions_array=[]
# Disable printing to the console
logging.getLogger('tensorflow').disabled = True
tf.get_logger().setLevel('ERROR')
transactions1 = []
trust_level=[0,0,0,0]
Trust_List = []
trust_to_num = {
    "HT": 1,
    "T": 0.75,
    "UT": 0.25,
    "DT": 0
}
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3
label_map = {0:"HT", 1:"T", 2:"UT", 3:"DT"}
lable=[0, 1, 2, 3]
ooo=0

class Trust:
    
    def __init__(self, payload=None):
        payload = []
        
    def TrustComputation_procedure(self, module_name, message_in, payload, module_List,FriendList,Tran,DT_List,OTList,num,Rec,transactions,feed,loop,AccSL):
        global ooo
        Km=Kmeans()
        global transactions1
        ooo+=1
        Trustee_List = self.FindTrusteeList(payload, module_List,DT_List)
        #Trust_List is the List of Potential providers along with their predicted trust 
        Trust_List=[]
        for module in module_List:
            key, attributes = next(iter(module.items()))
            if key == message_in.src:
                D1 = attributes
        #devices send request with probability of 0.5
        feat=None
        xx=random.random()
        if xx>0:
            #compute trust level for all devices in trustee list
            for T_Device in Trustee_List:
                if message_in.src!=T_Device:
                    X=self.ComputeTrust(module_name,message_in.src, T_Device, module_List,payload,FriendList,transactions,Rec)
                    for module in module_List:
                        key, attributes = next(iter(module.items()))
                        if key == message_in.src:
                            D1 = attributes
                        elif key == T_Device:
                            D2 = attributes
                    Trust_List.append([D1,D2,X ])
                    #find social freinds of D1
                    if self.compute_social_trust(D1, D2)>0 & (D2['Name'] not in list(FriendList[message_in.src])):
                        if D2['Name'] not in list(FriendList[message_in.src]):
                           FriendList[message_in.src].append(D2['Name'])
             
               
            #befor loop 2 select providers randomly 
            #after that Select Provider with highest trust
            if loop<2:
                a=random.randint(1, len(Trust_List))
                Trustee=Trust_List[a-1][1]
                Trustor=Trust_List[a-1][0]
                features=Trust_List[a-1][2]
                 
            else:    
                best_trustee_index = -1
                highest_trust_level = -1
    
                for index, trustee in enumerate(Trust_List):
                    potential_trustee = trustee[1]  # Get the potential trustee
                    
                    trust_level = trust_to_num[trustee[2][10]]  # Get the trust level value
                      
                    # Check if this trust level is higher than the current highest
                    if trust_level > highest_trust_level:
                        highest_trust_level = trust_level
                        best_trustee_index = index # Store the index of the best trustee
    
                # If There is no any trustable device return failed transaction
                if best_trustee_index == -1:
                    return 'DT',-1
                else:
                    Trustee=Trust_List[best_trustee_index][1]
                    Trustor=Trust_List[best_trustee_index][0]
                    flag=True
                    for i in transactions:
                            if (i[0]==Trustor['Name']) & (i[1]==Trustee['Name']) & (i[2]==payload) :
                                if len(i[8]) > 4:
                                   ttn = [trust_to_num[trust] for trust in i[8]]
                                   #If trust level computed for selected device is less than 0.5
                                   #do not select this providet and return failed transaction
                                   if sum(ttn)/len(i[8])<0.5:
                                       return None
                    
                    
                    features=Trust_List[best_trustee_index][2]
                
            Availability = self.compute_availability(Trustee)
            Responce_Time=self.compute_response_time(Trustee)
            QoS=self.quality_of_service(Trustor,Trustee,FriendList,payload)
            # compute the real trust level
            Trust_LevelR=self.trust_level(Availability,Responce_Time,QoS)
            
            #using te real trust level fot first 100 transaction as learning data
            if loop<2:
                feat=Trust_LevelR
                
            else: 
                #features[10] is the predicted trust level
                feat=features[10]
              
            if loop>1:
                kkk=Km.Kmeans_procedure(Trustor, Trustee,payload,Tran,feed,loop,AccSL) 
            else:
                kkk=1
            if kkk>0:
                #Update tansaction list update Tran and Transactions
                Tran.append([Trustor['Name'], Trustee['Name'],time.time(),features[2],features[3],features[4],features[5],features[6],features[7],features[8],features[9],Trust_LevelR,feat])
        
        
            #Update history list
            flag=True
            for i in transactions:
                    if (i[0]==Trustor['Name']) & (i[1]==Trustee['Name']) & (i[2]==payload) :
                        i[3]=self.compute_social_trust(Trustor,Trustee)
                        i[4]=self.computatinal_trust(Trustee)
                        i[5].append(Availability)
                        i[6].append(Responce_Time)
                        #save the last 10th value for QoS and Trust_LevelR
                        if len(i[7]) < 10:
                           i[7].append(QoS)  # Append to the end
                        else:
                           i[7].pop(0)     # Remove the first element
                           i[7].append(QoS)  # Append to the end
                        
                        if len(i[8]) < 10:
                           i[8].append(Trust_LevelR)  # Append to the end
                        else:
                           i[8].pop(0)     # Remove the first element
                           i[8].append(Trust_LevelR)  # Append to the end
                        flag=False
                        
                        break
            if flag==True: 
                transactions.append([Trustor['Name'], Trustee['Name'],payload,features[3],features[5],[Availability],[Responce_Time],[QoS],[Trust_LevelR]])
                                            
        transactions1=transactions
        return feat
    
    
    

    
    def FindTrusteeList(self, payload, module_list, distrusted_devices):
        """Finds capable devices that can provide the requested service,
        excluding distrusted devices.
    
        Args:
            payload: The service being requested
            module_list: List of device modules with attributes
            distrusted_devices: List of device IDs to exclude
    
        Returns:
            List[str]: List of trusted device IDs that can provide the service
        """
        capable_devices = [
            device_id 
            for module in module_list
            for device_id, attributes in module.items()
            if (device_id.startswith('Device') and 
                'Device_Type' in attributes and
                payload in attributes.get('Device_Services', []))
        ]
          
        return [
                device 
                for device in capable_devices 
                if device not in distrusted_devices
            ]
    
    def ComputeTrust(self, module_name, device1, device2, module_list, service_id, 
                 friend_list, transactions, recommendations):
        """Compute comprehensive trust metrics between two devices.
        
        Args:
            module_name: Name of the module/component
            device1: ID of the first device
            device2: ID of the second device
            module_list: List of device modules with attributes
            service_id: ID of the service being evaluated
            friend_list: Dictionary of device friendships
            transactions: List of historical transactions
            recommendations: List of existing recommendations
            
            
        Returns:
            tuple: Contains all computed trust metrics and levels
        """
        # Extract device attributes
        d1 = next((attrs for module in module_list 
                  for dev, attrs in module.items() if dev == device1), {})
        d2 = next((attrs for module in module_list 
                  for dev, attrs in module.items() if dev == device2), {})
        
        # Compute base trust metrics
        social_trust = self.compute_social_trust(d1, d2)
        centrality = self.compute_centrality(d1, d2, friend_list)
        computational_trust = self.computatinal_trust(d2) * centrality
        availability = self.compute_availability(d2)
        response_time = self.compute_response_time(d2)
        qos = self.quality_of_service(d1, d2, friend_list, service_id)
        recommendation = self.ComputeRecommendation(device1, device2, service_id, 
                                                   friend_list, transactions, recommendations)
        
        # Initialize default values
        metrics = {
            'goodness': 0.5,
            'usefulness': 0.5,
            'perseverance': 0.5,
            'fluctuation': 0.5,
            'trust_level_p': 'HT'  # Default high trust
        }
        
        # Check for matching transaction history
        for transaction in transactions:
            if (transaction[0] == device1 and 
                transaction[1] == device2 and 
                transaction[2] == service_id):
                
                trust_values = transaction[8]
                metrics.update({
                    'goodness': self.compute_goodness(trust_values),
                    'usefulness': (self.compute_usefulness(trust_values) 
                                  if len(trust_values) > 5 else metrics['goodness']),
                    'perseverance': self.compute_perseverance(trust_values),
                    'fluctuation': self.compute_fluctuation(trust_values)
                     })
                
                # Try to load ML model if available
                model_num = re.search(r'\d+', module_name).group()
                model_file = f"model{model_num}.json"
                weights_file = f"model{model_num}.h5"
                
                if os.path.isfile(model_file):
                    with open(model_file, 'r') as f:
                        model = model_from_json(f.read())
                    model.load_weights(weights_file)
                    model.compile(optimizer='adam', 
                                loss='categorical_crossentropy', 
                                metrics=['accuracy'])
                    
                    sample = [
                        centrality, social_trust, recommendation,
                        computational_trust, metrics['goodness'],
                        metrics['usefulness'], metrics['perseverance'],
                        metrics['fluctuation']
                    ]
                                       
                    trust_labels = {0: 'HT', 1: 'T', 2: 'UT', 3: 'DT'}
                    predictions = model.predict(np.array([sample]), verbose=0)
                    prediction = np.argmax(predictions, axis=1)[0]
                    metrics['trust_level_p'] = trust_labels.get(prediction, 'HT')
                    
                else:
                    metrics['trust_level_p'] = self.trust_level(
                        availability, response_time, qos)
                break
        
        # Update recommendation weights
        self.compute_recommendation_weight(
            device1, device2, friend_list, metrics['trust_level_p'], recommendations)
        
        return (
            device1, 
            device2,
            centrality,
            social_trust,
            recommendation,
            computational_trust,
            metrics['goodness'],
            metrics['usefulness'],
            metrics['perseverance'],
            metrics['fluctuation'],
            metrics['trust_level_p']
        )

        
    def compute_social_trust(self, device1, device2):
        """Calculate social trust score between two devices based on:
           - Same user (1.0)
           - Proximity < 10 units (0.9)
           - Friends (0.6)
           - Same device brand/model (0.5)
           - no relaton (0.0)
        """
        if device1['User_ID'] == device2['User_ID']:
            return 1.0
            
        distance = math.dist(device1['Position'], device2['Position'])
        if distance < 10:
            return 0.9
            
        if device1['User_ID'] in device2['User_Friend']:
            return 0.6
            
        if (device1['Device_Brand'] == device2['Device_Brand'] and 
            device1['Device_Model'] == device2['Device_Model']):
            return 0.5
            
        return 0.0
    
    def compute_centrality(self, Device1, Device2,FriendList):
        if len(FriendList[Device1['Name']])>0:
            return len(intersection(FriendList[Device1['Name']], FriendList[Device2['Name']]))/len(FriendList[Device1['Name']])
        else:
            return 0 
            
    def ComputeRecommendation(self, device1, device2,service_id,FriendList,transactions,recommendations):
        """Compute a recommendation score between two devices based on:
            - Friend relationships
            - Historical transaction trust values
            - Existing recommendation weights
        
        Args:
            device1: Source device ID
            device2: Target device ID
            service_id: Service identifier
            friend_list: Dictionary of device friends
            transactions: List of historical transactions
            recommendations: Existing recommendation records
        
        Returns:
            float: Recommendation score (0.5 if no data available)
        """
        # Get friends of device1
        friends = FriendList.get(device1, [])
        
        # Calculate trust scores from relevant transactions
        trust_scores = []
        
        for friend in friends:
            for transaction in transactions:
                # Check transaction matches friend->device2 with trust values
                if (transaction[0] == friend and 
                    transaction[1] == device2 and 
                    len(transaction[8]) > 0):
                    
                    # Convert trust values to numerical scores
                    trust_values = [trust_to_num[item] for item in transaction[8]]
                    avg_trust = sum(trust_values) / len(trust_values)
                    
                    # Find or create recommendation record
                    rec = next(
                        (r for r in recommendations 
                         if r[0] == device1 and r[1] == device2 and r[2] == friend),
                        None
                    )
                    
                    if rec:
                        # Update existing recommendation
                        rec[4] = avg_trust
                        weight = sum(rec[5])/len(rec[5]) if rec[5] else 1
                    else:
                        # Create new recommendation
                        rec = [device1, device2, friend, service_id, avg_trust, []]
                        recommendations.append(rec)
                        weight = 1
                    
                    trust_scores.append(avg_trust * weight)
        
        # Return average score or default 0.5 if no data
        return sum(trust_scores)/len(trust_scores) if trust_scores else 0.5      
        
    
    def compute_recommendation_weight(self, device1, device2, friend_list, trust_level, recommendations):
        """Updates recommendation weights based on trust level comparison.
        
        Args:
            device1: Source device ID
            device2: Target device ID
            friend_list: Dictionary mapping devices to their friends
            trust_level: Current trust level to compare against
            recommendations: List of existing recommendations to update
        """
        # Get friends of device1
        friends = friend_list.get(device1, [])
        
        # Convert trust level to numerical value
        current_trust = trust_to_num[trust_level]
        
        # Update matching recommendations
        for friend in friends:
            for rec in recommendations:
                if (rec[0] == device1 and 
                    rec[1] == device2 and 
                    rec[2] == friend):
                    
                    # Append 1 if trust difference < 0.5, else 0
                    rec[5].append(1 if abs(rec[4] - current_trust) < 0.5 else 0)
                   
        
    
    def computatinal_trust(self, device):
        """Calculate computational trust score based on device capability.
        
        Args:
            device: Device profile containing 'ComputatinalC' field
            
        Returns:
            float: Trust score (0, 0.5, or 1)
        """
        capability = device['ComputatinalC']
        return {
            1: 0.0,
            2: 0.5,
            3: 1.0
        }.get(capability, 0.0)

    
    def compute_availability(self, device):
        """Determine if device is available based on capability and random chance.
        
        Args:
            device: Device profile containing 'ComputatinalC' field
            
        Returns:
            int: 1 if available, 0 otherwise
        """
        capability = device['ComputatinalC']
        availability_prob = {
            1: 0.75,
            2: 0.85,
            3: 0.98
        }.get(capability, 0.0)
        
        return 1 if random.random() < availability_prob else 0    
        
    
    
    def compute_response_time(self, device):
        """Calculate response time quality score based on device capability.
        
        Args:
            device: Device dictionary containing 'ComputatinalC' field
            
        Returns:
            float: Normalized response time score (1.0 = best, 0.0 = worst)
        """
        capability = device['ComputatinalC']
        time_ranges = {
            3: (0.001, 0.3),
            2: (0.3, 0.7),
            1: (0.7, 1.0)
        }
        
        min_time, max_time = time_ranges.get(capability, (1.0, 1.0))
        return 1 - random.uniform(min_time, max_time)
        
    
    def quality_of_service(self, device1, device2, friend_list, service_id):
        """Determine quality of service based on device attack type and relationships.
        
        Args:
            device1: Requesting device dictionary
            device2: Target device dictionary
            friend_list: Dictionary mapping device names to friends
            service_id: Service identifier 
            
            
        Returns:
            int: 1 if good service quality, 0 otherwise
        """
        attack_type = device2['Attack_Type']
        
        # No attack and recommendation attack types
        if attack_type in {0, 2}:
            return 1
            
        # ME and WA unsafe attack types
        if attack_type in {'ME', 'WA'}:
            return 0
            
        # DA attack case
        if attack_type == 'DA':
            return int(device2['Name'] in friend_list.get(device1['Name'], set()))
            
        # OOA attack case
        if attack_type == 'OOA':
            return int(random.random() > 0.5)
            
        # OSA attack case
        if attack_type == 'OSA':
            reputation = self.compute_reputation(device2, service_id)
            return int(reputation <= 0.5)
            
        # Mixed attack case
        if attack_type == 'Mix':
            is_friend = device2['Name'] in friend_list.get(device1['Name'], set())
            random_chance = random.random() > 0.5
            reputation = self.compute_reputation(device2, service_id)
            return int(is_friend or random_chance or reputation <= 0.5)
            
        # Default case 
        return 0


    
    def ComputeReputation(self, device_2, service_id):
        """
        Calculate the reputation score for a device providing a specific service.
        
        Args:
            device_2 (dict): Target device dictionary containing 'Name' key
            service_id (str/int): Identifier for the specific service
            
        Returns:
            float: Reputation score between 0-1, returns 0.5 (neutral) if no transactions found
        """
        total_rep = 0.0
        transaction_count = 0
        
        for transaction in self.transactions1:
            # Check if transaction matches device and service
            if (transaction[1] == device_2['Name']) and (transaction[2] == service_id):
                # Only process if trust evaluation exists
                if transaction[8]:  # More pythonic than len() > 0
                    total_rep += self.trust_to_num[transaction[8][-1]]
                    transaction_count += 1
                    
        return total_rep / transaction_count if transaction_count > 0 else 0.5    
            
        
    
    
    def trust_level(self, availability, response, qos):
        """Determine trust level based on quality metrics.
        
        Args:
            availability (int): device availability (0 or 1)
            response (float): Response quality score (0-1)
            qos (int): Quality of service indicator (0 or 1)
        
        Returns:
            str: Trust level as one of:
                "DT" - Distrusted (when QoS is 0)
                "UT" - Untrusted (when unavailable or response ≤ 0.3)
                "T"  - Trusted (0.3 < response ≤ 0.7)
                "HT" - Highly Trusted (response > 0.7)
        """
        if not qos:
            return "DT"
        if not availability:
            return "UT"
        if response > 0.7:
            return "HT"
        if response > 0.3:
            return "T"
        return "UT"    
                
    def compute_goodness (self,trustL):
        sum=0
        for i in trustL:    
           if (i=="HT") | (i=="T"):
               sum+=1 
        if len(trustL)>0:
            return(sum/len(trustL))
        else:
            return(1)
    
    def compute_usefulness(self, trustL):
        if len(trustL)>0:
            usefulness = [trust_to_num[trust] for trust in trustL]
            Ls = 5
            p = 0.4
            Eres = sum([p * p**r for r in range(Ls+1)])
            u = sum([(p * p**l + Eres / Ls) * usefulness[-l] for l in range(1, Ls+1)])
            return u
        else:
            return(1)
    
    def compute_perseverance(self, trustL):
        if len(trustL)>0:
            f = [1 if i in ["HT", "T"] else 0 for i in trustL]
            u = 0.1
            d = 0.2
            p = 0.5
            for i in range(len(f)-1):
                if f[i] == f[i+1]:
                    if f[i] == 0:
                        p = max(0, p - d)
                    else:
                        p = min(1, p + u)
            return p
        else:
            return(1)
    
    def compute_fluctuation(self, trustL):
        trus=[]
        for i in range(0, len(trustL)):
            if (trustL[i]=='HT')|(trustL[i]=='T'):
                trus.append(1)
            else:
                trus.append(0)
        if len(trus)>0:
            p = sum(a != b for a, b in zip(trus, trus[1:]))
            return 1 - p / len(trus)
        else:
            return(1)
    
    def compute_average(self,lst):
        if len(lst)<1:
            return 0.0
        return sum(lst) / len(lst)
    
   
    
        