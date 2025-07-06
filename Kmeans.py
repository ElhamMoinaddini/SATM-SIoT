# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 14:01:14 2025

@author: charsoo
"""
import random
import os
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*KMeans is known to have a memory leak.*")
import math
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import csv
import numpy as np
import time
import joblib
feedback=[]
result=[]
counter=0
class Kmeans:
    
    def Kmeans_procedure(self, req, pro,payload,Tran,feed,loop,AccSL):
        global counter
        global feedback
        global result
        t=0
        f=1
        res='non'
        if len(Tran)>0:
            for i in Tran:
                if (i[0]==pro['Name']) & (i[1]==req['Name'])  :
                    t=Tran[0][2]
            #If time interval between transactions is less than 0.2, generate negetive feedback        
            if time.time()-t<0.2:
                f=0
                res='time'
            #If requster wants to access data out its role access, generate negetive feedback      
            if (req['User_Role']==2 )&(payload>12):
                f=0
                res='Role'
            if (req['User_Role']==3 )&(payload>10):
                f=0 
                res='Role'
            if (req['User_Role']==4 )&(payload>5):
                f=0 
                res='Role'
        a=0.5
        
        Role=req['User_Role']#Device Role
        Type=req['Device_Type']#Device_Type
        Comput=req['ComputatinalC']#device omputatinal Capability
        SocialR=self.ComputeSocialTrust(req,pro)#socila relationship between devices
        
        if len(Tran)>0:
            sum1=0
            num=0
            Tr=0
            
            for i in feed:
                    if i[1]==req['Name']  :
                        if len(i[3])>0:
                            sum1+=i[3][-1]
                            num+=1
                       
            if num>3:
                a=sum1/num            
            else:
                if len(feedback)>10:
                    
                    a=self.kmean(feedback,[Role,Type, Comput, SocialR])
                    
            #If requster is bejaving maliciously, generate negetive feedback  
            if(req['Attack_Type'] == 2) :
                f=0
                res='attack'  
                   
            flag=True
            #Update feed list
            for i in feed:
                    if (i[0]==pro['Name']) & (i[1]==req['Name'])  :
                        i[2].append(f)
                        #Compute Trust
                        Tr=self.ComputeTr(i[2],a)
                        i[3].append(Tr)
                        i[4].append(res)
                        i[5]=req['Attack_Type']
                        i[6].append(a)
                        feedback.append([Role,Type, Comput, SocialR,Tr])
                        flag=False
                        break
            if flag==True: 
                feed.append([pro['Name'], req['Name'],[f],[a],[res],req['Attack_Type'],[a]])
                feedback.append([Role,Type, Comput, SocialR,a])
            counter+=1 
        
        x = Tr if Tr > 0 else a
        p = 1 if x > 0.5 else 0
        r = 0 if req['Attack_Type'] == 2 else 1  
        AccSL.append([p,r])
        
        if f == 0:
            return 0
        return Tr if Tr > 0 else a
            
    
 
    
    def kmean(self, feedback,x):
        #if (len(feedback)<12)|(len(feedback)mod 10==0):
        data = np.array(feedback)

        # Separate features and trust values
        features = data[:, :4]  # Role, Type, Comput, SocialR
        trust_values = data[:, 4]  # Tr

        # Normalize the features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        # Elbow Method
        wcss = []
        for i in range(1, 11):  # Test for 1 to 10 clusters
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(features_normalized)
            wcss.append(kmeans.inertia_)
        
        
        
        # Function to determine the best number of clusters
        def find_best_cluster(wcss):
            # Calculate the difference between consecutive WCSS values
            diffs = np.diff(wcss)
            # Calculate the second derivative to find the elbow
            second_diffs = np.diff(diffs)
            # The elbow point is where the second derivative changes sign
            elbow_index = np.argmin(second_diffs) + 1  # +1 because of np.diff
            return elbow_index + 1  # +1 to match the cluster count
        
        # Get the best cluster number
        best_cluster_number = find_best_cluster(wcss)
        #print("Best number of clusters:", best_cluster_number)
        
        # Train the K-means model
        num_clusters = best_cluster_number
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(features_normalized)

        # Get the cluster labels
        labels = kmeans.labels_

        # Calculate average trust values for each cluster
        average_trust = []
        for i in range(num_clusters):
            cluster_trust = trust_values[labels == i]
            average_trust.append(np.mean(cluster_trust))

        # Function to find the nearest cluster and return its average trust
        def predict_trust(new_data):
            new_data_normalized = scaler.transform([new_data])
            nearest_cluster_index = kmeans.predict(new_data_normalized)[0]
            return average_trust[nearest_cluster_index]

        predicted_trust = predict_trust(x)

        return predicted_trust
        
        
    def ComputeTr(self, f,a):
        r=f.count(1)
        s=f.count(0)
        W=1
        b=r/(r+s+W)
        u=W/(r+s+W)
        return b+u*a
            
            
    def ComputeSocialTrust(self, Device1, Device2):
         if Device1['User_ID'] == Device2['User_ID']:
             return 1
         else:
             a = Device1['Position']
             b = Device2['Position']
             distance = math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
             if distance < 10:
                 return 0.9
             else:
                 if Device1['User_ID'] in Device2['User_Friend']:
                     
                    return 0.6
                 else:
                     if Device1['Device_Brand'] == Device2['Device_Brand'] & Device1['Device_Model'] == Device2['Device_Model']:
                         return 0.5
                
         return 0   