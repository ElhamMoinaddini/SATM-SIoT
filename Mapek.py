
import numpy as np
from collections import defaultdict
OTList = []
class Mapek:

    def Mapek_procedure(self,transactions,DT_List,OT):
        
        print("*************Enter Mapek*************")
        global OTList 
        
        
        # Mapping of feedback values
        feedback_mapping = {
            1: 'HT',
            0.75: 'T',
            0.25: 'UT',
            0: 'DT'
        }
        
        
       # Convert transactions to feedback_data
        feedback_data = []

        for transaction in transactions:
            device_a = transaction[0]
            device_b = transaction[1]
            x=transaction[-1]
            feedback_types = x[-1]  # Access the last element for feedback types
            
            # Loop through feedback types and add entries
            #for feedback_type in feedback_types:
            for key, value in feedback_mapping.items():
                    if feedback_types == value:
                        feedback_data.append((device_a, device_b, key))
                      
        # Step 1: Monitor
        feedback_aggregation = defaultdict(lambda: defaultdict(list))

        for trustor, trustee, satisfaction in feedback_data:
            feedback_aggregation[trustee][trustor].append(satisfaction)

        

        # Step 2: Calculate Average Feedback
        average_feedback = defaultdict(dict)

        for trustee, feedbacks in feedback_aggregation.items():
            for trustor, values in feedbacks.items():
                if values:
                    average_feedback[trustee][trustor] = np.mean(values)

        

        # Step 3: Analyze and Filter Outliers from Average Feedback
        

        for trustee, feedbacks in average_feedback.items():
            values = list(feedbacks.values())
            if len(values) < 4:  # Not enough data to calculate IQR
                continue
                
            # Calculate IQR and bounds
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Filter out values that are outside the bounds
            filtered_values = {}
            for k, v in feedbacks.items():
                if lower_bound <= v <= upper_bound:
                    filtered_values[k] = v
                else:
                    # If the value is an outlier, add the trustor to the OTList
                     OTList.append(k)


            average_feedback[trustee] = filtered_values

        # Step 4: Compute Hostility
        hostility_counts = {}
        total_feedback_counts = {}

        for trustee, feedbacks in average_feedback.items():
            dissatisfaction_count = sum(1 for v in feedbacks.values() if v == 0)
            total_feedback_count = len(feedbacks)
            
            hostility_counts[trustee] = dissatisfaction_count
            total_feedback_counts[trustee] = total_feedback_count

        # Calculate the dynamic threshold based on the total counts
        total_dissatisfaction = sum(hostility_counts.values())
        total_feedback = sum(total_feedback_counts.values())
        threshold = (total_dissatisfaction / total_feedback) if total_feedback > 0 else 0
        minTh = 0.1
        maxTh = 0.9

        Th = maxTh - (threshold * (maxTh - minTh))

        # Step 5: Identify Malicious Devices
        malicious_devices = [device for device, count in hostility_counts.items() if (count/total_feedback_counts[device] > Th)]
        """
        for device, count in hostility_counts.items():
            if device not in (DT_List):
                print(device, count,Th , total_feedback_counts[device])
        """
        for j in malicious_devices:
                if j not in (DT_List):
                    DT_List.append(j)
                    
        for item in set(OTList):  # Use set to avoid counting duplicates
            count = OTList.count(item)
            if count > 5 and item not in OT:
                OT.append(item)
                        
       
       