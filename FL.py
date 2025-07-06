
from lightgbm.sklearn import LGBMRegressor
from sklearn.metrics import f1_score
import logging
from tensorflow.keras.models import Sequential, model_from_json
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import numpy as np
import tensorflow.keras as keras
from keras.utils import to_categorical
from tensorflow.keras import backend as K
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
import csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Only show errors
import random
import re
from itertools import chain
import warnings
warnings.filterwarnings(
    "ignore", message="The load function will merged with load_all_node_attr function")
warnings.filterwarnings(
    "ignore", message="Your TensorFlow version is newer than 2.4.0 and so graph support has been removed in eager mode and some static graphs may not be supported.")
warnings.filterwarnings("ignore", message="`tf.keras.backend.set_learning_phase` is deprecated and will be removed after 2020-10-11. To update it, simply pass a True/False value to the `training` argument of the `__call__` method of your layer or model.")
# from imutils import paths




# Disable TensorFlow and Flwr logs
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("flwr").setLevel(logging.ERROR)
accuracy_list = []
ACC = [[] for _ in range(4)]
listL = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
comms_round=0

class FL:

    def FL_procedure(self):
        global listL
        global comms_round
        
        with open('lists.csv', 'r', errors='ignore') as file:
            reader = csv.reader(file)
            data = pd.DataFrame(reader)
        List = data.values
        X_np = []

        label = []
        W=[]
        ii = 0
        for List1 in List:
            y = []
            ii += 1
            for x in List1:
                if not isinstance(x, str):
                    continue
                x = x.replace("nan", "''")  # replace nan with empty string
                x = eval(x)  # evaluate string as Python expression
                y.append(x)
            X = np.array(y).reshape((-1, len(y[0])))  # change list to np.array
            last_column = X[:, -2]
            label.append(np.array(last_column))
            last_column = X[:, 3]
            W.append(np.array(last_column))
            X = np.delete(X, -1, axis=1)
            X = np.delete(X, -1, axis=1)
            X = np.delete(X, 0, axis=1)
            X = np.delete(X, 0, axis=1)
            X = np.delete(X, 0, axis=1)
            X_np.append(np.array(X, dtype=float))

        print("*****enter FL *****")
        X_test = []
        y_test = []
        X_np1 = X_np
        label1 = label
        
        
        #delete last transactions from X_np 
        for i in range(len(X_np)):
            X_np[i]=X_np[i][:100+listL[i]:]
            label[i]=label[i][:100+listL[i]:]
        
        
        for i in range(len(X_np)):
            listL[i] += len(X_np[i])
            X_train1, X_test1, y_train1, y_test1 = train_test_split(X_np[i],
                                                                    label[i],
                                                                    test_size=0.3,
                                                                    random_state=42)
            X_np1[i] = X_train1
            label1[i] = y_train1
            X_test.extend(X_test1)
            y_test.extend(y_test1)
            
        # create clients
        clients = self.create_clients(
            X_np1, label1, num_clients=len(X_np), initial='client')
        # process and batch the training data for each client
        clients_batched = dict()
        for (client_name, data) in clients.items():
            clients_batched[client_name] = self.batch_data(data)
        # process and batch the test set
        test_batched = tf.data.Dataset.from_tensor_slices(
            (X_test, y_test)).batch(len(y_test))

        lr = 0.01
        comms_round += 1
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
        NoFeatures = len(X_np[0][0])
        optimizer = tf.keras.optimizers.legacy.SGD(
            learning_rate=lr, decay=lr / comms_round, momentum=0.9)
        
        if os.path.exists('global_model.json'):
            json_file = open('global_model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("global_model.h5")
            global_model = loaded_model
        else:
            # initialize global model
            smlp_global = self.MLP()
            global_model = smlp_global.build(NoFeatures, 4)

        # commence global training loop
        
        global_weights = global_model.get_weights()

        # initial list to collect local model weights after scalling
        scaled_local_weight_list = list()

        # randomize client data - using keys
        client_names = list(clients_batched.keys())
        random.shuffle(client_names)
        i = 0
        # loop through each client and create new local model
        for client in client_names:
            
            smlp_local = self.MLP()
            local_model = smlp_local.build(NoFeatures, 4)
            local_model.compile(loss=loss, 
                          optimizer=optimizer, 
                          metrics=metrics)
            # set local model weight to the weight of the global model
            local_model.set_weights(global_weights)

            # fit local model with client's data
            local_model.fit(clients_batched[client],epochs=100, verbose=0)

            # scale the model weights and add to list
            scaling_factor = self.weight_scalling_factor(
                clients_batched, client)
            scaled_weights = self.scale_model_weights(
                local_model.get_weights(), scaling_factor)
            scaled_local_weight_list.append(scaled_weights)

            match = re.search(r'\d+', client)
            x = int(match.group())
            model_json = local_model.to_json()
            with open(f"model{x}.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            local_model.save_weights(f"model{x}.h5")
            # clear session to free memory after each communication round
            K.clear_session()
        # to get the average over all the local model, we simply take the sum of the scaled weights
        average_weights = self.sum_scaled_weights(scaled_local_weight_list)

        # update global model
        global_model.set_weights(average_weights)
        
        for (X_test, Y_test) in test_batched:
            global_acc, global_loss = self.test_model(
                X_test, Y_test, global_model, comms_round)    
            # get the global model's weights - will serve as the initial weights for all local models
            

        if global_model is not None:
            global_model_json = global_model.to_json()
            with open("global_model.json", "w") as json_file:
                json_file.write(global_model_json)
            global_model.save_weights("global_model.h5")
            
            
        dataset = np.array(list(chain.from_iterable(X_np)))
        labelT = np.array(list(chain.from_iterable(label)))    
          
        
        with open("globalFL.csv", "w") as f:
          writer = csv.writer(f, lineterminator='\n')
          writer.writerows(accuracy_list)

    def load_data(self, dataset, label):

        L = ['HT', 'T', 'UT', 'DT']
        # Count the occurrences of each label in y_train1
        label_counts = {}
        # Count the occurrences of each label in y_train1
        for i in label:
            if i in label_counts:
                label_counts[i] += 1
            else:
                label_counts[i] = 1

        less_than_two_labels = [x for x in L if label_counts.get(x, 0) < 2]
        
        for ii in less_than_two_labels:
            if ii == 'DT':
                dataset = np.vstack([dataset, dataset[0]])
                label = np.append(label, ['DT'])
                dataset = np.vstack([dataset, dataset[0]])
                label = np.append(label, ['DT'])
            elif ii == 'T':
                dataset = np.vstack([dataset, dataset[0]])
                label = np.append(label, ['T'])
                dataset = np.vstack([dataset, dataset[0]])
                label = np.append(label, ['T'])
            elif ii == 'HT':
                dataset = np.vstack([dataset, dataset[0]])
                label = np.append(label, ['HT'])
                dataset = np.vstack([dataset, dataset[0]])
                label = np.append(label, ['HT'])
            elif ii == 'UT':
                dataset = np.vstack([dataset, dataset[0]])
                label = np.append(label, ['UT'])
                dataset = np.vstack([dataset, dataset[0]])
                label = np.append(label, ['UT'])

        # split into train test sets
        x_train, x_test, y_train1, y_test1 = train_test_split(
            dataset, label, test_size=0.3, random_state=42, shuffle=True, stratify=label)
        label_map = {'HT': 0, 'T': 1, 'UT': 2, 'DT': 3}

        y_train = np.array([label_map[L] for L in y_train1], dtype=int)
        y_test = np.array([label_map[L] for L in y_test1], dtype=int)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        return (x_train, y_train, x_test, y_test)

    def create_clients(self, X_np, label, num_clients, initial='clients'):
        
        # create a list of client names
        client_names = ['{}_{}'.format(initial, i+1)
                        for i in range(num_clients)]
        shards = []
        for j in range(len(X_np)):
            x_train, y_train, x_test, y_test = self.load_data(X_np[j], label[j])

            data = list(zip(x_train, y_train))
            random.shuffle(data)
            shards.append(data)
            
        assert (len(shards) == len(client_names))

        return {client_names[i]: shards[i] for i in range(len(client_names))}

    def batch_data(self, data_shard, bs=32):
        data, label = zip(*data_shard)
        dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
        return dataset.shuffle(len(label)).batch(bs)
    
    class MLP:
        @staticmethod
        def build(shape, classes):
            model = Sequential()
            model.add(Dense(32, input_shape=(shape,)))
            model.add(Activation("relu"))
            model.add(Dense(16))
            model.add(Activation("relu"))
            model.add(Dense(classes))
            model.add(Activation("softmax"))
            return model
   
       
        
        
    def weight_scalling_factor(self, clients_trn_data, client_name):
        client_names = list(clients_trn_data.keys())
        # get the bs
        bs = list(clients_trn_data[client_name])[0][0].shape[0]
        # first calculate the total training data points across clinets
        global_count = sum([tf.data.experimental.cardinality(
            clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
        # get the total number of data points held by a client
        local_count = tf.data.experimental.cardinality(
            clients_trn_data[client_name]).numpy()*bs
        return local_count/global_count

    def scale_model_weights(self, weight, scalar):
        '''function for scaling a models weights'''
        weight_final = []
        steps = len(weight)
        for i in range(steps):
            weight_final.append(scalar * weight[i])
        return weight_final

    def sum_scaled_weights(self, scaled_weight_list):
        '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
        avg_grad = list()
        # get the average grad accross all client gradients
        for grad_list_tuple in zip(*scaled_weight_list):
            layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
            avg_grad.append(layer_mean)

        return avg_grad

    def test_model(self, X_test, Y_test,  model, comm_round):
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
        logits = model.predict(X_test)
        for i in logits:
            predicted_label = np.argmax(i)

            for j in range(4):
                i[j] = 0
            i[predicted_label] = 1

        y = []
        for i in range(len(Y_test)):
            if Y_test[i] == "HT":
                predicted_Lable = 0
            elif Y_test[i] == "T":
                predicted_Lable = 1
            elif Y_test[i] == "UT":
                predicted_Lable = 2
            elif Y_test[i] == "DT":
                predicted_Lable = 3
            y.append(predicted_Lable)
        y = to_categorical(y)
        loss = loss_fn(logits, y)
        acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(y, axis=1))
        print("Metrics for global FL:")
        print('comm_round: {} | global_acc: {:.3%} '.format(
            comm_round, acc))

        accuracy = accuracy_score(logits, y)
        print('Accuracy: %f' % accuracy)
        precision = precision_score(logits, y, average='weighted')
        f1 = f1_score(logits, y, average='weighted')
        print('F1 score: %f' % f1)
        loss1 = 1-accuracy
        recall = recall_score(
            logits, y, average='weighted', labels=np.unique(y))
        accuracy_list.append([loss1,accuracy,f1])
        return acc, loss
