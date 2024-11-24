

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
# In[73]:

num_channels = 3
max_sequence_length = 1000  # Set maximum sequence length

def dataPrep(folderPath):
    # Get a list of all .npy files in the folder
    file_list = [file for file in os.listdir(folderPath) if file.endswith('.npy')]

    print(len(file_list))
    # Initialize lists to store padded data
    padded_data_list = []

    # Process each .npy file
    for file_name in file_list:
        # Load data from .npy file
        data = np.load(os.path.join(folder_pathTrain, file_name))

        # Truncate or pad sequences to max_sequence_length
        truncated_data = [seq[:max_sequence_length] for seq in data]
        padded_data = pad_sequences(truncated_data, maxlen=max_sequence_length, padding='pre', dtype='float32')

        padded_data_list.append(padded_data)

    # Concatenate data from all files
    for i in range(142):
        padded_data_list.append(padded_data_list[0])
    padded_data_all = np.stack(padded_data_list, axis=0) #1136

    # Swap axes to match model input shape (None, 1000, 3)
    padded_data_all = np.swapaxes(padded_data_all, 1, 2)
    return padded_data_all


# In[74]:


folder_pathTrain="C:\\Users\\91629\\Desktop\\MF_train_Data"
test_data = dataPrep("C:\\Users\\91629\\Desktop\\MF_train_Data")


# In[75]:


print(test_data.shape)


# In[76]:


labels=pd.read_pickle("C:\\Users\\91629\\Downloads\\train_data_mf2_label.pkl")


# In[77]:


print(test_data.shape)
print(labels.shape)


# In[78]:


model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(max_sequence_length, num_channels)),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Assuming binary classification, adjust output units for multiclass
])

model.add(Flatten())
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(test_data, labels, epochs=10, batch_size=32, validation_split=0.2)


# In[79]:


num_channels = 3
max_sequence_length = 1000  # Set maximum sequence length

def dataPrep1(folderPath):
    # Get a list of all .npy files in the folder
    file_list = [file for file in os.listdir(folderPath) if file.endswith('.npy')]

    #print(len(file_list))
    # Initialize lists to store padded data
    padded_data_list = []

    # Process each .npy file
    for file_name in file_list:
        # Load data from .npy file
        data = np.load(os.path.join(folder_pathTest, file_name))

        # Truncate or pad sequences to max_sequence_length
        truncated_data = [seq[:max_sequence_length] for seq in data]
        padded_data = pad_sequences(truncated_data, maxlen=max_sequence_length, padding='pre', dtype='float32')

        padded_data_list.append(padded_data)
    padded_data_list.append(padded_data_list[0]) 
    # Concatenate data from all files
    padded_data_all = np.stack(padded_data_list, axis=0) #1136

    # Swap axes to match model input shape (None, 1000, 3)
    padded_data_all = np.swapaxes(padded_data_all, 1, 2)
    return padded_data_all


# In[80]:


folder_pathTest="C:\\Users\\91629\\Desktop\\MF_test_Data"
t_data=dataPrep1(folder_pathTest)


# In[81]:


label1=pd.read_pickle("C:\\Users\\91629\\Downloads\\test_data_mf2_label.pkl")


# In[100]:


s=model.predict(t_data)


# In[101]:


r=model.predict(test_data)


# In[168]:


train_acc = model.evaluate(t_data,label1, verbose=0)
test_acc = model.evaluate(test_data,labels, verbose=0)