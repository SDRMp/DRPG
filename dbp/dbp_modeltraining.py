#!/usr/bin/env python
# coding: utf-8

# In[1]:


dstype = 'dbp' 
mname = 'debertaV3'


# In[2]:


import pandas as pd


# In[3]:


# %%
# %%
import torch
import torchvision

# Check PyTorch version
print("PyTorch version:", torch.__version__)

# Check if CUDA is available (GPU support)
print("CUDA available:", torch.cuda.is_available())

# Check the number of GPUs
print("Number of GPUs:", torch.cuda.device_count())



# %%-----------------------------------++++++++++++++++++++++++++---------------------------------------

# %%
# %%
modelpath = 'microsoft/deberta-v3-base'
# modelpath = "bert-base-uncased"


datapath = None
saveDIR = f"/home/bhairavi/om/om5/{dstype}/{mname}_{dstype}"
print(saveDIR)
# %%


# In[4]:


modelpath  = saveDIR


# In[5]:


# %%
from datasets import load_dataset

dataset = load_dataset("DeveloperOats/DBPedia_Classes", name='default' )


# %%
dataset


# In[6]:


dataset


# In[7]:


train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])
validation_df = pd.DataFrame(dataset['validation'])


# In[8]:


train_df['split'] = 'train'

test_df['split'] = 'test'

validation_df['split'] = 'validation'


# In[9]:


df = pd.concat([train_df, test_df, validation_df], ignore_index=True)


# In[10]:


df.shape


# In[11]:


df.columns


# In[12]:


# %%
df['l3'].nunique()

# %%


# In[13]:


# %%
df['label'] = df['l3']


# In[14]:


# %%
from sklearn.preprocessing import LabelEncoder

# %%
le = LabelEncoder()
df['target'] = le.fit_transform(df['label'])


# In[15]:


# %%

# %%
df.columns

# %%


# In[16]:


df.shape

# %%


# In[17]:


import os
import torch 
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

torch.cuda.empty_cache() 

import pandas as pd
from datasets import Dataset, DatasetDict, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
 

# %%
numlabel = df['target'].nunique()
numlabel


# In[18]:


# %%

# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda"  # the device to load the model onto

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(modelpath)
model = AutoModelForSequenceClassification.from_pretrained(modelpath, num_labels=numlabel)

# Move the model to the specified device
model.to(device)


# In[19]:


# %%
# df['token_length'] = df['text'].apply(lambda x: len(tokenizer.tokenize(x)))

# # Calculate the maximum token length
# max_length = df['token_length'].max()
# Calculate the token length using the split method
df['token_length'] = df['text'].apply(lambda x: len(x.split()))

# Calculate the maximum token length
max_length = df['token_length'].max()



# In[20]:


max_length


# In[21]:


# Calculate the next maximum token length
next_max_token_length = df['token_length'].nlargest(2).iloc[1] 


# Calculate the average token length
average_token_length = df['token_length'].mean()

# Display the results
print(f"Maximum token length: {max_length}")
print(f"Next maximum token length: {next_max_token_length}") 
print(f"Average token length: {average_token_length:.2f}")

# %%
min(df['token_length'])


# In[22]:


# %%
fdf = df[df['token_length'] == 5]

# %%
fdf

# %%
df = df[df['token_length'] >= 5]

# %%
df.shape
 


# In[23]:


df.columns


# In[24]:


train_df = df[df['split'] == 'train'].drop(columns=['split'])

test_df = df[df['split'] == 'test'].drop(columns=['split'])

val_df = df[df['split'] == 'validation'].drop(columns=['split'])


# In[25]:


train_df.shape, test_df.shape, val_df.shape


# In[26]:


def tokenize_and_format(examples):
    # Tokenize the texts
    tokenized_inputs = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=max_length)
    tokenized_inputs['label'] = list(map(int, examples['target']))
    return tokenized_inputs





# %%
# Convert pandas DataFrame to Hugging Face's Dataset
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(val_df) 
test_dataset = Dataset.from_pandas(test_df)

# Map the tokenization function across the datasets
train_dataset = train_dataset.map(tokenize_and_format, batched=True,batch_size=16)
eval_dataset = eval_dataset.map(tokenize_and_format, batched=True,batch_size=16) 
test_dataset = test_dataset.map(tokenize_and_format, batched=True,batch_size=16)


# %%
 

# %%
from sklearn.metrics import f1_score, precision_score, recall_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    return {
        'eval_f1': f1,
        'eval_precision': precision,
        'eval_recall': recall,
    }



# In[27]:


test_df = test_df.reset_index(drop=True)


# In[28]:


df


# In[29]:


# import pandas as pd
# import numpy as np
# import submodlib
# from datasets import Dataset 


# def select_representative_samples(tokenized_data, num_samples_per_class):
#     selected_indices = []
     
#     for class_label in set(tokenized_data['label']):
#         print(f"Processing class: {class_label}")
#         class_indices = [i for i, label in enumerate(tokenized_data['label']) if label == class_label]
#         X_class = np.array(tokenized_data['input_ids'])[class_indices]
 
#         similarity_kernel = np.dot(X_class, X_class.T)
 
#         facility_location_function = submodlib.FacilityLocationFunction(n=len(X_class), mode="dense", sijs=similarity_kernel)

         
#         selected = facility_location_function.maximize(budget=num_samples_per_class, optimizer='NaiveGreedy')
 
#         selected_indices.extend([class_indices[i] for i in selected])

#     return selected_indices
 
 
# num_samples_per_class = 1000 // len(set(test_dataset['label']))   
# selected_indices = select_representative_samples(test_dataset, num_samples_per_class)

 
# test_dataset_subset = test_dataset.select(selected_indices)

# print(test_dataset_subset)


# In[30]:


print("""Data Preparation:

The text data is first extracted and then transformed into numerical features using TF-IDF vectorization. This process converts the text into a matrix of features where each feature represents the importance of a word in the document relative to the entire corpus.
Clustering for Representative Sample Selection:

The goal is to select a representative subset of samples for each class. To achieve this, the data is first clustered using the K-Means algorithm, which groups the data into a specified number of clusters.
Selecting Closest Points to Cluster Centroids:

For each cluster, the data points closest to the centroid (center) of the cluster are identified. These points are considered the most representative samples of the data within that cluster.""")


# submodelib fro each class

# In[31]:


# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
from apricot import FacilityLocationSelection
from datasets import Dataset

 

# Step 3: Define a function to select representative samples for each class using apricot
def select_representative_samples(tokenized_data, num_samples_per_class):
    selected_indices = []
    
    # Iterate over each unique class label
    for class_label in set(tokenized_data['label']):
        print(f"Processing class: {class_label}")
        class_indices = [i for i, label in enumerate(tokenized_data['label']) if label == class_label]
        X_class = np.array(tokenized_data['input_ids'])[class_indices]

        # Apply apricot's Facility Location Selection
        selector = FacilityLocationSelection(n_samples=num_samples_per_class, metric='euclidean', verbose=True)
        selector.fit(X_class)
        selected = selector.ranking

        # Collect the selected indices
        selected_indices.extend([class_indices[i] for i in selected])

    return selected_indices

 
 
num_samples_per_class = 1000 // len(set(test_dataset['label']))   
selected_indices = select_representative_samples(test_dataset, num_samples_per_class)

 
test_dataset_subset = test_dataset.select(selected_indices)

print(test_dataset_subset)


# In[32]:


# Convert to a pandas DataFrame first if not already one
df_subset = pd.DataFrame(test_dataset_subset)
df_subset.to_csv('/home/bhairavi/om/om5/dbp/test_dataset_subset.csv', index=False)


# In[31]:


# Step 1: Import Necessary Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np

# Step 2: Vectorize the Text Data
X = test_df['text']
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)
X_dense = X_vectorized.toarray()
y = test_df['target']

# Step 3: Define a function to select representative samples for each class
def select_representative_samples(X, y, num_samples_per_class):
    selected_indices = []
    for class_label in y.unique():
        print(class_label)
        class_indices = y[y == class_label].index
        X_class = X[class_indices]
        
        # Cluster the data
        num_clusters = num_samples_per_class
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X_class)
        
        # Find the closest samples to the cluster centroids
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_class)
        
        # Collect the selected indices
        selected_indices.extend(class_indices[closest])
    
    return selected_indices

# Step 4: Select samples for each class
num_samples_per_class = 1000 // len(y.unique())  # Adjust the number of samples per class as needed
selected_indices = select_representative_samples(X_dense, y, num_samples_per_class)

# Step 5: Create the subset DataFrame
df_subset = test_df.iloc[selected_indices]

# Display the subset DataFrame
print(df_subset)


# In[32]:


df_subset.shape


# In[33]:


df_subset['label'].nunique()


# In[35]:


for i in df_subset['target'].value_counts():
    print(i)


# In[36]:


saveDIR


# In[38]:


# %%

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Use 'epoch' to evaluate at the end of each epoch
    save_strategy="epoch",  # Also use 'epoch' to save at the end of each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,  # Load the best model at the end of training based on metric
    metric_for_best_model='f1',  # Define the metric for evaluating the best model
    logging_dir='./logs',
    logging_steps=10,
)


 

trainer = Trainer(
    model=model,
    args=training_args ,  # Here you will need to make sure that the Trainer is set up correctly
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

 


trainer.train()


# %%


# %%

save_directory = saveDIR
 

# Save the model
model.save_pretrained(save_directory)

# Save the tokenizer (optional, but recommended)
tokenizer.save_pretrained(save_directory)

 


# In[39]:


# %%


# %% [markdown]
# eval dataset performance so that keywords_classes can be fixed

# %%
results = trainer.evaluate()

# Predict using the trained model to get labels and predictions
predictions, labels, _ = trainer.predict(eval_dataset)
predictions = np.argmax(predictions, axis=1)


# %%
from sklearn.metrics import classification_report
# Generate the classification report
report = classification_report(
    labels,
    predictions,
    target_names=df['label'].unique() , # Adjust this line as per your dataset
    digits=4
)

print(report)


# %% [markdown]
# skyline

# %%
from colorama import Fore, Style

# %%
print(Fore.RED +"TEST DATA IS OUR SKYLINE RESULT")
 
results = trainer.evaluate()

# Predict using the trained model to get labels and predictions
predictions, labels, _ = trainer.predict(test_dataset)
predictions = np.argmax(predictions, axis=1)


# %%
from sklearn.metrics import classification_report
# Generate the classification report
report = classification_report(
    labels,
    predictions,
    target_names=df['label'].unique() , # Adjust this line as per your dataset
    digits=4
)

print(report)


# %%
 









