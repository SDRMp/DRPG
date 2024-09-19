#!/usr/bin/env python
# coding: utf-8

# In[30]:


mname = 'bert'
dstype = 'salad' 


# In[31]:


# skyline

# %%
from colorama import Fore, Style


# In[32]:


# %%
import torch
import torchvision

# Check PyTorch version
print("PyTorch version:", torch.__version__)

# Check if CUDA is available (GPU support)
print("CUDA available:", torch.cuda.is_available())

# Check the number of GPUs
print("Number of GPUs:", torch.cuda.device_count())



# In[33]:


# %%
# %%
modelpath = 'microsoft/deberta-v3-base'
modelpath = "bert-base-uncased"

datapath = None
saveDIR = f"/home/bhairavi/om/om5/{dstype}/{mname}_{dstype}"
print(saveDIR)
# %%


# In[34]:


from datasets import load_dataset

dataset = load_dataset("OpenSafetyLab/Salad-Data", name='base_set', split='train')


# In[35]:


dataset


# In[36]:


import pandas as pd


# In[37]:


df = pd.DataFrame(dataset)


# In[38]:


df['3-category'].nunique()


# In[39]:


df


# In[40]:


df['label'] = df['3-category']


# In[41]:


df['text'] = df['question']


# In[42]:


df = df[['text','label']]


# In[43]:


# %%
from sklearn.preprocessing import LabelEncoder

# %%
le = LabelEncoder()
df['target'] = le.fit_transform(df['label'])

# %%


# In[44]:


df.columns


# In[45]:


df.shape


# In[46]:


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
 


# In[47]:


numlabel = df['target'].nunique()
numlabel


# In[48]:


# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda"  # the device to load the model onto

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(modelpath)
model = AutoModelForSequenceClassification.from_pretrained(modelpath, num_labels=numlabel)

# Move the model to the specified device
model.to(device)


# In[49]:


df['token_length'] = df['text'].apply(lambda x: len(tokenizer.tokenize(x)))

# Calculate the maximum token length
max_length = df['token_length'].max()

# Calculate the next maximum token length
next_max_token_length = df['token_length'].nlargest(2).iloc[1]

# Calculate the average token length
average_token_length = df['token_length'].mean()

# Display the results
print(f"Maximum token length: {max_length}")
print(f"Next maximum token length: {next_max_token_length}")
print(f"Average token length: {average_token_length:.2f}")


# In[50]:


min(df['token_length'])


# In[51]:


fdf = df[df['token_length'] == 2]


# In[52]:


fdf


# In[53]:


df = df[df['token_length'] >= 5]


# In[54]:


df.shape


# In[55]:


# %%
from sklearn.model_selection import StratifiedShuffleSplit

# Splitting off the test set with 5% of the data
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=42)  # 5% for test
for train_val_idx, test_idx in sss.split(df, df['target']):
    train_val_df = df.iloc[train_val_idx]
    test_df = df.iloc[test_idx]

# Further split train_val_df into train and validation sets with validation set being 15.79% of the remaining data
# (which is equivalent to 15% of the original dataset size)
sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)  # ~15.79% of remaining data
for train_idx, val_idx in sss_val.split(train_val_df, train_val_df['target']):
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]
 

def tokenize_and_format(examples):
    # Tokenize the texts
    tokenized_inputs = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=max_length)
    tokenized_inputs['label'] = list(map(int, examples['target']))
    return tokenized_inputs




# In[56]:


# Convert pandas DataFrame to Hugging Face's Dataset
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(val_df) 
test_dataset = Dataset.from_pandas(test_df)

# Map the tokenization function across the datasets
train_dataset = train_dataset.map(tokenize_and_format, batched=True,batch_size=16)
eval_dataset = eval_dataset.map(tokenize_and_format, batched=True,batch_size=16) 
test_dataset = test_dataset.map(tokenize_and_format, batched=True,batch_size=16)


# In[57]:


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

 


# %%
 


# %%

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Use 'epoch' to evaluate at the end of each epoch
    save_strategy="epoch",  # Also use 'epoch' to save at the end of each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
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


# In[58]:


save_directory =  saveDIR
 

# Save the model
model.save_pretrained(save_directory)

# Save the tokenizer (optional, but recommended)
tokenizer.save_pretrained(save_directory)

 


# In[60]:


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
print(Fore.CYAN,"this is for keywords, The EVAL DATASET")
print(report)



# In[61]:


# %% [markdown]


# %%
 
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
print(Fore.RED +"TEST DATA IS OUR SKYLINE RESULT")


# %%
 





