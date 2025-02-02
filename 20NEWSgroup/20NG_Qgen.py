# %%
ngram_length = 3
dstype = '20NG' 
mname = 'debertaV3'
import pandas as pd
from colorama import Fore, Style
import os
path = f"/home/bhairavi/om/om3/{dstype}/{ngram_length}grams_{mname}/"
os.makedirs(path, exist_ok=True)
print("Directory created or already exists.")
file_path = path + f'{dstype}_{ngram_length}keys.csv' 
print(Fore.YELLOW,"csv_filePATH--->",file_path)
filepath_full = path + f'{dstype}_{ngram_length}que.csv' 
print(Fore.YELLOW,"QUE_filePATH--->",filepath_full)
modelpath = f"/home/bhairavi/om/om5/{dstype}/{mname}_{dstype}"
print(Fore.YELLOW,'modelPATH--->',modelpath)

# %%
df  = pd.read_csv(file_path)
print(df.shape)
# %%
numlabel = df['target'].nunique()
numlabel
# %%

# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda"  # the device to load the model onto

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(modelpath)
model = AutoModelForSequenceClassification.from_pretrained(modelpath, num_labels=numlabel)

# Move the model to the specified device
model.to(device)

max_length  = 512

# %%

# %%
from sklearn.model_selection import StratifiedShuffleSplit
    
import numpy as np
from apricot import FacilityLocationSelection
from datasets import Dataset 

def tokenize_and_format(examples):
    tokenized_inputs = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=max_length)
    tokenized_inputs['label'] = list(map(int, examples['target']))
    return tokenized_inputs

 
test_dataset = Dataset.from_pandas(df) 
df = test_dataset.map(tokenize_and_format, batched=True,batch_size=16)

# %%
# %%
def select_representative_samples(tokenized_data, num_samples_per_class):
    selected_indices = []
    
    for class_label in set(tokenized_data['label']):
        print(f"Processing class: {class_label}")
        class_indices = [i for i, label in enumerate(tokenized_data['label']) if label == class_label]
        X_class = np.array(tokenized_data['input_ids'])[class_indices]
        selector = FacilityLocationSelection(n_samples=num_samples_per_class, metric='euclidean', verbose=True)
        selector.fit(X_class)
        selected = selector.ranking
        selected_indices.extend([class_indices[i] for i in selected])

    return selected_indices

num_samples_per_class = 1000 // len(set(df['label']))   
selected_indices = select_representative_samples(df, num_samples_per_class)

 
df = df.select(selected_indices)

print(df.shape)

# %%
print(df['top3_predicted_labels'][0])
# %%
print(df['significant_words'][0])

# %%

import html
import re

def parse_sets(set_string):
    # Decode HTML entities
    decoded_string = html.unescape(set_string)

    # Find sets within the string using regex to capture text inside curly braces
    set_strings = re.findall(r'\{.*?\}', decoded_string)

    # List to hold actual sets
    actual_sets = []

    # Convert string representations of sets into actual sets
    for set_str in set_strings:
        # Remove the curly braces and split by ', '
        elements = set_str.strip('{}').split('\', \'') 
        cleaned_elements = {elem.strip('\'') for elem in elements}
        actual_sets.append(cleaned_elements)

    return actual_sets

df= pd.DataFrame(df)
df.head()
df['significant_words'] = df['significant_words'].apply(parse_sets) 
 

# %%
df['significant_words'][0]
# %%
df = df.reset_index()
df[0:3]
# %%



# %%

import requests 
import ast 
import re  
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 

import random
api_keys = [  

    "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", "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", 
    "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", "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",
    "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", "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",
    "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", "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",
    "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", "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",
    
]

# random.shuffle(api_keys)


def generate_question(api_key, partial_info, labels, keywords):
    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
    headers = {"Authorization": f"Bearer {api_key}"}
     
    input_text = """
 
    You are an AI Expert. You are provided with partial information along with the top-3 categories where this information could belong. Each category also has a list of keywords that represent the characteristic content covered by the category. Your task is to ask an information-seeking question based on the partial information and the category keywords such that when answered, one of the categories can be selected with confidence.

    Follow the following thinking strategy:
    First, eliminate the categories that are not probable based on the given information. Identify the main context of the partial information and see if similar content matches any of the keywords in a category. If it doesn't, then the category can be taken out of consideration.
    Now generate a question. This question should further probe for information that will help refine the identification of the most likely category. Your question should strategically use the keywords tied to each potential category, aiming to effectively differentiate between them.

    Strictly follow the format shown in examples for output generation. Double quote the final question.

    Here are a few examples to understand better:

    Example 1:
    Partial information: "I constantly sneeze and have a dry cough."

    Category: Allergy, Keywords: {headache, coughing, wet, sneeze, pain}
    Category: Diabetes, Keywords: {severe, feet, skin, rashes, infection}
    Category: Common Cold, Keywords: {swollen, cough, body, shivery, ache, dry}

    Note:
    Sneeze and dry cough are the main subjects of the partial information. Coughing is present in Allergy and common cold, but cough or sneeze is not present in Diabetes. Therefore, Diabetes can't be a possible label. Only two labels—Allergy and Common Cold—are considered. The keywords suggest that knowing about symptoms like headache, body pain, shivery, etc., will help refine the classification into one of the labels.

    Therefore only two labels, namely:
    Category: Allergy
    Category: Common Cold
    are used to form a question.

    QUESTION: "Besides fever, are you experiencing symptoms such as cough, severe headaches, localized pain, or inflammation? Also, can you describe the pattern of your fever—is it continuous or does it occur in intervals?"

    Example 2:
    Partial information: "The software keeps crashing."

    Category: Software Bug, Keywords: {crash, error, bug, glitch}
    Category: User Error, Keywords: {instructions, setup, incorrect, usage}
    Category: Hardware Issue, Keywords: {overheating, components, failure, malfunction}

    Note:
    The main subject of the partial information is the software crash. The keyword 'crash' is directly related to Software Bug but could also be indirectly related to User Error and Hardware Issue. However, to differentiate, asking about the conditions under which the crash happens or if any error messages appear could help narrow down the correct category.

    Therefore, all three categories, namely:
    Category: Software Bug
    Category: User Error
    Category: Hardware Issue
    are used to form a question.

    QUESTION: "When the software crashes, do you receive any specific error messages, or does it happen during particular tasks? Have you noticed any hardware malfunctions or overheating before the crashes?"

    Example 3:
    Partial information: "The car is making a strange noise."

    Category: Engine Problem, Keywords: {noise, misfire, engine, smoke}
    Category: Tire Issue, Keywords: {flat, noise, pressure, alignment}
    Category: Transmission Issue, Keywords: {shifting, noise, gears, slipping}

    Note:
    The main subject of the partial information is the strange noise. The keyword 'noise' is present in all three categories—Engine Problem, Tire Issue, Transmission Issue. Knowing more about the type of noise and when it occurs can help identify the correct category.

    Therefore, all three categories, namely:
    Category: Engine Problem
    Category: Tire Issue
    Category: Transmission Issue
    are used to form a question.

    QUESTION: "Can you describe the noise in more detail? Is it a grinding, squealing, or clicking sound? Does it happen while driving, when shifting gears, or when the car is stationary?"


    ***
    
    Now generate note and QUESTION for:
        
    """
           
    input_string = labels
    words = re.findall(r"'(.*?)'", input_string) 
    labels = ', '.join(words) 
    labels=labels.strip(" ").strip("'").split(",") 
    print(labels)
    
    input_text += f"Partial information: {partial_info}\n"
    for i in range(3):
        input_text += f"Category: {labels[i]}, Keywords: {keywords[i]}\n"
 
  
 
    response = requests.post(
        API_URL,
        headers=headers,
        json={
            "inputs": input_text,
            "options": {"use_cache": False},
            "parameters": {"max_new_tokens": 2048, "temperature": 0.7}
        }
    )
 
    if response.status_code == 200:
        response_data = response.json() 
        if isinstance(response_data, list) and 'generated_text' in response_data[0]:
            output_text = response_data[0]['generated_text'].strip()
    
            return output_text
        else:
            return f"Failed to query the model: {response.status_code} - {response.text}", None
    else:
        return None
 

chunk_size = len(df) // len(api_keys) + (len(df) % len(api_keys) > 0)
results = []

for i, key in enumerate(api_keys):
    start = i * chunk_size
    end = start + chunk_size
    df_chunk = df.iloc[start:end]
    def process_row(x):
        generated_q = None
        generated_text = generate_question(key, x['first_half'], x['top3_predicted_labels'], x['significant_words'])
        print("/n/n")
        print(key)
        print("/n/n")
        try:
            questions = re.findall(r'QUESTION: \"(.*?)\"', generated_text)

            # Eliminate the first 3 questions
            remaining_questions = questions[3:]

            # Find the next question with more than 5 words
            selected_question = None
            for question in remaining_questions:
                if len(question.split()) > 5:
                    selected_question = question
                    generated_q = selected_question
                    break
 
        except Exception as e:
            generated_q = None
        separator = "~" * 600
        print(separator)
        print("Generated Question:", generated_q)
    
        
        print(separator) 
        return generated_q

    questions = df_chunk.apply(process_row, axis=1)
 
    results.append(questions)
 

df['generated_question'] = pd.concat(results)

# %%
df.to_csv(filepath_full)
