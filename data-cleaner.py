""" I have used AI for this part and taken the help of the notes and other Homework assignments in the course """
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
import json


 # Tell people to upload these on their laptop as well. 
dataset = load_dataset("json", data_files= r"C:\Users\adity\OneDrive\Desktop\cs3114\S25P4GraphProject\instruction-vs-preference-alignment\code_alpaca_20k.json")["train"]

instruction = dataset["instruction"]
input = dataset["input"]
output = dataset["output"]

# Now data in list 
data = list(zip(instruction, input, output))

# Method to remoce whitespaces and clean the dataset to remove 
# Also changed the input to match the other file for conveneiece. 
def clear_whiteSpace(data):
    
    cleaned = []
    for j, i, k in data:
        
        j = j.strip()
        i = i.strip() 
        k = k.strip() 
        
        if i.lower() in ["<noinput>", "< noinput >", "<noinput >"]:
            i = ""  
        cleaned.append([j, i, k])
    return cleaned
            

final_data =  clear_whiteSpace(data)

# Printing the output 
json_data = [{"instruction": j, "input": i, "output": k} for j, i, k in final_data]
with open("cleaned_data.json", "w", encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

                        
