import json
from together import Together
import re
import random
random.seed(229)
from sklearn.utils import resample
client = Together()

with open('train_qa_ids.json', 'r') as f:
    train_qa_ids = json.load(f)

with open('labels.json', 'r') as f:
    labels = json.load(f)

train_labels = {qa_id: labels[qa_id] for qa_id in train_qa_ids if qa_id in labels}

def format_prompt(question, answer, explanation):

    task = f'''
    I took passages of text from the medical literature and converted each passage into Q&A problems with the goal to distill the medical knowledge
    of the literature into a set of Q&A problems. Having generated them, I noticed that some Q&As are bad because they do not contain medically or
    biomedically relevant information. 
    Ideally, a Q&A conveys a relevant piece of medical knowledge like this good example: 
    
    "Q: What is the first-line therapy for M. tuberculosis pneumonia? 
    A: Rifampin, isoniazid, pyrazinamide, and ethambutol are first-line antitubercular medications."
    
    By contrast, a bad Q&A would have one or more of the following issues:
    1. Refers to the details of a specific study (that is presented in the passage), such as specific details about the study's experimental design,
    the used statistical methods used in this study, tables or figures that appear in it, study dates, locations, funding sources, or other details
    that are not essential for understanding the underlying medical facts.
    2. Is heavily dependent on study-specific details that cannot be understood without the original passage, such as discussing the study's
    specific findings, limitations, or conclusions without providing sufficient background information.
    3. Focuses on experimental methods or protocols that, while medically related, are too specific to the referenced study and do not convey
    broadly relevant medical knowledge. 
    
    If the Q&A does not have any other bad aspects mentioned above, then the Q&A should be classified as good. A Q&A that effectively communicates a
    clinical procedure, treatment approach, or any clinical or biomedical knowledge in a clear and concise manner should be considered good.
    
    Use the above criteria to judge whether the following Q&A is good or bad. Classify the Q&A as either "good" or "bad" and provide a short
    explanation for the classification.
    
    Answer in this format
    
    Explanation: {{explanation}}
    Category: {{good or bad}}

    Question: {question}
    Answer: {answer}
    '''

    prompt = f'''
    <s>[INST] { task } [/INST]
    
    { explanation }</s> 
    '''

    return {"text": prompt}


good_prompts = []
bad_prompts = []
pattern = r"Category:\s*(good|bad)"
for qa_id, value in train_labels.items():

    question = value['question']
    answer = value['answer']
    gpt_response = value['response']
    formatted = format_prompt(question, answer, gpt_response)
    match = re.search(pattern, gpt_response, re.IGNORECASE)
    if match:
        category = match.group(1)
        if category.lower() == "good":
            good_prompts.append(formatted)
        elif category.lower() == "bad":
            bad_prompts.append(formatted)
    else:
        print("Error: could not find category!")

# Downsample the good prompts to match the number of bad prompts
good_prompts_downsampled = resample(good_prompts, 
                                    replace=False,
                                    n_samples=len(bad_prompts),  
                                    random_state=229)

# Ensure alternating good and bad prompts
def alternate_prompts(good_prompts, bad_prompts):
    combined = []
    good_index, bad_index = 0, 0
    while good_index < len(good_prompts) and bad_index < len(bad_prompts):
        combined.append(good_prompts[good_index])
        combined.append(bad_prompts[bad_index])
        good_index += 1
        bad_index += 1
    return combined

# Using downsampled good prompts
downsampled = alternate_prompts(good_prompts_downsampled, bad_prompts)

def dump_jsonl(filename, json_objects):
    with open(filename, 'w') as file:
        for json_object in json_objects:
            json.dump(json_object, file)
            file.write('\n')

dump_jsonl('training_set.jsonl', downsampled)
resp = client.files.upload(file="training_set.jsonl")
file_id = resp.id
model_name = 'mistralai/Mistral-7B-Instruct-v0.2'


resp = client.fine_tuning.create(
  training_file = file_id,
  model = model_name,
  n_epochs = 10,
  n_checkpoints = 1,
  batch_size = 8,
  learning_rate = 1e-5,
)
print(resp)
