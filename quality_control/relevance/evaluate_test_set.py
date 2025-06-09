import os
import json
from tqdm import tqdm
from collections import defaultdict
import re


def format_prompt(question, answer):

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
    '''

    return prompt


prompts_all = []
correct_responses = []

with open('labels.json', 'r') as f:
    labels = json.load(f)

with open('test_qa_ids.json', 'r') as f:
    test_qa_ids = json.load(f)

test_labels = {qa_id: labels[qa_id] for qa_id in test_qa_ids if qa_id in labels}

for qa_id, sample in test_labels.items():

    question = sample['question']
    answer = sample['answer']
    prompt = format_prompt(question, answer)
    prompts_all.append(prompt)
    correct_responses.append(sample['response'])


with open("test_set_relevance_predictions.json", "r") as f:
    results = json.load(f)


pattern = r"Category:\s*(good|bad|Good|Bad)"

def extract_category(text):
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        category = match.group(1)
        return category.lower()
    return None



total_instances = defaultdict(int)
true_positives = 0
false_positives = 0 
false_negatives = 0
true_negatives = 0
correct = 0
total = 0

negative_class = "bad"

for model_response, correct_response in zip(results["outputs"], correct_responses):
    model_category = extract_category(model_response)
    desired_category = extract_category(correct_response)
        
    total_instances[desired_category] += 1
    total += 1
    
    if model_category == desired_category:
        correct += 1
        
        if desired_category == negative_class:
            true_positives += 1  # Correctly identified as negative
        else:
            true_negatives += 1  # Correctly identified as not negative
    else:
        if model_category == negative_class:  # Model predicted negative but it's actually not
            false_positives += 1
        else:  # Model predicted not negative but it's actually negative
            false_negatives += 1

# Calculate metrics for negative class
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

print("Number of responses: ", len(results["outputs"]))
print(f"Negative Class Precision: {precision:.3f}")
print(f"Negative Class Recall: {recall:.3f}")