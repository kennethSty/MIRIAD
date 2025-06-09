import os
import json
import random
random.seed(229)
from tqdm import tqdm
from openai import OpenAI
from ast import literal_eval
import pandas as pd

client = OpenAI()

system_prompt = '''
I took passages of text from the medical literature and converted each passage into Q&A problems with the goal to distill the medical knowledge of the literature into a set of Q&A problems. Having generated them, I noticed that some Q&As are bad because they do not contain medically or biomedically relevant information. 
Ideally, a Q&A conveys a relevant piece of medical knowledge like this good example: 

"Q: What is the first-line therapy for M. tuberculosis pneumonia? 
A: Rifampin, isoniazid, pyrazinamide, and ethambutol are first-line antitubercular medications."

By contrast, a bad Q&A would have one or more of the following issues:
1. Refers to the details of a specific study (that is presented in the passage), such as specific details about the study's experimental design, the used statistical methods used in this study, tables or figures that appear in it, study dates, locations, funding sources, or other details that are not essential for understanding the underlying medical facts.
2. Is heavily dependent on study-specific details that cannot be understood without the original passage, such as discussing the study's specific findings, limitations, or conclusions without providing sufficient background information.
3. Focuses on experimental methods or protocols that, while medically related, are too specific to the referenced study and do not convey broadly relevant medical knowledge. 

If the Q&A does not have any other bad aspects mentioned above, then the Q&A should be classified as good. A Q&A that effectively communicates a clinical procedure, treatment approach, or any clinical or biomedical knowledge in a clear and concise manner should be considered good.

Use the above criteria to judge whether the following Q&A is good or bad. Classify the Q&A as either "good" or "bad" and provide a short explanation for the classification.

Answer in this format

Explanation: {explanation}
Category: {good or bad}
'''

def gpt(QA):
    response = client.chat.completions.create(
    model="gpt-4",
    messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": QA},
        ]
    )
    return response.choices[0].message.content

df = pd.read_csv('selected_passages.csv')

df['question'] = df['question'].apply(literal_eval)
df['answer'] = df['answer'].apply(literal_eval)
df['qa_id'] = df['qa_id'].apply(literal_eval)

labels = {}

for index, row in df.iterrows():

    passage = row['passage_text']
    qa_ids = row['qa_id']
    questions = row['question']
    answers = row['answer']

    qa = [[qa_id, q, a] for qa_id, q, a in zip(qa_ids, questions, answers)]

    for qa_id, q, a in qa:

        prompt = f"Question: {q}\nAnswer: {a}"

        response = gpt(prompt)

        labels[qa_id] = {
            "question": q,
            "answer": a,
            "response": response
        }
    
    with open("labels.json", "w") as f:
        json.dump(labels, f)

