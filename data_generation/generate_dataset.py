from openai import OpenAI
client = OpenAI()
import os

import time

from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
import json
import logging
from tqdm import tqdm
import re
import pickle
import time
import sys
import tiktoken
import shutil
import tempfile

tokenizer = tiktoken.get_encoding("cl100k_base")


logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6), retry=retry_if_exception_type(Exception))
def gpt(prompt):
    try:

        response = client.responses.create(
            model="gpt-3.5-turbo-0125",
            input=prompt,
            temperature=0,
            max_tokens=1000,
        )

        output = response.output_text

        print("Success!")
        
        return output
    
    except Exception as e:
        print(f"Caught an error: {e}. Sleeping before retry...")
        time.sleep(10)
        raise

main_prompt = '''

Please create three questions that are directly answerable from the passage's content. It's imperative that these questions do not focus on or refer to any specific studies, figures, or tables mentioned in the passage. Instead, they should encourage a deeper exploration and understanding of the passage's general content and ideas. They should be framed in a way that their answers can be clearly drawn from the context of the passage. They should be from the following categories:

Condition/Disease/Treatment/Symptom/Cause/Risk Factors/Prevention/Diagnosis/Prognosis/Pharmacology/Anatomy/Physiology/Biochemistry/Pathophysiology/Epidemiology/Surgical Procedures/Nutrition and Diet/Genetics and Genomics/Pediatrics/Geriatrics/Psychology/Psychiatry/Obstetrics and Gynecology/Dentistry/Immunology/Virology/Environmental and Occupational Health/Pharmacy and Drug Dispensation/Rehabilitation/Microbiology/Endocrinology/Neurology/Radiology/Oncology/Cardiology/Gastroenterology/Dermatology/Nephrology/Ophthalmology/Orthopedics/Hematology/Rheumatology/Pulmonology/Urology/Otorhinolaryngology (ENT)/Veterinary Medicine/Addiction Medicine/Chiropractic Medicine/Palliative Care/Bioinformatics/Transplantation/Toxicology/Parasitology/Stem Cell Biology/Podiatry/Hepatology/Sports Medicine/Family Medicine/Sleep Medicine/Critical Care Medicine/Medical Ethics/Forensic Medicine/Infectious Diseases/Emergency Medicine

Please structure your responses in the following format:

Question 1: {question that prompts a detailed exploration of a central theme or key concept found in the passage. This question should be answerable based solely on the passage's content, without needing to reference specific studies or data.}
Answer 1: {an associated in-depth answer grounded in the passage’s context, providing thorough information and explanation}

Question 2: {question that prompts a detailed exploration of a central theme or key concept found in the passage. This question should be answerable based solely on the passage's content, without needing to reference specific studies or data.}
Answer 2: {an associated in-depth answer grounded in the passage’s context, providing thorough information and explanation}

Question 3: {question that prompts a detailed exploration of a central theme or key concept found in the passage. This question should be answerable based solely on the passage's content, without needing to reference specific studies or data.}
Answer 3: {an associated in-depth answer grounded in the passage’s context, providing thorough information and explanation}


Passage to consider: \n
'''

negative_examples = '''
Examples of Inappropriate Questions:
"How does the study demonstrate the general effectiveness of the approach discussed in the passage?"
Why It's Inappropriate: This question inappropriately focuses on a specific "study," contrary to the aim of engaging with the passage's general content.

"What are the key findings of the study mentioned in the passage regarding the overall topic?"
Why It's Inappropriate: This question incorrectly seeks information from a specific "study," while the goal is to explore the passage's broader themes.

"In what ways does the study alter our understanding of the main subject discussed in the passage?"
Why It's Inappropriate: This question wrongly inquires about the impact of a specific "study," rather than encouraging engagement with the overall content of the passage.
'''


def robust_json_dump(data, file_path, indent=None):
    """
    Attempts to dump JSON data to a file robustly using a temporary file.
    
    :param data: The data to be dumped to the file.
    :param file_path: The path to the file where the data should be dumped.
    :param indent: The indentation level to use for pretty-printing the JSON data.
    """
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            json.dump(data, tmp_file, indent=indent)
            temp_file_path = tmp_file.name
        
        # Move the temporary file to the final destination
        # Ensuring the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Atomically move the temp file to the final destination
        shutil.move(temp_file_path, file_path)
        
    except Exception as e:
        print(f"An error occurred during JSON dumping: {e}")
        # Attempt to clean up the temporary file if it still exists
        try:
            os.remove(temp_file_path)
        except Exception:
            pass  # Fail silently if the temp file cannot be removed

def generate(start_shard, end_shard):

    if start_shard % 10 == 0 or start_shard == 21:

        return

    logging.info(f"Starting shard {start_shard}")

    output_file  = f'generated_data/shard_{start_shard}.json'

    if os.path.exists(output_file):

        with open(output_file, 'r') as f:

            paper_dict = json.load(f)

    else:

        paper_dict = {}


    with open(f'../preprocessing/medicine_passages/medicine_passages_{start_shard}.json', 'r') as f:
        shard_papers = json.load(f)

    for paper_index, paper in enumerate(shard_papers):

        paper_info = paper['paper']
        paper_id = paper_info['metadata']['paper_id']
        if paper_id in paper_dict:
            #logging.info(f"Skipping paper {paper_id} as it's already processed.")
            continue

        passages = paper['passages']
        paper_url = paper_info['metadata']['s2_url']
        paper_title = paper_info['metadata']['title']

        passage_dict = {}

        paper_dict[f'{paper_id}'] = {}

        for passage_index, passage in enumerate(passages):
        

            passage_id = f"{start_shard}_{paper_id}_{passage_index}"

            passage_dict[passage_id] = {}

            passage_dict[passage_id]['metadata'] = {
                'paper_url': paper_url, 
                'paper_title': paper_title, 
                'paper_id': paper_id,
                'passage_position': (passage_index+1)/len(passages)
            }

            passage_dict[passage_id]['passage_text'] = passage

            prompt = main_prompt + passage + negative_examples

            response = gpt(prompt)

            questions_and_answers = re.findall(r'Question \d+: (.*?)\nAnswer \d+: (.*?)(?=\n\n|$)', response, re.S)

            passage_dict[passage_id]['qa'] = {}

            for pair_index, (q, a) in enumerate(questions_and_answers):

                passage_dict[passage_id]['qa'][pair_index+1] = {}
                passage_dict[passage_id]['qa'][pair_index+1]['question'] = q
                passage_dict[passage_id]['qa'][pair_index+1]['answer'] = a

            passage_dict[passage_id]['llm_output'] = response


        paper_dict[f'{paper_id}']['passage_info'] = passage_dict
        paper_dict[f'{paper_id}']['paper_info'] = paper_info
        
        robust_json_dump(paper_dict, output_file)


def main(start_shard, end_shard):
    
    generate(start_shard, end_shard)

    logging.info(f"Finished {start_shard}")


if __name__ == "__main__":
    start_shard = int(sys.argv[1])
    end_shard = int(sys.argv[2])
    main(start_shard, end_shard)