## MIRIAD Dataset Generation

Here we provide the steps we've used for MIRIAD dataset generation. We've noticed that since Semantic Scholar has modified their dataset download settings, downloading raw S2ORC dataset might take slightly different steps from the steps provided below. We encourage you to check out [Semantic Scholar](https://github.com/allenai/s2orc) for the latest API access method.

### Step 1: Download Raw S2ORC Data
 ```bash
 source preprocessing/download_full.sh
```
##### Downloads metadata_*.jsonl.gz and pdf_parses_*.jsonl.gz from the S2ORC corpus. Files are stored in:
 ```bash
preprocessing/20200705v1/full/metadata/
preprocessing/20200705v1/full/pdf_parses/
 ```

### Step 2: Broad Filtering by Field of Study
 ```bash
 source preprocessing/broad_filter.sh
 ```
##### This script filters the raw data from step 1 to retain papers in medical, biological, or adjacent disciplines that  overlap with or contribute to biomedical research. Files are stored in:
 ```bash
preprocessing/20200705v1/selected/metadata/
preprocessing/20200705v1/selected/pdf_parses/
 ```

### Step 3: Filter Only Medical Documents
 ```bash
 source preprocessing/filter_medicine.sh
 ```
##### Further narrows to papers strictly in "Medicine" category. Files are stored in:
 ```bash
preprocessing/papers
 ```

### Step 4: Create Passages
 ```bash
 source preprocessing/create_passages.sh
```
 ##### Chunks full-text medical papers into smaller (1K token) passages that can be used for question generation. Files are stored in:
```bash
preprocessing/passages
 ```

 ### Step 5: Generate QA Pairs
 ```bash
 export OPENAI_API_KEY="sk-..."
 source data_generation/launch_generate.sh
 ```
 ##### Launches parallel screen sessions to generate QA pairs using the model `gpt-3.5-turbo-0125`. Files are stored in:
```bash
data_generation/generated_data
 ```
