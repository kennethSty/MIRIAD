## MIRIAD Dataset Quality Control

After data generation, we applied multiple rounds of quality control to ensure the high-quality of MIRIAD.

### Step 6: Quality Control

#### Step 6.1: Keyword Filtering
```bash
quality_control/keyword_filter.py
 ```
 ##### Filters generated data at path data_generation/generated_data by removing QA with answers with the prefix "the passage" or the study to make the QA pairs self contained and not referencing studies

#### Step 6.2: Relevance Filtering
#### Step 6.2A: Generating labels using GPT-4
```bash
python quality_control/relevance/generate_labels.py
 ```
#####  Files (including the train/test split) are stored in:
```bash
quality_control/relevance/labels.json
quality_control/relevance/train_qa_ids.json.json
quality_control/relevance/test_qa_ids.json.json
```

#### Step 6.2B: Finetuning `mistralai/Mistral-7B-Instruct-v0.2` as a relevance classifier
```bash
python quality_control/relevance/finetune.py
 ```

#### Step 6.2C: Filtering using the relevance classifier
```bash
python quality_control/relevance/relevance_inference.py
python quality_control/relevance/filter_relevance.py
 ```
##### Applies the relevance classifier to retain medically relevant QA pairs.

####  Step 6.3: Human Annotation App
Launch the Streamlit app for manual QA inspection and labeling:
```bash
streamlit run quality_control/streamlit_app/app.py
 ```
 ##### This allows human annotators to review generated QA pairs alongside their source passages and assign labels for Relevance, Factuality, and Groundedness in Passage.