import os 
import gzip
import io
import json
from IPython import embed
from tqdm import tqdm
import fire

def main(batch_id=0):
    # Hardcoded paths atm:
    # Input paths
    paths, metadata_outpath = get_paths()
    
    #1. Filter the metadata file for domain we want
    paper_ids_to_keep, field_to_id = filter_domain(paths, batch_id=batch_id)
    info = {'paper_ids': list(paper_ids_to_keep), 'field_to_paper_id': field_to_id}
    info_file = os.path.join(metadata_outpath, f'info_{batch_id}.json')
    with open(info_file, 'w') as f:
        json.dump(info, f)        

    #2. Extract papers from PDF extract files:
    # Selecting medical pdf parses (for current batch of data)
    ppath = paths['ppath'].format(batch_id)
    p_outpath = paths['p_outpath'].format(batch_id)
    with gzip.open(ppath, 'rb') as gz, open(p_outpath, 'wb') as f_out:
        f = io.BufferedReader(gz)
        for line in tqdm(f.readlines()):
            metadata_dict = json.loads(line)
            paper_id = metadata_dict['paper_id']
            if paper_id in paper_ids_to_keep:
                f_out.write(line)

def filter_domain(paths, batch_id=0):
    """
    Process metadata, select for field "Medicine" or "Biology"
    """
    # Unpack paths and specify batch:
    mpath = paths['mpath'].format(batch_id)
    ppath = paths['ppath'].format(batch_id)
    m_outpath = paths['m_outpath'].format(batch_id)
    p_outpath = paths['p_outpath'].format(batch_id)
    
    field_names = []
    paper_ids_to_keep = set()
    field_to_id = {}
    with gzip.open(mpath, 'rb') as gz_in, open(m_outpath, 'wb') as f_out:
        f = io.BufferedReader(gz_in)
        for i, line in enumerate(tqdm(f.readlines())):
            metadata_dict = json.loads(line)
            paper_id = metadata_dict['paper_id']
            mag_field_of_study = metadata_dict['mag_field_of_study']
            if mag_field_of_study not in field_names:
                field_names.append(mag_field_of_study)
            if mag_field_of_study and any([x in mag_field_of_study for x in ['Medicine', 'Biology']]):
                pass
            else:
                continue
            if metadata_dict['has_pdf_parse']:
                if metadata_dict['has_pdf_parsed_body_text']:
                    paper_ids_to_keep.add(paper_id)
                    f_out.write(line)
                    # add paper to field_to_id mapping 
                    # make field hashable:
                    mag_field_of_study = ' / '.join(mag_field_of_study)
                    if mag_field_of_study not in field_to_id.keys():
                        field_to_id[mag_field_of_study] = [paper_id]
                    else:
                        field_to_id[mag_field_of_study].append(paper_id)
    return paper_ids_to_keep, field_to_id



def get_paths():
    """
    Hard coded paths for now
    """
    metadata = '20200705v1/full/metadata'
    pdfs = '20200705v1/full/pdf_parses'
    # Output paths
    metadata_outpath = '20200705v1/selected/metadata'
    pdfs_outpath = '20200705v1/selected/pdf_parses'

    os.makedirs(metadata_outpath, exist_ok=True)
    os.makedirs(pdfs_outpath, exist_ok=True)

    # Compile dict containing all paths:
    paths = {}
    paths['mpath'] = os.path.join(metadata, 'metadata_{}.jsonl.gz')
    paths['ppath'] = os.path.join(pdfs, 'pdf_parses_{}.jsonl.gz')

    # Output paths: (metadata, pdf)
    paths['m_outpath'] = os.path.join(metadata_outpath, 'metadata_{}.jsonl.gz')
    paths['p_outpath'] = os.path.join(pdfs_outpath, 'pdf_parses_{}.jsonl.gz')

    return paths, metadata_outpath 

if __name__ == "__main__":

    fire.Fire(main)
