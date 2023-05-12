'''
preprocess_data.py contains methods for processing raw book data into all_data_masked.tsv for training
'''

import os
import click
import string
import csv
from tqdm import tqdm 

import spacy
nlp = spacy.load('en_core_web_lg')

MASK = '<MASK>'
IGNORE_CHARS = '0123456789' + string.punctuation
ENTITIES_TO_MASK = ['PERSON']


# mask entities within a text to remove special character names etc. exclusive to some books
def mask_entities(text, mask):
    doc = nlp(text)
    return " ".join([mask if t.ent_type_ and t.ent_type_ in ENTITIES_TO_MASK else t.text for t in doc])


def preprocess_file(filename, out_file, mask_ner, dialect, period, author):
    book_title = filename.split('-')[-1].strip().replace('.txt','')
    with open(filename, 'r', errors='ignore') as f:
        text = ''.join(f.readlines())
    if text.count('\n\n') > 10:
        paragraphs = text.split('\n\n')
    else:
        paragraphs = text.split('\n')
    print(f'{book_title}: {len(paragraphs)} paragraphs')
    with open(out_file, 'a', newline='') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        for par in tqdm(paragraphs):
            par = ' '.join(par.replace('\n', ' ').split())
            if mask_ner:
                par = mask_entities(par, MASK)
            if par.strip().translate(dict([(c, '') for c in IGNORE_CHARS])) and len(par.strip())>10:
                writer.writerow([par, dialect, period, author, book_title])


@click.command()
@click.option("--input-file", type=click.Path(readable=True), required=True, help='path for input .txt file or directory of files')
@click.option("--output-file", type=click.Path(readable=True), default='preprocessed_data.tsv', help='path output .tsv file (if exists, append)')
@click.option("--mask-named-entities", is_flag=True, help=f'whether to replace results of NER with {MASK}')
@click.option("--dialect", default=None, help=f'if passing in just one .txt file, dialect')
@click.option("--time-period", default=None, help=f'if passing in just one .txt file, time period')
@click.option("--author", default=None, help=f'if passing in just one .txt file, author')
def main(input_file, output_file, mask_named_entities, dialect, time_period, author):
    if not os.path.isfile(output_file):
        with open(output_file, 'w') as f:
            writer = csv.writer(f, delimiter='\t', lineterminator='\n')
            writer.writerow(['paragraph', 'dialect', 'time period', 'author', 'book title'])
    if os.path.isfile(input_file):
        preprocess_file(input_file, output_file, mask_named_entities, dialect, time_period, author)
    else:
        for root, d_names, f_names in os.walk(input_file):
            if not d_names:
                for f_name in f_names:
                    if f_name.endswith('.txt'):
                        dialect = root.split('/')[-2]
                        time_period = root.split('/')[-1]
                        author = f_name.split('-')[0].strip()
                        preprocess_file(os.path.join(root, f_name), output_file, mask_named_entities,  dialect, time_period, author)

if __name__ == '__main__':
    main()