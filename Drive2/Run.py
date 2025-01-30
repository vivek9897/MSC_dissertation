import os
import json
import argparse

from Dataset import Dataset


# load args
parser = argparse.ArgumentParser(description='Description of your script')
parser.add_argument('pair', help='Currency pair being profiled')
args = parser.parse_args()

# load params
with open('../params.json', 'r') as f:
    params = json.load(f)

# load full dataset
data = Dataset(f'../0_RawData/Concatenated/{args.pair}_concatenated.csv', args.pair, params)
