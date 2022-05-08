import os
import argparse
val_split_list = [
    'aardvark_female',
    'aardvark_juvenile',
    'aardvark_male',
    'african_elephant_female',
    'african_elephant_male',
    'african_elephant_juvenile',
    'binturong_female',
    'binturong_juvenile',
    'binturong_male',
    'grey_seal_female',
    'grey_seal_juvenile',
    'grey_seal_male',
    'bonobo_juvenile',
    'bonobo_male',
    'bonobo_female',
    'polar_bear_female',
    'polar_bear_juvenile',
    'polar_bear_male',
    'gray_wolf_female',
    'gray_wolf_juvenile',
    'gray_wolf_male',
    'common_ostrich_female',
    'common_ostrich_juvenile',
    'common_ostrich_male'
]

parser = argparse.ArgumentParser(description='help')

parser.add_argument('--index', default=0, type=int,
                    help='dataset path')

args = parser.parse_args()

name_list = val_split_list[args.index*6:(args.index+1)*6]

for name in name_list:
    command1 = 'bash scripts/template_batch.sh {} {}'.format(name,1257+args.index)
    os.system(command1)
    command2 = 'bash scripts/extract.sh {}-5 10 3 36 {} no no'.format(name,name)
    os.system(command2)
