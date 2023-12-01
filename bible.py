import json
from urllib.request import urlopen


kjv_lines = [b.decode('utf-8-sig') for b in urlopen('https://openbible.com/textfiles/kjv.txt').readlines()[2:]]
kjv_verses = [v.split('\t')[1] for v in kjv_lines]

erv_lines = [b.decode('utf-8-sig') for b in urlopen('https://openbible.com/textfiles/erv.txt').readlines()[2:]]
erv_verses = [v.split('\t')[1] for v in erv_lines]

rv1989_json = json.load(open('data/rv_1909.json'))
rv1989_verses = [v['text'] for v in rv1989_json['verses']]

