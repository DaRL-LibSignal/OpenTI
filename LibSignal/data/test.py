import numpy, re

str_ = "11111111, delay:0.6956879085964627, throughput:1635""-ahuisahdocdata/output_data/tsc/cityflow_frap/cityflow1x1/0/logger/2023_10_10-14_28_01_DTL.log"

def extract_filepath(s):
    match = re.search(r'data/.*\.log', s)
    return match.group(0) if match else None

print(extract_filepath(str_))