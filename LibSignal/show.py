from utils.painter import painter
path = "/Users/danielsmith/Documents/PythonFiles/workspace/libSignalDoc/LibSignalDoc/codesrc/LibSignal/data/output_data/tsc/cityflow_presslight/cityflow1x1/0/logger/2023_10_10-13_08_57_DTL.log"

painter({'hz1x1': path}, ['epoch', 'average travel time', 'rewards','delay'])
