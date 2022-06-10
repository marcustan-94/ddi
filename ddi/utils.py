import os

def get_rawdata_filepath(filename):
    return os.path.join(os.path.dirname(__file__), '..', 'raw_data', filename)


def get_data_filepath(filename):
    return os.path.join(os.path.dirname(__file__), 'data', filename)
