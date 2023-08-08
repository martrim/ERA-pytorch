import argparse


def parse_era_arguments():
    parser = argparse.ArgumentParser(description='Parsing Arguments.')
    parser.add_argument('--degree_denominator', choices=['2', '4'], default='2',
                        help='the degree of the denominator of the ERA')
    parser.add_argument('--initialization', choices=['leaky', 'relu', 'swish', 'gelu', 'random'], default='random',
                        help='the initialization of the ERA')
    return parser.parse_args()

