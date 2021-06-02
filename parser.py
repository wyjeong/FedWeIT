import argparse


class Parser:

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()

    def set_arguments(self):
        
        self.parser.add_argument('--work-type', type=str, help='to set work types e.g. gen-data, train ')
        self.parser.add_argument('--gpu', type=str, help='to set gpu ids to use')
        self.parser.add_argument('--gpu-mem-multiplier', type=int, help='to set gpu memory size (GB) ')
        
        self.parser.add_argument('--model', type=str, help='to set model to experiment')
        self.parser.add_argument('--task', type=str, help='to set tasks (e.g., non_iid_50, etc.)')
        self.parser.add_argument('--seed', type=int, help='to set seed')
        self.parser.add_argument('--num-rounds', type=int, help='to set number of rounds per task')
        self.parser.add_argument('--num-epochs', type=int, help='to set number of epochs per round')
        self.parser.add_argument('--batch-size', type=int, help='to set batch size')

    def parse(self):
        args, unparsed  = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        return args
