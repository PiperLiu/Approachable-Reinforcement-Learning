import os.path as osp
import sys
DIRNAME = osp.dirname(__file__)
sys.path.append(DIRNAME + '/..')

from AlphaZero.train import TrainPipeline

if __name__ == "__main__":
    training_pipeline = TrainPipeline(DIRNAME + '/current_policy.model')
    training_pipeline.run()

