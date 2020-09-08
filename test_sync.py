import torch
from options  import test_opts
from utils    import logger, recorders
from datasets import data_loader
from models   import model_builder

from modules import tester

args = test_opts.TestOpts().parse()
log  = logger.Logger(args)

def main(args):
    model_est = model_builder.buildEstModel(args)
    model_ref = model_builder.buildRefModel(args)
    models = [model_est, model_ref]
    recorder  = recorders.Records(args.log_dir)
    test_loader = data_loader.testDataloader(args) 
    tester.test(args, 'test', test_loader, models, log, recorder)
    log.plotCurves(recorder, 'test')

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    main(args)
