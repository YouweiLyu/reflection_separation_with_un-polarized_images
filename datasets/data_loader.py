import torch.utils.data
from datasets.Ref_Dataloader import Ref_Dataloader as dataloader
from datasets.Real_Dataloader import Real_Dataloader as dataloader_r
def testDataloader(args):
    args.log.printWrite("=> fetching test data in {}".format(args.data_dir_val))
    test_set = dataloader(args.data_dir_val)

    args.log.printWrite('Found Data:\t %d Val' % (len(test_set)))
    args.log.printWrite('\t Val Batch: %d' % (args.val_batch))

    test_loader  = torch.utils.data.DataLoader(test_set , batch_size=args.val_batch,
        num_workers=args.workers, pin_memory=args.cuda, shuffle=False)
    return test_loader

def realDataloader(args):
    test_set = dataloader_r(args.data_dir_val)
    test_loader  = torch.utils.data.DataLoader(test_set , batch_size=args.val_batch,
        num_workers=args.workers, pin_memory=args.cuda, shuffle=False)
    return test_loader
