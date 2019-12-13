from . import model_utils
import torch

def buildEstModel(args):
    print('Creating Est Model [{}]'.format(args.est_model))
    in_c = 2 if args.grayscale else 6    
    models = __import__('models.' + args.est_model)
    model_file = getattr(models, args.est_model)
    model = getattr(model_file, args.est_model)(in_c, args.num_para, args.fc_height, args.fc_width)
    if args.cuda: model = model.cuda()

    if args.pretrain_est:
        args.log.printWrite("#### Load pretrained Est-Model from '{}' ####".format(args.pretrain_est))
        model_utils.loadCheckpoint(args.pretrain_est, model, cuda=args.cuda)
        
    if args.resume_est:
        args.log.printWrite("#### Resume Est-Model from '{}' ####".format(args.resume_est))
        model_utils.loadCheckpoint(args.resume_est, model, cuda=args.cuda)
        args.resume = args.resume_est

    print(model)
    args.log.printWrite("=> Est-Model Parameters: {0:d}".format(model_utils.get_n_params(model)))
    return model

def buildRefModel(args):
    print('Creating Ref Model [{}]'.format(args.ref_model))
    in_c = model_utils.RefInputChanel(args)
    out_c = 2 if args.grayscale else 6
    models = __import__('models.' + args.ref_model)
    model_file = getattr(models, args.ref_model)
    model = getattr(model_file, args.ref_model)(in_c, out_c, args.use_BN)
    if args.cuda: model = model.cuda()

    if args.pretrain_ref: 
        args.log.printWrite("#### Load pretrained Ref-Model from '{}' ####".format(args.pretrain_ref))
        model_utils.loadCheckpoint(args.pretrain_ref, model, cuda=args.cuda)

    if args.resume_ref:
        args.log.printWrite("#### Resume Ref-Model from '{}' ####".format(args.resume_ref))
        model_utils.loadCheckpoint(args.resume_ref, model, cuda=args.cuda)
        args.resume = args.resume_ref

    print(model)
    args.log.printWrite("=> Ref-Model Parameters: {0:d}".format(model_utils.get_n_params(model)))
    return model
