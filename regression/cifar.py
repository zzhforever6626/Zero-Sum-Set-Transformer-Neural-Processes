import os
import os.path as osp
import argparse
import yaml
import torch
import time
from attrdict import AttrDict
from tqdm import tqdm

from data.image import img_to_task
from data.cifar import CIFAR10
from utils.misc import load_module
from utils.paths import results_path, evalsets_path
from utils.log import get_logger, RunningAverage

def main():
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument('--mode',
                        choices=['train', 'eval'],
                        default='train')
    parser.add_argument('--expid', type=str, default='default')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default='cuda:0')

    # Data
    parser.add_argument('--max_num_points', type=int, default=200)
    parser.add_argument('--max_num_train_target_points', type=int,
                        default=None)  # Sets the max number of training target points
    parser.add_argument('--resolution', type=int, default=32)

    # Model
    parser.add_argument('--model', type=str, default="zstnp")

    # Train
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--train_seed', type=int, default=42)
    parser.add_argument('--train_batch_size', type=int, default=100)
    parser.add_argument('--train_num_samples', type=int, default=4)
    parser.add_argument('--train_num_bs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)

    # Eval
    parser.add_argument('--eval_seed', type=int, default=42)
    parser.add_argument('--eval_num_bs', type=int, default=50)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--eval_num_samples', type=int, default=50)
    parser.add_argument('--eval_logfile', type=str, default=None)

    # OOD settings
    parser.add_argument('--t_noise', type=float, default=None)

    args = parser.parse_args()

    if args.expid is not None:
        args.root = osp.join(results_path, f'cifar10{args.resolution}', args.model, args.expid)
    else:
        args.root = osp.join(results_path, f'cifar10{args.resolution}', args.model)

    model_cls = getattr(load_module(f'models/{args.model}.py'), args.model.upper())
    with open(f'configs/cifar10{args.resolution}/{args.model}.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model = model_cls(**config)
    model.cuda(device=args.device)

    if args.mode == 'train':
        train(args, model)
    elif args.mode == 'eval':
        eval(args, model)


def train(args, model):
    if osp.exists(args.root + '/ckpt.tar'):
        if args.resume is None:
            raise FileExistsError(args.root)
    else:
        os.makedirs(args.root, exist_ok=True)

    with open(osp.join(args.root, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)

    train_ds = CIFAR10(train=True, resolution=args.resolution)
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=args.train_batch_size,
                                               shuffle=True, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * args.num_epochs)

    if args.resume:
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_epoch = ckpt.epoch
    else:
        logfilename = osp.join(args.root, 'train_{}.log'.format(
            time.strftime('%Y%m%d-%H%M')))
        start_epoch = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    if not args.resume:
        logger.info('Total number of parameters: {}\n'.format(
            sum(p.numel() for p in model.parameters())))

    for epoch in tqdm(range(start_epoch, args.num_epochs + 1)):
        model.train()
        for (x, _) in tqdm(train_loader, ascii=True):
            x = x.cuda(device=args.device)
            batch = img_to_task(x,
                                max_num_points=args.max_num_points,
                                max_num_target_points=args.max_num_train_target_points)
            optimizer.zero_grad()
            if args.model in ["np", "anp", "bnp", "banp"]:
                outs = model(batch, num_samples=args.train_num_samples)
            else:
                outs = model(batch)
            outs.loss.backward()
            optimizer.step()
            scheduler.step()

            for key, val in outs.items():
                ravg.update(key, val)

        line = f'{args.model}:{args.expid} epoch {epoch} '
        line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
        line += ravg.info()
        logger.info(line)

        if epoch % args.eval_freq == 0:
            logger.info(eval(args, model) + '\n')

        ravg.reset()

        if epoch % args.save_freq == 0 or epoch == args.num_epochs:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.epoch = epoch + 1
            torch.save(ckpt, osp.join(args.root, 'ckpt.tar'))

    args.mode = 'eval'
    eval(args, model)


def gen_evalset(args):
    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    eval_ds = CIFAR10(train=False, resolution=args.resolution)
    eval_loader = torch.utils.data.DataLoader(eval_ds,
                                              batch_size=args.eval_batch_size,
                                              shuffle=False, num_workers=4)

    batches = []
    for x, _ in tqdm(eval_loader, ascii=True):
        # No max_num_target_points bc it evaluates on all datapoints.
        batches.append(img_to_task(
            x, max_num_points=args.max_num_points,
            t_noise=args.t_noise)
        )

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    path = osp.join(evalsets_path, f'cifar10{args.resolution}')
    if not osp.isdir(path):
        os.makedirs(path)

    filename = 'no_noise.tar' if args.t_noise is None else \
        f'{args.t_noise}.tar'
    torch.save(batches, osp.join(path, filename))


def eval(args, model):
    if args.mode == 'eval':
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        if args.eval_logfile is None:
            eval_logfile = f'eval'
            if args.t_noise is not None:
                eval_logfile += f'_{args.t_noise}'
            eval_logfile += '.log'
        else:
            eval_logfile = args.eval_logfile
        filename = osp.join(args.root, eval_logfile)
        logger = get_logger(filename, mode='w')
    else:
        logger = None

    path = osp.join(evalsets_path, f'cifar10{args.resolution}')
    if not osp.isdir(path):
        os.makedirs(path)
    filename = f'no_noise.tar' if args.t_noise is None else \
        f'_{args.t_noise}.tar'
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args)

    eval_batches = torch.load(osp.join(path, filename))

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    ravg = RunningAverage()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(eval_batches, ascii=True):
            for key, val in batch.items():
                batch[key] = val.cuda(device=args.device)

            if args.model in ["np", "anp", "bnp", "banp"]:
                outs = model(batch, args.eval_num_samples)
            else:
                outs = model(batch)

            for key, val in outs.items():
                ravg.update(key, val)

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = f'{args.model}:{args.expid} '
    if args.t_noise is not None:
        line += f'tn {args.t_noise} '
    line += ravg.info()

    if logger is not None:
        logger.info(line)

    return line


if __name__ == '__main__':
    main()
