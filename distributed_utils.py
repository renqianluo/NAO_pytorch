from collections import namedtuple
import os
import pickle
import subprocess
import logging
import utils
import torch
import torch.distributed as dist
from torch import nn


def is_master(args):
    return args.distributed_rank == 0


def infer_init_method(args):
    if  args.distributed_init_method is not None:
        return

    # support torch.distributed.launch
    if all(key in os.environ for key in [
        'MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK'
    ]):
        if (not os.environ['MASTER_ADDR'] or os.environ['MASTER_ADDR'].isspace()):
            print('| WARNING: master addr unknown')
            if os.path.isfile(args.master_address_file):
                os.environ['MASTER_ADDR'] = open(args.master_address_file, 'r').read().strip()
                print('Set it to {}'.format(os.environ['MASTER_ADDR']))
            else:
                raise FileNotFoundError('File {} not found'.format(args.master_address_file))
        args.distributed_init_method = 'tcp://{addr}:{port}'.format(
            addr=os.environ['MASTER_ADDR'],
            port=os.environ['MASTER_PORT'],
        )
        args.distributed_world_size = int(os.environ['WORLD_SIZE'])
        args.distributed_rank = int(os.environ['RANK'])

    # we can determine the init method automatically for Slurm
    elif args.distributed_port > 0:
        node_list = os.environ.get('SLURM_JOB_NODELIST')
        if node_list is not None:
            try:
                hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', node_list])
                args.distributed_init_method = 'tcp://{host}:{port}'.format(
                    host=hostnames.split()[0].decode('utf-8'),
                    port=args.distributed_port)
                args.distributed_rank = int(os.environ.get('SLURM_PROCID'))
                args.device_id = int(os.environ.get('SLURM_LOCALID'))
            except subprocess.CalledProcessError as e:  # scontrol failed
                raise e
            except FileNotFoundError:  # Slurm is not installed
                pass


def setup_init_philly_shared_system(args):
    args.distributed_rank = int(os.environ['RANK'])
    args.distributed_world_size = int(os.environ['WORLD_SIZE'])
    import time
    time.sleep(int(os.environ['OMPI_COMM_WORLD_RANK']))
    # args.device_id = (int(os.environ['LOCAL_PROCESS_RANK']))
    args.device_id = args.distributed_rank % torch.cuda.device_count()
    args.distributed_init_method = "file://{}".format(args.distributed_init_method)  # Currently hard-cored here
    args.distributed_world_size = int(os.environ['WORLD_SIZE'])
    logging.info(
        'Dist On Philly, DistRank {}, DistWorldSize {}, Device Id {}'.format(args.distributed_rank,
                                                                             args.distributed_world_size,
                                                                             args.device_id))


def synchronize():
    """Helper function to synchorize between multiple processes when using distributed training."""

    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    if get_world_size() == 1:
        return
    dist.barrier()


def distributed_init(args):
    if args.distributed_world_size == 1:
        raise ValueError('Cannot initialize distributed with distributed_world_size=1')

    logging.info('| distributed init (rank {}): {}'.format(
        args.distributed_rank, args.distributed_init_method))

    dist.init_process_group(
        backend=args.distributed_backend,
        init_method=args.distributed_init_method,
        world_size=args.distributed_world_size,
        rank=args.distributed_rank,
    )

    print('| distributed init success (rank {})'.format(args.distributed_rank))

    # # Run ``synchronize`` right after ``init_process_group`` to fix "Resource temporarily unavailable" error.
    # # See <https://github.com/facebookresearch/maskrcnn-benchmark/issues/172> for more details.
    # synchronize()

    suppress_output(is_master(args))

    return args.distributed_rank


def suppress_output(is_master):
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


def get_default_group():
    return dist.group.WORLD


def all_reduce(tensor, group=None):
    if group is None:
        group = get_default_group()
    return dist.all_reduce(tensor, group=group)


def all_gather_list(data, group=None, max_size=16384):
    """Gathers arbitrary data from all nodes into a list.
    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.
    Args:
        data (Any): data from the local worker to be gathered on other workers
        group (optional): group of the collective
        max_size (int, optional): maximum size of the data to be gathered
            across workers
    """
    rank = get_rank()
    world_size = get_world_size()

    buffer_size = max_size * world_size
    if not hasattr(all_gather_list, '_buffer') or \
            all_gather_list._buffer.numel() < buffer_size:
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
    buffer = all_gather_list._buffer
    buffer.zero_()

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError('encoded data exceeds max_size: {}'.format(enc_size + 2))
    assert max_size < 255*256

    buffer_rank = buffer[rank * max_size : (rank + 1) * max_size]
    buffer_rank[0] = enc_size // 255  # this encoding works for max_size < 65k
    buffer_rank[1] = enc_size % 255
    buffer_rank[2:enc_size+2] = torch.ByteTensor(list(enc))

    all_reduce(buffer, group=group)

    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size : (i + 1) * max_size]
            size = (255 * utils.item(out_buffer[0])) + utils.item(out_buffer[1])
            if size > 0:
                result.append(
                    pickle.loads(bytes(out_buffer[2:size+2].tolist()))
                )
        return result
    except pickle.UnpicklingError:
        raise Exception(
            'Unable to unpickle data from other workers. all_gather_list requires all '
            'workers to enter the function together, so this error usually indicates '
            'that the workers have fallen out of sync somehow. Workers can fall out of '
            'sync if one of them runs out of memory, or if there are other conditions '
            'in your training script that can cause one worker to finish an epoch '
            'while other workers are still iterating over their portions of the data.'
        )