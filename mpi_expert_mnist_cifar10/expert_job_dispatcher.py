'''
Created on Nov 24, 2018

@author: jinz27
'''

from mpi4py import MPI
import socket
import argparse
import expert_moe
import os
from utility import SystemUtilLogger
import atexit

utilityLogger = None
# def stopLogger():
#     if utilityLogger:
#         utilityLogger.stop()
#         utilityLogger.join()

# atexit.register(stopLogger)

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="expert_mnist")
    
    parser.register("type", "bool", lambda v: v.lower() == "true")
    
    parser.add_argument(
        '--batch_size', type=int, default = 512, 
        help="batch size")
    '''
    parser.add_argument(
        '--max_steps', type=int, default = 50000, 
        help="number of batches to run")
    '''
    parser.add_argument(
        '--max_epochs', type=int, default = 300, 
        help="maximum number of epochs to run")
    
    parser.add_argument(
        '--train_dataset_size', type=int, default = 50000, 
        help="train dataset size")
    
    parser.add_argument(
        '--log_frequency', type=int, default = 10, 
        help="how often to log results to the console")
    
    parser.add_argument(
        '--model_dir', default = "expert_2_cifar10/ckpt", 
        help="train directory")
     
    parser.add_argument(
        '--data_dir', default = "expert_cifar10/data", 
        help="data directory")
    
    parser.add_argument(
        '--tmp_dir', default = "expert_2_cifar10/tmp", 
        help="temporary directory")
    
    parser.add_argument(
        '--log_dir', default = "expert_2_cifar10/log", 
        help="train directory")
    
    parser.add_argument(
        '--log_device_placement', default = False, 
        help="whether to log device placement")
    
    parser.add_argument(
        '--stale_interval', type=int, default = 50, 
        help="stale interval")
    
    parser.add_argument(
        '--mc_steps', type=int, default = 1, 
        help="number of mc steps")

    parser.add_argument(
        '--input_shape', default = (None, 32, 32, 3), 
        help="input shape")

    parser.add_argument(
        '--model', default = "expert_shake_shake", 
        help="model")
    
    parser.add_argument(
        '--model_cls', default = None, 
        help="model class")
    
    parser.add_argument(
        '--problem', default = "image_cifar10", 
        help="problem")
    
    parser.add_argument(
        '--hparams', default = "expert_shakeshake_big", 
        help="hparams")
    
    parser.add_argument(
        '--generate_data', default = False, 
        help="generate_data")
    
    parser.add_argument(
        '--reshuffle_each_epoch', default = False, 
        help="reshuffle for each epoch")
    
    parser.add_argument(
        '--mode', default = "train", 
        help="mode")
    
    parser.add_argument(
        '--logger', default = None, 
        help="logger")
    
    parser.add_argument(
        '--device', default = "gpu", 
        help="device")
    
    parser.add_argument(
        '--verbose', default = False, 
        help="verbose")

    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
        )
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Comma-separated list of hostname:port pairs"
        )
    parser.add_argument(
        "--job_name",
        type=str,
        default="worker",
        help="Comma-separated list of hostname:port pairs"
        )
    parser.add_argument(
        "--use_mpi",
        type="bool",
        default=True,
        help="if use MPI"
        )
    parser.add_argument(
        "--protocol",
        type=str,
        default="grpc+mpi",
        help="Comma-separated list of hostname:port pairs"
        )
    parser.add_argument(
        "--task",
        type=str,
        default="mnist",
        help="model to run"
        )
    parser.add_argument(
        '--num_experts', type=int, default = 2, 
        help="num_experts")
    return parser

def main(params):
    pid = os.getpid()
    print("pid: %d" % pid)
    print("Use MPI:%s" % params.use_mpi)

    # utilityLogger = SystemUtilLogger(pid, task = "expert_%d_cifar10" % (params.num_experts))
    # utilityLogger.start()

    # utilityLogger.stop()
    # utilityLogger.join()

    if params.use_mpi:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        worker_hosts_list = params.worker_hosts.split(',')
        num_worker_hosts = len(worker_hosts_list)
        num_hosts = num_worker_hosts

        params.protocol="grpc+mpi"

        if(size != num_hosts):
            print("size:",size,'# OF HOSTS',num_hosts)
            print("ERROR")
            return

        for rank_rotate in range(num_hosts):
            if rank == rank_rotate:
                print("I am rank " + str(rank_rotate) + "...")
                hostname = socket.gethostname()
                
                print("My job ID is:worker" + str(rank))

                params.task_index = rank
                # utilityLogger.start()
                expert_moe.main(params, comm)
                # utilityLogger.stop()
                # utilityLogger.join()

    else:
        params.protocol="grpc"
        expert_moe.main(params)

if __name__ == "__main__":
    parser = create_parser()
    params = parser.parse_args()


    if params.task == "mnist":
        if params.num_experts == 2:
            hparams = "expert_basic_fc_4"
            batch_size = 128
        elif params.num_experts == 4:
            hparams = "expert_basic_fc_2"
            batch_size = 128

        params.max_epochs = 800
        params.input_shape = (None, 28, 28, 1)
        params.reshuffle_each_epoch = True
        params.model= "expert_basic_fc_relu"
        params.train_dataset_size = 60000
        params.problem = "image_mnist"
        params.model_dir = "expert_%d_mnist/ckpt_%s_%s"%(params.num_experts,hparams,'rs' if params.reshuffle_each_epoch else 'nrs')
        params.data_dir = "expert_mnist/data"
        params.log_dir = "expert_%d_mnist/log"%(params.num_experts)
        params.tmp_dir = "expert_%d_mnist/tmp"%(params.num_experts)

    elif params.task == "cifar10":
        if params.num_experts == 2:
            hparams = "expert_shakeshake_big_l14"
            batch_size = 32
            params.max_epochs = 1200
        elif params.num_experts == 4:
            hparams = "expert_shakeshake_big_l8"
            batch_size = 32
            params.max_epochs = 4800
        else:
            hparams = "expert_shakeshake_big_l8"
            batch_size = 32
            params.max_epochs = 4800
        
        params.model_dir = "expert_%d_cifar10/ckpt_%s_%s"%(params.num_experts,hparams,'rs' if params.reshuffle_each_epoch else 'nrs')
        # params.data_dir = "expert_%d_cifar10/data"%(params.num_experts)
        params.log_dir = "expert_%d_cifar10/log"%(params.num_experts)
        params.tmp_dir = "expert_%d_cifar10/tmp"%(params.num_experts)
    else:
        print("Wrong task")


    
    params.hparams = hparams
    params.batch_size = batch_size

    params.mode = "predict"
    params.logger = None

    pid = os.getpid()
    print("pid: %d" % pid)
    utilityLogger = SystemUtilLogger(pid, device = params.devce, task = "expert_%d_%s" % (params.num_experts, params.task))
    utilityLogger.start()

    main(params)

    utilityLogger.stop()
    utilityLogger.join()
