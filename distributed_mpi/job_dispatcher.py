'''
Created on Jun 21, 2018

@author: fangy5
'''
'''
Created on Jun 21, 2018

@author: fangy5
'''


from mpi4py import MPI
import socket
import argparse
import moe
import os
from utility import SystemUtilLogger


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="job_dispatcher")
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        '--batch_size', type=int, default = 512, 
        help="batch size")
    '''
    parser.add_argument(
        '--max_steps', type=int, default = 10000, 
        help="number of batches to run")
    '''
    parser.add_argument(
        '--max_epochs', type=int, default = 150, 
        help="maximum number of epochs to run")
    
    parser.add_argument(
        '--train_dataset_size', type=int, default = 60000, 
        help="train dataset size")
        
    parser.add_argument(
        '--log_frequency', type=int, default = 10, 
        help="how often to log results to the console")
    
    parser.add_argument(
        '--model_dir', default = "mnist/ckpt", 
        help="train directory")
     
    parser.add_argument(
        '--data_dir', default = "mnist/data", 
        help="data directory")
    
    parser.add_argument(
        '--tmp_dir', default = "mnist/tmp", 
        help="temporary directory")
    
    parser.add_argument(
        '--log_dir', default = "mnist/log", 
        help="log directory")    

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
        '--input_shape', default = (None, 28, 28, 1), 
        help="input shape")

    parser.add_argument(
        '--model', default = "basic_fc_relu", 
        help="model")
    
    parser.add_argument(
        '--model_cls', default = None, 
        help="model class")
    
    parser.add_argument(
        '--problem', default = "image_mnist", 
        help="problem")
    
    parser.add_argument(
        '--hparams', default = "basic_fc_small", 
        help="hparams")
    
    parser.add_argument(
        '--generate_data', default = False, 
        help="generate_data")
    
    parser.add_argument(
        '--reshuffle_each_epoch', default = True, 
        help="reshuffle for each epoch")
        
    parser.add_argument(
        '--mode', default = "train", 
        help="mode")
    
    parser.add_argument(
        '--logger', default = None, 
        help="logger")
    
    parser.add_argument(
        '--device', 
        type=str,
        default = "gpu", 
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
        help="Comma-separated list of hostname:port pairs"
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
        "--by_conv",
        type="bool",
        default=False,
        help="split the computation by_conv or by_branch "
        )
    return parser


def main(params):
    print("Use MPI:%s" % params.use_mpi)

    pid = os.getpid()
    print("pid: %d" % pid)
    utilityLogger = SystemUtilLogger(pid, task = params.task, device = params.device)

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
                #print("My hostname is: " + hostname)
                # for ps_hosts_rotate in range(num_ps_hosts):
                #     if hostname == ps_hosts_list[ps_hosts_rotate].split(':')[0]:
                #         print("My job ID is: ps" + str(ps_hosts_rotate))
                #         os.system("python -u " + FLAGS.script + " --ps_hosts=" + FLAGS.ps_hosts + " --worker_hosts=" + FLAGS.worker_hosts + " --job_name=ps --task_index=" + str(ps_hosts_rotate))
                #for worker_hosts_rotate in range(num_worker_hosts):
                    #if hostname == worker_hosts_list[worker_hosts_rotate].split(':')[0]:
                print("My job ID is:worker" + str(rank))
                #command = "python3 " + FLAGS.script + " --worker_hosts=" + FLAGS.worker_hosts + " --job_name=worker --task_index=" + str(rank)

                params.task_index = rank
                utilityLogger.start()
                moe.main(params, comm)
                utilityLogger.stop()
                utilityLogger.join()

    else:
        params.protocol="grpc"
        moe.main(params)


if __name__ == "__main__":
    parser = create_parser()
    params = parser.parse_args()

    #params.device= [0]

    if params.task == "mnist":
        params.hparams = "basic_fc_8"
        params.max_epochs = 100
        params.model_dir = "mnist/ckpt_%s"%(params.hparams)
        params.batch_size = 128
        '''
        for j in range(5):
            params.max_epochs = 20*(j+1)
           
            params.model_dir = "mnist/ckpt_%s"%(hparams)
            params.hparams = hparams
            params.batch_size = batch_size
            
            params.mode = "train"
            params.logger = None
            moe.main(params)
            params.mode = "predict"
            params.logger = None
            moe.main(params)
        '''
    elif params.task == "cifar10":
        params.hparams = "shakeshake_big_l26"
        params.max_epochs = 1200
        params.batch_size = 32
        params.train_dataset_size = 50000
        params.reshuffle_each_epoch = False
        params.model_dir = "cifar10/ckpt_%s_%s"%(params.hparams,'rs' if params.reshuffle_each_epoch else 'nrs')
        params.data_dir = "cifar10/data"
        params.tmp_dir = "cifar10/tmp"
        params.log_dir = "cifar10/log"

        params.input_shape = (None, 32, 32, 3)
        params.model = "shake_shake"
        params.problem = "image_cifar10"

    
    params.mode = "predict"
    params.logger = None
    main(params)
