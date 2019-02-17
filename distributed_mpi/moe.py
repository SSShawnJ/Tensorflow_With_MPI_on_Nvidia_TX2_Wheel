'''
Created on Jul 18, 2018

@author: yfang
'''

from multiprocessing import Process, Manager
import time
import os
import numpy as np
from datetime import datetime
from os import listdir
from os.path import isfile, isdir, join
import re
import logging

import tensorflow as tf
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import registry
from tensor2tensor import problems
from tensor2tensor.utils import t2t_model
from tensorflow.python.client import session
from tensorflow.python.training import saver as saver_mod


# class GPUtil (Process):
#     def __init__(self):
#         Process.__init__(self)
#         self.manager = Manager()
#         self.r_queue = self.manager.Queue(1)
        
#     def run(self):
#         self.r_queue.put(len(self.get_available_gpus()))
        
#     def get_available_gpus(self):
#         from tensorflow.python.client import device_lib
#         local_device_protos = device_lib.list_local_devices()
#         return [x.name for x in local_device_protos if x.device_type == 'GPU']

# gputil = GPUtil()
# gputil.start()
# NUM_GPUS = gputil.r_queue.get()
# gputil.join()


def get_ckpt_iters(params):
    ds = [d for d in listdir(params.model_dir) if isdir(join(params.model_dir, d))]
    all_num_iters = []
    for d in ds:
        d_full = join(params.model_dir, d)
        fs=[f for f in listdir(d_full) if isfile(join(d_full, f))] 
        num_iters = []
        for f in fs:
            m = re.search('\-([0-9]+).meta', f)
            if m:
                num_iters.append(int(m.group(1)))
        num_iter = np.max(num_iters) if len(num_iters) > 0 else 0
        all_num_iters.append(num_iter)
    all_num_iter = np.max(all_num_iters) if len(all_num_iters) > 0 else 0
    return all_num_iter

class DataReader ():
    def __init__(self, name, device, params):
        
        self.name = name
        self.device = device
        self.params = params
        #self.manager = Manager()
        #self.r_queue = self.manager.Queue(1)
        
    def run(self):
        self.params.logger.info("begin DataReader")
        if self.device is None:
            device_id = ""
        else:
            device_id = self.device[len(self.device)-1]
        os.environ["CUDA_VISIBLE_DEVICES"]=device_id
        
        # Enable TF Eager execution
        # config = tf.ConfigProto(
        #         log_device_placement=self.params.log_device_placement,
        #         allow_soft_placement=True,
        #         device_count = {'GPU': 0}
        #         )
        #config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction=0.5

        tfe = tf.contrib.eager
        tfe.enable_eager_execution()
        
        # Setup the training data
        
        # Fetch the MNIST problem
        problem = problems.problem(self.params.problem)
        # The generate_data method of a problem will download data and process it into
        # a standard format ready for training and evaluation.
        if self.params.generate_data == True:
            problem.generate_data(self.params.data_dir, self.params.tmp_dir)

        Modes = tf.estimator.ModeKeys
        
        if self.params.mode == "train":
            mode = Modes.TRAIN
            max_epochs = self.params.max_epochs
            start_epoch = get_ckpt_iters(self.params)*self.params.batch_size//self.params.train_dataset_size
            num_repeats = max_epochs-start_epoch

        elif self.params.mode == "predict":
            mode = Modes.EVAL
            max_epochs = self.params.max_epochs
            start_epoch = get_ckpt_iters(self.params)*self.params.batch_size//self.params.train_dataset_size
            num_repeats = 1
            self.params.logger.info("epoch #%d"%self.params.max_epochs)
            
        model_data = []
        if num_repeats > 0:
            dataset = problem.dataset(mode, self.params.data_dir)
            
            dataset = dataset.shuffle(buffer_size = 256, reshuffle_each_iteration=self.params.reshuffle_each_epoch)
            dataset = dataset.repeat(num_repeats).batch(self.params.batch_size)
            
            pre_r = -1
            for count, example in enumerate(tfe.Iterator(dataset)):
                if self.params.mode == "train":
                    r = start_epoch + count*self.params.batch_size//self.params.train_dataset_size
                elif self.params.mode == "predict":
                    r = start_epoch
                    
                if r > pre_r:
                    self.params.logger.info("epoch #%d"%(r+1))
                    pre_r = r
                
                inputs, targets = example["inputs"], example["targets"]
                model_data.append([inputs.numpy(), targets.numpy()])

        self.params.logger.info("end DataReader")   
        return model_data
  

class Expert ():
    def __init__(self, name, device, params, data, server, cluster):
        Process.__init__(self)
        
        self.name = name
        self.device = device
        self.params = params
        self.data = data
        self.server = server
        self.cluster = cluster
        
        #self.manager = Manager()
        #self.c_queue = self.manager.Queue(1)
        #self.r_queue = self.manager.Queue(1)


    def run(self):
        self.params.logger.info("begin Expert")
        if self.device is None:
            device_id = ""
        else:
            device_id = self.device[len(self.device)-1]
        os.environ["CUDA_VISIBLE_DEVICES"]=device_id 

        """Train CIFAR-10 for a number of steps."""
        with tf.Graph().as_default():
            with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % self.params.task_index,
            cluster=self.cluster)):
                global_step = tf.train.get_or_create_global_step()
            
                inputs = tf.placeholder(tf.float32, shape=self.params.input_shape)
                targets = tf.placeholder(tf.int32, shape=(None))

                problem = problems.problem(self.params.problem)
                # Create your own model
                hparams = trainer_lib.create_hparams(self.params.hparams, data_dir=self.params.data_dir, problem_name=self.params.problem)
                hparams.add_hparam("by_conv", self.params.by_conv)
                Modes = tf.estimator.ModeKeys
                   
                if self.params.model_cls is not None:
                    model = self.params.model_cls(hparams, Modes.TRAIN)
                else:
                    model = registry.model(self.params.model)(hparams, Modes.TRAIN)

                    
                example = {}
                example["inputs"] = inputs
                example["targets"] = targets
                example["targets"] = tf.reshape(example["targets"], [-1, 1, 1, 1])
                    
                logits, losses_dict = model(example)
                logits = tf.reshape(logits, (-1, logits.get_shape()[-1]))
                    
                #optimizer = tf.train.AdamOptimizer()
                    
                # Accumulate losses
                loss = tf.add_n([losses_dict[key] for key in sorted(losses_dict.keys())])
                    
                #train_op = optimizer.minimize(loss, global_step)
                train_op = model.optimize(loss)
                    
                probs = tf.nn.softmax(logits, axis=1)               

                params = self.params
    
            class _LoggerHook(tf.train.SessionRunHook):
                """Logs loss and runtime."""
    
                def begin(self):
                    self._start_time = time.time()
    
                def before_run(self, run_context):
                    
                    return tf.train.SessionRunArgs([loss, global_step])    # Asks for loss value.
    
                def after_run(self, run_context, run_values):
                    loss_value, global_step_value = run_values.results
                    if global_step_value % params.log_frequency == 0:
                        current_time = time.time()
                        duration = current_time - self._start_time
                        self._start_time = current_time
    
                        examples_per_sec = params.log_frequency * params.batch_size / duration
                        sec_per_batch = float(duration / params.log_frequency)
    
                        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                                    'sec/batch)')
                        params.logger.info (format_str % (datetime.now(), global_step_value, loss_value,
                                                                 examples_per_sec, sec_per_batch))
            
            def get_predictive_entropy(probs):
                probs = np.maximum(probs, 1e-37)
                entropy = -np.sum(probs * np.log(probs), axis = 1)
                return entropy 
            
            
            saver = tf.train.Saver()

            scaffold = tf.train.Scaffold(saver=saver)
            
            checkpoint_dir = self.params.model_dir + "/" + self.name
            saver_hook = tf.train.CheckpointSaverHook(
                checkpoint_dir, save_steps=self.params.stale_interval, scaffold=scaffold)
                   
            '''
            if self.device is None:
                config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, 
                        allow_soft_placement=True, device_count = {'CPU': 1},
                        log_device_placement=self.params.log_device_placement)
            else:
            '''
            config = tf.ConfigProto(
                log_device_placement=self.params.log_device_placement,
                allow_soft_placement=True
                )
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction=0.8
            #tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            
            print("Start monitored session")
            with tf.train.MonitoredTrainingSession(
                    master=self.server.target,
                    is_chief=(self.params.task_index==0),
                    checkpoint_dir=checkpoint_dir,
                    scaffold=scaffold,
                    hooks=[saver_hook, 
                           tf.train.NanTensorHook(loss),
                            _LoggerHook()
                          ],
                    config=config,
                    save_checkpoint_secs=None,
                    save_summaries_secs=None, 
                    save_summaries_steps=None, 
                    ) as mon_sess:


                #self.r_queue.put("ready")
                all_accs = []
                all_uncs = []
                all_ts = []

                for batch in self.data:
                    bt = time.time()

                    batch_xs, batch_ys = batch
                            
                    ps_sum = None
                    for _ in range(self.params.mc_steps):
                        ps = mon_sess.run(probs, {inputs: batch_xs, targets: batch_ys})
                        ps_sum = (ps if ps_sum is None else ps_sum + ps)
                            
                    ps = ps_sum/self.params.mc_steps

                    unc = get_predictive_entropy(ps)
                                
                    acc = np.equal(np.argmax(ps, axis=1), np.reshape(batch_ys, -1))
                                
                    et = time.time()
                    t = et - bt

                    all_accs.append(acc)
                    all_uncs.append(unc)
                    all_ts.append(t)
                    params.logger.info('elapsed time: %.3f s' % t)

                            
                print("Return results")
                return [all_accs, all_uncs, all_ts]

        self.params.logger.info("end Expert")
                    # sess_exit = False
                    # while not sess_exit:
                    #     command = self.c_queue.get()
                    #     if command[0] == "train":
                    #         if command[1] is not None:
                    #             batch_xs, batch_ys = command[1]
                    #             mon_sess.run(train_op, {inputs: batch_xs, targets: batch_ys})

                    #             self.r_queue.put(None)
                                
                    #     if command[0] == "predict":
                    #         if command[1] is not None:       
                    #             bt = time.time()
                                
                    #             batch_xs, batch_ys = command[1] 
                                
                    #             ps_sum = None
                    #             for _ in range(self.params.mc_steps):
                    #                 print("here 3")
                    #                 ps = mon_sess.run(probs, {inputs: batch_xs, targets: batch_ys})
                    #                 ps_sum = (ps if ps_sum is None else ps_sum + ps)
                            
                    #             print("after sess run")
                    #             ps = ps_sum/self.params.mc_steps

                    #             unc = get_predictive_entropy(ps)
                                
                    #             acc = np.equal(np.argmax(ps, axis=1), np.reshape(batch_ys, -1))
                                
                    #             et = time.time()
                    #             t = et - bt
                                
                    #             print("return results")
                    #             self.r_queue.put([acc, unc, t])
                        
                    #     elif command[0] == "restore":
                    #         '''
                    #         ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                    #         if ckpt and ckpt.model_checkpoint_path:
                    #             self.params.logger.info("Restores from checkpoint: %s" % ckpt.model_checkpoint_path)
                    #             saver.restore(mon_sess, ckpt.model_checkpoint_path) 
                    #         '''
                            
                    #         #mon_sess._coordinated_creator._session_creator._get_session_manager()._restore_checkpoint(mon_sess._coordinated_creator._session_creator._master, mon_sess._coordinated_creator._session_creator._scaffold._saver, checkpoint_dir)    
                            
                    #         session_creator = mon_sess._coordinated_creator._session_creator
                    #         session_manager = session_creator._get_session_manager()
                            
                    #         session_manager._target = session_creator._master
                    #         sess = session.Session(session_manager._target, graph=session_manager._graph, config=config)
                            
                    #         ckpt = saver_mod.get_checkpoint_state(checkpoint_dir)
                    #         if ckpt and ckpt.model_checkpoint_path:
                    #             self.params.logger.info("Restores from checkpoint: %s" % ckpt.model_checkpoint_path)
                    #             # Loads the checkpoint.
                    #             session_creator._scaffold._saver.restore(sess, ckpt.model_checkpoint_path)    
                                
                    #         self.r_queue.put(None)     
                    #     elif command[0] == "exit":
                    #         sess_exit = True
                    #         self.r_queue.put(None)  
                        
 
    
def get_device_id(device, i):
    if device is None:
        device_id = i%NUM_GPUS
    else:
        device_id = device[i%len(device)]
    return device_id



'''
num_experts must be 1
'''
def fast_train(params):
    reader = DataReader("reader", None, params)
    reader.start()
    
    params.logger.info("creating model sessions")
    model_name = params.model

    expert = Expert(model_name , "/device:GPU:%d"%get_device_id(params.device, 0), params)
    expert.start()
    
    while True:
        values = reader.r_queue.get()
        
        if values is None:
            break
        batch_xs, batch_ys = values
        expert.c_queue.put(["train",[batch_xs, batch_ys]])
        expert.r_queue.get()
           
    expert.c_queue.put(["exit"])
    expert.r_queue.get()
    expert.join()
  

    
'''
num_experts must be 1
'''
def fast_predict(params, server, cluster, comm):
    if params.job_name == "ps":
        server.join()
    elif params.job_name == "worker":
        if (params.task_index==0):
            reader = DataReader("reader", None, params)
            data = reader.run()
            
            params.logger.info("creating model sessions")
            model_name = params.model
            
            expert = Expert(model_name , None, params, data, server, cluster)
            
            all_accs, all_uncs, all_ts = expert.run()
            
            all_accs = np.concatenate(all_accs, axis=0)
            all_uncs = np.concatenate(all_uncs, axis=0)

            numpy_all_ts = np.array(all_ts)

            print('%s: precision: %.3f (std:%.3f) , elapsed time: %.3f ms (std:%.3f)' % (datetime.now(), np.mean(all_accs), np.std(all_accs), 
                                                                                        1e3*np.sum(all_ts)/len(all_accs), np.std(1e3*numpy_all_ts/len(all_accs))))
            params.logger.info('%s: precision: %.3f (std:%.3f) , elapsed time: %.3f ms (std:%.3f)' % (datetime.now(), np.mean(all_accs), np.std(all_accs), 
                                                                                        1e3*np.sum(all_ts)/len(all_accs), np.std(1e3*numpy_all_ts/len(all_accs))))
        
        comm.Barrier()
        

def get_logger(params):
    logger = logging.getLogger("%s_%s_%s_%s"%(params.mode, params.hparams, 'rs' if params.reshuffle_each_epoch else 'nrs', params.device))
    logger.setLevel(logging.DEBUG if params.verbose else logging.INFO)
    if not logger.hasHandlers():
        # create file handler which logs even debug messages
        fh = logging.FileHandler(join(params.log_dir, "%s_%s_%s_%s.log"%(params.mode, params.hparams, 'rs' if params.reshuffle_each_epoch else 'nrs', params.device)))
        fh.setLevel(logging.DEBUG if params.verbose else logging.INFO)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(process)d/%(threadName)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


def main(params, comm):   

    if not os.path.exists(params.model_dir):
        os.makedirs(params.model_dir)
    if not os.path.exists(params.tmp_dir):
        os.makedirs(params.tmp_dir)
    if not os.path.exists(params.data_dir):
        os.makedirs(params.data_dir)
    if not os.path.exists(params.log_dir):
        os.makedirs(params.log_dir)

    if params.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        
    max_epochs = params.max_epochs
    start_epoch = get_ckpt_iters(params)*params.batch_size//params.train_dataset_size
    
        
    if params.logger is None:
        params.logger = get_logger(params)

    worker_hosts = params.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"worker": worker_hosts})

    print("Starting tf server")
    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                            job_name=params.job_name,
                            task_index=params.task_index,
                            protocol=params.protocol)
    comm.Barrier()

    if params.mode == "train":
        if max_epochs-start_epoch > 0: 
            fast_train(params)
            
    if params.mode == "predict":
        if max_epochs-start_epoch >= 0: 
            fast_predict(params, server, cluster, comm)

if __name__ == '__main__':
    main()
