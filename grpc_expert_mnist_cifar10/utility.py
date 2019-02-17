# import os
# import sys
# from subprocess import Popen, PIPE, STDOUT

# script_path = os.path.join(get_script_dir(), 'parent.py')
# p = Popen([sys.executable, '-u', script_path],
#           stdout=PIPE, stderr=STDOUT, bufsize=1)
# with p.stdout:
#     for line in iter(p.stdout.readline, b''):
#         print line,
# p.wait()
import re
import os
from datetime import datetime
import subprocess
from threading import Thread
import numpy as np
import psutil
import copy

BIN_PATH = '~/tegrastats'

class SystemUtilLogger(Thread):
	def __init__(self, pid, task = "", device = "cpu", LOG_FILE_PATH = "/log/system_utility_log", STATS_FILE_PATH = "/log/system_utility_stats"):
		Thread.__init__(self)
		self.RAM_CPU_GPU = re.compile(r"^RAM (\d+/\d+)MB .+ CPU \[(\d+)%@\d+,off,off,(\d+)%@\d+,(\d+)%@\d+,(\d+)%@\d+\] .+ GR3D_FREQ (\d+)%@.+")
		self.pid = pid
		self.shoudStop = False
		self.task = task
		self.device = device
		self.LOG_FILE_PATH = LOG_FILE_PATH
		self.STATS_FILE_PATH = STATS_FILE_PATH

		self.complete_time = None
		self.memory = []
		self.memory_top = []
		self.top_cpu = []
		self.cpu1 = []
		self.cpu2 = []
		self.cpu3 = []
		self.cpu4 = []
		self.gpu  = []

	def stop(self):
		self.shoudStop = True

	def run(self):
		print("Start system utility logger")
		reference = copy.deepcopy(psutil.virtual_memory().used)
		cmds = ["echo 'nvidia' |sudo -S " + BIN_PATH +' --interval 200']
		p = subprocess.Popen(cmds, stdout=subprocess.PIPE, shell=True)
		p_mem = subprocess.Popen("top -b -d 0.2 -p %d | grep %d" % (self.pid,self.pid), stdout=subprocess.PIPE, shell=True)
		LOG_FILE = open(self.task+self.LOG_FILE_PATH+"_"+self.device+".txt","a")
		while not self.shoudStop:
			current_stat = p.stdout.readline().decode().strip()
			mem_stat = p_mem.stdout.readline().decode().strip()
			mem_stats = psutil.virtual_memory()
			
			mem_splited = mem_stat.split()

			mem_stat_1 = mem_splited[-3]
			top_cpu = mem_splited[-4]
			timestamp = datetime.now()
			if current_stat == '':
				print("tegrastats error")
				break
			memory, cpu1, cpu2, cpu3, cpu4, gpu = self.RAM_CPU_GPU.sub(r'\1,\2,\3,\4,\5,\6',current_stat).split(",")
			memory = (mem_stats.used - reference)/mem_stats.total * 100
			text = "%s,%s%%,%s%%,%s%%,%s%%,%s%%,%s%%" % (str(timestamp), memory, cpu1, cpu2, cpu3, cpu4, gpu)
			#print(text)
			LOG_FILE.write(text+"\n")

			self.complete_time = str(timestamp)
			#[memory_usage,memory_total] = memory.split("/")
			try:
				m = float(mem_stat_1)
				c = float(top_cpu)
				if m>100 or c>300 or memory>100:
					continue

				self.memory.append(memory)
				self.memory_top.append(m)
				self.top_cpu.append(c)
				self.cpu1.append(int(cpu1))
				self.cpu2.append(int(cpu2))
				self.cpu3.append(int(cpu3))
				self.cpu4.append(int(cpu4))
				self.gpu.append(int(gpu))
			except ValueError:
				print(mem_stat)
				continue
	    
		LOG_FILE.close()
		self.calculate_statistics()
		#os.killpg(os.getpgid(p.pid), signal.SIGTERM)
		print("System utility logger stopped")

	def calculate_statistics(self):
		print("Calculating system utility statstics...")
		np_memory = np.array(self.memory)
		np_memory_top = np.array(self.memory_top)
		np_top_cpu = np.array(self.top_cpu)
		np_cpu1 = np.array(self.cpu1)
		np_cpu2 = np.array(self.cpu2)
		np_cpu3 = np.array(self.cpu3)
		np_cpu4 = np.array(self.cpu4)
		np_gpu  = np.array(self.gpu)

		STATS_FILE = open(self.task+self.STATS_FILE_PATH+"_"+self.device+".txt","a")
		STATS_FILE.write("Complete Time:%s, Avg Memory Usage (%%):%.1f, Std Memory Usage:%.1f, Max Memeory Usage(%%):%.1f, Max Memeory(top) Usage(%%):%.1f, Avg CPU(top) Usage (%%):%.1f, Std CPU(top) Usage:%.1fAvg CPU1 Usage (%%):%.1f, Std CPU1 Usage:%.1f, Avg CPU2 Usage (%%):%.1f, Std CPU2 Usage:%.1f, Avg CPU3 Usage (%%):%.1f, Std CPU3 Usage:%.1f, Avg CPU4 Usage (%%):%.1f, Std CPU4 Usage:%.1f, Avg GPU Usage (%%):%.1f, Std GPU Usage:%.1f\n" 
			% ( self.complete_time,
				np.mean(np_memory_top),
				np.std(np_memory_top),
				np.max(np_memory),
				np.max(np_memory_top),
				np.mean(np_top_cpu),
				np.std(np_top_cpu),
				np.mean(np_cpu1),
				np.std(np_cpu1),
				np.mean(np_cpu2),
				np.std(np_cpu2),
				np.mean(np_cpu3),
				np.std(np_cpu3),
				np.mean(np_cpu4),
				np.std(np_cpu4),
				np.mean(np_gpu),
				np.std(np_gpu)
				)
			)
		STATS_FILE.close()
		print("%s %.1f%%(%.1f) %.1f%% %.1f%%(%.1f) %.1f%%(%.1f) %.1f%%(%.1f) %.1f%%(%.1f) %.1f%%(%.1f) %.1f%%(%.1f)\n" 
			% ( self.complete_time,
				np.mean(np_memory_top),
				np.std(np_memory_top),
				#np.max(np_memory),
				np.max(np_memory_top),
				np.mean(np_top_cpu),
				np.std(np_top_cpu),
				np.mean(np_cpu1),
				np.std(np_cpu1),
				np.mean(np_cpu2),
				np.std(np_cpu2),
				np.mean(np_cpu3),
				np.std(np_cpu3),
				np.mean(np_cpu4),
				np.std(np_cpu4),
				np.mean(np_gpu),
				np.std(np_gpu)
				)
			)




# os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
# def exit_pro(signum, frame):
# 	if LOG_FILE:
# 		LOG_FILE.close()
# 		print("Utility logger stopped")
# 		exit()


# signal.signal(signal.SIGINT, exit_pro)
# signal.signal(signal.SIGTERM, exit_pro)
# where the tegrastats binary file is

# output_file = open(LOG_FILE_PATH,"a")


if __name__ == '__main__':
	import time
	# import os

	# pid = os.getpid()
	# logger = SystemUtilLogger(pid, task = "mnist")
	# logger.start()
	# time.sleep(5)
	# logger.stop()
	# logger.join()
	import os

	pid = os.getpid()
	print(pid)
	while True:
		p = subprocess.Popen("top -b -d 0.1 -p %d | grep %d" % (pid,pid), stdout=subprocess.PIPE, shell=True)
		stat = p.stdout.readline().decode().strip()
		mem = stat.split()[-3]

		print(stat)
		print(mem)
		time.sleep(1)
