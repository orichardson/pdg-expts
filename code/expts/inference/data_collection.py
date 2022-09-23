"""
Given a PDG, collet 
"""

from collections import namedtuple
from doctest import OutputChecker
import json
import numpy as np

from pgmpy.inference import BeliefPropagation
from pgmpy.utils import get_example_model

import sys
#sys.path.append("../../..")
sys.path.append("../..")
print(sys.path)

from pdg.pdg import PDG
from pdg.store import TensorLibrary
from pdg.rv import Variable as Var
from pdg.dist import CPT, RawJointDist as RJD, Dist

from pdg.alg import interior_pt as ip
from pdg.alg import torch_opt


## TIMING / LOGGING UTILS
import psutil
from psutil._common import bytes2human
# from multiprocessing import Process
import multiprocessing as multiproc
import time, datetime
import pickle
import logging
import os


# logging.basicConfig(format='%(asctime)s %(message)s', 
# 	filename='example.log', encoding='utf-8', level=logging.DEBUG)



#### INDEPENDENT VARIABLES / INPUTS #######
#  - method (lir / cvx opt / ...)
#  - input stats (size of graph, etc.)
#  - hyperparameters (learning rate, iterations, tol, optimizer)
#  - gamma
#
#### DEPENDENT VARIABLES / OUTPUTS ########
#  - time taken
#  - memory taken
#  - training curve (if available): loss over time
#  - (Inc, Idef) of final product
DataPt = namedtuple('Datum', 
	# ['result', 'total_time', 'max_mem'])
	['method', 'input_stats', 'input_name', 'parameters', 'gamma',
		'inc', 'idef', 'total_time', 'max_mem', 'timestamp']
)

def run_expt_log_datapt_worker(
			input_name, job_number, input_stats, rslt_connection,
			fn,	args, kwargs, output_processor=None
		) -> DataPt:
	""" this is the worker method.
	"""


	init_time = time.time()
	# init_mem  = psutil.Process(os.getpid()).memory_info().rss
	init_mem = total_mem_recursive(os.getpid())

	try:
		# rslt = fn(M, *args, **kwargs)
		rslt = fn(*args, **kwargs)
	except:
		with open(f"datapts/{input_name}-{job_number}.err", "w") as f:
			json.dump(datapt, f)

	# prefix = f"{input_name+'-'+str(job_number):>20}|"
	# print(prefix, "requesting memory")
	# connection.send("done!")
	# mem_connection.send(os.getpid())

	total_time = time.time() - init_time
	# print(prefix, "about to wait for memory usage")
	# max_mem = mem_connection.recv() # should have sent back memory usage.
	# print(prefix, "recieved memory usage")
	# mem_diff = max_mem - init_mem

	if output_processor is None:
		M = args[0] # assume M is first argument
		inc = M.Inc(rslt)
		idef = M.IDef(rslt)
	else:
		inc,idef = output_processor(rslt)

	datapt = DataPt(
		method=fn.__name__,
		input_stats = input_stats,
		input_name = input_name,
		parameters=(args, kwargs),
		gamma=kwargs['gamma'] if 'gamma' in kwargs else 0,
		# inc=M.Inc(rslt).real,
		# idef = M.IEef(rslt),
		inc = inc,
		idef = idef,
		total_time=total_time,
		# max_mem=mem_diff,
		max_mem=-1,
		# timestamp=datetime.datetime.now().strftime("%Y")
		timestamp=str(datetime.datetime.now())
	)

	print('finished!')
	print(datapt)

	with open(f"datapts/{input_name}-{job_number}.pt", "w") as f:
		json.dump(datapt._asdict(), f)

	rslt_connection.send(datapt)
	# return datapt



def total_mem_recursive(pid):
	return psutil.Process(pid).memory_info().rss
	## TODO actually make this recursive

def mem_track( proc_id_recvr, response_line ):
	"""
	takes a queue
	"""

	maxmem_log = {}
	closed = False

	sleep_time = 1E-3

	while len(maxmem_log) > 0 or not closed:
		if not closed and proc_id_recvr.poll():
			new_pid = proc_id_recvr.recv()

			if new_pid == "END": 
				closed = True
				proc_id_recvr.close()
			elif new_pid in maxmem_log: 
				response_line.send(maxmem_log[new_pid])
				del maxmem_log[new_pid]				
			else:
				# processes.append(new_pid) 
				maxmem_log[new_pid] = 0
				sleep_time = 1E-3

		
		for pid in maxmem_log.keys():
			curmem = total_mem_recursive(pid)
			maxmem_log[pid] = max(maxmem_log[pid], curmem)

		time.sleep(sleep_time)
		if sleep_time < 0.5:
			sleep_time *= 1.4

	with open("memory_summary.json", 'w') as f:
		json.dump(maxmem_log)
	response_line.send(maxmem_log)


example_bn_names =  [ 
	"asia", "cancer", "earthquake", "sachs", "survey", "alarm", "barley", "child",
	# "insurance", "mildew", "water", "hailfinder", "hepar2", "win95pts", "andes", "diabetes",
	# "link", "munin1", "munin2", "munin3", "munin4", "pathfinder", "pigs", "munin" 
]

zerofn = lambda r: (0,0)

def main():
	store = TensorLibrary()

	if not os.path.exists('./datapts'):
		os.makedirs('./datapts')	

	memtrack_recvr, main_sender = multiproc.Pipe(False)
	main_recvr, memtrack_sender = multiproc.Pipe(False)

	mem_tracker = multiproc.Process(target=mem_track, args=(memtrack_recvr, memtrack_sender))

	# is this a good idea? I have no idea.
	memtrack_sender.close()
	memtrack_recvr.close()

	loose_ends = {} # (id_name, jobnumber) -> rslt_recvr, process
	pid_map = {} # pid -> (id_name, jobnumber)
	results = {} # (id_name, jobnumber) -> DataPt

	# with multiproc.Pool() as pool:
	jobnum = [0]

	# global available_cores
	available_cores = [ os.cpu_count() - 1 ]  # max with this many threads
	print("total cpu count: ", available_cores[0])

	def sweep(waiting_time=1E-2):

		""" returns True if there was any result that freed """
		for namenum, (rslt_recvr, proc) in loose_ends.items():
			proc.join(waiting_time)
			if not proc.is_alive():
				results[namenum] = rslt_recvr.recv()
				main_sender.send(proc.pid)
				results[namenum].max_mem = main_recvr.recv()
				break

		else:
			return False

		# nonlocal available_cores, loose_ends
		available_cores[0] += 1
		del loose_ends[namenum]
		print('cleaned up ', namenum)
		return True

	def enqueue_expt(input_name, input_stats, fn, *args, output_processor=None, **kwargs):
		rslt_recvr, rslt_sender = multiproc.Pipe()

		# nonlocal available_cores
		print()
		while available_cores[0] <= 0:
			print(' zzz ',end='')
			if not sweep():
				time.sleep(0.5)
		
		p = multiproc.Process(target=run_expt_log_datapt_worker,
				args=(bn_name, jobnum[0], input_stats), kwargs=dict(
					rslt_connection = rslt_sender,
					fn=fn, args=args, kwargs=kwargs,
					output_processor=output_processor
			))
		
			
			# raise NotImplemented
			# wait for next thread to finish ... with join? but which one?

		p.start()
		available_cores[0] -= 1

		# rslt_later = pool.apply_async(run_expt_log_datapt_worker, 
		# 	args=(bn_name, jobnum), 
		# 	kwds=dict(
		# 		rslt_connection = rslt_sender,
		# 		fn=fn, args=args, kwargs=kwargs,
		# 		output_processor=output_processor
		# 	),
		# 	callback=print)

		rslt_sender.close()

		main_sender.send(p.pid)
		pid_map[p.pid] = (input_name,jobnum[0])
		loose_ends[(input_name,jobnum[0])] = (rslt_recvr, p)

		jobnum[0] += 1

	

	for bn_name in example_bn_names:
		bn = get_example_model(bn_name)
		pdg = PDG.from_BN(bn)

		stats = dict(
			n_vars = len(pdg.varlist),
			n_worlds = int(np.prod(pdg.dshape)), # without cast, json cannot interperet int64 -.-
			n_params = int(sum(p.size for p in pdg.edges('P'))), #here also
			n_edges = len(pdg.Ed)
		)

		bp = BeliefPropagation(bn)
		# glog(bn_name+"-as-FG.bp", bp.calibrate)
		enqueue_expt(bn_name+"-belief-prop",stats, bp.calibrate, output_processor=zerofn)
				
		enqueue_expt(bn_name+"-as-pdg.ip.-idef",stats, ip.cvx_opt_joint, pdg, also_idef=False)
		enqueue_expt(bn_name+"-as-pdg.ip.+idef",stats, ip.cvx_opt_joint, pdg, also_idef=True)
		# collect_inference_data_for(bn_name+"-as-pdg", pdg, store)

		for gamma in [1E-12, 1E-8, 1E-4, 1E-2, 1, 2]:
			enqueue_expt(bn_name+"-as-pdg.cccp.gamma%.0e"%gamma, stats,
					 ip.cccp_opt_joint, pdg, gamma=gamma)
			
			for ozrname in ['adam', "lbfgs", "asgd"]:
				enqueue_expt(bn_name+"-as-pdg.torch.gamma%.0e"%gamma, stats,
					torch_opt.opt_dist, pdg,
					gamma=gamma, optimizer=ozrname)
				
				


	# with open("library.pickle", 'w') as f:
	# 	pickle.dump(store, f)
	with open("RESULTS.json", 'w') as f:
		json.dump(results, f)

if __name__ == '__main__':
	main()

