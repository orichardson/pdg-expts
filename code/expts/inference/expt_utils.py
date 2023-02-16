from collections import namedtuple
import json

from pdg.pdg import PDG
# from pdg.store import TensorLibrary
# from pdg.rv import Variable as Var
from pdg.dist import RawJointDist as RJD


## TIMING / LOGGING UTILS
import psutil
from psutil._common import bytes2human
import multiprocessing as multiproc
import time, datetime
import traceback
import os, sys # for stderr, getpid()


import numpy
numpy.warnings.filterwarnings('error', category=numpy.VisibleDeprecationWarning)


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


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

DataPt = namedtuple('DataPt', 
	# ['result', 'total_time', 'max_mem'])
	['method', 'input_stats', 'input_name', 'parameters', 'gamma',
		# 'inc', 'idef',
		'rslt_metrics',  'total_time', 'init_mem', 'max_mem', 'timestamp']
)

def run_expt_log_datapt_worker( DATA_DIR,
			input_name, job_number, input_stats, rslt_connection,
			fn,	args, kwargs, output_processor=None, IGNORE=set()
		):
	""" this is the worker method.
	"""
	fileprefix = f"{DATA_DIR}/{input_name}-{job_number}"

	print(f"[pid {os.getpid()} @ {datetime.datetime.now().strftime('%H:%M:%S') }] STARTING #{job_number};"+
		f"output to be saved in \"{fileprefix}\"")
	

	# init_mem  = psutil.Process(os.getpid()).memory_info().rss
	init_mem = total_mem_recursive(os.getpid())
	init_time = time.time()

	try:
		# rslt = fn(M, *args, **kwargs)
		rslt = fn(*args, **kwargs)

		if isinstance(rslt, RJD) and rslt._torch:
			rslt = rslt.npify()


		total_time = time.time() - init_time
		# print(prefix, "about to wait for memory usage")
		# max_mem = mem_connection.recv() # should have sent back memory usage.
		# print(prefix, "recieved memory usage")
		# mem_diff = max_mem - init_mem

		if output_processor is None:
			M = args[0] # assume M is first argument
			# assumre rslt is either RJD | CliqueForest, and can be npified
			rslt.npify(inplace=True)
			inc = M.Inc(rslt)
			idef = M.IDef(rslt)
			if numpy.ma.is_masked(inc): inc = numpy.inf
			if numpy.ma.is_masked(idef): idef = numpy.nan
			rslt_metrics = dict(inc=inc,idef=idef)
		else:
			rslt_metrics = output_processor(rslt)

		datapt = DataPt(
			method=fn.__name__,
			input_stats = input_stats,
			input_name = input_name,
			parameters=(tuple(a for a in args if not isinstance(a, PDG)), {k:v for k,v in kwargs.items() if k not in IGNORE}),
			gamma=kwargs['gamma'] if 'gamma' in kwargs else 0,
			# inc=M.Inc(rslt).real,
			# idef = M.IEef(rslt),
			# inc = inc,
			# idef = idef,
			rslt_metrics=rslt_metrics,
			total_time=total_time,
			# max_mem=mem_diff,
			init_mem=init_mem,
			max_mem=-1,
			# timestamp=datetime.datetime.now().strftime("%Y")
			timestamp=str(datetime.datetime.now())
		)

		print('finished!')
		print(datapt)

		with open(fileprefix+".pt", "w") as f:
			# json.dump(datapt._asdict(), f)
			json.dump(datapt, f)

		if rslt_connection is None:
			return rslt
		else:
			rslt_connection.send(datapt)

	except Exception as e:
		with open(fileprefix+".err", "w") as f:
			sys.stderr.write(f"==== ERROR WHILE HANDLING {input_name}, {job_number}, {fn.__name__}, {kwargs}\n\n"
				 + "".join(traceback.TracebackException.from_exception(e).format()))
			# json.dump(datapt, f)
			f.writelines(traceback.TracebackException.from_exception(e).format())
		if rslt_connection is not None:
			rslt_connection.send(None)
	
	finally:
		if rslt_connection is not None:
			rslt_connection.close()

	# prefix = f"{input_name+'-'+str(job_number):>20}|"
	# print(prefix, "requesting memory")
	# connection.send("done!")
	# mem_connection.send(os.getpid())
	# return datapt



def total_mem_recursive(pid):
	return psutil.Process(pid).memory_info().rss
	## TODO actually make this recursive

def mem_track( proc_id_recvr, response_line ):
	"""
	takes two pipes as input.
	Intended to be run as a separate process.
	"""

	maxmem_log = {}
	closed = False

	sleep_time = 1E-3

	while len(maxmem_log) > 0 or not closed:
		if proc_id_recvr.poll():
			pid = proc_id_recvr.recv()

			if pid == "END": 
				print("\n ===== memory tracker recieved END =====", flush=True)
				closed = True
				# proc_id_recvr.close()

			elif pid in maxmem_log: 
				print("MEMTRACK: removing ", pid, flush=True)
				mm = maxmem_log[pid]
				del maxmem_log[pid]				
				response_line.send(mm)
			elif not closed:
				# processes.append(new_pid) 
				print("MEMTRACK: new pid ", pid, flush=True)
				maxmem_log[pid] = 0
				sleep_time = 1E-3

		
		dead = 0
		for pid in maxmem_log.keys():
			if psutil.pid_exists(pid):
				try:
					curmem = total_mem_recursive(pid)
					maxmem_log[pid] = max(maxmem_log[pid], curmem)
				except psutil.NoSuchProcess as ex:
					sys.stderr.write("".join(traceback.TracebackException.from_exception(ex).format()))
					dead += 1
			else:
				# sys.stderr.write("oopsies, PID %d does not exist (anymore).\n"%pid)
				# sys.stderr.flush()
				# pass
				dead += 1

		time.sleep(sleep_time)
		if sleep_time < 0.2:
			sleep_time *= 1.4
		else:
			if dead > 0:
				print(' dead processes: [%d/%d]'%(dead, len(maxmem_log)))
		# else: print("<memtracker sleeping>")

	proc_id_recvr.close()
	
	with open("memory_summary.json", 'w') as f:
		json.dump(maxmem_log, f)

	response_line.send(maxmem_log)
	response_line.close()


# Infrastructure = namedtuple("Infrastructure", ['enqueue'])

class MultiExptInfrastructure: 
	def __init__(self, datadir='datapts', n_threads=None, kw_params_to_ignore=set()):
		self.datadir = datadir 
		if not os.path.exists(datadir):
			os.makedirs(datadir)

		self.kwparams2ignore = kw_params_to_ignore

		memtrack_recvr, main_sender = multiproc.Pipe(False)
		main_recvr, memtrack_sender = multiproc.Pipe(False)

		mem_tracker = multiproc.Process(target=mem_track, args=(memtrack_recvr, memtrack_sender))

		mem_tracker.start()
		# is this a good idea? I have no idea.
		memtrack_sender.close()
		memtrack_recvr.close()

		self.to_memtracker = main_sender
		self.from_memtracker = main_recvr
		self.mem_tracker = mem_tracker

		self.loose_ends = {} # (id_name, jobnumber) -> rslt_recvr, process
		self.pid_map = {} # pid -> (id_name, jobnumber)
		self.results = {} # (id_name, jobnumber) -> DataPt

		self.finish_now = False

		# with multiproc.Pool() as pool:
		# jobnum = [0]
		self.jobnum = 0

		# global available_cores
		# available_cores = [ os.cpu_count() - 1 ]  # max with this many threads
		self.available_cores = multiproc.Value('i',
			os.cpu_count()-2 if (n_threads is None or n_threads <= 0) else n_threads )
		print("total cpu count: ", os.cpu_count(), ';  using: ', self.available_cores.value)

	def sweep(self):
		""" returns True if there was any result that freed """
		if self.finish_now:
			raise InterruptedError

		for namenum, (rslt_recvr, proc) in self.loose_ends.items():
			if not proc.is_alive():
				proc.join()
				self.to_memtracker.send(proc.pid)

				try:
					assert rslt_recvr.poll(), "process is dead, but there's no result??; "+str(namenum)

					m_m = self.from_memtracker.recv()

					result = rslt_recvr.recv()

					if result is None:
						self.results[namenum] = None
					else:
						self.results[namenum] = result._replace(max_mem = m_m)
						with open(self.datadir+"/"+namenum[0]+"-"+str(namenum[1])+".mpt", 'w') as fh:
							json.dump(self.results[namenum]._asdict(), fh)
				
				except EOFError:
					sys.stderr.write(f"EOFError! @process: {namenum}; already in results? "
						+ str(namenum in self.results)+"\n",flush=True)
				except Exception as ex:
					sys.stderr.write(
						f"\n @ PROCESS {namenum}; "+
						"".join(traceback.TracebackException.from_exception(ex).format()))

				finally:
					break

		else:
			return False

		# nonlocal available_cores, loose_ends
		with self.available_cores.get_lock():
			self.available_cores.value += 1 

		del self.loose_ends[namenum]
		print('cleaned up ', namenum)
		return True

	def execute(self, input_name, input_stats, fn, *args, output_processor=None, **kwargs):
		# nonlocal available_cores
		print('requested execute: ', input_name, fn.__name__, kwargs)
		
		rslt = run_expt_log_datapt_worker(self.datadir,input_name,self.jobnum,
			input_stats, rslt_connection=None, fn=fn,args=args,kwargs=kwargs,
			output_processor=output_processor,IGNORE=self.kwparams2ignore)
		
		self.jobnum += 1

		return rslt
	def enqueue(self, input_name, input_stats, fn, *args, output_processor=None, **kwargs):
		rslt_recvr, rslt_sender = multiproc.Pipe()

		# nonlocal available_cores
		print('requested enqueue: ', input_name, fn.__name__, kwargs)
		while self.available_cores.value <= 0:
			# print(' zzz (%d)' % self.available_cores.value)
			if not self.sweep():
				time.sleep(0.5)
		
		p = multiproc.Process(target=run_expt_log_datapt_worker,
				args=(self.datadir, input_name, self.jobnum, input_stats), kwargs=dict(
					rslt_connection = rslt_sender,
					fn=fn, args=args, kwargs=kwargs,
					output_processor=output_processor,
					IGNORE=self.kwparams2ignore
			))
		
			
			# raise NotImplemented
			# wait for next thread to finish ... with join? but which one?

		p.start()
		rslt_sender.close()

		with self.available_cores.get_lock():
			self.available_cores.value -= 1

		# rslt_later = pool.apply_async(run_expt_log_datapt_worker, 
		# 	args=(bn_name, jobnum), 
		# 	kwds=dict(
		# 		rslt_connection = rslt_sender,
		# 		fn=fn, args=args, kwargs=kwargs,
		# 		output_processor=output_processor
		# 	),
		# 	callback=print)


		self.to_memtracker.send(p.pid)
		self.pid_map[p.pid] = (input_name, self.jobnum)
		self.loose_ends[(input_name, self.jobnum)] = (rslt_recvr, p)

		self.jobnum += 1

	def done(self):
		self.to_memtracker.send("END")
		print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] waiting for remaining processes ...")
		
		while len(self.loose_ends):
			# print('.', end='')
			# if self.finish_now:  raise InterruptedError
			time.sleep(0.5)
			self.sweep()

		print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] waiting for memory tracking thread to finish up...")
		self.mem_tracker.join()
		print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ... done!")

	# return dotdict(enqueue=enqueue_expt, sweep=sweep, done=when_finished, jobnum=jobnum, results=results)