import argparse
parser = argparse.ArgumentParser(description=
	"aggregate all files from target directory, into a single json file")

parser.add_argument("-d", "--data-dir", default="") # help="target directory")
parser.add_argument("-s", "--source-files", nargs="*")
parser.add_argument("-t" , "--target-file", default="")
parser.add_argument("-i" , "--ignore-ext", nargs='*', default=[])
parser.add_argument("-e",  "--use-ext", nargs='*', default=[".mpt", ".pt"], help="list of extensions to incorporate; if there are two matching file names with different extensions, takes the earlist one in the list.")


args = parser.parse_args()

if args.target_file == "":
	if len(args.data_dir) > 0:
		args.target_file = args.data_dir + "-aggregated.json"
	else:
		args.target_file = "aggregate.json"

whitelist = len(args.use_ext) > 0 and args.use_ext[0] != '*' 

import os, json

aggregate = []
filelist = args.source_files if args.source_files else [args.data_dir+'/'+f for f in os.listdir(args.data_dir)]
filelistset = set(filelist)

def can_do_better(fname):
	for j,ext in enumerate(args.use_ext):
		if fname.endswith(ext):
			for i in range(j):
				if fname[:-len(ext)]+args.use_ext[i] in filelistset:
					return True
	return False

for fname in filelist:

	# first ignore blacklist
	if any(fname.endswith(ext) for ext in args.ignore_ext):
		continue

	# now, if there's a whitelist, do that also.
	if whitelist and (
			not any(fname.endswith(ext) for ext in args.use_ext)
			or can_do_better(fname)):
		continue
	
	# print(fname)

	with open(fname, 'r') as fh:
		obj = json.load(fh)
		aggregate.append(obj)


with open(args.target_file, 'w') as fh:
	json.dump(aggregate, fh)

print('finished! all %d files aggregated to `%s`.'%(len(aggregate), args.target_file))
