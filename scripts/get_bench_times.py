import sys
import re

def get_throughput(filename):
	with open(filename) as fp:
		line = fp.readline()
		while line:
			m=re.search('^Throughput: (.*)$', line)
			if (m == None):
				m=re.search('^Throughput (.*)$', line)
			if (m != None):
				throughput = m.group(1)
				return throughput
			line = fp.readline()
	return None

for i in range(len(sys.argv)-1):
	filename = sys.argv[i+1]
	m=re.search('.*-b(\d+)-n.*.txt', filename)
	if (m != None):
		batch_size = m.group(1)
		tput = get_throughput(filename)
		if (tput != None):
			print(batch_size, tput)
