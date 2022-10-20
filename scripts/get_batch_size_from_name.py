import sys
import re

filename = sys.argv[1]
m=re.search('.*-b(\d+)-n\d+.*', filename)
if (m == None):
	m=re.search('.*-b(\d+)-raw.txt', filename)
if (m == None):
	m=re.search('.*-b(\d+)-sass.txt', filename)
if (m != None):
	batch_size = m.group(1)
	print(batch_size)
else:
	print('0')
