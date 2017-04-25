#!/usr/bin/env python
from pexpect import spawn,run
import sys

def connect(arg):
	if arg=='scp':
		child= spawn("scp input_data2.txt cs12b032@libra.iitm.ac.in:DDP")
	else:
		child= spawn("ssh cs12b032@libra.iitm.ac.in")	
	child.expect(".+ password: ")
	child.sendline("ask@abhi$123")
	if arg=='ssh':
		try:
			child.sendline('ssh cn006')
			child.sendline('cd DDP')
			child.interact()

		except:
			child.terminate()
	else:
		line=child.read().strip()
		print line,

if __name__=='__main__':
	arg= sys.argv[1]
	connect(arg)