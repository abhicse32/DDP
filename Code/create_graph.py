#!/usr/bin/env python
import random
import os,sys
nums= 1000000
def create(filename):
	lst= random.sample(xrange(-1*nums*10,nums*10),nums)
	lst2= random.sample(xrange(-1*nums*10,nums*10),nums)
	with open(filename,"w") as w:
		w.write(str(nums)+"\n")
		for i in range(len(lst)):
			w.write(str(lst[i])+" "+str(lst2[i])+"\n")

if __name__=='__main__':
	filename= sys.argv[1]
	create(filename)