#!/usr/bin/env python	
import sys
import matplotlib.pyplot as plt

def create_graph(filename):
	lst_serial,lst_concurrent,lst_paper,lst_num=[],[],[],[]
	data=[]
	with open(filename,"r") as reader:
		for line in reader:
			print line
			lst=map(float,line.strip().split())
			lst_serial.append(lst[0])
			lst_concurrent.append(lst[1])
			lst_paper.append(lst[2])
			lst_num.append(lst[3])

	fig, ax= plt.subplots()
	ax.plot(lst_num,lst_serial,linewidth='1.5',label='Serial CPU execution')
	ax.plot(lst_num,lst_concurrent,linewidth='1.5',label='GPU execution')
	ax.plot(lst_num,lst_paper,linewidth='1.5',label='Concurrent CPU execution')
	plt.xlabel('number of queries')
	plt.ylabel('time(ms)')

	legend = ax.legend(loc='upper center', shadow=True)
	plt.show()
	
def create_bar(filename):
	lst_serial,lst_concurrent,lst_paper,lst_num=[],[],[],[]
	with open(filename,"r") as reader:
		for line in reader:
			lst= map(float,line.strip().split())
			lst_serial.append(lst[0])
			lst_concurrent.append(lst[1])
			lst_paper.append(lst[2])
			lst_num.append(lst[3])

	width=0.2
	fig=plt.figure()
	ax= fig.add_subplot(111)
	ax.bar(lst_num,lst_concurrent,width,color='b')
	plt.xlabel('Size')
	plt.ylabel('time(ms)')
	plt.legend()
	# plt.tight_layout()
	plt.show()

if __name__=='__main__':
	filename= sys.argv[1]
	# create_graph(filename)
	create_bar(filename)