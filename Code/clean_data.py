#!/usr/bin/env python
import sys,os
import re,string,random

def reformat_file(filename):
	out_file= "formatted.txt"
	with open(filename,"r") as reader:
		with open(out_file,"w") as writer:
			for lines in reader:
				str_= lines.strip()
				writer.write(str_+'\n')
	return out_file

def remove_bracket_lines(filename):
	out_file= "input_data.txt"
	obj1= re.compile(r'^\[')
	with open(filename,"r") as reader:
		with open(out_file,"w") as writer:
			for lines in reader:
				if not obj1.match(lines) and not lines.startswith("by") and "by" in lines:
					lst=lines.split(r', by ')
					if len(lst)==2:
						lst[0]= re.sub(r'[, ]*$','',lst[0].strip())
						lst[0]= re.sub(r'[^\x00-\x7F]+','',lst[0])
						lst[1]= re.sub(r'[ \n]+(\d+[a-zA-Z]*)?$','',lst[1].strip())
						lst[1]= re.sub(r'[^\x00-\x7F]+','',lst[1])
						writer.write(lst[0]+"$"+ lst[1]+'\n')

def add_books(filename):
	dest_file="input_data3.txt"
	os.system("cp "+filename+" "+dest_file)
	with open(dest_file,"a") as writer:
		for i in range(500000):
			str_1=''.join(random.choice(string.ascii_uppercase + string.digits + 
				string.ascii_lowercase) for _ in range(random.randint(1,70)))
			str_2=''.join(random.choice(string.ascii_uppercase + string.digits + 
				string.ascii_lowercase) for _ in range(random.randint(1,70)))
			writer.write(str_1+"$"+str_2+"\n")

if __name__=='__main__':
	filename= sys.argv[1]
	#out_file= reformat_file(filename)
	# remove_bracket_lines(filename)
	add_books(filename)