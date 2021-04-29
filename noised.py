import csv
import sys

# noise_ratio=0.03
# size=5000

# with open('MyData_3.csv',mode='r+') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     csv_writer = csv.writer(csv_file,delimiter=',')
#     line_count = 0
#     for row in csv_reader:
#     	row2=row
#         if (line_count%(size*noise_ratio)==0):
# 			if (int(row2[0])==0):
# 				row2[0]=1
# 			else:
# 				row2[0]=0
# 			csv_writer.write


#         line_count+=1

# import csv
def add_noise(noise_ratio,inpath,outpath,size=5000):
	
	f = open(inpath)
	csv_f = csv.reader(f, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)

	g = open(outpath,'w')
	csv_g= csv.writer(g, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
	line_count=0
	noise_num=size//(size*noise_ratio)
	for row in csv_f:
		if (line_count==0):
			line_count+=1
			pass

		elif (line_count%noise_num==0):
			if (int(row[0])==0):
				row2=1
			elif (int(row[0])==1):
				row2=0
			new_row=[row2]+row[1:]
			csv_g.writerow(new_row)
		else:
			new_row=row
			csv_g.writerow(new_row)
		line_count+=1

arglist=sys.argv
if len(arglist)>3:
	add_noise(double(arglist[1]),str(arglist[2]),str(arglist[3]))


