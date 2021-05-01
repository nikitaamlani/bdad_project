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
def filterx(inpath,outpath):
	
	f = open(inpath)
	csv_f = csv.reader(f, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)

	g = open(outpath,'w')
	csv_g= csv.writer(g, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
	
	for row in csv_f:
		if ("" not in row):
			new_row=row
			csv_g.writerow(new_row)
		
			
		# line_count+=1

arglist=sys.argv
if len(arglist)>1:
	# print(arglist[0])
	filterx(arglist[1],arglist[2])


