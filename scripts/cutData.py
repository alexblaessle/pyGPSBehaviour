"""Short script to shorten data."""

import sys
import csv

# Get input
nPoints=int(sys.argv[2])
fn=sys.argv[1]

# Read file
with open(fn,'rb') as f:

	fcsv=csv.reader(f,delimiter=',')
	
	rows=[]
	for row in fcsv:
		rows.append(row)

# Shorten data		
rowsNew=rows[:nPoints]

# Write back into new file
with open(fn.replace('.csv','_short.csv'),'wb') as f:
	
	fcsv=csv.writer(f,delimiter=',')
	for row in rowsNew:
		fcsv.writerow(row)