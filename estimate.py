import numpy
import sys

def main():
	try:
		t = numpy.loadtxt("output.csv", delimiter = ',')
	except:
		print ("output.csv file missing, setting thetas to 0")
		t = [0, 0]
	try:
		mileage = int(input("Choose mileage (or first parameter) : "))
	except:
		print ("type error")
		sys.exit()
	print (t[0] + (t[1] * mileage))

if (__name__ == "__main__"):
	main()