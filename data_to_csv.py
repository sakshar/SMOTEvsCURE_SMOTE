import csv

file_name = 'xyz' #put the downloaded data file's name here for converting into csv

with open(file_name + '.data') as input_file:
   lines = input_file.readlines()
   newLines = []
   for line in lines:
      newLine = line.strip().split()
      print(newLine)
      newLines.append( newLine )

with open(file_name + '.csv', 'w', newline='') as test_file:
   file_writer = csv.writer(test_file)
   file_writer.writerows( newLines )