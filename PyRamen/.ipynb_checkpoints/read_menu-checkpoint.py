# Import the pathlib and csv library
from pathlib import Path
#from pandas import pd
import csv


menu = []
csvpath = Path('./menu_data.csv')
with open(csvpath, 'r') as csvfile:

    csvreader = csv.reader(csvfile, delimiter=',')
   
    header = next(csvreader)

    for item in csvreader:
        menu.append(item)
        
        
sales = []
csvpath = Path('./sales_data.csv')
with open(csvpath, 'r') as csvfile:

    csvreader = csv.reader(csvfile, delimiter=',')
   
    header = next(csvreader)

    for item in csvreader: 
        sales.append(item)
        

report = {
"01-count": 0,
"02-revenue": 0,
"03-cogs": 0,
"04-profit": 0,
}


menu_Item = ""
quantity = 0     
for item in sales: 
    menu_Item = item[0]
    quantity = item[0]
        
        

        
        
        
        
        
        
        
        
        
        
output_path = Path('./output.csv')
with open(output_path, 'w') as outfile:

    outfile.write(f"Total Months: {total_months}")
    outfile.write(f"Total: ${total_value}")
    outfile.write(f"Average  Change: ${average_change}\n")

    outfile.write(f"Greatest Increase in Profits: {max_increase_month} \n")
    outfile.write(f"Greatest Decrease in Profits: {max_decrease_month} \n")

