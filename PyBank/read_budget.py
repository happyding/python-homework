# Import the pathlib and csv library
from pathlib import Path
#from pandas import pd
import csv

# Set the file path
csvpath = Path('./budget_data.csv')

# Initialize variable to hold salaries
total_months = 0
total_value = 0

total_change = 0
total_change_months = 0
average_change = 0


amount = 0
month = ""
prior_amount = 0

change_amount = 0

# Initialize metric variables
max_increase = 0
max_increase_month = ""

max_decrease = 0
max_decrease_month = ""

# Open the input path as a file object
with open(csvpath, 'r') as csvfile:

    csvreader = csv.reader(csvfile, delimiter=',')
   
    header = next(csvreader)

    for row in csvreader:
        
        month = row[0]
        amount = int(row[1])
        
        total_months += 1
        total_value += amount
        
        change_amount = int(amount - prior_amount)       
        prior_amount = amount
        
        if total_months == 2 :
            
            total_change += change_amount
            
            max_increase = change_amount
            max_increase_month = month
        
            max_decrease = change_amount
            max_decrease_month = month
            
          
        
        elif total_months > 2 :
             
            total_change += change_amount
            
            if change_amount > max_increase :
                max_increase = change_amount
                max_increase_month = month
        
            elif change_amount < max_decrease :
                max_decrease = change_amount
                max_decrease_month = month
                

total_change_months = total_months - 1
average_change = round(total_change / total_change_months, 0)

print("\nFinancial Analysis")
print("----------------------------")

print(f"Total Months: {total_months}")
print(f"Total: ${total_value}")
print(f"Average  Change: ${average_change}")

print(f"Greatest Increase in Profits: {max_increase_month} ${max_increase}")
print(f"Greatest Decrease in Profits: {max_decrease_month} ${max_decrease}")


output_path = Path('./budget_output.csv')

with open(output_path, 'w') as outfile:
    outfile.write("Financial Analysis\n")
    outfile.write("----------------------------\n")

    outfile.write(f"Total Months: {total_months}\n")
    outfile.write(f"Total: ${total_value}\n")
    outfile.write(f"Average  Change: ${average_change}\n")

    outfile.write(f"Greatest Increase in Profits: {max_increase_month} \(${max_increase}\)\n")
    outfile.write(f"Greatest Decrease in Profits: {max_decrease_month} \(${max_decrease}\)\n")

#Financial Analysis
#----------------------------
#Total Months: 86
#Total: $38382578
#Average  Change: $-2315.12
#Greatest Increase in Profits: Feb-2012 ($1926159)
#Greatest Decrease in Profits: Sep-2013 ($-2196167)
