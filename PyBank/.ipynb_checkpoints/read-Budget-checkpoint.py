# Import the pathlib and csv library
from pathlib import Path
import csv

# Set the file path
csvpath = Path('/Users/admin/test/w2/ucb-sfc-fin-pt-08-2020-u-c/02-Homework/02-Python/Instructions/PyBank/Resources/budget_data.csv')

# Initialize variable to hold salaries
salaries = []

# Initialize line_num variable
line_num = 0

# Open the input path as a file object
with open(csvpath, 'r') as csvfile:

    # Print the datatype of the file object
    print(type(csvfile))

    # Pass in the csv file to the csv.reader() function
    # (with ',' as the delmiter/separator) and return the csvreader object
    csvreader = csv.reader(csvfile, delimiter=',')
    # Print the datatype of the csvreader
    print(type(csvreader))

    # Go to the next row from the start of the file
    # (which is often the first row/header) and iterate line_num by 1
    header = next(csvreader)
    line_num += 1
    # Print the header
    print(f"{header} <---- HEADER")

    # Read each row of data after the header
    for row in csvreader:
        # Print the row
        print(row)
        # Set salary variable equal to the value in the 4th column of each row
        salary = int(row[3])
        # Append the row salary value to the list of salaries
        salaries.append(salary)

# Initialize metric variables
total_months = 0
total_value = 0
average_value = 0

max_increase = 0
max_increase_month = ""

max_decrease = 0
max_decrease_month = ""

# Calculate the max, mean, and average of the list of salaries
for salary in salaries:

    # Sum the total and count variables
    total_salary += salary
    count_salary += 1

    # Logic to determine min and max salaries
    if min_salary == 0:
        min_salary = salary
    elif salary > max_salary:
        max_salary = salary
    elif salary < min_salary:
        min_salary = salary

# Calculate the average salary, round to the nearest 2 decimal places
avg_salary = round(total_salary / count_salary, 2)

# Print the metrics
print(max_salary, min_salary, avg_salary)

# Set the output header
header = ["Max_Salary", "Min_Salary", "Avg_Salary"]
# Create a list of metrics
metrics = [max_salary, min_salary, avg_salary]

# Set the output file path
output_path = Path('budget_output.csv')

# Open the output path as a file object
with open(output_path, 'w') as csvfile:
    # Set the file object as a csvwriter object
    csvwriter = csv.writer(csvfile, delimiter=',')
    # Write the header to the output file
    csvwriter.writerow(header)
    # Write the list of metrics to the output file
    csvwriter.writerow(metrics)

            
            

# Print the output of increase/decrease and its month
print("Financial Analysis")
print("----------------------------")

print(f"Total Months: {total_months}")
print(f"Total: ${total_value}")
print(f"Average  Change: ${average_value}")

print(f"Greatest Increase in Profits: {max_increase_month} (${max_increase})")
print(f"Greatest Decrease in Profits: {max_decrease_month} (${max_decrease})")


Financial Analysis
----------------------------
Total Months: 86
Total: $38382578
Average  Change: $-2315.12
Greatest Increase in Profits: Feb-2012 ($1926159)
Greatest Decrease in Profits: Sep-2013 ($-2196167)
