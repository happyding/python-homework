# Import the pathlib and csv library
from pathlib import Path
#from pandas import pd
import csv

row_count = 0

menu = []
csvpath = Path('./menu_data.csv')
with open(csvpath, 'r') as csvfile:

    csvreader = csv.reader(csvfile, delimiter=',')
   
    header = next(csvreader)

    for item in csvreader:
        menu.append(item)
        
row_count = len(menu)
print(f"Total lines of menu: {row_count}!")
       
sales = []
csvpath = Path('./sales_data.csv')
with open(csvpath, 'r') as csvfile:

    csvreader = csv.reader(csvfile, delimiter=',')
   
    header = next(csvreader)

    for item in csvreader: 
        sales.append(item)
        
row_count = len(sales)
print(f"Total times of sales: {row_count}!")

sales_quantity = 0  
sales_item_name = ""
report = {}

for sales_item in sales:
    
    sales_quantity = sales_item[3]
    sales_item_name = sales_item[4]
    
    if report.get(sales_item_name) == None:
        report[sales_item_name] = {'01-count': 0,
                                 '02-revenue': 0.0,
                                 '03-cogs': 0.0,
                                 '04-profit': 0.0}

row_count = len(report)
print(f"Total length of report: {row_count}!")

        
        
menu_price = 0.0
menu_cost= 0.0
menu_profit= 0.0

found = 0

for sales_item in sales:
    
    found = 0
    
    row_count += 1
    
    sales_quantity = sales_item[3]
    sales_item_name = sales_item[4]
    
    for menu_item in menu: 
        
        if sales_item_name == int(menu_item[0]): 
            
            found = 1  
            menu_price = int(menu_item[3])
            menu_cost= int(menu_item[4])
            menu_profit = menu_price - menu_cost
            
            print
    
            break
        
    if found == 1:
        
        report[sales_item_name]["01-count"] += int(sales_quantity)
        report[sales_item_name]["02-revenue"] += menu_price * int(sales_quantity)
        report[sales_item_name]["03-cogs"] += menu_cost * int(sales_quantity)
        report[sales_item_name]["04-profit"] += menu_profit * int(sales_quantity)
        
    else:
        
        print(f"{sales_item_name} : {sales_quantity} does not equal any menu item! NO MATCH!")
        

output_path = Path('./output.csv')
with open(output_path, 'w') as outfile:
    
    for item_name, item_value in report.items(): 
        print(f"{item_name} {item_value}")
        outfile.write(f"{item_name} {item_value}\n")
    

#spicy miso ramen {'01-count': 9238, '02-revenue': 110856.0, '03-cogs': 110856.0, '04-profit': 64666.0}
#tori paitan ramen {'01-count': 9156, '02-revenue': 119028.0, '03-cogs': 54936.0, '04-profit': 64092.0}

