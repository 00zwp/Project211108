import os
import xlwt
path = "D:\Desktop\物联网实验报告\第一次实验-2021-10-22"

dir = os.listdir(path)
workbook = xlwt.Workbook()
worksheet = workbook.add_sheet("My new Sheet")
for i,name in enumerate(dir):
    print(name)
    x = name.split('-')
    worksheet.write(i, 0, x[0])
    worksheet.write(i, 1, x[1])
    worksheet.write(i, 2, x[2])
workbook.save("./1.xls")