import openpyxl
import os

class test:
    def __init__(self, name="test.xlsx"):
        self.filename = name
        if os.path.isfile(self.filename):
            self.book = openpyxl.load_workbook(filename=self.filename)
        else:
            self.book = openpyxl.Workbook()
        self.sheet = self.book.active
        self.sheet['A1'] = 'YOLOv3'
        self.sheet['B1'] = "HawkEye"
        self.sheet['C1'] = "SORT"
    def write(self,result):
        self.sheet.append(result)
    def endWrite(self):
        self.book.save(filename=self.filename)
if __name__ == '__main__':
    test = test()
    test.write()