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
        self.sheet['B1'] = "come"
        self.sheet['C1'] = "warning"
        self.sheet['D1'] = "Accuracy"
    def write(self,yolo,hawkEye,y_axis):
        yolo = list(filter(lambda x: x[3]>y_axis,yolo))
        come = list(filter(lambda x: x[-1]==1,hawkEye))
        warn = list(filter(lambda x: x[-1]==2,hawkEye))
        if len(yolo) == 0:
            self.sheet.append([len(yolo), len(come), len(warn),0])
        else:
            self.sheet.append([len(yolo),len(come),len(warn),(len(warn)+len(come))/len(yolo)])
    def endWrite(self):
        self.book.save(filename=self.filename)
if __name__ == '__main__':
    test = test()
    test.write()