class Node:
    def __init__(self,dataval=None):
        self.dataval = dataval
        self.nextval = None

class SLinkedList:
    def __init__(self):
        self.headval = None

    def listprint(self):
        printvalue = self.headval
        while printvalue is not None:
            print(printvalue.dataval)
            printvalue = printvalue.nextval

    def __str__(self):
        strt = ""
        printvalue = self.headval
        while printvalue is not None:
            if strt == "":
                strt = printvalue.dataval
            else:
                strt = strt +"->" +printvalue.dataval
            printvalue = printvalue.nextval

        return strt
    def addEleAtStart(self, newdata):
        NewNode = Node(newdata)
        NewNode.nextval = self.headval
        self.headval = NewNode


list1 = SLinkedList()
list1.headval = Node("Mon")
e2 = Node("Tue")
e3 = Node("Wed")


list1.headval.nextval = e2
e2.nextval = e3

print(list1)
print(e2)
print(e3)

list1.listprint()
print(list1)

list1.addEleAtStart("Sun")
print(list1)
list1.listprint()
