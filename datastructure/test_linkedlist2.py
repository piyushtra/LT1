class Node:
    def __init__(self, dataval=None):
        self.dataval = dataval
        self.nextval = None

class SLinkedList:
    def __init__(self):
        self.headval = None

# Print the linked list
    def listprint(self):
        printval = self.headval
        while printval is not None:
            print (printval.dataval)
            printval = printval.nextval
    def AtBegining(self,newdata):
        NewNode = Node(newdata)

# Update the new nodes next val to existing node
        NewNode.nextval = self.headval
        self.headval = NewNode

    def AtEnd(self, newdata):
        NewNode = Node(newdata)
        if self.headval is None:
            self.headvale = NewNode
            return
        laste = self.headval

        while(laste.nextval):
            laste = laste.nextval

        laste.nextval = NewNode

    def AtMiddle(self,node,newdata):
        NewNode = Node(newdata)
        if node.nextval is None:
            print("Node does not exist")
            return

        swapvar = node.nextval
        node.nextval = NewNode
        NewNode.nextval = swapvar
        return

    def removeNode(self,data):
        if self.headval is None:
            print("XXXXXXXXX")
            return
        DataNode = self.headval
        while 

list = SLinkedList()
list.headval = Node("Mon")
e2 = Node("Tue")
e3 = Node("Wed")

list.headval.nextval = e2
e2.nextval = e3

list.AtBegining("Sun")

list.listprint()

list.AtEnd("Thu")
list.listprint()

list.AtMiddle(e2,"Fri")
list.listprint()