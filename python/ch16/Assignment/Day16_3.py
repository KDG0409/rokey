class queue:
    def __init__(self,n):
        self.queue=[]
        self.num = n
    def is_empty(self):
        if len(self.queue) == 0:
            return True
        return False
    def is_full(self):
        if len(self.queue)-self.num >= 0 :
            return True
        return False
    def enqueue(self,x):
        if self.is_full():
            return self.queue()
        elif self.is_empty():
            return -1
        return self.queue.append(x)
    def dequeue(self):
        if not self.is_empty():
            return self.queue.pop(0)
        return -1
    def front(self):
        if not self.is_empty():
            return self.queue[0]
        return -1