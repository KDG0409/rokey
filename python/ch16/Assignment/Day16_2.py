class queue:
    def __init__(self):
        self.queue = []
    def is_empty(self):
        if len(self.queue) == 0 :
            return True
        return False
    def enqueue(self,data):
        return self.queue.append(data)
    def dequeue(self):
        if not self.is_empty() :
            return self.queue.pop(0)
        return -1
    def front(self):
        if not self.is_empty():
            return self.queue[0]
        return -1

