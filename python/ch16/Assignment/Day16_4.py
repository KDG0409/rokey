class deque:
    def __init__(self):
        self.deque = []
    def is_empty(self):
        if len(self.deque) == 0 :
            return True
        return False
    def push_front(self,x):
        i = 1
        if not self.is_empty():
            for i in range(len(self.deque)):
                self.deque[i+1]=self.deque[i]
                self.deque[0] = x
        else:
            self.deque.append(x)
        return self.deque
    def push_back(self,x):
        return self.deque.append(x)
    def pop_front(self):
        if not self.is_empty():
            return self.deque.pop(0)
        return -1
    def pop_back(self):
        if not self.is_empty():
            return self.deque.pop()
        return -1
