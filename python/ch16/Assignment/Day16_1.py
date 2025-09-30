class stack:
    def __init__(self):
        self.stack = []
    def is_empty(self):
        if len(self.stack) == 0 :
            return True
        return False
    def push(self,data):
        self.stack.append(data)
    def pop(self):
        if not self.is_empty:
            self.stack.pop()
            return self.stack
        return -1
    def top(self):
        if not self.is_empty:
            return self.stack[-1]
        return -1
