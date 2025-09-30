import tkinter as tk
import random

# 게임 설정
CELL_SIZE = 30   # 셀 크기
COLS = 10        # 가로 칸 수
ROWS = 20        # 세로 칸 수
DELAY = 500      # 블록 낙하 속도(ms)

# 블록 모양 (회전 고려하지 않은 기본 형태)
SHAPES = {
    "I": [[1, 1, 1, 1]],
    "O": [[1, 1],
          [1, 1]],
    "T": [[0, 1, 0],
          [1, 1, 1]],
    "S": [[0, 1, 1],
          [1, 1, 0]],
    "Z": [[1, 1, 0],
          [0, 1, 1]],
    "J": [[1, 0, 0],
          [1, 1, 1]],
    "L": [[0, 0, 1],
          [1, 1, 1]]
}

COLORS = {
    "I": "cyan",
    "O": "yellow",
    "T": "purple",
    "S": "green",
    "Z": "red",
    "J": "blue",
    "L": "orange"
}

class Tetris:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=COLS*CELL_SIZE, height=ROWS*CELL_SIZE, bg="black")
        self.canvas.pack()
        
        self.board = [[None for _ in range(COLS)] for _ in range(ROWS)]
        self.score = 0

        self.current_shape = None
        self.current_x = 0
        self.current_y = 0

        self.spawn_new_block()
        self.running = True
        self.root.bind("<Key>", self.key_handler)
        self.update()

    def draw_cell(self, x, y, color):
        """셀 하나를 그림"""
        x1, y1 = x*CELL_SIZE, y*CELL_SIZE
        x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray20")

    def draw_board(self):
        self.canvas.delete("all")
        # 고정된 블록
        for y in range(ROWS):
            for x in range(COLS):
                if self.board[y][x]:
                    self.draw_cell(x, y, self.board[y][x])
        # 현재 움직이는 블록
        if self.current_shape:
            for y, row in enumerate(self.current_shape):
                for x, val in enumerate(row):
                    if val:
                        self.draw_cell(self.current_x + x, self.current_y + y, self.color)

    def rotate_shape(self):
        """시계 방향 회전"""
        rotated = list(zip(*self.current_shape[::-1]))
        return [list(row) for row in rotated]

    def check_collision(self, shape, offset_x, offset_y):
        """충돌 여부 검사"""
        for y, row in enumerate(shape):
            for x, val in enumerate(row):
                if val:
                    new_x = offset_x + x
                    new_y = offset_y + y
                    if new_x < 0 or new_x >= COLS or new_y >= ROWS:
                        return True
                    if new_y >= 0 and self.board[new_y][new_x]:
                        return True
        return False

    def freeze_block(self):
        """블록 고정"""
        for y, row in enumerate(self.current_shape):
            for x, val in enumerate(row):
                if val:
                    self.board[self.current_y + y][self.current_x + x] = self.color
        self.clear_lines()
        self.spawn_new_block()

    def clear_lines(self):
        """가득 찬 줄 삭제"""
        lines_cleared = 0
        for y in range(ROWS-1, -1, -1):
            if all(self.board[y]):
                del self.board[y]
                self.board.insert(0, [None for _ in range(COLS)])
                lines_cleared += 1
        if lines_cleared > 0:
            self.score += lines_cleared * 100
            self.root.title(f"테트리스 - 점수: {self.score}")

    def spawn_new_block(self):
        """새 블록 생성"""
        shape_name = random.choice(list(SHAPES.keys()))
        self.current_shape = SHAPES[shape_name]
        self.color = COLORS[shape_name]
        self.current_x = COLS // 2 - len(self.current_shape[0]) // 2
        self.current_y = 0

        if self.check_collision(self.current_shape, self.current_x, self.current_y):
            self.running = False
            self.root.title("게임 오버")

    def move_down(self):
        if not self.check_collision(self.current_shape, self.current_x, self.current_y + 1):
            self.current_y += 1
        else:
            self.freeze_block()

    def move_left(self):
        if not self.check_collision(self.current_shape, self.current_x - 1, self.current_y):
            self.current_x -= 1

    def move_right(self):
        if not self.check_collision(self.current_shape, self.current_x + 1, self.current_y):
            self.current_x += 1

    def rotate(self):
        rotated = self.rotate_shape()
        if not self.check_collision(rotated, self.current_x, self.current_y):
            self.current_shape = rotated

    def key_handler(self, event):
        if not self.running:
            return
        if event.keysym == "Left":
            self.move_left()
        elif event.keysym == "Right":
            self.move_right()
        elif event.keysym == "Down":
            self.move_down()
        elif event.keysym == "Up":
            self.rotate()
        self.draw_board()

    def update(self):
        if self.running:
            self.move_down()
            self.draw_board()
            self.root.after(DELAY, self.update)

# 실행
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Tetris")
    game = Tetris(root)
    root.mainloop()