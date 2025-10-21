import pymysql

conn = pymysql.connect(
    host = "localhost",
    user = "root",
    password = "1234",
    database = "exampledb",
    charset = "utf8mb4",
    cursorclass = pymysql.cursors.DictCursor
)
cursor = conn.cursor()

cursor.execute("SELECT DATABASE()")
print("현재 데이터베이스:", cursor.fetchone())

cursor.execute("SELECT @@AUTOCOMMIT")
autocommit_status = cursor.fetchone()
print("자동 커밋 상태:", autocommit_status)

# 테이블 생성

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age INT,
    email VARCHAR(100) UNIQUE
)
""")
conn.commit() # 변경 사항 저장

cursor.execute("""
    CREATE TABLE Employees (
    employee_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) NOT NULL,
    department_name VARCHAR(50),
    salary DECIMAL(10,2) CHECK (salary > 0),
    hiredate DATE DEFAULT (CURRENT_DATE)
);
               """)

cursor.execute("""
    CREATE TABLE new_Employees (
    employee_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) NOT NULL,
    department_name VARCHAR(50),
    salary DECIMAL(10,2) CHECK (salary > 0),
    hiredate DATE DEFAULT (CURRENT_DATE)
);
               """)
conn.commit() # 변경 사항 저장

# 데이터 삽입
cursor.execute("INSERT INTO Employees (name, department_name, salary) VALUES ('김철수', '영업팀', 5000)")
cursor.execute("INSERT INTO Employees (employee_id, name, salary) VALUES (2, '이영희', 4000)")
cursor.execute("INSERT INTO Employees (employee_id, name, hiredate) VALUES (3, '박민수', '2025-02-05')")
cursor.execute("INSERT INTO employees VALUES (4, '박명수', '영업팀', 4500, '2025-02-06')")
cursor.execute("INSERT INTO Employees VALUES (5, '유재석', '개발팀', 7000, '2025-02-06')")
conn.commit() # 변경 사항 저장
print("데이터 삽입 완료")

cursor.execute("INSERT INTO users (name, age, email) VALUES ('Alice', 25,'alice@example.com');")
conn.commit() # 변경 사항 저장
print("데이터 삽입 완료")

# 데이터 삽입

cursor.execute("INSERT INTO new_employees SELECT * FROM employees WHERE salary < 5000")
cursor.execute("SELECT * FROM new_employees")
users = cursor.fetchall()

for user in users:
    print(user)

# 데이터 조회

cursor.execute("SELECT * FROM users")
users = cursor.fetchall()

for user in users:
    print(user)

cursor.execute("SELECT * FROM employees")
users = cursor.fetchall()

for user in users:
    print(user)

# 데이터 수정

cursor.execute("UPDATE users SET name = %s WHERE id = %s", ("Bob", 1))
conn.commit()
print("데이터 수정 완료")

cursor.execute("SELECT * FROM users")
users = cursor.fetchall()

for user in users:
    print(user)

# 데이터 삭제

cursor.execute("DELETE FROM users WHERE id = 2")
conn.commit()
print("데이터 삭제 완료")

cursor.execute("SELECT * FROM users")
users = cursor.fetchall()

for user in users:
    print(user)

# 테이블 삭제

cursor.execute("DROP TABLE users")
cursor.execute("DROP TABLE employees")
cursor.execute("DROP TABLE new_employees")

conn.close()