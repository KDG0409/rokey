from pathlib import Path

print(Path.cwd)

# 경로 생성
path=Path(r".\python\ch22\task\folder")
#print(path/"file.txt")

# 존재 확인
path=Path(r".\python\ch22\task\file.txt")
print(path.exists())
print(path.is_file())

path=Path(r".\python\ch22\task\folder")
print(path.exists())
print(path.is_dir())

# 디렉토리 생성,삭제
path=Path(r".\python\ch22\task\new_folder")
path.mkdir(exist_ok=True)
path.rmdir()

# 파일 생성,삭제
path=Path(r".\python\ch22\task\new_file.txt")
path.touch()
path.unlink()

# 목록,확장자 조회
path=Path(r".\python\ch22\task\folder")
for item in path.iterdir():
    print(item)
path=Path(r".\python\ch22\task\file")
print(path.suffix)

# 파일 이름 변경 및 이동
path=Path(r".\python\ch22\task\example.txt")
path.touch()
destination=Path(r".\python\ch22\task\new_folder\example.txt")
destination.parent.mkdir(exist_ok=True)
path.rename(destination)