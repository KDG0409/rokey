import shutil
import os

path = r"./python/ch22/task"
src=f"{path}/file.txt"
dst=f"{path}/copiedfile.txt"

if not os.path.exists(dst):
    shutil.copy(src,dst)

#디렉토리 생성
folder = f"{path}/test_dir"
if not os.path.exists(folder):
    os.mkdir(folder)

#파일 복사
src=f"{path}/file.txt"
dst=f"{path}/test_dir/file.txt"
if not os.path.exists(dst):
    shutil.copy(src,dst)

src=f"{path}/test_dir"
dst=f"{path}/copied_test_dir"
if not os.path.exists(dst):
    shutil.copytree(src,dst)

#파일 이동
src=f"{path}/copiedfile.txt"
dst=f"{path}/copied_test_dir/copiedfile.txt"
if not os.path.exists(dst):
    shutil.move(src,dst) 

#파일 삭제
dir1=f"{path}/test_dir"
dir2=f"{path}/copied_test_dir"

shutil.rmtree(dir1)
shutil.rmtree(dir2)

