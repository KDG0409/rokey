import os
import shutil

#1. txt 파일을 검색하여 리스트로 반환
def list_text_files(folder_path):
    f_list=[]
    files=os.listdir(folder_path)
    for file in files:
        if file[-4:]==".txt":
            f_list.append(file)
    return f_list

#2. txt 파일을 찾아 다른 폴더로 복사
def copy_text_file(source_folder,destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for file in list_text_files(source_folder):
        shutil.copy(os.path.join(source_folder,file),
                    os.path.join(destination_folder,file))
        print("copy_file")

#3. txt 파일을 찾아 다른 폴더로 이동
def move_text_file(source_folder,destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for file in list_text_files(source_folder):
        shutil.move(os.path.join(source_folder,file),
                    os.path.join(destination_folder,file))
        print("move_file")
#4. txt 파일을 찾아 삭제
def del_text_file(source_folder):
    for file in list_text_files(source_folder):
        shutil.rmtree(os.path.join(source_folder,file))
    print("del_file")

#5. 함수를 호출
if __name__=="__main__":
    folder = "./python/ch22/task/folder1"
    source = "./python/ch22/task/folder1"
    destination = "./python/ch22/task/folder2"
    destination2 = "./python/ch22/task/folder3"
    if not os.path.exists(folder):
        os.mkdir(folder)
    file1=f"{folder}/file1.txt"
    file2=f"{folder}/file2.csv"
    with open(file1,'w')as f:
        f.write("file1\n")
    with open(file2,'w')as f:
        f.write("file2\n")
    print(list_text_files(folder))
    copy_text_file(source,destination)
    move_text_file(source,destination2)
    del_text_file(source)

shutil.rmtree(source)
shutil.rmtree(destination)
shutil.rmtree(destination2)   
