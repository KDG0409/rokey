from pathlib import Path

# 1. 특정 폴더 내의 모든 파일 목록을 반환하는 함수
def get_files(folder_path):
    f_list=[]
    folder = Path(folder_path)
    # for file in folder.iterdir():
    #     f_list.append(str(file))
    # return f_list
    return [file for file in folder.iterdir() if file.is_file()]

#2. 특정 폴더 내의 파일을 확장자 별로 폴더를 생성하여 이동하는 함수
def organize_file_by_extension(source_folder):
    source=Path(source_folder)
    if not source.exists():
        print("폴더가 존재하지않음")
        return
    for file in get_files(source):
        extension = file.suffix[1:]
        ext_folder=source/extension # 확장자명으로 폴더 생성
        ext_folder.mkdir(exist_ok=True)
        file.rename(ext_folder/file.name) # 이름변경/이동


#3. 함수 호출
if __name__ == "__main__":
    source = r".\python\ch22\task\files"
    print(get_files(source))
    print(organize_file_by_extension(source))