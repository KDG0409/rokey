from pathlib import Path

source=r".\python\ch22\Assignment\sample_folder\sample.txt"
path=Path(source)
path.touch()

source1=r".\python\ch22\Assignment\sample_folder"
path=Path(source1)
for file in path.iterdir():
    print(file)
    txt_list=[]
    if file.suffix == ".txt":
        txt_list.append(file.name)
print(txt_list)

source2=r".\python\ch22\Assignment\new_folder"
path=Path(source2)
if not path.exists():
    path.mkdir(exist_ok=True)