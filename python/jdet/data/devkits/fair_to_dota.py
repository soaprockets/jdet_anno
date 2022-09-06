import os
import cv2
from xml.dom.minidom import parse
from tqdm import tqdm
import sys


#xml 通过树的形式嵌套内容，根据html的标签字符索引内容"""
def solve_xml(src, tar):
    domTree = parse(src)
    rootNode = domTree.documentElement # 根节点是document对象
    objects = rootNode.getElementsByTagName("objects")[0].getElementsByTagName("object") #获取根节点下的所有子代节点---objects
    box_list=[]
    #循环遍历objects中的对象
    for obj in objects: 
        name=obj.getElementsByTagName("possibleresult")[0].getElementsByTagName("name")[0].childNodes[0].data #通过<name>字段值索引label ,
        points=obj.getElementsByTagName("points")[0].getElementsByTagName("point") # 通过<points>字段值 索引bbox上下左右四个框的坐标值
        bbox=[]
        for point in points[:4]:
            x=point.childNodes[0].data.split(",")[0]
            y=point.childNodes[0].data.split(",")[1]
            bbox.append(float(x))
            bbox.append(float(y))
        box_list.append({"name":name, "bbox":bbox}) #{name:label,"bbox":[上，下，左，右] }
    
    file=open(tar,'w')
    print("imagesource:GoogleEarth",file=file)
    print("gsd:0.0",file=file)
    for box in box_list:
        ss=""
        for f in box["bbox"]:
            ss+=str(f)+" " # ss+ 各点坐标+ 空格“ ”
        name=  box["name"]  # name就是 label
        name = name.replace(" ", "_") #将label字段中的空格字段换成"_"
        ss+=name+" 0"  # ss = bbox + label + '0'
        print(ss,file=file)
    file.close()


def fair_to_dota(in_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(os.path.join(out_path, "images"), exist_ok=True)



#对ext.tiff 图像进行转换，target是ext.png"""

    tasks = []
    for root, dirs, files in os.walk(os.path.join(in_path, "images")): #os.walk 返回当前操作的目录root，root下的子目录dirs和一些文件files
        for f in files:
            src=os.path.join(root, f)
            tar="P"+f[:-4].zfill(4)+".png" 
            tar=os.path.join(out_path,"images", tar) # 指定目标文件下的图像命名格式。
            tasks.append((src, tar))
    print("processing images")
    for task in tqdm(tasks):# task = (src目录，target目录)
        file = cv2.imread(task[0], 1) #RGB读取图像
        cv2.imwrite(task[1], file)


#对labelXml下的label进行处理，调用solve_xml方法处理，只读取<name>,<point>两个字段的值"""
    if (os.path.exists(os.path.join(in_path, "labelXml"))):
        os.makedirs(os.path.join(out_path, "labelTxt"), exist_ok=True)
        tasks = []
        for root, dirs, files in os.walk(os.path.join(in_path, "labelXml")):
            for f in files:
                src=os.path.join(root, f)
                tar="P"+f[:-4].zfill(4)+".txt"
                tar=os.path.join(out_path,"labelTxt", tar)
                tasks.append((src, tar))
        print("processing labels")
        for task in tqdm(tasks):
            solve_xml(task[0], task[1])

if __name__ == '__main__':
    src = sys.argv[1] #对应配置文件下的source_fair_dataset_path 目录
    tar = sys.argv[2] # 对应配置文件下的source_dataset_path 目录，即dota目录
    fair_to_dota(src, tar)
