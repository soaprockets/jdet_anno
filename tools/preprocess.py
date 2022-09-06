import cv2
import argparse
import os
import shutil
from jdet.config import init_cfg, get_cfg
from jdet.data.devkits.ImgSplit_multi_process import process
from jdet.data.devkits.convert_data_to_mmdet import convert_data_to_mmdet
from jdet.data.devkits.fair_to_dota import fair_to_dota
from jdet.utils.general import is_win

from jdet.data.devkits.ssdd_to_dota import ssdd_to_dota  #jdet.data.devkits下存储ssdd-dota,fair-dota互相转换几种方法。


def clear(cfg):
    if is_win():
        """windows系统下使用shutil.retree删除文件目录及其下所有文件"""
        shutil.rmtree(os.path.join(cfg.source_dataset_path, 'trainval'),ignore_errors=True)
        shutil.rmtree(os.path.join(cfg.target_dataset_path),ignore_errors=True)
    else:
        """ linux 系统下使用rm -rf 删除文件目录及其下所有文件 """
        os.system(f"rm -rf {os.path.join(cfg.source_dataset_path, 'trainval')}")
        os.system(f"rm -rf {os.path.join(cfg.target_dataset_path)}")

def run(cfg):
    """根据输入的数据类型指定相应的转换分支，当前预处理的数据集类型为fair1m_1_5"""
     # SSDD及SSDD+为舰船检测数据集
    if cfg.type=='SSDD+' or cfg.type=='SSDD':
        for task in cfg.convert_tasks:
            print('==============')
            print("convert to dota:", task)
            # create the dir of the converted data
            out_path = os.path.join(cfg.target_dataset_path, task)  # 创建训练/验证任务的数据输出目录，根据config下 preprocessed.py指定的target_dataset_path指定
            if task == 'test':
                out_path = os.path.join(cfg.target_dataset_path, 'val') # 测试集的数据输出目录，指定的preprocessed目录下的文件
            out_path += '_' + str(cfg.resize) #加上是否经过resize操作flag值，1.0表示经过resize，0.0表示未经过resize操作
            if cfg.type=='SSDD+':
                """ssdd_to_dota将img，anno分别转化成ext.png,ext.txt两种格式，
                dota目录的img，anno是一一对应的。
                convert_data_to_mmdet是将dota目录下的.png,.txt通过字典的方式封装在pkl文件中
                """
                ssdd_to_dota(
                    os.path.join(cfg.source_dataset_path, f'JPEGImages_{task}'),
                    os.path.join(cfg.source_dataset_path, f'Annotations_{task}'),
                    out_path,
                    cfg.resize,
                    plus=True
                )
            else:
                ssdd_to_dota(
                    os.path.join(cfg.source_dataset_path, f'JPEGImages_{task}'),
                    os.path.join(cfg.source_dataset_path, f'Annotations_{task}'),
                    out_path,
                    cfg.resize,
                    plus=False
                )

            convert_data_to_mmdet(out_path, os.path.join(out_path, 'labels.pkl'), type=cfg.type)
        return
    """ """

    if (cfg.type=='FAIR' or cfg.type=='FAIR1M_1_5'):
        for task in cfg.convert_tasks:
            print('==============')
            print("convert to dota:", task)
            fair_to_dota(os.path.join(cfg.source_fair_dataset_path, task), os.path.join(cfg.source_dataset_path, task))

    for task in cfg.tasks:
        label = task.label #获取train，test两个标签值
        cfg_ = task.config
        print('==============')
        print("processing", label)


        """
         config=dict(
            subimage_size=1024,
            overlap_size=200,
            multi_scale=[1.],
            horizontal_flip=False,
            vertical_flip=False,
            rotation_angles=[0.] 
        )
        """
        subimage_size=600 if cfg_.subimage_size is None else cfg_.subimage_size 
        overlap_size=150 if cfg_.overlap_size is None else cfg_.overlap_size
        multi_scale=[1.] if cfg_.multi_scale is None else cfg_.multi_scale
        horizontal_flip=False if cfg_.horizontal_flip is None else cfg_.horizontal_flip
        vertical_flip=False if cfg_.vertical_flip is None else cfg_.vertical_flip
        rotation_angles=[0.] if cfg_.rotation_angles is None else cfg_.rotation_angles
        
        """ 遥感图无法进行旋转，翻转等操作，否则对应的bbox的坐标也需要进行相应的变换 """
        assert(rotation_angles == [0.]) #TODO support multi angles
        assert(horizontal_flip == False) #TODO support horizontal_flip
        assert(vertical_flip == False) #TODO support vertical_flip

        assert(label in ['trainval', 'train', 'val', 'test'])
        in_path = os.path.join(cfg.source_dataset_path, label) #dota dirs
        out_path = os.path.join(cfg.target_dataset_path, label) # preprocessed dirs
        # generate trainval 针对上级目录是trainval做的处理，重新定义out_img,out_label 的路径
        if (label == 'trainval' and (not os.path.exists(in_path))): #输入路径不存在条件判断才能成立
            out_img_path = os.path.join(cfg.source_dataset_path, 'trainval', 'images')
            out_label_path = os.path.join(cfg.source_dataset_path, 'trainval', 'labelTxt')
            os.makedirs(out_img_path,exist_ok=True)
            os.makedirs(out_label_path,exist_ok=True)
            # TODO support Windows etc.
            if is_win():
                """windows下使用shutil.copytree拷贝整个文件目录及其下的所有文件到指定目录"""
                shutil.copytree(os.path.join(cfg.source_dataset_path, 'train', 'images'),out_img_path,dirs_exist_ok=True) 
                shutil.copytree(os.path.join(cfg.source_dataset_path, 'val', 'images'),out_img_path,dirs_exist_ok=True)
                shutil.copytree(os.path.join(cfg.source_dataset_path, 'train', 'labelTxt'),out_label_path,dirs_exist_ok=True)
                shutil.copytree(os.path.join(cfg.source_dataset_path, 'val', 'labelTxt'),out_label_path,dirs_exist_ok=True)
            else:
                """linux下使用cp拷贝整个文件目录及其下的所有文件到指定目录"""
                os.system(f"cp {os.path.join(cfg.source_dataset_path, 'train', 'images', '*')} {out_img_path}")
                os.system(f"cp {os.path.join(cfg.source_dataset_path, 'val', 'images', '*')} {out_img_path}")
                os.system(f"cp {os.path.join(cfg.source_dataset_path, 'train', 'labelTxt', '*')} {out_label_path}")
                os.system(f"cp {os.path.join(cfg.source_dataset_path, 'val', 'labelTxt', '*')} {out_label_path}")
       
       
        target_path = process(in_path, out_path, subsize=subimage_size, gap=overlap_size, rates=multi_scale) #多批次处理数据
        if (label != "test"):
            print("converting to mmdet format...")
            print(cfg.type)
            convert_data_to_mmdet(target_path, os.path.join(target_path, 'labels.pkl'), type=cfg.type)

def main():
    parser = argparse.ArgumentParser(description="Jittor DOTA data preprocess")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--clear",
        default=False,
        action='store_true'
    )
    args = parser.parse_args()
    if args.config_file:
        init_cfg(args.config_file)
    cfg = get_cfg()
    print(cfg.dump())

    if (args.clear):
        clear(cfg)
    else:
        run(cfg)

if __name__ == "__main__":
    main()