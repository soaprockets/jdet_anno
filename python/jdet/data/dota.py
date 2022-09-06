from numpy.lib import save
from jdet.data.devkits.voc_eval import voc_eval_dota
from jdet.models.boxes.box_ops import rotated_box_to_poly_np, rotated_box_to_poly_single
from jdet.utils.general import check_dir
from jdet.utils.registry import DATASETS
from jdet.config.constant import get_classes_by_name 
from jdet.data.custom import CustomDataset
from jdet.ops.nms_poly import iou_poly
import os
import jittor as jt
import numpy as np
from tqdm import tqdm

def s2anet_post(result):
    dets,labels = result 
    labels = labels+1 
    scores = dets[:,5]
    dets = dets[:,:5]
    polys = rotated_box_to_poly_np(dets)  # 将poly 转成 rotated_box=[x_ctr,y_ctr,w,h,angle].astype(np.float32)
    return polys,scores,labels

@DATASETS.register_module()
#继承jdet.data.custom 中的CustomDataset"""
class DOTADataset(CustomDataset):

    def __init__(self,*arg,balance_category=False,version='1',**kwargs):
        assert version in ['1', '1_5', '2']
        """ jdet.config.constant 中存储所有数据集的标签类别和 """
        self.CLASSES = get_classes_by_name('DOTA'+version)

        super().__init__(*arg,**kwargs)
        if balance_category:# 如果需要进行类别平衡，调用_balance_categories(self)方法
            self.img_infos = self._balance_categories()
            self.total_len = len(self.img_infos)


 #通过图像idx重复采样使得label之间的样本数达到基本平衡，关键操作是 new_d = d*l1+d[:l2]"""
    def _balance_categories(self):
        img_infos = self.img_infos
        cate_dict = {}
        for idx,img_info in enumerate(img_infos):
            unique_labels = np.unique(img_info["ann"]["labels"]) #获取 img_info["ann"]["labels"]中的所有label，返回无重复元素的label的list
            for label in unique_labels:
                if label not in cate_dict:
                    cate_dict[label]=[] 
                cate_dict[label].append(idx)  # cate_dicts中的label对应一个列表，列表该类别bbox在img_info列表中的索引idx，label是int类型
        new_idx = []
        balance_dict={
            "storage-tank":(1,526), #该元组表示idx的缩放因子和偏移量
            "baseball-diamond":(2,202),
            "ground-track-field":(1,575),
            "swimming-pool":(2,104),
            "soccer-ball-field":(1,962),
            "roundabout":(1,711),
            "tennis-court":(1,655),
            "basketball-court":(4,0),
            "helicopter":(8,0),
            "container-crane":(50,0)
        }

        for k,d in cate_dict.items():
            classname = self.CLASSES[k-1] #获得类别名
            l1,l2 = balance_dict.get(classname,(1,0)) #dict中存在classname，返回对应的tuple，无则返回（1,0）
            """ 数据集扩充关键代码"""
            new_d = d*l1+d[:l2] # list*l1+list[:l2]表示列表的复制l1次并加上列表中前l2个图像的idx，表示数据重复采样，达到每个类别样本数近似一致
            new_idx.extend(new_d)
        img_infos = [self.img_infos[idx] for idx in new_idx] # 表示img_infos的扩充
        return img_infos

#保存输出结果到txt文档中
    def parse_result(self,results,save_path):
        check_dir(save_path) # os.makedir(save_path,exist_ok = True)判定save_path是否存在，不存在直接创建路径目录
        data = {}
        for (dets,labels),img_name in results:
            img_name = os.path.splitext(img_name)[0] # 根据"."将文件名和扩展名分开，类似str.split(".")[0]，分成 [file_path,py]
            for det,label in zip(dets,labels):
                bbox = det[:5]  # det[0:5]存储bbox坐标
                score = det[5]  # det[5]存储预测分数
                classname = self.CLASSES[label]  # label 是int类型，即 constant.py中类别列表的索引
                bbox = rotated_box_to_poly_single(bbox) # bbox =[x_center,y_center,width,height,rotated_angle]
                temp_txt = '{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
                            img_name, score, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4],
                            bbox[5], bbox[6], bbox[7])
                if classname not in data:
                    data[classname] = [] #data={classname:temp_text}
                data[classname].append(temp_txt)
        for classname,lines in data.items():
            f_out = open(os.path.join(save_path, classname + '.txt'), 'w')
            f_out.writelines(lines)
            f_out.close()



    def evaluate(self,results,work_dir,epoch,logger=None,save=True):
        print("Calculating mAP......")
        if save:
            save_path = os.path.join(work_dir,f"detections/val_{epoch}")
            check_dir(save_path)
            jt.save(results,save_path+"/val.pkl")
        dets = []
        gts = []
        diffcult_polys = {}
        for img_idx,(result,target) in enumerate(results):
            det_polys,det_scores,det_labels =  result
            det_labels += 1
            if det_polys.size>0: # 判断det_ploy非空
                idx1 = np.ones((det_labels.shape[0],1))*img_idx
                print(f'定位在dota.py第108行代码输出:idx1={idx1}')
                det = np.concatenate([idx1,det_polys,det_scores.reshape(-1,1),det_labels.reshape(-1,1)],axis=1) #对应行的拼接
                print(f'定位在dota.py第110行代码输出:det={det}')
                dets.append(det)
            
            scale_factor = target["scale_factor"]
            gt_polys = target["polys"]
            gt_polys /= scale_factor

            if gt_polys.size>0:
                gt_labels = target["labels"].reshape(-1,1) 
                idx2 = np.ones((gt_labels.shape[0],1))*img_idx
                gt = np.concatenate([idx2,gt_polys,gt_labels],axis=1)
                gts.append(gt)
            diffcult_polys[img_idx] = target["polys_ignore"]/scale_factor
        if len(dets) == 0:
            aps = {}
            for i,classname in tqdm(enumerate(self.CLASSES),total=len(self.CLASSES)):
                aps["eval/"+str(i+1)+"_"+classname+"_AP"]=0 
            map = sum(list(aps.values()))/len(aps)
            aps["eval/0_meanAP"]=map
            return aps
        dets = np.concatenate(dets)
        gts = np.concatenate(gts)
        aps = {}
        for i,classname in tqdm(enumerate(self.CLASSES),total=len(self.CLASSES)):
            c_dets = dets[dets[:,-1]==(i+1)][:,:-1]
            c_gts = gts[gts[:,-1]==(i+1)][:,:-1]
            img_idx = gts[:,0].copy()
            classname_gts = {}
            for idx in np.unique(img_idx):
                g = c_gts[c_gts[:,0]==idx,:][:,1:]
                dg = diffcult_polys[idx].copy().reshape(-1,8)
                diffculty = np.zeros(g.shape[0]+dg.shape[0])
                diffculty[int(g.shape[0]):]=1
                diffculty = diffculty.astype(bool)
                g = np.concatenate([g,dg])
                classname_gts[idx] = {"box":g.copy(),"det":[False for i in range(len(g))],'difficult':diffculty.copy()}
            rec, prec, ap = voc_eval_dota(c_dets,classname_gts,iou_func=iou_poly)
            aps["eval/"+str(i+1)+"_"+classname+"_AP"]=ap 
        map = sum(list(aps.values()))/len(aps)
        aps["eval/0_meanAP"]=map
        return aps
            
            
def test_eval():
    results= jt.load("projects/s2anet/work_dirs/s2anet_r50_fpn_1x_dota/detections/val_0/val.pkl")
    results = jt.load("projects/s2anet/work_dirs/s2anet_r50_fpn_1x_dota/detections/val_rotate_balance/val.pkl")
    # results = results
    dataset = DOTADataset(annotations_file='/mnt/disk/lxl/dataset/DOTA_1024/trainval_split/trainval1024.pkl',
        images_dir='/mnt/disk/lxl/dataset/DOTA_1024/trainval_split/images/')
    dataset.evaluate(results,None,None,save=False)
    
    # data = []
    # for result,target in results:
    #     img_name = target["filename"]
    #     data.append((result,img_name))

    # dataset.parse_result(data,"test_")



if __name__ == "__main__":
    test_eval()