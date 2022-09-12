import argparse
import jittor as jt
jt.flags.use_cuda_managed_allocator=1
from jdet.runner import Runner 
from jdet.config import init_cfg
from time import time
def main():
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="/home/hexf/data/jdet_anno/configs/s2anet/s2anet_r50_fpn_1x_fair1m_1_5.py",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--task",
        default="train",
        help="train,val,test",
        type=str,
    )

    parser.add_argument(
        "--no_cuda",
        action='store_true'
    )

    parser.add_argument(
        "--save_dir",
        default="/home/hexf/data/jdet_anno",
        type=str,
    )
    
    args = parser.parse_args()

    if not args.no_cuda:
        jt.flags.use_cuda=1

    assert args.task in ["train","val","test","vis_test"],f"{args.task} not support, please choose [train,val,test,vis_test]"
    
    if args.config_file:
        init_cfg(args.config_file)

    runner = Runner()

    if args.task == "train":
        runner.run()
    elif args.task == "val":
        runner.val()
    elif args.task == "test":
        runner.test()
    elif args.task == "vis_test":
        runner.run_on_images(args.save_dir)

if __name__ == "__main__":
    main()
    # print(f'train/test time is{end_time-start_time}')
