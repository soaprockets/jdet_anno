type='FAIR1M_1_5'
source_fair_dataset_path='/home/hexf/data/dataset/remote_det/data'
source_dataset_path='/home/hexf/data/dataset/remote_det/dota'
target_dataset_path='/home/hexf/data/dataset/remote_det/preprocessed'
convert_tasks=['train','test']

# available labels: train, val, test, trainval
tasks=[
    dict(
        label='train',
        config=dict(
            subimage_size=1024,
            overlap_size=200,
            multi_scale=[1.],
            horizontal_flip=False,
            vertical_flip=False,
            rotation_angles=[0.] 
        )
    ),
    dict(
        label='test',
        config=dict(
            subimage_size=1024,
            overlap_size=200,
            multi_scale=[1.],
            horizontal_flip=False,
            vertical_flip=False,
            rotation_angles=[0.] 
        )
    )
]