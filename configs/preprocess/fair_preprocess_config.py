type='FAIR'
source_fair_dataset_path='/home/cxjyxx_me/workspace/JAD/datasets/FAIR/fair'
source_dataset_path='/home/cxjyxx_me/workspace/JAD/datasets/FAIR/fair_DOTA'
target_dataset_path='/home/cxjyxx_me/workspace/JAD/datasets/FAIR/processed'
convert_tasks=['train','val','test']

# available labels: train, val, test, trainval
tasks=[
    dict(
        label='trainval',
        config=dict(
            subimage_size=600,
            overlap_size=150,
            multi_scale=[1.],
            horizontal_flip=False,
            vertical_flip=False,
            rotation_angles=[0.] 
        )
    ),
    dict(
        label='test',
        config=dict(
            subimage_size=600,
            overlap_size=150,
            multi_scale=[1.],
            horizontal_flip=False,
            vertical_flip=False,
            rotation_angles=[0.] 
        )
    )
]