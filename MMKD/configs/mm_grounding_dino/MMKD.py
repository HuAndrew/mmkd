_base_ = './mmdetection/configs/mm_grounding_dino/coco/grounding_dino_swin-t_finetune_16xb4_1x_coco.py'

data_root = './'
# class_name = ('cat', )
# num_classes = len(class_name)
# metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

class_name = ('car', 'bus','truck', 'car_reg', 'car_big_reg', 'car_front', 'car_big_front', 'person', 'bicyclist', 'motorcyclist','trafficlight','sign','licence')
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])
teacher_ckpt = './mmdetection/epoch_20.pth'

model = dict(type='GroundingDINOKD',bbox_head=dict(num_classes=num_classes),teacher_config='',teacher_ckpt=teacher_ckpt)

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='./mmdetection/data/AEB/GDDA1.5M.json',
        data_prefix=dict(img='data1/ADAS_DATASET_FOR_MMgdino')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='./mmdetection/data/AEB/GDDA1.5M_test_1.7W.json',
        data_prefix=dict(img='data1/ADAS_DATASET_FOR_MMgdino')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file='./mmdetection/data/AEB/GDDA1.5M_test_1.7W.json')
test_evaluator = val_evaluator

max_epoch = 20

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=500))
train_cfg = dict(max_epochs=max_epoch, val_interval=20)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[13,15],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.00005),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0),
        }))

auto_scale_lr = dict(base_batch_size=20)

work_dir = 'MMKD/GDDA1.5M_Main'
