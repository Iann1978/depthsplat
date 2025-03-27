#!/usr/bin/env bash


# base model
# train on 8x GPUs, batch size 4 on each gpu (>=45GB memory)
python -m src.main +experiment=re10k \
data_loader.train.batch_size=4 \
dataset.test_chunk_interval=10 \
trainer.val_check_interval=0.5 \
trainer.max_steps=150000 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitb \
model.encoder.gaussian_regressor_channels=32 \
model.encoder.color_large_unet=true \
model.encoder.feature_upsampler_channels=128 \
model.encoder.return_depth=true \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vitb.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
wandb.project=depthsplat \
output_dir=checkpoints/re10k-depthsplat-base


# evaluate on re10k
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
dataset.test_chunk_interval=1 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitb \
model.encoder.gaussian_regressor_channels=32 \
model.encoder.color_large_unet=true \
model.encoder.feature_upsampler_channels=128 \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-re10k-256x256-044fdb17.pth \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.mode=disabled \
test.save_image=false \
test.save_gt_image=false \
output_dir=output/tmp


# render video on re10k (need to have ffmpeg installed)
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
dataset.test_chunk_interval=100 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitb \
model.encoder.gaussian_regressor_channels=32 \
model.encoder.color_large_unet=true \
model.encoder.feature_upsampler_channels=128 \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-re10k-256x256-044fdb17.pth \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json \
test.save_video=true \
test.compute_scores=false \
wandb.mode=disabled \
test.save_image=false \
test.save_gt_image=false \
output_dir=output/tmp


# evaluate on re10k
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
dataset.test_chunk_interval=1 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitb \
checkpointing.pretrained_model=../depthsplat/pretrained/depthsplat-camera-ready-release/depthsplat-gs-base-re10k-256x256-view2-fbe87117.pth \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.mode=disabled \
test.save_image=false \
test.save_gt_image=false \
output_dir=output/tmp

