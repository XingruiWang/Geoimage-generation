CUDA_VISIBLE_DEVICES=1,3 python train.py \
    --dataset="geoimage" \
    --batch_size=8 \
    --test_batch_size=4 \
    --output_dir="./visualization/geoimage" \
    --niter=100 \
    --net_G="./checkpoint/geoimage/netG_model_epoch_300.pth" \
    --net_D="./checkpoint/geoimage/netD_model_epoch_300.pth"