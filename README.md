# Lightweight Few-shot View Gaussian Splatting reconstruction method
## Environmental Setups
We provide install method based on Conda package and environment management:
```bash
conda env create --file environment.yml
conda activate L-GS
```

## Data Preparation
For different datasets, divide the training set and use Colmap to extract point clouds based on the training views. Mip-NeRF 360 uniformly divides 24 views as input, while LLFF uses 3 views.Note that extracting point clouds using this method requires a GPU-supported version of Colmap.

``` 
cd LFSVGS
mkdir dataset 
cd dataset

# download LLFF dataset
gdown 16VnMcF1KJYxN9QId6TClMsZRahHNMW5g

# run colmap to obtain initial point clouds with limited viewpoints
python tools/colmap_llff.py

# download MipNeRF-360 dataset
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip -d mipnerf360 360_v2.zip

# run colmap on MipNeRF-360 dataset
python tools/colmap_360.py
```
If you encounter difficulties during data preprocessing, you can download point cloud data that has been preprocessed using Colmap. You may download them [through this link](https://drive.google.com/drive/folders/1VymLQAqzXtrd2CnWAFSJ0RTTnp25mLgA?usp=share_link). We use dense point cloud during training but you can still try sparse point cloud on your own.

## Training
LLFF datasets. You can set `--sample_pseudo_interval` to 10 to achieve good results, which can significantly reduce the training time.
``` 
python train.py  --source_path dataset/nerf_llff_data/flower --model_path output/flower --eval --n_views 3 --sample_pseudo_interval 1 --comp --store_npz
```

MipNerf360 datasets
``` 
python train.py  --source_path dataset/mipnerf360/garden --model_path output/garden --eval --n_views 24 --depth_pseudo_weight 0.03 --comp --store_npz
```
We train on views that are downsampled by a factor of eight in both datasets. In this case, it is not recommended to use the C-T-F method, as excessive downsampling may lead to performance degradation during evaluation in some scenes. You can manually set `resize_period` to 0 and `resize_scale` to 1 in `arguments_init_.py`.

## Rendering

```
python render.py --source_path dataset/nerf_llff_data/flower/  --model_path  output/flower --iteration 10000 
```
If you want to obtain depth maps predicted by a monocular depth estimator.

```
python render.py --source_path dataset/nerf_llff_data/flower/  --model_path  output/flower --iteration 10000 --render_depth
```
You can customize the rendering path as same as NeRF by adding `video` argument

```
python render.py --source_path dataset/nerf_llff_data/flower/  --model_path  output/flower --iteration 10000  --video  --fps 30
```

## Evaluation
You can just run the following script to evaluate the model.  

```
python metrics.py --source_path dataset/nerf_llff_data/flower/  --model_path  output/flower --iteration 10000
```

## Acknowledgement

Our method benefits from these excellent works.

- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [FSGS](https://github.com/VITA-Group/FSGS)
- [SparseNeRF](https://github.com/Wanggcong/SparseNeRF)
- [Compact-3DGS](https://github.com/maincold2/Compact-3DGS)
- [MipNeRF-360](https://github.com/google-research/multinerf)
- [efficientgaussian](https://github.com/Sharath-girish/efficientgaussian)
