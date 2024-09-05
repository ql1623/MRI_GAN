# MRI_GAN

to train

`python train.py --save_dir_name "save_name" --fold 0 --num_features 32 --batch_size 32 --data_png_dir "/data_path" --num_epochs 150`


to test
`python test.py --load_dir_name "save_name" --fold 0 --num_features 32 --batch_size 32 --data_png_dir "/data_path" --num_epochs 150`
