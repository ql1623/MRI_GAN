# MRI_GAN

to train reported model, the two conditional encoder approach using convolution conditioning:

`python train.py --save_dir_name "save_name" --fold 0 --num_features 32 --batch_size 32 --data_png_dir "/data_path" --num_epochs 150`


to test:
`python test.py --load_dir_name "save_name" --fold 0 --num_features 32 --batch_size 32 --data_png_dir "/data_path" --load_epoch 150`



other model can be trained using:
`python train_3_encode.py --save_dir_name "save_name" --fold 0 --num_features 32 --batch_size 32 --data_png_dir "/data_path" --num_epochs 150`
`python train_2_encode_linear.py --save_dir_name "save_name" --fold 0 --num_features 32 --batch_size 32 --data_png_dir "/data_path" --num_epochs 150`
