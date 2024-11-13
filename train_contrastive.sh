source /.tensorflow2/bin/activate
python scripts/train_contrastive.py --path_pickle_train=train/1fps_100_0.pickle \
--path_pickle_val=val/1fps_0.pickle \
--path_dataset=/DATA/iae_frames/ --data_center=stras --num_frames_input=10

