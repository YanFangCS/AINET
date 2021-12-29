gpu=0
worker=4
python main.py --data='/home/name/AINet/BSDS500/' --savepath='./ckpt/' --workers $worker --input_img_height 208 --input_img_width 208 --print_freq 20 --gpu $gpu --batch-size 16  --suffix '_myTrain' 
