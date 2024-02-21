CUDA_VISIBLE_DEVICES=0 python3 train.py -g 0 -o result_0 -e 300 -b 16 -t 1 -s 0 --cuda
CUDA_VISIBLE_DEVICES=0 python3 train.py -g 0 -o result_1 -e 300 -b 16 -t 1 -s 1 --cuda
CUDA_VISIBLE_DEVICES=0 python3 train.py -g 0 -o result_2 -e 300 -b 16 -t 1 -s 2 --cuda
CUDA_VISIBLE_DEVICES=0 python3 test.py -g 0 -o result_0 -s 0 --cuda
CUDA_VISIBLE_DEVICES=0 python3 test.py -g 0 -o result_1 -s 1 --cuda
CUDA_VISIBLE_DEVICES=0 python3 test.py -g 0 -o result_2 -s 2 --cuda

