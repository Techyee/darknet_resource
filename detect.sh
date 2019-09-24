#./darknet detector periodic ./cfg/coco.data ./yolov3-tiny.cfg ./yolov3-tiny.weights -res_cfg res_cfg.part -process_num 1 -filename coco/100.part
./darknet classifier predict ./cfg/imagenet1k.data ./alexnet.cfg ./alexnet.weights -res_cfg res_cfg.part -process_num 1 ./data/dog.png








#-thread_num 2 #< ./data/test_mult.txt

