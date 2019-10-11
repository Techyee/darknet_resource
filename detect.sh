./darknet detector periodic ./cfg/coco.data ./yolov3-tiny.cfg ./yolov3-tiny.weights -res_cfg res_cfg.part -period 220 -process_num 14 -filename coco/100.part
./darknet detector periodic -models models.list -weights weights.list -data data.list -res_cfg res_cfg.list -period 220 -process_num 4 -filename coco/100.part
#./darknet classifier predict ./cfg/imagenet1k.data ./extraction.cfg ./extraction.weights -res_cfg res_cfg.part -process_num 1 ./data/eagle.jpg








#-thread_num 2 #< ./data/test_mult.txt

