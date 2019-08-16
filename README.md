# Cascade-RCNN_Tensorflow

## Abstract
This is a tensorflow re-implementation of [Cascade R-CNN Delving into High Quality Object Detection ](https://arxiv.org/abs/1712.00726).       

This project is completed by [YangXue](https://github.com/yangxue0827) and [WangYashan](https://github.com/toubasi).      

## Train on VOC 2007 trainval and test on VOC 2007 test (PS. This project also support coco training.)     
![1](voc_2007.gif)      

## Comparison
### use_voc2012_metric
| Stage | AP50 | AP60 | AP70 | AP75 | AP80 | AP85 | AP90 | AP95 |
|------------|:---:|:--:|:--:|:--:|:---:|:--:|:--:|:--:|
|baseline|75.80|67.25|52.15|41.41|27.98|12.63|2.73|0.11|
|1+2+3|75.80|**68.74**|**57.09**|**48.68**|37.70|22.52|7.51|0.54|
|1+2|**75.98**|68.40|56.01|46.89|35.67|20.42|6.44|0.39|
|1|74.89|65.98|52.45|40.63|27.79|13.22|2.94|0.11|
|2|75.67|68.69|56.73|47.82|35.5|20.29|6.46|0.38|
|3|74.35|67.62|56.64|48.65|**38.02**|**23.19**|**8.05**|**0.54**|

### use_voc2007_metric
| Stage | AP50 | AP60 | AP70 | AP75 | AP80 | AP85 | AP90 | AP95 |
|------------|:---:|:--:|:--:|:--:|:---:|:--:|:--:|:--:|
|baseline|73.62|65.28|51.93|42.52|29.48|16.2|5.84|1.32|
|1+2+3|73.69|66.59|56.19|48.82|39.47|25.57|12.09|2.5|
|1+2|**74.01**|66.5|55.53|46.53|36.96|23.6|11.33|2.15|
|1|72.92|64.29|52.41|48.8|30.36|16|5.64|2.15|
|2|73.55|**66.75**|55.78|48.35|37.39|23.61|10.66|**2.69**|
|3|71.58|65.73|**56.64**|**49.08**|**39.68**|**26.25**|**12.28**|2.32|

## Requirements
1、tensorflow >= 1.2     
2、cuda8.0     
3、python2.7 (anaconda2 recommend)    
4、[opencv(cv2)](https://pypi.org/project/opencv-python/)    

## Download Model
1、please download [resnet50_v1](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)、[resnet101_v1](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz) pre-trained models on Imagenet, put it to $PATH_ROOT/data/pretrained_weights.     
2、please download [mobilenet_v2](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz) pre-trained model on Imagenet, put it to $PATH_ROOT/data/pretrained_weights/mobilenet.     
3、please download [trained model](https://github.com/DetectionTeamUCAS/Models/tree/master/Cascade_R-CNN_Tensorflow) by this project, put it to $PATH_ROOT/output/trained_weights.   

## Data Format
```
├── VOCdevkit
│   ├── VOCdevkit_train
│       ├── Annotation
│       ├── JPEGImages
│   ├── VOCdevkit_test
│       ├── Annotation
│       ├── JPEGImages
```

## Compile
```  
cd $PATH_ROOT/libs/box_utils/cython_utils
python setup.py build_ext --inplace
```

## Demo

**Select a configuration file in the folder ($PATH_ROOT/libs/configs/) and copy its contents into cfgs.py, then download the corresponding [weights](https://github.com/DetectionTeamUCAS/Models/tree/master/Cascade_R-CNN_Tensorflow).**      

```   
cd $PATH_ROOT/tools
python inference.py --data_dir='/PATH/TO/IMAGES/' 
                    --save_dir='/PATH/TO/SAVE/RESULTS/' 
                    --GPU='0'
```

## Eval
```  
cd $PATH_ROOT/tools
python eval.py --eval_imgs='/PATH/TO/IMAGES/'  
               --annotation_dir='/PATH/TO/TEST/ANNOTATION/'
               --GPU='0'
```

## Train

1、If you want to train your own data, please note:  
```     
(1) Modify parameters (such as CLASS_NUM, DATASET_NAME, VERSION, etc.) in $PATH_ROOT/libs/configs/cfgs.py
(2) Add category information in $PATH_ROOT/libs/label_name_dict/lable_dict.py     
(3) Add data_name to line 76 of $PATH_ROOT/data/io/read_tfrecord.py 
```     

2、make tfrecord
```  
cd $PATH_ROOT/data/io/  
python convert_data_to_tfrecord.py --VOC_dir='/PATH/TO/VOCdevkit/VOCdevkit_train/' 
                                   --xml_dir='Annotation'
                                   --image_dir='JPEGImages'
                                   --save_name='train' 
                                   --img_format='.jpg' 
                                   --dataset='pascal'
```     

3、train
```  
cd $PATH_ROOT/tools
python train.py
```

## Tensorboard
```  
cd $PATH_ROOT/output/summary
tensorboard --logdir=.
``` 
![2](scalars.png)
![1](images.png)

## Reference
1、https://github.com/endernewton/tf-faster-rcnn   
2、https://github.com/zengarden/light_head_rcnn   
3、https://github.com/tensorflow/models/tree/master/research/object_detection
