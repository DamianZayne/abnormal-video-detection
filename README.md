# flow

此项目：在MLEP模型添加一个部署功能，此功能可以让用户从一个简易网站使用使用模型的功能。通过在网页输入图片上传，经过模型检测再输出在网页。  
此项目功能特色是模型单次输入只需5张图片，即可完成一次检测输出，并展示预测的图片，这能更直观地体会到模型的作用。  
MLEP：Margin Learning Embedded Prediction for Video Anomaly Detection with A Few Anomalies, IJCAI 2019  
运行test3.py的命令  
```
python test3.py --dataset
avenue \
--prednet
cyclegan_convlstm \
--num_his
4 \
--label_level
temporal \
--gpu
0 \
--interpolation \
--summary_dir
./outputs/summary/temporal/avenue/prednet_cyclegan_convlstm_folds_10_kth_1_/MARGIN_1.0_LAMBDA_1.0 \
--psnr_dir
./outputs/psnrs/temporal/avenue/prednet_cyclegan_convlstm_folds_10_kth_1_/MARGIN_1.0_LAMBDA_1.0
```
