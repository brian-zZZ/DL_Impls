## Dataset
Flower classification dataset

## Fine-tuning
Only fine-tuning code. Excute the ```train.py``` with scripts below to fine-tune with pre-trained weights.
* Run with single GPU. Specify with ```device``` setting.
* Change **```from vit_model import vit_base_patch16_224_in21k as create_model```** to switch model.
```
python train.py --num_classes 5 \
    --epochs 10 --batch-size 8 --lr 0.001 \
    --data-path '/home/brian/datasets/flower_data/flower_photos' \
    --weights './weights/vit_base_patch16_224_in21k.pth' \
    --device 'cuda:0'
```
## Predict
Excute the ```predict.py``` to predict a single pic with the fine-tuned weights abtained above.
* Specify the ```img_path``` value in main function.
* Specify the ```model_weight_path``` value toward fine-tuned weights.
```
python predict.py
```
