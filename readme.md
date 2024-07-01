# Defect type classification
Data type : Grayscale image 

Type of defects : 4
- 4 classes classification 

Statistics:

|Type|#|
|-|-|
|Type0|1427|
|Type1|46|
|Type2|420|
|Type3|579|

- Training-validation-testing split $\approx$ 75:5:20
    |Type|# training |# validation|# testing|
    |-|-|-|-|
    |Type0|1026|115|286|
    |Type1|32|4|10|
    |Type2|302|34|84|
    |Type3|416|47|116|

## Method :

## Use simple resnet50 for classification:

### Hardware:
4 $\times$ 1080 Ti
### hyperparameters:
- batchsize : 68
- max iteration: 50
    - Will stop early if there's no improvement on the validation set for 30 iterations.
- optimizer: Adam
    - initial learning rate : 0.001

Marco-F1 score

|loss function / Dataset|baseline|aug Type1~Type3| down samping Type0 , aug Type1|
|-|-|-|-|
|CE|0.828|0.934|0.890|
|WCE|0.876|0.923| 0.899|
|Focal Loss|0.855|0.883|0.846|

## Use simple resnet34 for classification the connected component pathes and vote:
Marco-F1 score: 0.826