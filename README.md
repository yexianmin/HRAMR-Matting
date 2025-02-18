# HRAMR-Matting

### Requirements
The codes are tested in the following environment:
- python 3.9
- pytorch 1.8.0
- CUDA 11.1 & CuDNN 8.0.5

~~~python
pip3 install -r requirements.txt
~~~

The model is trained on Composition-1K train dataset, and tested on Transparent-460 test dataset.
| Models | SAD | MSE | Grad | Conn | Link|
|  ----  | ----  |  ----  | ----  |  ----  | ----  |
| HRAMR-Matting | 173.43 | 13.09 | 55.84 | 156.35 | [Google Drive](https://drive.google.com/file/d/1DRHaoBi7-emo9EQBfbLe9D8Bw8PWYEsv/view?usp=sharing) |



## Testing on Transparent-460
Download the model file 'checkpoints/' and place it in the root directory.

1.Run the test code
~~~python
python3 infer.py
~~~

2.Evaluate the results by the official evaluation python code evaluation.py (provided by [MatteFormer](https://github.com/webtoon/matteformer.git))

Obtain the dataset from [TransMatting](https://github.com/AceCHQ/TransMatting). 

Run the test code. 

Evaluate the results using the evaluation code provided in Transparent-460.

## Acknowledgment
This repo borrows code from several repos, like [GCA](https://github.com/Yaoyi-Li/GCA-Matting) and [MatteFormer](https://github.com/webtoon/matteformer.git)
