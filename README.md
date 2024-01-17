# 多模态情感分析

使用bert预训练模型对文本进行特征抽取，resnet152对图像进行特征抽取，使用不同的特征融合方法共构建四个模型

## 项目结构

```
|── Multimodal-Sentiment-Analysis  #项目根目录
│   ├── label                      #存放标签的目录
│   │   ├── test_without_label.txt #无标签的测试集
│   │   └── train.txt              #有标签的训练集
│   ├── main.py                    #主程序
│   ├── model                      #存放所有模型的目录
│   │   ├── AddModel.py            #特征相加进行融合的模型
│   │   ├── AttentionAddModel.py   #使用attention机制加特征融合的模型
│   │   ├── AttentionCatModel.py   #使用attention机制加特征concat的模型
│   │   └── CatModel.py            #特征concat的模型
│   ├── output                     #输出目录
│   │   ├── add_model_test_with_label.txt            #AddModel的预测结果
│   │   ├── attention_add_model_test_with_label.txt  #AttentionAddModel的预测结果
│   │   ├── attention_cat_model_test_with_label.txt  #AttentionCatModel的预测结果
│   │   ├── cat_model_test_with_label.txt            #CatModel的预测结果
│   │   └── result.txt                               #四个模型进行硬投票得到的最终结果
│   ├── pre_trained                #预训练模型的存放目录
│   │   ├── bert-base-uncased      #bert预训练模型
│   │   └── resnet-152             #resnet152预训练模型       
│   ├── project_result.ipynb       #保存了所有输出结果的notebook文件
│   ├── README.md                  #README
|   ├── requirements.txt           #项目依赖
│   └── util                       #存放工具类的目录
│       └── data_preprocess.py     #数据预处理的处理器类
```

注：data目录和pre_trained目录由于太大，没有传到github上，如果需要运行main.py，请将实验5数据集解压到该项目目录下，并将microsoft/resnet152和bert-base-uncased预训练模型下载好放在pre_trained目录中

## Requirements

```sh
numpy==1.24.4
pandas==2.0.3
Pillow==10.2.0
scikit_learn==1.3.2
torch==1.10.0+cu102
transformers==4.18.0
```

```sh
pip install -r requirements.txt
```

## 训练与测试

```sh
usage: main.py [-h] [--text_only] [--image_only] [--do_test] [--lr LR] [--weight_decay WEIGHT_DECAY] [--epochs EPOCHS] [--model MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --text_only           only use text to train model, default false
  --image_only          only use image to train model, default false
  --do_test             use trained model to predict test dataset, default false
  --lr LR               set the learning rate, default 3e-5
  --weight_decay WEIGHT_DECAY
                        set weight decay, default 1e-5
  --epochs EPOCHS       set train epochs, default 5
  --model MODEL         set the type of model, default AttentionCatModel
```

可选参数列表如下：

1. text_only：仅使用text进行训练，默认为False
2. image_only：仅使用image进行训练，默认为False
3. do_test：使用训练得到的.pth模型文件进行预测，默认为False
4. lr：学习率，默认为3e-5
5. weight_decay：权重衰减，默认为1e-5
6. epochs：迭代轮数，默认5轮
7. model：使用的模型，可选的模型包括：AddModel CatModel AttentionAddModel AttentionCatModel，默认AttentionCatModel

以下是命令示例：

```sh
python main.py --lr 1e-5 --weight_decay 1e-4 --epochs 10 --model CatModel
python main.py --text_only --model AttentionCatModel
python main.py --do_test --model AttentionAddModel
```

训练会比较耗时，epoch为5的话 我在云主机上跑了大概20分钟

测试完成后会在output文件夹中生成对应文件，如果是用AttentionCatModel测试的话，运行结果文件名为AttentionCatModel_test_with_label.txt，其他model类似

## 实验结果

| 模型              | 平均正确率 | 最高正确率 |
| ----------------- | ---------- | ---------- |
| CatModel          | 73.95      | 75.5       |
| AddModel          | 74.45      | 76.25      |
| AttentionCatModel | 75.15      | 75.5       |
| AttentionAddModel | 73.25      | 75.75      |

## 消融实验

| 单模态                       | 平均正确率 | 最高正确率 |
| ---------------------------- | ---------- | ---------- |
| Text Only（AttentionModel）  | 69.4       | 71.00      |
| Image Only（AttentionModel） | 59.75      | 62.25      |
| Text Only（NaiveModel）      | 73.5       | 74.75      |
| Image Only（NaiveModel）     | 64.75      | 66.25      |

## 参考

vista-net: https://github.com/PreferredAI/vista-net

多模态情感分析VistaNet：https://zhuanlan.zhihu.com/p/345713441?utm_id=0