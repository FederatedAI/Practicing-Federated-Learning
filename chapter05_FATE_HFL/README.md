# 第5章：用FATE从零实现横向逻辑回归

在第三章我们介绍了如何使用Python构建简单的横向联邦学习模型。但应该注意到，联邦学习的开发，特别是工业级的产品开发，涉及的工程量却远不止于此，一个功能完备的联邦学习框架的设计细节是相当复杂的，幸运的是，随着联邦学习的发展，当前市面上已经出现了越来越多的开源平台。

本章我们将介绍使用FATE从零开始构建一个简单的横向逻辑回归模型，经过本章的学习，读者能够了解利用FATE进行横向建模的基本流程。鉴于本书的篇幅有限，以及本书的写作目的，我们不对FATE的具体原理实现进行详细的讲解。



**注：由于FATE平台在不断迭代和更新的过程中，本章的内容撰写截稿时间较早（2020年9月截稿），可能会因版本变化而导致配置方式和运行方式发生改变，因此，强烈建议读者如果想进一步了解FATE的最新使用和安装，可以直接参考FATE官方文档教程：**

* [FATE的安装部署](https://github.com/FederatedAI/DOC-CHN/tree/master/%E9%83%A8%E7%BD%B2)
* [FATE的官方文档](https://github.com/FederatedAI/DOC-CHN)

**如果FATE的安装和使用遇到任何问题，可以添加FATE小助手，有专门的工程团队人员帮忙解决。**

<div align=center>
<img width="300" src="figures/FATE_logo.jpg" alt="FATE小助手"/>
</div>

## 5.1 实验准备

在开始本章实验之前，请读者确保已经安装[Python](https://www.anaconda.com/products/individual)和[FATE单机版](https://github.com/FederatedAI/DOC-CHN/blob/master/%E9%83%A8%E7%BD%B2/FATE%E5%8D%95%E6%9C%BA%E9%83%A8%E7%BD%B2%E6%8C%87%E5%8D%97.rst)。

**注意：本书编写时，FATE主要支持的是使用dsl和conf配置文件来构建联邦学习模型，而在之后的最新版本中，FATE也进行了很多方面的改进，特别是最新引入了pipeline的建模方式，更加方便，有关FATE pipeline的训练流程，读者可以参考文档：[FATE-Pipeline](https://github.com/FederatedAI/FATE/tree/master/examples/pipeline)。**



## 5.2 数据集获取

本章我们使用威斯康星州临床科学中心开源的乳腺癌肿瘤数据集来测试横向联邦模型，数据集可以从[Kaggle](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)网站中下载，也可以直接使用sklearn库的内置数据集获取：

```
from sklearn.datasets import load_breast_cancer
import pandas as pd 


breast_dataset = load_breast_cancer()

breast = pd.DataFrame(breast_dataset.data, columns=breast_dataset.feature_names)

breast['y'] = breast_dataset.target

breast.head()
```



## 5.3 横向联邦数据集切分

为了模拟横向联邦建模的场景，我们首先在本地将乳腺癌数据集切分为特征相同的横向联邦形式，当前的breast数据集有569条样本，我们将前面的469条作为训练样本，后面的100条作为评估测试样本。

* 从469条训练样本中，选取前200条作为公司A的本地数据，保存为breast\_1\_train.csv，将剩余的269条数据作为公司B的本地数据，保存为breast\_2\_train.csv。
* 测试数据集可以不需要切分，两个参与方使用相同的一份测试数据即可，文件命名为breast\_eval.csv。

数据集切分代码请查看：[split_dataset.py](split_dataset.py)



## 5.4 利用FATE构建横向联邦学习Pipeline

用FATE构建横向联邦学习Pipeline，涉及到三个方面的工作：

* 数据转换输入
* 模型训练
* 模型评估：（可选）

为了方便后面的叙述统一，我们假设读者安装的FATE单机版本目录为：

```
fate_dir=/data/projects/fate-1.3.0-experiment/standalone-fate-master-1.4.0/
```



### 5.4.1 数据转换输入

该步骤是将5.3中切分的本地数据集文件转换为FATE的文件格式DTable，DTable是一个分布式的数据集合：

<div align=center>
<img width="500" src="./figures/local_2_dtable.png" alt="数据格式转换"/>
</div>

如书中所述，将数据格式转换为DTable，需要执行下面几个步骤即可：

* 将本地的切分数据上传到$fate_dir/examples/data目录下
* 进入$fate_dir/examples/federatedml-1.x-examples目录，打开upload_data.json文件进行更新, 以breast_1_train.csv文件为例：

```
{
  "file": "examples/data/breast_1_train.csv",
  "head": 1,
  "partition": 10,
  "work_mode": 0,
  "table_name": "homo_breast_1_train",
  "namespace": "homo_host_breast_train"
}
```

有关上面数据上传各字段的解析，可以参考FATE的Github官方文档：

https://github.com/FederatedAI/DOC-CHN/blob/master/Federatedml/%E6%95%B0%E6%8D%AE%E4%B8%8A%E4%BC%A0%E8%AE%BE%E7%BD%AE%E6%8C%87%E5%8D%97.rst

最后在当前目录下（$fate_dir/examples/federatedml-1.x-examples），在命令行中执行下面的命令，即可自动完成上传和格式转换：

```
python $fate_dir/fate_flow/fate_flow_client.py -f upload -c upload_data.json
```



### 5.4.2 模型训练

借助FATE，我们可以使用组件的方式来构建联邦学习，而不需要用户从新开始编码，FATE构建联邦学习Pipeline是通过自定义dsl和conf两个配置文件来实现：

* dsl文件：用来描述任务模块，将任务模块以有向无环图（DAG）的形式组合在一起。

* conf文件：设置各个组件的参数，比如输入模块的数据表名；算法模块的学习率、batch大小、迭代次数等。

有关dsl和conf文件的设置，读者可以参考FATE的官方文档：

https://github.com/FederatedAI/DOC-CHN/blob/master/Federatedml/%E8%BF%90%E8%A1%8C%E9%85%8D%E7%BD%AE%E8%AE%BE%E7%BD%AE%E6%8C%87%E5%8D%97.rst

本案例中，读者可以直接使用本文件夹提供的配置文件：

* [test_homolr_train_job_conf.json](https://github.com/FederatedAI/Practicing-Federated-Learning/blob/main/chapter05_FATE_HFL/test_homolr_train_job_conf.json)
* [test_homolr_train_job_dsl.json](https://github.com/FederatedAI/Practicing-Federated-Learning/blob/main/chapter05_FATE_HFL/test_homolr_train_job_dsl.json)



将dsl和conf文件放置在任意目录下，并在该目录下执行下面的命令进行训练：

```
python $fate_dir/fate_flow/fate_flow_client.py -f submit_job -d test_homolr_train_job_dsl.json -c test_homolr_train_job_conf.json
```

