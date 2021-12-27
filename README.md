# 联邦学习实战 (Practicing-Federated-Learning)

 [![Awesome](https://img.shields.io/badge/Awesome-Federated%20Learning-blue)](https://github.com/innovation-cat/Awesome-Federated-Machine-Learning) [![Book](https://img.shields.io/badge/Book-Purchase-brightgreen)](https://item.jd.com/13206070.html)    



<font size=4>联邦学习是一种新型的、基于数据隐私保护技术实现的分布式计算范式，自提出以来，就受到学术界和工业界的广泛关注。近年来，随着联邦学习的飞速发展，使得其成为解决数据孤岛和用户隐私问题的首选方案，但当前市面上这方面的实战书籍却尚不多见。本书是第一本权威的联邦学习实战书籍，结合联邦学习案例，有助于读者更深入的理解联邦学习这一新兴的学科。</font>

<font size=4>**本项目将长期维护和更新《联邦学习实战》书籍对应的章节代码。书或代码中遇到的问题，可以邮件联系：huanganbu@gmail.com**</font>

由于受到出版刊物限制，本书不能在纸质书页面上放置网络链接，本书的链接对应可查看这里（[书中链接](figures/link.md)）。

书中可能存在印刷或撰写的错误，勘误列表读者可点击：[勘误列表](errata/README.md)，同时也欢迎读者反馈书中的文字错误问题，以便我们改进。

&nbsp;

## 联邦学习材料

- [联邦学习最新研究论文、书籍、代码、视频等详细资料汇总 (Everything about Federated Learning)](https://github.com/innovation-cat/Awesome-Federated-Machine-Learning)

- [香港科技大学“联邦学习”课程](https://ising.cse.ust.hk/fl/index.html)

&nbsp;

## 简  介

<font size=4>本书是联邦学习系列书籍的第二本，共由五大部分共19章构成。既有理论知识的系统性总结，又有详细的案例分析，本书的组织结构如下：</font>

- <font size=4>第一部分简要回顾联邦学习的理论知识点，包括联邦学习的分类、定义；联邦学习常见的安全机制等。</font>

- <font size=4>第二部分介绍如何使用Python和FATE进行简单的联邦学习建模；同时，我们也对当前一些常见的联邦学习平台进行总结。</font>

- <font size=4>第三部分是联邦学习的案例分析，本部分我们挑选了包括视觉、个性化推荐、金融保险、攻防、医疗等领域的案例，我们将探讨联邦学习如何在这些领域中进行应用和落地。</font>

- <font size=4>第四部分主要介绍和联邦学习相关的高级知识点，包括联邦学习的架构和训练的加速方法；联邦学习与区块链、split learning、边缘计算的异同。</font>

- <font size=4>第五部分是对本书的总结，以及对联邦学习技术的未来展望。</font>



<font size=4>本书可以与《联邦学习》书籍配套使用，书的链接地址：</font>

- <font size=4>联邦学习实战：[https://item.jd.com/13206070.html](https://item.jd.com/13206070.html)</font>

- <font size=4>联邦学习：[https://item.jd.com/12649191.html](https://item.jd.com/12649191.html)</font>



注：本项目含有部分数学公式，为了不影响阅读效果，在Github网页中能正常显示公式，需要读者首先下载相应的浏览器插件，如MathJax插件（[MathJax Plugin for Github](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima)）

&nbsp;

## 推荐语

本书的编写也得到了来自学术界、金融投资界和工业界的知名人士推荐，在此深表感谢：

> <p>"国务院在2020年将数据作为新型生产要素写入法律文件中，与土地、劳动力、资本、技术并列为五个生产要素。这意味着一方面，个人数据隐私将受到法律的严格保护；另一方面，数据与其它生产要素一样，可以进行开放、共享和交易。如何有效解决数据隐私与数据共享之间的矛盾成为了当前人工智能领域的研究热点问题。联邦学习作为一种新型的分布式机器学习训练和应用范式，从提出以来就备受广泛的关注，也被认为是当前产业界解决数据隐私与数据共享之间矛盾的一种有效方案。同时作为可信计算的新成员，书中特别提到，联邦学习还可以与区块链进行强强联合，例如借助区块链记录的不可篡改特性，帮助对联邦学习可能面临的恶意攻击进行追溯；借助区块链的共识机制和智能合约，对联邦学习创造的价值进行利益分配等。《联邦学习实战》一书，对联邦学习的理论和应用案例做了系统性的阐述和分析，相信能够为广大的科研工作者、企业开发人员提供有效的指导和参考。"</p>
> <b>&mdash; 陈  纯，中国工程院院士</b>

&nbsp;

> <p>"人工智能时代的到来已经不可逆转，在算力算法和机器学习蓬勃发展的大背景下，数据资产成为了非常重要的技术资源，服务企业的长远发展。如何利用好、保护好数据资产是人工智能能否创造出更大经济和社会价值的关键因子。联邦学习理念的提出和发展，在这个方面探索出一条可行之路并做出了重要的示范。杨强教授是世界级人工智能研究专家，在学术和产业两端都对这一领域有非常深的造诣，希望他这本《联邦学习实战》可以为更多业内人士和机器学习的从业者与爱好者带来更多的启发与思考。"</p>
>
> <b>&mdash;  沈南鹏，红杉资本全球执行合伙人</b>

&nbsp;

> <p>"数据资产化是实现人工智能产业价值的核心环节，而联邦学习是其中的关键技术。书中严谨而深入浅出阐述为读者们提供了非常有效的工具。"</p>
>
> <b>&mdash; 陆  奇，奇绩创坛创始人</b>

&nbsp;

> <p>"为了互联网更好的未来，我们需要建立负责任的数据经济体系，使我们既能充分实现数据价值，又能很好的保护用户的数据隐私，并能够公平分配数据创造的价值。联邦学习正是支撑这一愿景的重要技术。本书描述了该领域的实际应用案例，为将联邦学习付诸实践提供了重要的指导意义。"</p>
>
> <b>&mdash; Dawn Song，美国加州大学伯克利分校教授</b>


&nbsp;

## 代码章节目录

 * [第3章：用Python从零实现横向联邦图像分类](chapter03_Python_image_classification)
 * [第5章：用FATE从零实现横向逻辑回归](chapter05_FATE_HFL)
 * [第6章：用FATE从零实现纵向线性回归](chapter06_FATE_VFL)
 * [第9章：个性化推荐案例实战](chapter09_Recommendation)
 * [第10章：视觉案例实战](chapter10_Computer_Vision)
 * [第15章：联邦学习攻防实战](chapter15_Attack_and_Defense)
 * [第15章：攻防实战 - 后门攻击](chapter15_Backdoor_Attack)
 * [第15章：攻防实战 - 差分隐私](chapter15_Differential_Privacy)
 * [第15章：攻防实战 - 同态加密](chapter15_Homomorphic_Encryption)
 * [第15章：攻防实战 - 稀疏化](chapter15_Sparsity)
 * [第15章：攻防实战 - 模型压缩](chapter15_Compression) 

