# 更快,更高,更强---YOLO 10 速看


- 代码: [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- 论文: [https://arxiv.org/abs/2107.08430](https://arxiv.org/abs/2107.08430)



## 碎碎念的前言


7月20日旷世挂到 arxiv 的一篇文章,标题很唬人,标题下不等摘要就急不可待的挂了一张比较炸裂的效果图.


![yolo3.png](https://cdn.nlark.com/yuque/0/2021/png/2433458/1626962502499-16506486-5959-4040-9e1e-68e5595fb989.png#clientId=u94003346-7279-4&from=ui&id=ube0b10df&margin=%5Bobject%20Object%5D&name=yolo3.png&originHeight=808&originWidth=1350&originalType=binary&ratio=1&size=204501&status=done&style=none&taskId=ufa9dcbc0-71e7-458b-a724-c26c6d8134c)


当时看到时,冲着这 YOLO 的名头,心想肯定是爆点,于是21号小编就火急火燎的速刷完了整篇论文,准备给各位读者老爷带来一波速看解析,但不巧被突如其来的疫情耽搁,索性再鸽一天,又把代码囫囵吞枣式的扒拉了一遍.


既然东西看了,那是必须要拉出来给各位读者老爷来个误人子弟.下面,小编就把自己的所见所得简单汇总一下,若是有不妥之处,欢迎大佬在线留言掰头.


## 总体评价


首先论文方面,整体看下来,干货十足,文章不长,藏金不少,各种改造思路,训练小妙招堆起来简直就是大型调参教科书,工程性非常强.上一次让人有这种感觉的,大概就是那篇 YOLOv4 的论文了.但是这篇文章直接起名叫 YOLOX 了,十分霸气测漏,而且文章少有的将效果图直接摆在了摘要上面,简直就差在文章里写:"我要打十个!!!"


![beat10.jpeg](https://cdn.nlark.com/yuque/0/2021/jpeg/2433458/1626962523100-81f203a5-9669-4ffe-b7de-77b32ae361cf.jpeg#clientId=u94003346-7279-4&from=ui&id=u4934f7f4&margin=%5Bobject%20Object%5D&name=beat10.jpeg&originHeight=225&originWidth=225&originalType=binary&ratio=1&size=7114&status=done&style=none&taskId=u1be18f1e-ea0d-42f8-b53b-f48f9c5714b)


代码方面这次旷世也是良心十足了,不仅 python 代码架构清晰,还直接给安排了现成的 TensorRT,Openvino,ncnn 部署示例,GPU,CPU,移动端直接一步到位,让我等伸手党简直激动感动得要叫爸爸.


那吹也吹完了,接下来,咱们就来真正梳理下 YOLOX 这次都有哪些闪光之处.当然, 由于涉及的要素过多,还有一些上游论文,这一次咱们就简单说说了.等后面有空,小编再出一期真想教会你系列的详解,咱们细细的聊一聊具体代码实现和各个细节的原理.


## 亮点一:投靠 anchor-free


anchor-base 自从被 YOLOv2 看上后,直到 YOLOX 之前,都未曾被抛弃过,v4,v5版本更是将其发挥到了一个极致,毕竟能有效提召回,减小回归难度,有谁不爱呢.但是 anchor 的引入本身也会带来很多问题,比如正负样本不平衡,额外超参,网络结构复杂计算量大,anchor 本身超参不具普适性等各种不优雅问题.再加上近年来 fcos,center net 等工作开启了新世界的大门,因此作者索性这次也将 anchor 抛开,从 YOLOv3 版本开始大刀阔斧的改造,借鉴 fcos 的一些工作,将 YOLOv3 直接改造成了 anchor-free.


做法上就是原来 v3 在每个层特征图点上预测了3个 anchor 框,然后回归实际目标和 anchor 之间的误差,这里直接改成预测1一个目标框,直接预测目标框相对网格的左上角偏置以及目标框长宽.这一块的基础工作和 fcos 如出一辙.


## 亮点二:更好的标签分配策略 SimOTA


这一波算是旷世宣传一下本家工作并结合工程进行了简化.我们知道过去标签分配的常见做法是根据 IoU 阈值来分配哪些是正哪些是负,但是这种做法是相对简单,没从全局角度去考虑问题的.那涉及到这种分配配对问题,1对1咱们可以匈牙利算法直接上,那1对多,我们就可以考虑最优传输问题了.而 OTA 就是将标签分配问题视为最优传输问题,连优化算法Sinkhorn-Knopp都一口气直接套用了.而Sinkhorn-Knopp 算法本身会增加额外 25% 左右的训练时间.因此旷世找了一个动态 top-k 的方法来替代 Sinkhorn-Knopp 算法近似求解.具体的做法和实现,等小编刷完全部代码再来谈谈吧.


## 亮点三:解耦分类,回归头


其实这个方法在其他检测方法中,早就普及开来了,但是YOLO系列却一直是一个头将定位分类一把梭.具体不多说了,看图就懂.


![yolox2.png](https://cdn.nlark.com/yuque/0/2021/png/2433458/1626962537656-822dcf4a-61b3-48cd-b95f-41931b65ac23.png#clientId=u94003346-7279-4&from=ui&id=u59edfec1&margin=%5Bobject%20Object%5D&name=yolox2.png&originHeight=743&originWidth=1333&originalType=binary&ratio=1&size=161106&status=done&style=none&taskId=u7e292f70-76df-4cd8-94e6-7fdd502ef12)


## 亮点四:Mosaic,MixUp一起上


这其实也没啥好说的,优良传统必须保持.值得注意的是,最后 15 轮作者把这两策略关闭了,但是文中没提到的是,关闭的同时,作者在 Loss 上偷偷加了一个 L1 正则.个人就这点问过作者,作者表示无他尔,就是这是一个能涨0.1~0.2% AP 的经验.另外文中反映这种增强之后, ImageNet 的预训练模型似乎拉了后腿,所以作者从头训练了整个网络模型.


## 其余细节


最后,那还剩下的一些有价值的细节就是:


1. 作者其实还做了一个端到端实验,企图把 NMS 也干掉,但是实际发现性能掉点,就从最终模型去掉了.
2. 在实际权重更新时,使用了指数移动平均数的方法来更新权重,保证权重变化的平滑.
3. 用了多尺度训练,为此作者把 dataloader 又包了一层,每次迭代都会改变输入图片大小来提升性能.

其实再看代码的时候,还发现了诸如网络偏置初始化专门做了处理等等细节,这些等小编看完全部代码吃透再回过头讲讲吧.
## 部署踩坑


最后再罗嗦下实际部署,按照官方教程完全没啥问题,唯一需要注意的是,pip 安装的 torch 版本若和本机的 cuda 版本不一致,会导致 apex 编不过去,把 torch 版本换对了就行.
