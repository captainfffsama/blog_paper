[toc]

# 前言


来自ICLR 2019的一篇关于离群检测的论文,论文本身理论很简单甚至没有什么推导就直接一小段拍出了公式,提出了离群点暴露这个概念.但细想过来,确是很启发的人的一个炼丹小妙招.故而在这里简单记录一下核心内容,很多内容都为小编个人理解,若有不足之处,欢迎在线"掰头".


# 现存问题


之前有很多工作,比如 CVPR 2015 那篇有名的 [Deep Neural Networks are Easily Fooled](https://arxiv.org/abs/1412.1897) 都曾提到过,现有模型都偏向于对未知样本给出高的预测值. 即模型通常不区分分布内和分布外数据,对于分布外数据也常常有极"迷之自信",给出了一个不可靠的预测.


举个栗子,比如我们使用大量猫狗图片训练了一个猫狗二分类的模型供给宠物店应用.但在实际使用时,偏偏店主是个团长的铁粉,一不小心给模型喂了一张团长的照片:


![](https://mmbiz.qpic.cn/mmbiz_jpg/a2UrmovRkiayibbLobEHhRpScwsu5VJbyAF4SComS2icTaejbiaYdb27Flz6I1d9x9SLY7fOIHrORY9QNua6a4B6BQ/0?wx_fmt=jpeg#id=Y5uXv&originHeight=280&originWidth=496&originalType=binary&ratio=1&status=done&style=none)


显然此时团长是完全不在训练集分布中的.但由于我们使用了 softmax 来概率化最后输出,模型极有可能""理直气壮"的说:"有0.9的概率是一只喵,0.1的概率是个汪." 当然,也有可能是其他答案.但是无论答案是什么,显然模型都很难告诉我们:"这既不是喵也不是汪,俺是真的不认识她."


当然,各路大神也想出了各种方法来教会模型说"俺是真的不认识她"这句话.但是比起其他方法,这篇论文给出的炼丹小妙招似乎都更加简单易行,以致在后来其他几篇论文中,我又再看到离群点暴露这个点子.


# 论文解法


回顾之前问题,其实我们不难发现之前问题产生的两大原因:


1. 模型只见过猫狗,没见过其他的.
1. 在实际应用中,模型给出的分类置信度并不足以用来真正的表明输入的答案.



本着"打不过就加入,缺啥补啥"的原则,自然而然想到的解法就是:


1. 正好现在有类似 ImageNet 这类超大型数据集,那我们就带模型去看看那灯红酒绿的花花世界.
1. 既然分类置信度只管分内事,那我们就再给模型添张嘴,让它也说说当前事这算不算分内的事.



以上两点,落实到实际数学公式中,就成了论文中给出的核心公式:

$$
E_{{(x,y)} \backsim D_{in}} [L(f(x),y)+ \lambda E_{x' \backsim D^{OE}_{out}} [L_{OE}(f(x'),f(x),y)]]
$$
以猫狗分类为例,那么这里 $D_{in}$ 就是猫狗数据集,$D^{OE}_{out}$ 就是 ImageNet,我们要带模型去看的花花世界.$L(f(x),y)$ 就是之前模型的学习目标,比如最小化输出概率和 one-hot 之间的交叉熵.


值得注意的是花花世界通常是没有现成标签答案的,而且公式后一项是一个辅助模型训练的正则项,不用那么呵责,因此公式中后一项中的 $y$ 我们忽略就好.搁在猫狗分类中,这一项就是 ImageNet 图片的概率输出和原始训练集类别分布的交叉熵.


说到这,鸡贼的小伙伴肯定会说:"就这? 花花世界看过了,那模型添的那管分不分内事张嘴在哪?"


这点在论文中,也给了个简单粗暴的做法,后面那个正则项,就是在实际推理阶段用来衡量是不是分布内样本的分数.


至此,整篇文章的思想我们就撸通的差不多了.至于正则项里例子用了交叉熵,那我换成 JS 散度行不行?换成 W 距离行不行?


显然都行,公式只是个指导思想,里面的核心,想怎么换就怎么换.实际上论文后续实验,在不同的任务中,正则项的具体公式都不一样.


# 代码实现


其实看完以上,很多小伙伴应该都知道猫狗分类里面怎么用上这招.但是看完 issue 之后,我发现代码虽然简单,但还是要水一下.原始论文在分类实验中训练时实现代码是这样的:


```python
# forward
x = net(data)

# backward
scheduler.step()
optimizer.zero_grad()

loss = F.cross_entropy(x[:len(in_set[0])], target)
# cross-entropy from softmax distribution to uniform distribution,lambda is 0.5
loss += 0.5 * -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()
```


这时候很多看完的小伙伴肯定说:"小编你贴错代码了吧?"


确实,这里 `-(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1))` 怎么看都不太像交叉熵的公式:


$$
H=- \sum_x p(x) log q(x)
$$


但实际上,这里只是作者做了一步数学上的简化.


我们知道 $p(x)$ 是训练集分布,那么训练集文章假设是类别平衡的,是个均匀分布. $q(x)$ 是花花世界样本 softmax 概率化之后的预测分布,那么公式改写过程如下:

$$
\begin{align}
H&=- \sum_N \frac{1}{N} log \frac{e^{x_j}}{\sum^N_{i=1}e^{x_i}}  \\
&= -\frac{\sum_N log \frac{e^{x_j}}{\sum^N_{i=1}e^{x_i}} }{N}  \\
&= -\frac{\sum_N (log e^{x_j}-log{\sum^N_{i=1}e^{x_i}} )}{N}  \\
&= -( \frac{\sum^N x_j}{N}-\frac{\sum^Nlog{\sum^N_{i=1}e^{x_i}}}{N})  \\
&= -(E(x)-log{\sum^N_{i=1}e^{x_i}})
\end{align}
$$


显然,前一项就是 `len(in_set[0]):].mean(1)`,后一项就是 `torch.logsumexp(x[len(in_set[0]):], dim=1)`.


至此,全文奥义我们基本吃透.那么各位小伙伴,你们学费了吗?


# 相关资料


- 代码:[https://github.com/hendrycks/outlier-exposure](https://github.com/hendrycks/outlier-exposure)
- 文章:[https://arxiv.org/abs/1812.04606](https://arxiv.org/abs/1812.04606)



