# DeepLearning
**relearn**


## Question & Answer 
**1. Pytorch构建网络模型时super(__class__, self).__init__()的作用**
<details>
<summary> </summary>

- 在类的继承时, super()方法并不是必须的。但是在涉及自动运行的魔术方法时, 必须使用该语句, 这时就必须在B类中使用super()方法, 注明B类要继承A类中的__init__()方法或其他方法。
- super()方法中, 括号内的内容是可以不用写的。可以直接写super().__init__()或者super(__class__, self).__init__()。
- torch.nn.Module.__init__()的作用是初始化内部模型状态(Initializes internal Module state)。具体地, 就是初始化training, parameters..._modules这些在Pytorch中内部使用的属性。所以，在Pytorch框架下，所有的神经元网络模型子类，都必须要继承这些内部属性的初始化过程。

参考
- https://developer.aliyun.com/article/1467527
</details>

**2. nn.BCEWithLogitsLoss()与nn.BCELoss()**
<details>
<summary> </summary>
二元交叉熵损失(Binary Cross Entropy Loss)

- 

参考
- https://blog.csdn.net/qq_22210253/article/details/85222093
</details>

**3. sigmoid与softmax**
<details>
<summary> </summary>
- Sigmoid是逐元素的二元激活函数, 将任意实数映射到(0,1)区间, 适合独立的二分类或多标签任务; Softmax是逐向量的归一化函数, 将一个向量映射为概率分布(和为 1), 适合互斥的多分类任务。
!()


</details>


**4. sigmoid与softmax**
![](https://github.com/weiwei4396/DeepLearning/blob/main/picture/sigmoid_softmax.jpg)













