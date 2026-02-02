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

- nn.BCELoss要求输入是已经经过 Sigmoid 激活的概率值(0~1 之间);
- nn.BCEWithLogitsLoss直接接受未激活的logits(Logits是神经网络最后一层输出的原始、未归一化的分数(raw scores), 即模型线性变换后的直接结果, 没有经过Sigmoid或Softmax等激活函数处理), 内部自动完成Sigmoid + BCE 计算, 数值更稳定, 是实际项目中的首选。 
https://github.com/weiwei4396/DeepLearning/blob/main/picture/BCELoss_BCEWithLogitsLoss.jpg

参考
- https://blog.csdn.net/qq_22210253/article/details/85222093
</details>

**3. Sigmoid与Softmax**
<details>
<summary> </summary>
- Sigmoid是逐元素的二元激活函数, 将任意实数映射到(0,1)区间, 适合独立的二分类或多标签任务; Softmax是逐向量的归一化函数, 将一个向量映射为概率分布(和为1), 适合互斥的多分类任务。
https://github.com/weiwei4396/DeepLearning/blob/main/picture/sigmoid_softmax.jpg
</details>

**4. 神经网络训练的基本步骤**
<details>
<summary> </summary>

- A.将数据分解为训练集和测试集; 验证集, 用于调参和早停;
- B.定义网络结构, 初始化模型部件, forward函数定义模型结构;
- C.定义超参数和工具, epoch, batch size, 学习率, 优化器, 损失函数;
- D.模型训练(model.train()), 循环epoch, 循环minibatch, 前向传播(模型预测, 计算损失), 反向传播(清零梯度准备计算, 反向传播, 更新训练参数)
</details>

```python
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.train() 
for epoch in range(num_epochs):
    running_loss = 0.0
    # 前向传播
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    # 反向传播 + 优化
    optimizer.zero_grad() # 清零梯度准备计算
    loss.backward() # 反向传播
    optimizer.step() # 更新训练参数
    running_loss = loss.item()
```


**5. Batch Normalization和 Layer Normalization**
<details>
<summary> </summary>



</details>













