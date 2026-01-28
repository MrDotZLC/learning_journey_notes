## 1. softmax 定义（向量形式）
$$\alpha_i = \frac{e^{x_i}}{\sum_k e^{x_k}}$$
## 2. Jacobian（必须记住）
$$\frac{\partial \alpha_i}{\partial x_j} = \alpha_i(\delta_{ij} - \alpha_j),\quad \delta_{ij} = \begin{cases} 1 & i=j \\ 0 & i\neq j \end{cases}$$
- i：输出索引
- j：输入索引
## 3. attention 输出回顾
$$O = \sum_j \alpha_j V_j$$
## 4. 反向目标
求：
$$\frac{\partial L}{\partial x_i}$$
## 5. 链式法则展开
$$\frac{\partial L}{\partial x_i} = \sum_j \frac{\partial L}{\partial \alpha_j} \frac{\partial \alpha_j}{\partial x_i}$$
代入 Jacobian：
$$= \sum_j \frac{\partial L}{\partial \alpha_j} \alpha_j(\delta_{ij} - \alpha_i)$$
## 6. 利用 δ 拆分
$$= \alpha_i \frac{\partial L}{\partial \alpha_i} - \alpha_i \sum_j \alpha_j \frac{\partial L}{\partial \alpha_j}$$
## 7. 用 attention 结构代换
$$\frac{\partial L}{\partial \alpha_i} = \left(\frac{\partial L}{\partial O}\right)^\top V_i$$
$$\sum_j \alpha_j \frac{\partial L}{\partial \alpha_j} = \left(\frac{\partial L}{\partial O}\right)^\top O$$
## 8. 最终 softmax backward（attention 场景）
$$\boxed{ \frac{\partial L}{\partial x_i} = \alpha_i \left(\frac{\partial L}{\partial O}\right)^\top (V_i - O) }$$
