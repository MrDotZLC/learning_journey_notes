# 0. 先验知识
## 0.1. pytorch中的维度与llama.cpp中的内存跨步对比
![[Pasted image 20260110030940.png]]
## 0.2. Token输入
输入：Hello my name is
token数量：5（Hello前默认有一个起始符 \<s\>）
## 0.3. 广播机制
在对不同形状（shape）的张量进行逐元素运算时，框架在不显式复制数据的前提下，**自动扩展维度较小的张量，使其在逻辑上与较大张量对齐**。
### 以 PyTorch / NumPy 为例）
假设对两个张量 `A` 和 `B` 进行逐元素运算（如加、减、乘、除）：
1. 从**最后一个维度开始对齐**
2. 对齐的维度满足以下条件之一即可广播：
    - 两个维度相等
    - 其中一个维度为 1
3. 若某个维度不满足上述条件 → **无法广播，直接报错**
4. 缺失的高维度视为 1
### 示例
`A: (batch, seq_len, hidden) B: (hidden)`
逻辑上等价于：
`B → (1, 1, hidden) → (batch, seq_len, hidden)`
### 注意：LLaMA.cpp中不能运行时广播，因为内存都是提前分配好的。LLaMA.cpp解决唯独不匹配的
1. graph构建前，就计算好tensor的维度，提前消除维度不一致
2. 在kernel中


## 0.4. 维度折叠（LLaMA）
将多个逻辑维度合并为一个物理维度，或在计算前后进行 reshape/view，使计算在更低维或更规则的张量形态上完成。
典型形式：
`(B, H, L, D) → (B·H, L, D)`
`(B, L, H, D) → (B·L, H·D)`
前提条件：
- 折叠前后的 **元素总数不变**
- 折叠的维度在语义上 **彼此独立或可等价处理**
### 从系统与硬件角度看，原因非常明确：
#### 1. 符合 BLAS / GEMM 接口
- GEMM 本质是 **二维矩阵运算**
- 高维张量需要先 reshape 才能高效调用
#### 2. 提升 GPU 利用率
- 连续内存访问
- 更好的 warp / tile 映射
- 减少 stride 带来的 cache miss
#### 3. 简化算子与代码路径
- 少写多维循环
- 复用成熟 kernel
- 降低实现复杂度与维护成本
#### 4. 与广播机制配合使用
- 折叠维度 → 广播参数
- 广播 mask → 折叠 batch/head

二者是 LLM 张量工程的“左右手”。
### 注意事项：
1. 数据一定要内存连续，非连续数据需要 contiguous()
2. 维度顺序不能乱
3. 折叠逻辑要与其他变量的维度对齐，如KV cache

# 1. GET_ROWS
根据索引从weight矩阵中取值拼接。
GET_ROWS算子是CPU后端进行计算。
采用cuda并行计算方式，每个线程处理多个数据。
核心代码：等价于dst\[i\] = src0\[src1\[i\]\]
```
static void ggml_compute_forward_get_rows_f16(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0]; // 源tensor(2048,92544),词表长度*单词特征维度
    const struct ggml_tensor * src1 = dst->src[1]; // 索引tensor(5,1)，输入的5个 token

    GGML_TENSOR_BINARY_OP_LOCALS // 宏定义，展开后是将src和dst中的ne,nb赋值给局部变量，方便使用，见下图

    const int64_t nc = ne00; // 每一行的元素个数，即2048
    const int64_t nr = ggml_nelements(src1); // 总行数，即token个数

    assert(ne0  == nc);
    assert(ne02 == ne11);
    assert(nb00 == sizeof(ggml_fp16_t));
    assert(ggml_nrows(dst) == nr);

    const int ith = params->ith; // 第 ith 个线程，类似cuda中的threadIdx
    const int nth = params->nth; // 一共 nth 个线程，类似cuda的blockDim

    // rows per thread
    const int dr = (nr + nth - 1)/nth; // 每个线程处理的行数

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int64_t i = ir0; i < ir1; ++i) { // 遍历负责的行，即dst的逻辑索引，维度[i10, i11, i12, 1]，跨步[nb10,nb11,nb12,0]=[4, 20, 20, 20]
        const int64_t i12 = i/(ne11*ne10); // 第 2 维索引
        const int64_t i11 = (i - i12*ne11*ne10)/ne10; // 第 1 维索引
        const int64_t i10 = (i - i12*ne11*ne10 - i11*ne10); // 第 0 维索引
        const int64_t i01 = *(int32_t *) ((char *) src1->data + i10*nb10 + i11*nb11 + i12*nb12); // 索引张量中的索引数据，即src0 中的第 i01 行 

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_fp16_to_fp32_row( // 复制数据
                // i01是token_id, 乘第0维步长，token_id只和第0位有关，其他维是并行维保持一致，才能保证src0和dst属于同一子空间内。
                (const void *) ((char *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
	            // nb00是一个元素的字节数，i10是0维索引，所以要乘一个第0维度的总字节数 nb01，同理i11乘一个第1维度的总字节数    
			    (float *) ((char *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}
```
![[Pasted image 20260110033758.png]]

# 2. RMS_NORM
[[RMS_norm介绍]]均方根归一化。
![[Pasted image 20260111025330.png]]
对2048进行mean，得到[1,5,1]，再对源输入进行广播。 
代码解析：
![[Pasted image 20260111031403.png]]
block形状[1024,1,1]，一个 block有1024个线程和32个warp。数据形状为5 * 2048，即5个block，每个block处理一行，每个线程处理2个数据。 
![[Pasted image 20260111033734.png]]
```
template <int block_size>
static __global__ void rms_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;  // 行数
    const int tid = threadIdx.x;                          // 线程ID

    float tmp = 0.0f; // partial sum for thread in warp

	// 将一行数规约成1024个数
    for (int col = tid; col < ncols; col += block_size) { // 两个数的平方和
        const float xi = x[row*ncols + col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp); // warp内使用shuffle进行规约求和，求得32个数
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32]; // 共享内存，存32个warp里的第0个数
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) { // 是否是处理第0个数的线程
            s_sum[warp_id] = tmp;
        }
        __syncthreads(); // 所有线程同步
        tmp = s_sum[lane_id]; // 同个warp的每个线程取一个数
        tmp = warp_reduce_sum(tmp); // 每个warp都对32个数进行规约求和
    }

    const float mean = tmp / ncols; // 均值，block内所有线程的tmp值相同
    const float scale = rsqrtf(mean + eps); // 1/sqrt(mean+eps)

    for (int col = tid; col < ncols; col += block_size) { // 每个线程负责2个数
        dst[row*ncols + col] = scale * x[row*ncols + col]; // scale 乘上 xi
    }
}
```

# 3. MUL


![[Pasted image 20260112150529.png]]