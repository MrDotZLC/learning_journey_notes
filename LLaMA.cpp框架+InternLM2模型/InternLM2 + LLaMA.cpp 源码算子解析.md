环境：[InternLM 概述与环境准备](Learning/LLaMA.cpp框架+InternLM2模型/InternLM%20概述与环境准备.md)
# 0. 先验知识
## 0.1. pytorch中的维度与llama.cpp中的内存跨步对比
![Pasted image 20260110030940](Pasted%20image%2020260110030940.png)
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
### 注意：LLaMA.cpp中不能运行时广播，因为内存都是提前分配好的。LLaMA.cpp解决维度不匹配的方式：
1. graph构建前，**计算好tensor的维度**，提前消除维度不一致。
2. 利用**维度折叠**，“消灭”不匹配的维度。
3. 在kernel中显式做“**逻辑广播**”，“写死”在index计算里。
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
## 0.5. llama.cpp中Attention结构
![Pasted image 20260113211925](Pasted%20image%2020260113211925.png)
## 0.6. InternLM2 F16精度的推理Graph结构
num_seq是不断增加的，到达上限后，删除最旧的数据，再存放新seq的数据。
kv cache是提前分配了足够的维度，只是多了与无效数据的计算。
![Graph-2026-01-18-0557.excalidraw](Learning/LLaMA.cpp框架+InternLM2模型/Graph-2026-01-18-0557.excalidraw.md|666)

# 1. GET_ROWS（取值拼接）
## 1.1 InternLM python代码：
![Pasted image 20260113213148](Pasted%20image%2020260113213148.png)
![Pasted image 20260113213346](Pasted%20image%2020260113213346.png)
## 1.2 LLaMA.cpp 核心代码：等价于dst\[i\] = src0\[src1\[i\]\]
根据索引从weight矩阵中取值拼接。
GET_ROWS算子是CPU后端进行计算。
采用cuda并行计算方式，每个线程处理多个数据。
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
![Pasted image 20260110033758](Pasted%20image%2020260110033758.png)

# 2. RMS_NORM（均方根归一化）
[RMS_norm介绍](RMS_norm%E4%BB%8B%E7%BB%8D.md)
![Pasted image 20260111025330](Pasted%20image%2020260111025330.png)
对2048进行mean，得到[1,5,1]，再对源输入进行广播。 
## 2.1 InternLM python代码解析：
![Pasted image 20260113211108](Pasted%20image%2020260113211108.png)
## 2.2 LLaMA.cpp 实现代码解析：实际只完成了红框部分的计算，红框部分乘上**权重**是下一个算子（广播乘法）。
![Pasted image 20260111031403](Pasted%20image%2020260111031403.png)
block形状[1024,1,1]，一个 block有1024个线程和32个warp。数据形状为5 * 2048，即5个block，每个block处理一行，每个线程处理2个数据。 
![Pasted image 20260111033734](Pasted%20image%2020260111033734.png)
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

# 3. MUL（RMSNorm中的广播乘法）
## 3.1 统一维度（维度折叠）
![Pasted image 20260112173351](Pasted%20image%2020260112173351.png)
```
// 模板参数：bin_op 是一个二元 float 运算函数指针（如 add / mul）
template<float (*bin_op)(const float, const float)>
struct bin_bcast_cuda {

    // 泛型算子：支持不同 src0/src1/dst 数据类型
    template<typename src0_t, typename src1_t, typename dst_t>
    void operator()(
        const struct ggml_tensor * src0,   // 输入张量0（host 侧描述）
        const struct ggml_tensor * src1,   // 输入张量1（host 侧描述）
        struct ggml_tensor * dst,          // 输出张量（host 侧描述）
        const src0_t * src0_dd,             // 输入0 的 device 指针
        const src1_t * src1_dd,             // 输入1 的 device 指针
        dst_t * dst_dd,                     // 输出的 device 指针
        cudaStream_t stream) {              // CUDA stream

        // 展开 GGML tensor 的维度与 stride：
        // ne*  表示每个维度的元素数量
        // nb*  表示每个维度的字节 stride
        GGML_TENSOR_BINARY_OP_LOCALS // 定义 ne0..ne3, ne00..ne03, ne10..ne13, nb*

        // 计算 src1 相对于 dst 在每个维度上的 broadcast 比率
        int nr0 = ne10/ne0;
        int nr1 = ne11/ne1;
        int nr2 = ne12/ne2;
        int nr3 = ne13/ne3;

        // nr[i] == 1 表示该维度没有 broadcast
        int nr[4] = { nr0, nr1, nr2, nr3 };

        // ====== 用于“维度折叠（collapse）”的临时数组 ======

        // dst 的 shape
        int64_t cne[]  = {ne0, ne1, ne2, ne3};
        // src0 的 shape
        int64_t cne0[] = {ne00, ne01, ne02, ne03};
        // src1 的 shape
        int64_t cne1[] = {ne10, ne11, ne12, ne13};

        // dst 的 stride
        size_t cnb[]  = {nb0, nb1, nb2, nb3};
        // src0 的 stride
        size_t cnb0[] = {nb00, nb01, nb02, nb03};
        // src1 的 stride
        size_t cnb1[] = {nb10, nb11, nb12, nb13};

        // 将高维折叠进低维：
        // [d0, d1, d2, d3] -> [d0*d1, d2, d3, 1]
        auto collapse = [](int64_t cne[]) {
            cne[0] *= cne[1];
            cne[1] = cne[2];
            cne[2] = cne[3];
            cne[3] = 1;
        };

        // 同步更新 stride（注意 stride 与 shape 的乘法关系）
        auto collapse_nb = [](size_t cnb[], const int64_t cne[]) {
            cnb[1] *= cne[1];
            cnb[2] *= cne[2];
            cnb[3] *= cne[3];
        };

        // ====== 维度折叠优化 ======
        // 仅当 src0 / src1 / dst 都是 contiguous 时才允许
        // 并且从最低维开始，只要该维度不需要 broadcast（nr[i] == 1）
        if (ggml_is_contiguous(src0) &&
            ggml_is_contiguous(src1) &&
            ggml_is_contiguous(dst)) {

            for (int i = 0; i < 4; i++) {
                // 一旦遇到需要 broadcast 的维度就停止折叠
                if (nr[i] != 1) {
                    break;
                }

                // 从第 1 维开始才真正折叠
                if (i > 0) {
                    collapse_nb(cnb,  cne);
                    collapse_nb(cnb0, cne0);
                    collapse_nb(cnb1, cne1);

                    collapse(cne);
                    collapse(cne0);
                    collapse(cne1);
                }
            }
        }

        // ====== 使用折叠后的 shape / stride 重新绑定局部变量 ======
        {
            int64_t ne0 = cne[0];
            int64_t ne1 = cne[1];
            int64_t ne2 = cne[2];
            int64_t ne3 = cne[3];

            // src1 的 shape（src0 的 shape 在 kernel 中不再需要）
            int64_t ne10 = cne1[0];
            int64_t ne11 = cne1[1];
            int64_t ne12 = cne1[2];
            int64_t ne13 = cne1[3];

            // dst 的 stride（字节）
            size_t nb0 = cnb[0];
            size_t nb1 = cnb[1];
            size_t nb2 = cnb[2];
            size_t nb3 = cnb[3];

            // src0 的 stride（字节）
            size_t nb00 = cnb0[0];
            size_t nb01 = cnb0[1];
            size_t nb02 = cnb0[2];
            size_t nb03 = cnb0[3];

            // src1 的 stride（字节）
            size_t nb10 = cnb1[0];
            size_t nb11 = cnb1[1];
            size_t nb12 = cnb1[2];
            size_t nb13 = cnb1[3];

            // ====== 将 stride 从“字节”转换为“元素数” ======

            size_t s0 = nb0 / sizeof(dst_t);
            size_t s1 = nb1 / sizeof(dst_t);
            size_t s2 = nb2 / sizeof(dst_t);
            size_t s3 = nb3 / sizeof(dst_t);

            size_t s10 = nb10 / sizeof(src1_t);
            size_t s11 = nb11 / sizeof(src1_t);
            size_t s12 = nb12 / sizeof(src1_t);
            size_t s13 = nb13 / sizeof(src1_t);

            size_t s00 = nb00 / sizeof(src0_t);
            size_t s01 = nb01 / sizeof(src0_t);
            size_t s02 = nb02 / sizeof(src0_t);
            size_t s03 = nb03 / sizeof(src0_t);

            // CUDA block 的最大线程数
            const int block_size = 128;

            // ne0 的一半（通常用于 vectorized / half2 优化）
            int64_t hne0 = std::max(ne0 / 2LL, 1LL);

            // ====== 计算 block 维度 ======
            dim3 block_dims;
            block_dims.x = std::min<unsigned int>(hne0, block_size);
            block_dims.y = std::min<unsigned int>(ne1, block_size / block_dims.x);
            block_dims.z = std::min(
                std::min<unsigned int>(
                    ne2 * ne3,
                    block_size / block_dims.x / block_dims.y),
                64U
            );

            // ====== 计算 grid 维度 ======
            dim3 block_nums(
                (hne0        + block_dims.x - 1) / block_dims.x,
                (ne1         + block_dims.y - 1) / block_dims.y,
                (ne2 * ne3   + block_dims.z - 1) / block_dims.z
            );

            // ====== kernel 选择 ======
            // CUDA 的 z 维 grid 上限是 65535
            if (block_nums.z > 65535) {
                // 超出限制时，退化为 1D grid 的 unravel kernel
                int block_num =
                    (ne0 * ne1 * ne2 * ne3 + block_size - 1) / block_size;

                k_bin_bcast_unravel<bin_op><<<block_num, block_size, 0, stream>>>(
                    src0_dd, src1_dd, dst_dd,
                    ne0, ne1, ne2, ne3,
                    ne10, ne11, ne12, ne13,
                    /* dst stride */  s1,  s2,  s3,
                    /* src0 stride */ s01, s02, s03,
                    /* src1 stride */ s11, s12, s13
                );
            } else {
                // 正常使用 3D grid kernel
                k_bin_bcast<bin_op><<<block_nums, block_dims, 0, stream>>>(
                    src0_dd, src1_dd, dst_dd,
                    ne0, ne1, ne2, ne3,
                    ne10, ne11, ne12, ne13,
                    /* dst stride */  s1,  s2,  s3,
                    /* src0 stride */ s01, s02, s03,
                    /* src1 stride */ s11, s12, s13
                );
            }
        }
    }
};
```
## 3.2 广播乘法
### 3.2.1 python代码中，调用的是库函数。
![Pasted image 20260114005850](Pasted%20image%2020260114005850.png)
核函数的grid、block的线程分布示意：
z表示3/4维，用ne3做除法和取模，结果分别为第3维和第4维。
![Pasted image 20260112190722](Pasted%20image%2020260112190722.png)
### 3.2.2 LLaMA.cpp 代码解析：
```
// CUDA kernel：对两个张量执行逐元素二元运算（支持 broadcast）
// bin_op : 二元 float 运算（如 add / mul / max）
// src0_t / src1_t / dst_t : 实际数据类型（可能是 fp16 / fp32 等）
template<float (*bin_op)(const float, const float),
         typename src0_t,
         typename src1_t,
         typename dst_t>
static __global__ void k_bin_bcast(
        const src0_t * src0,   // 输入张量0（device 指针，可为 null）
        const src1_t * src1,   // 输入张量1（device 指针）
        dst_t * dst,           // 输出张量（device 指针）

        // dst 的 shape（折叠后的 4 维）
        int ne0, int ne1, int ne2, int ne3,

        // src1 的 shape（用于 broadcast）
        int ne10, int ne11, int ne12, int ne13,

        // dst 的 stride（单位：元素）
        /*int s0, */ int s1,  int s2,  int s3,

        // src0 的 stride（单位：元素）
        /*int s00,*/ int s01, int s02, int s03,

        // src1 的 stride（单位：元素）
        /*int s10,*/ int s11, int s12, int s13) {

    // ====== 计算线程对应的逻辑索引 ======

    // i0s：当前线程负责的 ne0 维起始索引
    // x 维 thread + block 映射到 ne0
    const int i0s = blockDim.x * blockIdx.x + threadIdx.x;

    // i1：ne1 维索引（y 维 grid/block）
    const int i1  = blockDim.y * blockIdx.y + threadIdx.y;

    // z 维同时覆盖 ne2 和 ne3：
    // 先映射到线性索引，再拆成 (i2, i3)
    const int iz  = blockDim.z * blockIdx.z + threadIdx.z;
    const int i2  = iz / ne3;   // ne2 维索引
    const int i3  = iz % ne3;   // ne3 维索引

    // ====== 越界保护 ======
    if (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    // ====== broadcast 维度取模 ======
    // 当 src1 在某几个维度上是 broadcast 时：
    // ne1? < ne?，通过 % 映射回有效索引

    const int i11 = i1 % ne11;  // src1 在 ne1 维的索引
    const int i12 = i2 % ne12;  // src1 在 ne2 维的索引
    const int i13 = i3 % ne13;  // src1 在 ne3 维的索引

    // ====== 计算 base offset（不包含 ne0 维） ======

    // src0 的行起始偏移（元素单位）
    const size_t i_src0 =
            i3 * s03 +
            i2 * s02 +
            i1 * s01;

    // src1 的行起始偏移（元素单位，使用 broadcast 后的索引）
    const size_t i_src1 =
            i13 * s13 +
            i12 * s12 +
            i11 * s11;

    // dst 的行起始偏移（元素单位）
    const size_t i_dst =
            i3 * s3 +
            i2 * s2 +
            i1 * s1;

    // ====== 取得当前 (i1,i2,i3) 对应的指针 ======

    const src0_t * src0_row = src0 + i_src0;
    const src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row         = dst  + i_dst;

    // ====== ne0 维上的主循环 ======
    // 采用 grid-stride loop 覆盖整个 ne0 维
    for (int i0 = i0s; i0 < ne0; i0 += blockDim.x * gridDim.x) {

        // src1 在 ne0 维上的 broadcast 映射（0维是连续的，几何直觉上是列，放在循环里广播映射）
        const int i10 = i0 % ne10;

        // 执行二元运算：
        // - src0 可能为 null（如一元算子退化情形）
        // - src0/src1 转换为 float
        // - 结果再 cast 为 dst_t
        dst_row[i0] = (dst_t)bin_op(
            src0 ? (float)src0_row[i0] : 0.0f,
            (float)src1_row[i10]
        );
    }
}

```
## 3.3 为什么 GGML 不把stride设为0
1. **保持 stride 的“物理含义不变”**：把 broadcast 作为“逻辑索引规则”处理，而不是“内存布局规则”。
2. **保证 collapse / reshape / view 的正确性**：避免 index 从空间塌缩成一个点，进而 debug 困难。
3. **让 kernel 保持统一、可组合、可维护**：避免分支特判，或生成多套 kernel 。
4. **用极小的 `%` 成本换取整体架构稳定性**：不是性能瓶颈。
# 4. MAT_MUL（Attention 中的矩阵乘法，Linear / GEMM）
用Linear将权重和归一化结果，生成QKV。
![Pasted image 20260113212008](Pasted%20image%2020260113212008.png)
## 4.1 InternLM python代码：
torch融合QKV权重为一个大矩阵，一次Linear中进行一次矩阵乘法，生成QKV融合矩阵，再分割成QKV。
调用的是torch库函数，这里不再深入解析。
![Pasted image 20260113215407](Pasted%20image%2020260113215407.png)
## 4.2 LLaMA.cpp代码：
权重$W_Q$、$W_K$、$W_V$在内存拷贝时分配，Attention中分别矩阵乘$X_{RMSNorm}$，得到QKV。
![Pasted image 20260113204201](Pasted%20image%2020260113204201.png)
每个矩阵乘法都是调用cuda库函数，不在深入解析，cuda 矩阵乘法可参考[CUDA：SGEMM单精度矩阵乘法（待整理）](Learning/CUDA/CUDA：SGEMM单精度矩阵乘法（待整理）.md)。
![Pasted image 20260114041038](Pasted%20image%2020260114041038.png)
# 5. ROPE（旋转位置编码）
详见[Positional Encoding 位置编码介绍](Learning/AI%20Infra/Positional%20Encoding%20位置编码介绍.md)。
假设 token embedding 向量 $x \in \mathbb{R}^d$，将其拆成每两个维度一组$(x_{2i}, x_{2i+1})$，对每组应用旋转矩阵：
$$\begin{bmatrix} x'_{2i} \\ x'_{2i+1} \end{bmatrix} =
\begin{bmatrix} \cos \theta_i & -\sin \theta_i \\ \sin \theta_i & \cos \theta_i \end{bmatrix}
\begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix}$$
其中：
$$
\theta_i = \text{position} \times \omega_i
$$
- $\text{position}$：token 在序列中的位置（0, 1, 2, …）  
- $\omega_i = 10000^{-2i/d}$：每组维度的频率，低维度频率高，高维度频率低。
## 5.1 InternLM python代码解析
![](Learning/LLaMA.cpp框架+InternLM2模型/Pasted%20image%2020260114054025.png)
### 5.1.1 获取所有维度对的旋转角度
![](Learning/LLaMA.cpp框架+InternLM2模型/Pasted%20image%2020260114203720.png)
### 5.1.2 计算旋转后的QK矩阵
公式里写的是“矩阵乘法”，但在 PyTorch 实现中，它被等价地展开成了逐元素运算（Hadamard 运算 + 广播），从而避免显式构造旋转矩阵。
**对整个向量 $x \in \mathbb{R}^d$，RoPE 是 对每一对维度独立旋转**：
$$x' = x \odot \cos\theta + \text{rotate\_half}(x) \odot \sin\theta$$
![](Learning/LLaMA.cpp框架+InternLM2模型/Pasted%20image%2020260116191050.png)

## 5.2 LLaMA.cpp 代码解析
框架写死了block_dim(1,256,1)，在核函数中，实际只有(1,64,1)参与ROPE计算。
![](Learning/LLaMA.cpp框架+InternLM2模型/Pasted%20image%2020260116211949.png)
1. $\omega_i = 10000^{-2i/d}$中与维度 i 无关的部分，提前计算成固定值theta_scale。
2. 每个线程计算1对维度对 i0，代表相邻的2个维度$i_{2n}$和$i_{2n+1}$。
3. 在(1,5,16,128)的矩阵中，以最后1个维度为1行，row 为第 row 行。
4. 在(1,5,16,128)的矩阵中，i 为矩阵的位置索引下标。
5. 在(1,5,16,128)的矩阵中，i2 为第2维的索引下标，即第几个token位置下表
6. 这一行对应公式$\theta_{i0} = \text{position}_{i2} \times \omega_{i0}$
7. 将维度对 i0 拆分为维度$i_{2n}$和$i_{2n+1}$。
8. 旋转维度对。
![](Learning/LLaMA.cpp框架+InternLM2模型/Pasted%20image%2020260116233545.png)
# 6. 维度变换
## 6.1 实践建议（经验法则）
1. **首选 `reshape`**
   除非你明确需要 `view` 的零拷贝语义
2. **`transpose` / `permute` 后如果要 `view`**
    `x = x.permute(...).contiguous().view(...)`
3. **调试维度问题**
    `print(x.shape, x.stride(), x.is_contiguous())`
## 6.2 pytorch 的实现
| 方法          | 是否改 shape | 是否改维度顺序 | 是否拷贝 | 要求 contiguous | 常见用途        |
| ----------- | --------- | ------- | ---- | ------------- | ----------- |
| `view`      | 是         | 否       | 否    | 是             | 高性能 reshape |
| `reshape`   | 是         | 否       | 可能   | 否             | 安全 reshape  |
| `transpose` | 是         | 是（2 维）  | 否    | 否             | 矩阵/特征转置     |
| `permute`   | 是         | 是（任意）   | 否    | 否             | 任意维度重排      |
## 6.3 LLaMA.cpp 代码实现
本质上，这些操作都是视图节点，不涉及数据操作。
### 6.3.1 view/reshape
- **不改变 data 指针**
- **不允许破坏 stride 规则**
- **不会触发 copy**
- 如果 stride 不合法 → **构图阶段直接拒绝**
- reshape可能比view慢，因为可能会隐式调用 CONT 算子进行数据拷贝。
> 换句话说：
> ggml 的 reshape = PyTorch 的 `view`，  
> 而不是 `reshape`

![](Learning/LLaMA.cpp框架+InternLM2模型/Pasted%20image%2020260117020723.png)
```
struct ggml_tensor * ggml_reshape_3d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    GGML_ASSERT(ggml_is_contiguous(a)); // 内存连续断言
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2); // 元素个数断言

    const int64_t ne[3] = { ne0, ne1, ne2 };
    // reshape前后buffer均为空
    struct ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 3, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    return result;
}
```
### 6.3.2 transpose
- **只支持 2D**
- 仅修改 shape + stride
- **不 copy 数据**
![](Learning/LLaMA.cpp框架+InternLM2模型/Pasted%20image%2020260117021310.png)
### 6.3.3 premute
- 允许多维度重排**
- 仅修改 shape + stride
- **不 copy 数据**
![](Learning/LLaMA.cpp框架+InternLM2模型/Pasted%20image%2020260117021847.png)

# 7. CPY
Graph中，以洋红框中的算子为例。
1. cache_k_l0：K的缓存
2. k_cache_view-0：是cache_k_l0 的一部分，它们共享同一块内存。
3. k_cache_view-0(copy_of_Kcur-0)：将 Kcur-0 复制到 k_cache_view-0，等价于复制到cache_k_l0。
![](Learning/LLaMA.cpp框架+InternLM2模型/Pasted%20image%2020260117172430.png)
## 7.1 LLaMA.cpp 代码解释
1. block_num=80，block_dim=64，正好是5120个线程，每个线程处理一个数据。
2. 计算维度索引（兼容不连续内存）
3. 计算数据源 x 和目标 dst 的开始位置偏移。
4. 使用 cpy_1_f32_f16 将 f32 的 x 数据复制到 f16 的 dst，该核函数很复杂不深入讲解。
![](Learning/LLaMA.cpp框架+InternLM2模型/Pasted%20image%2020260117195155.png)

# 8. BATCH_MAT_MUL（batch矩阵乘）
![](Learning/LLaMA.cpp框架+InternLM2模型/Pasted%20image%2020260119232500.png)
实际是，运行时创建2个容器指针，容器中分别存放1个batch的矩阵地址和目标矩阵地址，按数组进行矩阵乘。

```
static void ggml_cuda_mul_mat_batched_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    
    ...

    void * src0_ddq = src0->data;
    half * src0_f16 = (half *) src0_ddq; // src0是F16，直接强转
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    // convert src1 to fp16
    ggml_cuda_pool_alloc<half> src1_f16_alloc(ctx.pool()); // 存储src1每个batch矩阵指针
    if (src1->type != GGML_TYPE_F16) {
        const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src1->type);
        const int64_t ne_src1 = ggml_nelements(src1);
        src1_f16_alloc.alloc(ne_src1);                      // 调用RunTime API分配显存
        // 包含一个核函数进行强制类型转换，将F32转成F16
        to_fp16_cuda(src1_ddf, src1_f16_alloc.get(), ne_src1, main_stream);
    }
    half * src1_f16 = src1->type == GGML_TYPE_F16 ? (half *) src1_ddf : src1_f16_alloc.get();

	...

    if (dst->op_params[0] == GGML_PREC_DEFAULT) {
        dst_t = (char *) dst_f16.alloc(ne_dst);

        nbd2 /= sizeof(float) / sizeof(half);
        nbd3 /= sizeof(float) / sizeof(half);
    } else {                                                   // F16*F16直接生成F32
        dst_t = (char *) dst_ddf;

        cu_compute_type = CUBLAS_COMPUTE_32F;
        cu_data_type    = CUDA_R_32F;

        alpha = &alpha_f32;
        beta  = &beta_f32;
    }
    
	...    
    
    if (r2 == 1 && r3 == 1 && ggml_is_contiguous_2(src0) && ggml_is_contiguous_2(src1)) {
        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const char *) src0_f16, CUDA_R_16F,   nb01/nb00, nb02/nb00,  // strideA
                       (const char *) src1_f16, CUDA_R_16F,   nb11/nb10, nb12/nb10,  // strideB
                beta,  (      char *)    dst_t, cu_data_type, ne01,       nb2/nb0,   // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // use cublasGemmBatchedEx
        const int ne23 = ne12*ne13;                  // 前2个维度乘积1*16，即batch总数
		// 为 src0,src1 和 dst创建容器，存放数据地址指针
		// ptrs_src 包含 src0 和src1，所以需要*2
        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);
        // 按维度将数据地址指针存入容器，每个线程存1个数据
        dim3 block_dims(ne13, ne12);
        k_compute_batched_ptrs<<<1, block_dims, 0, main_stream>>>(
                src0_f16, src1_f16, dst_t,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                nb02, nb03,
                src1->type == GGML_TYPE_F16 ? nb12 : nb12/2,
                src1->type == GGML_TYPE_F16 ? nb13 : nb13/2,
                nbd2, nbd3,
                r2, r3);
        CUDA_CHECK(cudaGetLastError());

		// 调用cublas库：进行 ne23 次 F16*F16→F32 的矩阵乘
        CUBLAS_CHECK(
        cublasGemmBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), CUDA_R_16F,   nb01/nb00,
                       (const void **) (ptrs_src.get() + 1*ne23), CUDA_R_16F,   nb11/nb10,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type, ne01,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    
    ...
}
```

# 9. SOFT_MAX
pytorch中是单个算子，llama.cpp中包含了：
1. 对分数矩阵缩放$\frac{1}{\sqrt{d_k}}$：防止维度大导致梯度消失
2. 累加mask上三角矩阵：将无效注意力（padding位）权重置为负无穷，exp为0，即对soft_max分母无贡献。
3. 计算soft_max：将分数转成权重（概率分布）
![](Learning/LLaMA.cpp框架+InternLM2模型/Pasted%20image%2020260120015744.png)
- ggml中，softmax 通常是 **自定义 kernel**
- 做了：
    - max-reduction（防溢出）
    - exp
    - sum
    - normalize
- 会显式跳过：
    - padding token
    - causal mask 的未来位置
![](Learning/LLaMA.cpp框架+InternLM2模型/Pasted%20image%2020260120023037.png)
![](Learning/LLaMA.cpp框架+InternLM2模型/Pasted%20image%2020260120040532.png)

# 10. CONT(contiguous连续内存)
使数据内存连续，本质是CPY算子，使用场景：
- 可能会破坏连续性的操作后
- 需要连续性操作前

|操作|是否改 shape|是否拷贝|是否保证连续|
|---|---|---|---|
|`view()`|是|否|否|
|`reshape()`|是|可能|否|
|`contiguous()`|否|可能|是|

# 11. SILU
一个 **平滑、非线性、非对称** 的激活函数。[Attention中的激活函数（SILU）](Learning/AI%20Infra/Attention中的激活函数（SILU）.md)
$$\text{SiLU}(x) = x \cdot \sigma(x) = x \cdot \frac{1}{1 + e^{-x}}$$
![](Learning/LLaMA.cpp框架+InternLM2模型/Pasted%20image%2020260120043155.png)
