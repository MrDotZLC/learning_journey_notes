# 0. 先验知识
## 0.1. pytorch中的维度与llama.cpp中的内存跨步对比
![[Pasted image 20260110030940.png]]
## 0.2. Token输入
输入：Hello my name is
token数量：5（Hello前默认有一个起始符 \<s\>）

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
                // i01是token_id, 
                (const void *) ((char *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
	            // nb00是一个元素的字节数，i10是0维索引，所以要乘一个第0维度的总字节数 nb01，同理i11乘一个第1维度的总字节数    
			    (float *) ((char *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}
```
![[Pasted image 20260110033758.png]]

# 2. RMS_NORM


# 3. MUL
