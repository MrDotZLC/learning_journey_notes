C++问答题：
## Continuous Batching  
## 计算：  bank
  ``` C[M,N] = A[M,K] × B[K,N]  
      
    M=N=K=4096  
    FP16  

假设使用：  
  
    Block Tile = 128 × 128  
    K Tile = 32  

每一轮需要：  
  
    A_tile = 128 × 32  
    B_tile = 32 × 128  

sky 2026/9/28 14:12:18  
假设某 GPU 的 Shared Memory 具有如下特性：  
  
    Bank 数量：32  
    Bank Width：4 Byte  
    Warp 大小：32 Threads  
    sizeof(float) = 4 Byte  

Shared Memory 的 Bank 映射可以简化为：  
  
    bank_id = (address / 4) % 32  

现在定义：  
  
    __shared__ float tile[32][32];  

假设一个 Warp 中：  
  
    threadIdx.x = 0, 1, 2, ..., 31  

32 个线程同时执行：  
  
    float x = tile[threadIdx.x][0];  
  ```
   
## 计算 S
```
题目  
  
标准 Attention：  
  
    S = Q × Kᵀ  
    P = softmax(S)  
    O = P × V  

给定：  
  
    Batch = 1  
    Heads = 32  
    Sequence Length = 4096  
    Head Dimension = 128  
    FP16  

赵连成 2026/9/28 14:21:32  
batch * heads * seq * seq * 2   
  
sky 2026/9/28 14:24:16  
给定一个 Transformer：  
  
    Layers = 32  
    Batch = 8  
    Sequence Length = 4096  
      
    Attention Heads = 32  
    KV Heads = 32  
    Head Dimension = 128  
      
    KV 数据类型 = FP16  

请计算 KV Cache 总大小。  
  
然后模型改成：  
  
    Query Heads = 32  
    KV Heads = 8  

也就是使用 Grouped Query Attention。  
  
重新计算 KV Cache 大小。  

赵连成 2026/9/28 14:25:09  
2 * Batch * layer * Heads * head_dim * seq * 2  
```

## size_t 溢出
```
sky 2026/9/28 14:26:49  
#include <iostream>  
#include <vector>  
  
void print_reverse(const std::vector<int>& vec) {  
    for (std::size_t i = vec.size() - 1; i >= 0; --i) {  
        std::cout << vec[i] << " ";  
    }  
}  
```

## delete 数组，用delete[]
```
sky 2026/9/28 14:33:16  
class Buffer {  
public:  
    Buffer(size_t n)  
        : size_(n), data_(new int[n]) {  
    }  
  
    ~Buffer() {  
        delete data_;  
    }  
  
private:  
    size_t size_;  
    int* data_;  
};  
  
int main() {  
    Buffer a(100);  
    Buffer b = a;  
}  
```

##  Rule of Zero
  
## copy-and-swap  

## 迭代器删除
```
sky 2026/9/28 14:39:31  
#include <iostream>  
#include <vector>  
  
void removeEvens(std::vector<int>& vec) {  
    for (auto it = vec.begin(); it != vec.end(); ++it) {  
        if (*it % 2 == 0) {  
            vec.erase(it);  
        }  
    }  
} 
```
 
##  const& 返回值 指针空悬、临时对象
```
sky 2026/9/28 14:44:51  
#include <iostream>  
#include <string>  
  
const std::string& get_longer(const std::string& a,  
                              const std::string& b) {  
    return (a.size() >= b.size()) ? a : b;  
}  
  
int main() {  
    std::string str1 = "Hello";  
    const std::string& longer = get_longer(str1, "World_Longer");  
    std::cout << "Longer string: " << longer << "\n";  
}
```  
  
## map[] 默认插入
```
sky 2026/9/28 14:48:20  
#include <map>  
#include <string>  
  
class AccessController {  
private:  
    // 用户 ID -> 角色（例如 "ADMIN"、"MEMBER"）  
    std::map<std::string, std::string> user_roles_;  
  
public:  
    AccessController() {  
        user_roles_["admin_888"] = "ADMIN";  
        user_roles_["user_001"]  = "MEMBER";  
    }  
  
    // 校验某个用户是否是管理员  
    bool is_admin(const std::string& user_id) {  
        if (user_roles_[user_id] == "ADMIN") {  
            return true;  
        }  
        return false;  
    }  
  
    // 获取当前已记录的用户总数  
    std::size_t get_total_users() const {  
        return user_roles_.size();  
    }  
};  
```

## 死锁
用scope_mutex。
```
sky 2026/9/28 14:52:19  
#include <mutex>  
#include <thread>  
  
struct Account {  
    std::mutex mtx;  
    int balance = 1000;  
};  
  
void transfer_money(Account& from, Account& to, int amount) {  
    std::lock_guard<std::mutex> lock_from(from.mtx);  
    std::lock_guard<std::mutex> lock_to(to.mtx);  
  
    from.balance -= amount;  
    to.balance += amount;  
}  
```

## String
```
#include <cstddef>
#include <cstring>
#include <utility>
#include <stdexcept>
#include <iostream>

class String {
public:
    // 默认构造：空字符串，仍保证 c_str() 返回有效指针
    String() noexcept
        : data_(new char[1]{'\0'}), size_(0) {}

    // 从 C 风格字符串构造
    String(const char* s) {
        if (s == nullptr) {
            throw std::invalid_argument("String: null pointer");
        }
        size_ = std::strlen(s);
        data_ = new char[size_ + 1];
        std::memcpy(data_, s, size_ + 1);
    }

    // 析构
    ~String() {
        delete[] data_;
    }

    // 拷贝构造：深拷贝，两个对象互不影响
    String(const String& other)
        : size_(other.size_) {
        data_ = new char[size_ + 1];
        std::memcpy(data_, other.data_, size_ + 1);
    }

    // 拷贝赋值：copy-and-swap，异常安全且自赋值安全
    String& operator=(const String& other) {
        if (this != &other) {
            String tmp(other);
            swap(tmp);
        }
        return *this;
    }

    // 移动构造：窃取资源，noexcept 很关键
    String(String&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    // 移动赋值：先释放自身资源，再窃取对方资源
    String& operator=(String&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    std::size_t size() const noexcept {
        return size_;
    }

    const char* c_str() const noexcept {
        return data_;
    }

    // 下标访问（可读）
    const char& operator[](std::size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("String: index out of range");
        }
        return data_[index];
    }

    // 下标访问（可写）
    char& operator[](std::size_t index) {
        if (index >= size_) {
            throw std::out_of_range("String: index out of range");
        }
        return data_[index];
    }

    void swap(String& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }

private:
    char* data_;
    std::size_t size_;
};
```
  
sky 2026/9/28 15:05:20  
题目  
  
实现一个拥有动态字符数组的 String 类。  
  
加分项：写出移动构造和移动赋值函数，添加 noexcept，并解释原因。  
  
需求边界  
  
为避免题意不清，面试时建议至少约定：  
  
- 字符串拥有自己的连续内存，并以 \0 结尾；  
- 支持默认构造、从字符序列构造、析构、拷贝和移动；  
- 支持 size()、c_str() 和下标访问；  
- 拷贝后两个对象互不影响；  
- 移动后的源对象仍然有效、可析构、可再次赋值。  

sky 2026/9/28 15:10:54  
std::string_view
假设某 GPU 的 Shared Memory 具有如下特性：
    Bank 数量：32
    Bank Width：4 Byte
    Warp 大小：32 Threads
    sizeof(float) = 4 Byte
    
Shared Memory 的 Bank 映射可以简化为：

    bank_id = (address / 4) % 32

现在定义：
    __shared__ float tile [32] [32];

假设一个 Warp 中：

    threadIdx.x = 0, 1, 2, ..., 31
    

32 个线程同时执行：

    float x = tile[threadIdx.x][0];

4096
128 * 128 C

GRID

32 * 32 , 128 * 128
