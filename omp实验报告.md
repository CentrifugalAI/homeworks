# OPM 实验报告

## 文件目录

- 源代码文件
  
  - `origin.c` : 原始源代码
  - `mul1.c` : 修改后的第一个版本的源代码
  - `mul2.c` : 修改后的第二个版本的源代码
  - `mul3.c` : 修改后的第三个版本的源代码
  
- 编译输出文件
  
  - `origin`
  - `mul1`
  - `mul2`
  - `mul3`
  
- `bash` 脚本文件

  - `mul1.sh` : 编译并运行修改后的第一个版本的源代码
  - `mul2.sh` : 编译并运行修改后的第一个版本的源代码
  - `mul3.sh` : 编译并运行修改后的第一个版本的源代码

## 实验思路

### 步骤 0 : 理解程序代码逻辑

对于源代码的解释和分析附在源代码文件的注释中

### 步骤 1 : 可以通过编译优化得到一定的加速

本实验 `bash` 脚本中采用 `-O2` 选项进行优化

### 步骤 2 : 可以利用多线程对循环语句进行优化

> 该代码中存在大量的循环代码结构。优化举例如下：

- 对数据初始化部分进行优化：

比如：

```C
    // maxDisSumPivots : the top M pivots combinations
    int *maxDisSumPivots = (int *)malloc(sizeof(int) * k * (M + 1));
#pragma omp parallel for
    for (i = 0; i < M; i++)
    {
        int ki;
        for (ki = 0; ki < k; ki++)
        {
            maxDisSumPivots[i * k + ki] = 0;
        }
    }
```

- 一些循环不涉及顺序，可以直接使用编译指导语句进行优化

    > 下面的代码是根据循环变量的索引进行操作，这种根据循环变量的索引进行的操作可以使用 omp for 并行优化：因为它保证了循环的顺序以外的逻辑正常

```C
#pragma omp parallel for num_threads(6)
    for (i = 0; i < n; i++)
    {
        int ki;
        for (ki = 0; ki < k; ki++)
        {
            double distance = 0;
            int pivoti = pivots[ki];
            int j;
            for (j = 0; j < dim; j++)
            {
                distance += pow(coord[pivoti * dim + j] - coord[i * dim + j], 2);
            }
            rebuiltCoord[i * k + ki] = sqrt(distance);
        }
    }
```
- 涉及顺序的循环可以使用 `ordered` 进行基于顺序的并行

如下：
```C
#pragma omp parallel for ordered num_threads(4) schedule(guided)
    for (a = M; a > 0; a--)
    {
        if (maxDistanceSum[a] > maxDistanceSum[a - 1])
        {
            //! 交换值
            double temp = maxDistanceSum[a];
            maxDistanceSum[a] = maxDistanceSum[a - 1];
            maxDistanceSum[a - 1] = temp;
            int kj;

            //! 交换索引
            for (kj = 0; kj < k; kj++)
            {
                int temp = maxDisSumPivots[a * k + kj];
                maxDisSumPivots[a * k + kj] = maxDisSumPivots[(a - 1) * k + kj];
                maxDisSumPivots[(a - 1) * k + kj] = temp;
            }
        }
    }
```

- 使用 `reduction` 子句对循环代码逻辑进行优化

> 下面的代码中，将二维循环先展开为一维循环
> 
> 对于新的一维循环，由于代码的最终任务是**加和**，因此可以直接对 `for` 循环进行并行优化
> 
> 可以直接将每个线程中得到的切尔可夫距离作为副本保存，最后用 `reduction` 子句将所有的副本进行加和优化

```C
int ij, ki;
    double chebyshevSum = 0;
#pragma omp parallel for private(ki) reduction(+ \
                                            : chebyshevSum)
    for (ij = 0; ij < n * n; ij++)
    {
        // printf("(%d, %d)", i, j);
        for (ki = 0; ki < k; ki++)
        {
            double dis = fabs(rebuiltCoord[ij / n * k + ki] - rebuiltCoord[ij % n * k + ki]);
            chebyshevSum = dis > chebyshevSum ? dis : chebyshevSum;
        }
    }
```

## 运行方式

原始代码运行方式：

```
>>> bash origin.sh
```

修改后的最终版本代码的运行方式：

```
>>> bash mul3.sh
```

> 相应的运行结果写在 `bash` 脚本的注释中

## 附：知识基础 OPENMP语言

笔记整理于

留档时间：2022/8/31





