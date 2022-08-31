//! ======================================================
//! 版本1：主要完成了: 1.开启 o3 编译优化选项; 2.输入部分的并行加速; 3.参数初始化部分的并行加速
//!
//! max : 143 351 117761.647418
//! min : 83 226 43769.849602
//!
//! ======================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

// translate:当组合不同的枢轴时，计算距离的总和
// Calculate sum of distance while combining different pivots. Complexity : O( n^2 )

// translate:当组合不同的枢轴时，计算距离的总和
// Calculate sum of distance while combining different pivots. Complexity : O( n^2 )
double SumDistance(const int k, const int n, const int dim, double *coord, int *pivots)
{
    double *rebuiltCoord = (double *)malloc(sizeof(double) * n * k);
    int i;
    for (i = 0; i < n * k; i++)
    {
        rebuiltCoord[i] = 0;
    }

    // Rebuild coordinates. New coordinate of one point is its distance to each pivot.
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

    // Calculate the sum of Chebyshev distance with rebuilt coordinates between every points
    double chebyshevSum = 0;
    for (i = 0; i < n; i++)
    {
        int j;
        for (j = 0; j < n; j++)
        {
            double chebyshev = 0;
            int ki;
            for (ki = 0; ki < k; ki++)
            {
                double dis = fabs(rebuiltCoord[i * k + ki] - rebuiltCoord[j * k + ki]);
                chebyshev = dis > chebyshev ? dis : chebyshev;
            }
            chebyshevSum += chebyshev;
        }
    }

    free(rebuiltCoord);
    return chebyshevSum;
}

// Recursive function Combination() : combine pivots and calculate the sum of distance while combining different pivots.
// ki  : current depth of the recursion
// k   : number of pivots
// n   : number of points
// dim : dimension of metric space
// M   : number of combinations to store
// coord  : coordinates of points
// pivots : indexes of pivots
// maxDistanceSum  : the largest M distance sum
// maxDisSumPivots : the top M pivots combinations
// minDistanceSum  : the smallest M distance sum
// minDisSumPivots : the bottom M pivots combinations
void Combination(int ki, const int k, const int n, const int dim, const int M, double *coord, int *pivots,
                 double *maxDistanceSum, int *maxDisSumPivots, double *minDistanceSum, int *minDisSumPivots)
{
    // 边界条件：当前递归深度已经达到了（支撑点集数量 - 1）
    if (ki == k - 1)
    {
        int i;
        // pivots
        for (i = pivots[ki - 1] + 1; i < n; i++)
        {
            pivots[ki] = i; // 即 pivots[k-1] = i

            // Calculate sum of distance while combining different pivots.
            double distanceSum = SumDistance(k, n, dim, coord, pivots);

            // put data at the end of array
            maxDistanceSum[M] = distanceSum;
            minDistanceSum[M] = distanceSum;
            int kj;
            for (kj = 0; kj < k; kj++)
            {
                maxDisSumPivots[M * k + kj] = pivots[kj];
            }
            for (kj = 0; kj < k; kj++)
            {
                minDisSumPivots[M * k + kj] = pivots[kj];
            }
            // sort
            int a;
            for (a = M; a > 0; a--)
            {
                if (maxDistanceSum[a] > maxDistanceSum[a - 1])
                {
                    double temp = maxDistanceSum[a];
                    maxDistanceSum[a] = maxDistanceSum[a - 1];
                    maxDistanceSum[a - 1] = temp;
                    int kj;
                    for (kj = 0; kj < k; kj++)
                    {
                        int temp = maxDisSumPivots[a * k + kj];
                        maxDisSumPivots[a * k + kj] = maxDisSumPivots[(a - 1) * k + kj];
                        maxDisSumPivots[(a - 1) * k + kj] = temp;
                    }
                }
            }
            for (a = M; a > 0; a--)
            {
                if (minDistanceSum[a] < minDistanceSum[a - 1])
                {
                    double temp = minDistanceSum[a];
                    minDistanceSum[a] = minDistanceSum[a - 1];
                    minDistanceSum[a - 1] = temp;
                    int kj;
                    for (kj = 0; kj < k; kj++)
                    {
                        int temp = minDisSumPivots[a * k + kj];
                        minDisSumPivots[a * k + kj] = minDisSumPivots[(a - 1) * k + kj];
                        minDisSumPivots[(a - 1) * k + kj] = temp;
                    }
                }
            }
        }
        return;
    }

    // 递归调用 Combination() 函数实现支撑点组合
    int i; // i 的起点是上一次调用递归函数
    for (i = pivots[ki - 1] + 1; i < n; i++)
    {
        pivots[ki] = i; // 这里写个递归的作用就是依照顺序地设置 pivots 的索引
        Combination(ki + 1, k, n, dim, M, coord, pivots, maxDistanceSum, maxDisSumPivots, minDistanceSum, minDisSumPivots);

        /** Iteration Log : pivots computed, best pivots, max distance sum, min distance sum pivots, min distance sum
        *** You can delete the logging code. **/
        // 下面这些代码是日志代码，用于记录数据结果
        if (ki == k - 2)
        {
            int kj;
            for (kj = 0; kj < k; kj++)
            {
                printf("%d ", pivots[kj]);
            }
            putchar('\t');
            for (kj = 0; kj < k; kj++)
            {
                printf("%d ", maxDisSumPivots[kj]);
            }
            printf("%lf\t", maxDistanceSum[0]);
            for (kj = 0; kj < k; kj++)
            {
                printf("%d ", minDisSumPivots[kj]);
            }
            printf("%lf\n", minDistanceSum[0]);
        }
    }
}

int main(int argc, char *argv[])
{
    // filename : input file namespace
    char *filename = (char *)"uniformvector-2dim-5h.txt";
    if (argc == 2)
    {
        filename = argv[1];
    }
    else if (argc != 1)
    {
        printf("Usage: ./pivot <filename>\n");
        return -1;
    }
    // M : number of combinations to store
    const int M = 1000;
    // dim : dimension of metric space
    int dim; // 这里指的是原有度量空间的维度
    // n : number of points
    int n;
    // k : number of pivots
    int k; // 隐含条件：变化后的 x 的坐标的维度就是支撑点的个数

    // Read parameter
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        printf("%s file not found.\n", filename);
        return -1;
    }
    fscanf(file, "%d", &dim);
    fscanf(file, "%d", &n);
    fscanf(file, "%d", &k);
    printf("dim = %d, n = %d, k = %d\n", dim, n, k);

    //********************************************优化时间记从这里开始记录************************************************//
    // Start timing
    struct timeval start;
    gettimeofday(&start, NULL);

    // Read Data
    //! coord指的是我们输入的数据点，这里整合成一维数组的形式
    double *coord = (double *)malloc(sizeof(double) * dim * n);
    int i;

//!===============================下面的代码是程序的输入读取部分，并行地读取它们可以加速程序==================================================//
// file read path exist optimization space：because the coord correspoding to the index
//! danger<已解决>：二维循环可以采用下面这种 omp 语言的加速方式，已经证明不影响结果
#pragma omp parallel shared(n, dim, coord)
    {
        {
#pragma omp for schedule(guided)
            for (i = 0; i < n; i++)
            {
                int j;
                for (j = 0; j < dim; j++)
                {
#pragma omp critical
                    fscanf(file, "%lf", &coord[i * dim + j]);
                }
            }
        }
    }
    fclose(file);

    // maxDistanceSum : the largest M distance sum
    double *maxDistanceSum = (double *)malloc(sizeof(double) * (M + 1));

#pragma omp parallel for shared(maxDistanceSum)
    for (i = 0; i < M; i++)
    {
        maxDistanceSum[i] = 0;
    }
    // maxDisSumPivots : the top M pivots combinations
    int *maxDisSumPivots = (int *)malloc(sizeof(int) * k * (M + 1));
    // danger : 2-dim circle maybe use other parallel function，here may cause wrong

#pragma omp parallel shared(maxDisSumPivots, M, k)
    {
#pragma omp for
        for (i = 0; i < M; i++)
        {
            int ki;
            for (ki = 0; ki < k; ki++)
            {
                maxDisSumPivots[i * k + ki] = 0;
            }
        }
    }

    // minDistanceSum : the smallest M distance sum
    double *minDistanceSum = (double *)malloc(sizeof(double) * (M + 1));
#pragma omp parallel for
    for (i = 0; i < M; i++)
    {
        minDistanceSum[i] = __DBL_MAX__;
    }
    // minDisSumPivots : the bottom M pivots combinations

    int *minDisSumPivots = (int *)malloc(sizeof(int) * k * (M + 1));
#pragma omp parallel shared(minDisSumPivots, i, M, k)
    {
#pragma omp for
        for (i = 0; i < M; i++)
        {
            int ki;
            for (ki = 0; ki < k; ki++)
            {
                minDisSumPivots[i * k + ki] = 0;
            }
        }
    }

    //!===================================================================================================================================//

    // temp : indexes of pivots with dummy array head
    int *temp = (int *)malloc(sizeof(int) * (k + 1));
    temp[0] = -1;
    // Main loop. Combine different pivots with recursive function and evaluate them. Complexity : O( n^(k+2) )
    Combination(0, k, n, dim, M, coord, &temp[1], maxDistanceSum, maxDisSumPivots, minDistanceSum, minDisSumPivots);

    // End timing
    struct timeval end;
    gettimeofday(&end, NULL);

    //********************************************优化时间记时截至到这里，后面的部分不再进行优化************************************************//

    printf("Using time : %f ms\n", (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0);

    // Store the result
    FILE *out = fopen("result.txt", "w");
    for (i = 0; i < M; i++)
    {
        int ki;
        for (ki = 0; ki < k - 1; ki++)
        {
            fprintf(out, "%d ", maxDisSumPivots[i * k + ki]);
        }
        fprintf(out, "%d\n", maxDisSumPivots[i * k + k - 1]);
    }
    for (i = 0; i < M; i++)
    {
        int ki;
        for (ki = 0; ki < k - 1; ki++)
        {
            fprintf(out, "%d ", minDisSumPivots[i * k + ki]);
        }
        fprintf(out, "%d\n", minDisSumPivots[i * k + k - 1]);
    }
    fclose(file);

    // Log
    int ki;
    printf("max : ");
    for (ki = 0; ki < k; ki++)
    {
        printf("%d ", maxDisSumPivots[ki]);
    }
    printf("%lf\n", maxDistanceSum[0]);
    printf("min : ");
    for (ki = 0; ki < k; ki++)
    {
        printf("%d ", minDisSumPivots[ki]);
    }
    printf("%lf\n", minDistanceSum[0]);
    // for(i=0; i<M; i++){
    // int ki;
    // for(ki=0; ki<k; ki++){
    // printf("%d\t", maxDisSumPivots[i*k + ki]);
    // }
    // printf("%lf\n", maxDistanceSum[i]);
    // }
    // for(i=0; i<M; i++){
    // int ki;
    // for(ki=0; ki<k; ki++){
    // printf("%d\t", minDisSumPivots[i*k + ki]);
    // }
    // printf("%lf\n", minDistanceSum[i]);
    // }

    return 0;
}
