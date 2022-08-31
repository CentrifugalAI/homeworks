
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

// translate:当组合不同的枢轴时，计算距离的总和
// Calculate sum of distance while combining different pivots. Complexity : O( n^2 )
//! 该函数传入的是所有数据点整合成的一维数组 coord 和支撑点在数据点中的索引数组 pivots
double SumDistance(const int k, const int n, const int dim, double *coord, int *pivots)
{
    double *rebuiltCoord = (double *)malloc(sizeof(double) * n * k);
    int i;
    //! 初始化数组函数，可以进行优化
#pragma omp parallel for num_threads(6) schedule(guided)
    for (i = 0; i < n * k; i++)
    {
        rebuiltCoord[i] = 0;
    }

// Rebuild coordinates. New coordinate of one point is its distance to each pivot.
//*这段代码的作用是根据支撑点计算重建坐标，这里采用欧几里得距离s重建坐标：即 d(x,p_i) = /sqrt(/sum_{j=0}^{dim-1}(x[j]-p_i[j])^2)
//* n 是数据点集的个数
//* k 是支撑点集的个数（支撑点的维度和数据点的维度应该是一致的，都等于度量空间的维度 dim）
//* rebuiltCoord 的涵义是：将所有支撑点和数据点求出距离值（这里是将它整合成一维数组）
//! rebuiltCoord[i * k + ki] 表示第 i 个原始数据点和第 ki 个支撑点之间的距离
//! rebuiltCoord[i * k + ki] 也表示第 i 个重建后的数据点的第 ki 维分量

//! 下面的代码是根据循环变量的索引进行操作，这种根据循环变量的索引进行的操作可以使用 omp for 并行优化：因为它保证了循环的顺序以外的逻辑正常
#pragma omp parallel for num_threads(6) schedule(guided)
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
    //! 下面这个函数的流程是：
    //! 1. 所有数据点之间两两组合，计算任意两个点之间的切比雪夫距离
    //! 2. 如果当前的绝对值之差比最大值大，就取这个新的最大值
    //! 3. 如果当前的绝对值之差比最大值小，将原本的最大值赋值给最大值
    //! 4. 对于任意两两数据点重复 2，3 操作，每次计算出切比雪夫距离，并进行加和

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

    //!改写思路：将 i 和 j 整合成一个变量，将 chebyshevSum 作为线程私有变量，然后借用 reduction 加和

    //     int ij, ki;
    //     double chebyshevSum = 0;
    // #pragma omp parallel for private(ki) reduction(+ \
    //                                            : chebyshevSum)
    //     for (ij = 0; ij < n * n; ij++)
    //     {
    //         // printf("(%d, %d)", i, j);
    //         for (ki = 0; ki < k; ki++)
    //         {
    //             double dis = fabs(rebuiltCoord[ij / n * k + ki] - rebuiltCoord[ij % n * k + ki]);
    //             chebyshevSum = dis > chebyshevSum ? dis : chebyshevSum;
    //         }
    //     }

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
    //! 如果当前程序达到了最后一次迭代
    if (ki == k - 1)
    {
        int i;
        for (i = pivots[ki - 1] + 1; i < n; i++)
        {
            pivots[ki] = i; //! 确认第 ki 个支撑点的索引

            // Calculate sum of distance while combining different pivots.
            double distanceSum = SumDistance(k, n, dim, coord, pivots); //! 求已经确认的 k 个支撑点的切比雪夫距离和

            // put data at the end of array
            maxDistanceSum[M] = distanceSum;
            minDistanceSum[M] = distanceSum;
            int kj;
            //! 将索引记录下来
#pragma omp parallel for num_threads(6) schedule(guided)
            for (kj = 0; kj < k; kj++)
            {
                maxDisSumPivots[M * k + kj] = pivots[kj];
            }
#pragma omp parallel for num_threads(6) schedule(guided)
            for (kj = 0; kj < k; kj++)
            {
                minDisSumPivots[M * k + kj] = pivots[kj];
            }

            //! 对 1000 个最大最小值及其索引进行排序
            //! 排序非常依赖调用顺序，因此应该强调按顺序并行
            int a;
#pragma omp parallel for ordered num_threads(6) schedule(guided)
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
#pragma omp parallel for ordered num_threads(6) schedule(guided)
            for (a = M; a > 0; a--)
            {
                if (minDistanceSum[a] < minDistanceSum[a - 1])
                {
                    //! 交换值
                    double temp = minDistanceSum[a];
                    minDistanceSum[a] = minDistanceSum[a - 1];
                    minDistanceSum[a - 1] = temp;
                    int kj;

                    //! 交换索引
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

    // Recursively call Combination() to combine pivots
    int i;

    //* 这个函数可以进行优化：因为选择
    //!从剩下的点的点选择一个
    for (i = pivots[ki - 1] + 1; i < n; i++)
    {
        pivots[ki] = i;

        //! 开启下一轮递归（选择下一个点）
        Combination(ki + 1, k, n, dim, M, coord, pivots, maxDistanceSum, maxDisSumPivots, minDistanceSum, minDisSumPivots);

        /** Iteration Log : pivots computed, best pivots, max distance sum, min distance sum pivots, min distance sum
        *** You can delete the logging code. **/
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
    omp_set_num_threads(4);
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
    int dim;
    // n : number of points
    int n;
    // k : number of pivots
    int k;

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

    // Start timing
    struct timeval start;
    gettimeofday(&start, NULL);

    // Read Data
    double *coord = (double *)malloc(sizeof(double) * dim * n);
    int i;
    for (i = 0; i < n; i++)
    {
        int j;
        for (j = 0; j < dim; j++)
        {
            fscanf(file, "%lf", &coord[i * dim + j]);
        }
    }
    fclose(file);

    // maxDistanceSum : the largest M distance sum
    double *maxDistanceSum = (double *)malloc(sizeof(double) * (M + 1));
#pragma omp parallel for num_threads(6) schedule(guided)
    for (i = 0; i < M; i++)
    {
        maxDistanceSum[i] = 0;
    }
    // maxDisSumPivots : the top M pivots combinations
    int *maxDisSumPivots = (int *)malloc(sizeof(int) * k * (M + 1));
#pragma omp parallel for num_threads(6) schedule(guided)
    for (i = 0; i < M; i++)
    {
        int ki;
        for (ki = 0; ki < k; ki++)
        {
            maxDisSumPivots[i * k + ki] = 0;
        }
    }
    // minDistanceSum : the smallest M distance sum
    double *minDistanceSum = (double *)malloc(sizeof(double) * (M + 1));
#pragma omp parallel for num_threads(6) schedule(guided)
    for (i = 0; i < M; i++)
    {
        minDistanceSum[i] = __DBL_MAX__;
    }
    // minDisSumPivots : the bottom M pivots combinations
    int *minDisSumPivots = (int *)malloc(sizeof(int) * k * (M + 1));
#pragma omp parallel for num_threads(6) schedule(guided)
    for (i = 0; i < M; i++)
    {
        int ki;
        for (ki = 0; ki < k; ki++)
        {
            minDisSumPivots[i * k + ki] = 0;
        }
    }

    // temp : indexes of pivots with dummy array head
    int *temp = (int *)malloc(sizeof(int) * (k + 1));
    temp[0] = -1;

    // Main loop. Combine different pivots with recursive function and evaluate them. Complexity : O( n^(k+2) )
    Combination(0, k, n, dim, M, coord, &temp[1], maxDistanceSum, maxDisSumPivots, minDistanceSum, minDisSumPivots);

    // End timing
    struct timeval end;
    gettimeofday(&end, NULL);
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
