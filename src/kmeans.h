#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <chrono>

/*
-k num_cluster: an INTEGER specifying the number of clusters
-d dims: an INTEGER specifying the dimension of the points
-i inputfilename: a STRING specifying the input filename
-m max_num_iter: an INTEGER specifying the maximum number of iterations
-t threshold: a DOUBLE specifying the threshold for convergence test.
-c: a FLAG to control the output of your program. If -c is specified, your program should output the centroids of all clusters. If -c is not specified, your program should output the labels of all points. See details below.
-s seed: an INTEGER specifying the seed for rand(). This is used by the autograder to simplify the correctness checking process. See details below.
-f function_num: an INTEGER specifying the function to implement. 0 -> cpu, 1 -> trust, 2 -> cuda basic, 3 -> cuda shared
-D solution_file: DEBUG option will compare results with provided solution file.
*/

struct options_t {
    int k;
    int dims;
    char *in_file;
    int max_iter;
    double threshold;
    bool print_center;
    int rseed;
    int func;
    char *DEBUG;
};

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand();
void kmeans_srand(unsigned int seed);

double euclidean_distance(int dims, double* pt1, double *pt2);
bool check_converged(int k, int dims, double *old_centroids, double *centroids, double threshold);

void compute_kmeans_cpu(struct options_t *args, int num_pts, double* points, double* sol_cent);
void compute_kmeans_gpu(struct options_t *args, int num_pts, double* points, double* sol_cent, bool gpu_basic);
void compute_kmeans_thrust(struct options_t *args, int num_pts, double* points_arr, double* sol_cent);