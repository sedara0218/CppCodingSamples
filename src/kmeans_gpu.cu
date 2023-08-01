#include <kmeans.h>
#include <cuda.h>
#include <cuda_runtime.h>

/********************* GLOBAL DEFINES *********************/

#define THREADS_PER_BLOCK 32
#define THREADS_PER_BLOCK_SM 128

/********************* KERNEL FUNCTIONS *********************/

__device__ double euclidean_distance_g(int dims, double* pt1, double *pt2)
{
    double sum = 0;
    for (int i = 1; i <= dims; ++i) {
        sum += (pt1[i] - pt2[i])*(pt1[i] - pt2[i]); // positive sum ensured
    }

    return sqrt(sum); // positive distance
}

__global__ void init_k_random_centers(double *centroids, double *points, int rand_index, int dims)
{
    int i = threadIdx.x / (dims+1);
    int j = threadIdx.x % (dims+1);
    
    centroids[i*(dims+1) + j] = points[rand_index*(dims+1) + j];
}

__global__ void assign_labels_g(int k, int dims, int num_points, double *centroids, double *points, double *new_centroids_sum, int *pts_per_label)
{
    // Check if this thread should be doing work
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_points)
        return;

    
    // For this point at index i determine closest centroid
    int cur_label = 0;
    double min_dist = euclidean_distance_g(dims, &points[i*(dims+1)], &centroids[0]);
    for (int j = 1; j < k; ++j) {
        double tmp_dist = euclidean_distance_g(dims, &points[i*(dims+1)], &centroids[j*(dims+1)]);
        if (tmp_dist < min_dist) {
            min_dist = tmp_dist;
            cur_label = j;
        }
    }

    points[i*(dims+1)] = cur_label;

    for (int j = 1; j <= dims; ++j) {
       atomicAdd(&new_centroids_sum[cur_label*dims + j-1], points[i*(dims+1) + j]);
    }
    atomicAdd(&pts_per_label[cur_label], 1);
}

__global__ void assign_labels_g_shm(int k, int dims, int num_points, double *centroids, double *points, double *new_centroids_sum, int *pts_per_label)
{
    // Check if this thread should be doing work
    int i = threadIdx.x + blockIdx.x * blockDim.x; //global index
    if (i >= num_points)
        return;

    extern __shared__ double points_shm[]; // ?Add centroids to shmem --> bank conflicts will happen since each thread will access all centroids
    for (int d = 0; d <= dims; ++d)
        points_shm[threadIdx.x*(dims+1) + d] = points[i*(dims+1) + d];

    __syncthreads();

    // For this point at index i determine closest centroid
    int cur_label = 0;
    double min_dist = euclidean_distance_g(dims, &points_shm[threadIdx.x*(dims+1)], &centroids[0]);
    for (int j = 1; j < k; ++j) {
        double tmp_dist = euclidean_distance_g(dims, &points_shm[threadIdx.x*(dims+1)], &centroids[j*(dims+1)]);
        if (tmp_dist < min_dist) {
            min_dist = tmp_dist;
            cur_label = j;
        }
    }

    points[i*(dims+1)] = cur_label;

    for (int j = 1; j <= dims; ++j) {
       atomicAdd(&new_centroids_sum[cur_label*dims + j-1], points[i*(dims+1) + j]); 
    }
    atomicAdd(&pts_per_label[cur_label], 1);
}

/********************* WRAPPER FUNCTIONS *********************/

void compute_kmeans_gpu(struct options_t *args, int num_pts, double* points, double* sol_cent, bool gpu_basic)
{
    // Declare HOST Vars
    int iter = 0;
    double ms_per_iter = 0;
    double *centroids, *old_centroids;

    // ALLOC Memory on HOST for HOST Vars
    centroids = (double*) malloc(args->k * (args->dims+1) * sizeof(double));
    old_centroids = (double*) malloc(args->k * (args->dims+1) * sizeof(double));

    // Declare GPU Vars
    double *points_g, *centroids_g;

    // ALLOC and COPY Memory on GPU for GPU Vars
    cudaMalloc((void**) &points_g, num_pts * (args->dims+1) * sizeof(double));
    cudaMemcpy(points_g, points, num_pts * (args->dims+1) * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &centroids_g, args->k * (args->dims+1) * sizeof(double));

    // Set Random Seed
    kmeans_srand(args->rseed); 

    // Choose k Random Centers (TODO: Does it make sense to parallelize given K is small?)
    /* // Parallel Implementation of Init k random centers
    cudaEventRecord(start);
    init_k_random_centers<<<1,args->k*(args->dims+1)>>>(centroids_g, points_g, kmeans_rand() % num_pts, args->dims);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf(" *** CUDA execution time for init k random centers: %f *** \n", milliseconds); 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    */

    // Sequential Implementation of Init k random centers
    for (int i = 0; i < args->k; ++i)
    {
        int index = kmeans_rand() % num_pts;

        // copy point to centroids array
        for (int j = 0; j <= args->dims; ++j)
            centroids[i*(args->dims+1) + j] = points[index*(args->dims+1) + j];
    }

    // LOOP UNTILL CONVERGENCE
    do {
        auto start = std::chrono::high_resolution_clock::now();

        // Timers
        float milliseconds = 0;
        cudaEvent_t start_g, stop_g;
        cudaEventCreate(&start_g);
        cudaEventCreate(&stop_g);

        for (int i = 0; i < args->k; ++i) {
            for (int j = 0; j <= args->dims; ++j) {
                old_centroids[i*(args->dims+1) + j] = centroids[i*(args->dims+1) + j];
            }
        }
        
        // New Centroid Calculation Vars
        double *new_centroids_sum = (double*) malloc(args->k * args->dims * sizeof(double));
        int *pts_per_label = (int*) malloc(args->k * sizeof(int));

        double *new_centroids_sum_g;
        int *pts_per_label_g;

        auto end = std::chrono::high_resolution_clock::now();
        ms_per_iter += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        cudaEventRecord(start_g);

        cudaMalloc((void**) &new_centroids_sum_g, args->k * args->dims * sizeof(double));
        cudaMalloc((void**) &pts_per_label_g, args->k * sizeof(int));

        cudaMemset(new_centroids_sum_g, 0, args->k * args->dims * sizeof(double));
        cudaMemset(pts_per_label_g, 0, args->k * sizeof(int));

        // Memcpy centroid h2d
        cudaMemcpy(centroids_g, centroids, args->k * (args->dims+1) * sizeof(double), cudaMemcpyHostToDevice);

        // Assign New Labels
        if (gpu_basic)
            assign_labels_g<<<num_pts / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(args->k, args->dims, num_pts, centroids_g, points_g, new_centroids_sum_g, pts_per_label_g);
        else
            assign_labels_g_shm<<<num_pts / THREADS_PER_BLOCK_SM, THREADS_PER_BLOCK_SM, THREADS_PER_BLOCK_SM*(args->dims+1)*sizeof(double)>>>(args->k, args->dims, num_pts, centroids_g, points_g, new_centroids_sum_g, pts_per_label_g);

        // Determine New Centers
        //cudaMemcpy(centroids, centroids_g, args->k * (args->dims+1) * sizeof(double), cudaMemcpyDeviceToHost);
        //cudaMemcpy(points, points_g, num_pts * (args->dims+1) * sizeof(double), cudaMemcpyDeviceToHost);        
        cudaMemcpy(new_centroids_sum, new_centroids_sum_g, args->k * args->dims * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(pts_per_label, pts_per_label_g, args->k * sizeof(int), cudaMemcpyDeviceToHost);

        cudaEventRecord(stop_g);
        cudaEventSynchronize(stop_g);
        cudaEventElapsedTime(&milliseconds, start_g, stop_g);
        cudaEventDestroy(start_g);
        cudaEventDestroy(stop_g);

        //printf("Iter: %d, CUDA ms: %lf\n", iter, milliseconds);

        ms_per_iter += milliseconds;

        start = std::chrono::high_resolution_clock::now();

        // Calculate new center
        for (int i = 0; i < args->k; ++i) {
            for (int j = 1; j <= args->dims; ++j) {
                centroids[i*(args->dims+1) + j] = new_centroids_sum[i*args->dims + j-1] / pts_per_label[i];
            }
        }

        end = std::chrono::high_resolution_clock::now();
        ms_per_iter += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // Free Data
        free(new_centroids_sum);
        free(pts_per_label);
        cudaFree(new_centroids_sum_g);
        cudaFree(pts_per_label_g);

        // Update Vars
        iter++;
       //if (iter == 1) break;
    } while (!check_converged(args->k, args->dims, old_centroids, centroids, args->threshold) && iter < args->max_iter); // NOT COVERGED OR NOT MAX ITER

    cudaMemcpy(points, points_g, num_pts * (args->dims+1) * sizeof(double), cudaMemcpyDeviceToHost);

    /************ PRINT OUTPUT ************/
    // Iterations and Elapsed Time
    printf("%d,%lf\n", iter, ms_per_iter / iter);

    // -c Option Given
    if (args->print_center) {
        for (int clusterId = 0; clusterId < args->k; ++clusterId){
            printf("%d ", clusterId);
            for (int d = 1; d <= args->dims; ++d)
                printf("%lf ", centroids[clusterId*(args->dims+1) + d]);
            printf("\n");
        }
    } else {
    // -c Option Not Given
        printf("clusters:");
        for (int p = 0; p < num_pts; ++p)
            printf(" %d", (int) points[p*(args->dims+1)]);
        printf("\n");
    }

    // DEBUG: SOLUTION COMPARE
    if (args->DEBUG != NULL) {
        // Compare Solution
        double EPSILON = 0.0001;
        for (int i = 0; i < args->k; ++i) {
            bool match_found = false;
            for (int j = 0; j < args->k; ++j) {
                if (sol_cent[j*(args->dims+1)] != -1) continue;

                int equal = true;
                for (int k = 1; k <= args->dims; ++k) {
                    if (abs(sol_cent[j*(args->dims+1) + k] - centroids[i*(args->dims+1) + k]) > EPSILON) {
                        equal = false;
                        break;
                    }
                }

                if (equal) {
                    match_found = true;
                    sol_cent[j*(args->dims+1)] = i;
                    printf("%d:%d\n", i,j);
                    break;
                }
            }
            if (!match_found) {
                printf("\n!!!SOLUTION MISMATCH!!!.  Could not find match for centroid: %d\n", i);
                break;
            }
        }
    }

    // Free Memory
    // TODO: CUDA FREE & HOST FREE
    free(centroids);
    free(old_centroids);
    cudaFree(points_g);
    cudaFree(centroids_g);
}
