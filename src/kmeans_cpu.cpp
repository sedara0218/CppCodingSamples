#include <kmeans.h>

int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}

void kmeans_srand(unsigned int seed) {
    next = seed;
}

double euclidean_distance(int dims, double* pt1, double *pt2)
{
    double sum = 0;
    for (int i = 1; i <= dims; ++i) {
        sum += (pt1[i] - pt2[i])*(pt1[i] - pt2[i]); // positive sum ensured
    }

    return sqrt(sum); // positive distance
}

// Checks if old and cur centroids have converged
bool check_converged(int k, int dims, double *old_centroids, double *centroids, double threshold)
{
    int converged = 0;
    for (int i = 0; i < k; ++i) {
        if (euclidean_distance(dims, &old_centroids[i*(dims+1)], &centroids[i*(dims+1)]) <= threshold)
            converged++;
    }

    return converged == k;
}

void assign_labels_and_recalc_centroids(int k, int dims, int num_pts, double* points, double* centroids)
{
    // New Centroid Calculation Vars
    double *new_centroids_sum = (double*) malloc(k * dims * sizeof(double));
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < dims; ++j)
            new_centroids_sum[i*dims + j] = 0;
    }

    int *pts_per_label = (int*) malloc(k * sizeof(int));
    for (int i = 0; i < k; ++i)
        pts_per_label[i] = 0;

    // Loop through all points to assign labels
    for (int i = 0; i < num_pts; ++i) {
        int cur_label = 0;
        double min_dist = euclidean_distance(dims, &points[i*(dims+1)], &centroids[0]);
        for (int j = 1; j < k; ++j) {
            double tmp_dist = euclidean_distance(dims, &points[i*(dims+1)], &centroids[j*(dims+1)]);
            if (tmp_dist < min_dist) {
                min_dist = tmp_dist;
                cur_label = j;
            }
        }

        points[i*(dims+1)] = cur_label;

        for (int j = 1; j <= dims; ++j) {
            new_centroids_sum[cur_label*dims + j-1] += points[i*(dims+1) + j];
        }
        pts_per_label[cur_label]++;
    }

    // Calculate new center
    for (int i = 0; i < k; ++i) {
        for (int j = 1; j <= dims; ++j) {
            centroids[i*(dims+1) + j] = new_centroids_sum[i*dims + j-1] / pts_per_label[i];
        }
    }

    // Free Data
    free(new_centroids_sum);
    free(pts_per_label);
}

void compute_kmeans_cpu(struct options_t *args, int num_pts, double* points, double* sol_cent)
{
    // DEBUG: PRINTS EXECUTION TIME
    //auto tmp_start = std::chrono::high_resolution_clock::now();
    //auto tmp_end = std::chrono::high_resolution_clock::now();
    //printf("\n\nDEBUG::Execution time CPU: %lu\n\n", std::chrono::duration_cast<std::chrono::milliseconds>(tmp_end - tmp_start).count());
    // Declare Init Vars
    int iter;
    double *old_centroids, *centroids;
    double ms_per_iter = 0;

    // Init Setup
    iter = 0;
    old_centroids = (double*) malloc (args->k * (args->dims+1) * sizeof(double));
    centroids = (double*) malloc (args->k * (args->dims+1) * sizeof(double));

    // Set Random Seed
    kmeans_srand(args->rseed); 

    // Choose k Random Centers
    for (int i = 0; i < args->k; ++i)
    {
        int index = kmeans_rand() % num_pts;

        // copy point to centroids array
        for (int j = 0; j <= args->dims; ++j) {
            centroids[i*(args->dims+1) + j] = points[index*(args->dims+1) + j];
        }
    }

    // KMEANS CONVERGENCE LOOP
    do {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < args->k; ++i) {
            for (int j = 0; j <= args->dims; ++j) {
                old_centroids[i*(args->dims+1) + j] = centroids[i*(args->dims+1) + j];
            }
        }

        // Assign New Labels && Determine New Centers
        assign_labels_and_recalc_centroids(args->k, args->dims, num_pts, points, centroids);

        iter++;
        auto end = std::chrono::high_resolution_clock::now();
        ms_per_iter += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        // Check Convergence
    } while (!check_converged(args->k, args->dims, old_centroids, centroids, args->threshold) && iter < args->max_iter);

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

    // Free Data
    free(old_centroids);
    free(centroids);
}