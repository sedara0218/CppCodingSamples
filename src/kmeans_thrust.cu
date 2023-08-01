#include <kmeans.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/sequence.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

/********************* FUNCTORS *********************/
struct printf_labels_per_point_functor
{
    __host__ void operator()(int x) const
    {
        printf(" %d", x);
    }
};

template <typename T>
struct mod_by_n
{
    int n;

    __host__ __device__ T operator()(const T &x) const 
    { 
        return x % n;
    }
};

template <typename T>
struct divide_by_n
{
    int n;

    __host__ __device__ T operator()(const T &x) const 
    { 
        return x / n;
    }
};

template <typename T>
struct divide_by_n_mod_a
{
    int n, a;

    __host__ __device__ T operator()(const T &x) const 
    { 
        return (x / n) % a;
    }
};

struct square_root
{
    __host__ __device__ double operator()(const double &x) const 
    { 
        return sqrt(x);
    }
};

struct tup_diff_square
{
    __host__ __device__ double operator()(const thrust::tuple<double, double> &x) const 
    { 
        return (thrust::get<0>(x) - thrust::get<1>(x)) * (thrust::get<0>(x) - thrust::get<1>(x));
    }
};

struct point_ind_cent_comp_calc_functor
{
    int k, dims;

    __host__ __device__ int operator()(const int &x) const 
    { 
        return (x / (k * dims)) * dims + (x % dims);
    }
};

struct point_cent_pairs_find_min_functor
{

    __host__ __device__ thrust::tuple<int, double> operator()(const thrust::tuple<int, double> &lhs, const thrust::tuple<int, double> &rhs) const 
    { 
        return (thrust::get<1>(lhs) < thrust::get<1>(rhs)) ? lhs : rhs;
    }
};

struct get_elem0_labels_tuple_functor
{
    __host__ __device__ int operator()(const thrust::tuple<int, double> &x) const 
    { 
        return thrust::get<0>(x);
    }    
};

struct get_elem1_labels_tuple_functor
{
    __host__ __device__ int operator()(const thrust::tuple<int, double> &x) const 
    { 
        return thrust::get<1>(x);
    }    
};

struct convert_to_cent_dim_functor
{
    int dims;

    __host__ __device__ int operator()(const int &dim_add, const int &cent_id) const 
    { 
        return dim_add + cent_id * dims;
    }
};

/********************* FUNCTIONS *********************/

thrust::device_vector<int> assign_labels_and_recalc_centroids(int k, int dims, int num_pts, thrust::device_vector<double> &points, thrust::device_vector<double> &centroids)
{
    /****** Set Up Vec Iterators for Comparison ******/

    // Set up point ID vec for comparison
    // thrust::device_vector<int> point_ind(num_pts * dims * k);
    // thrust::sequence(point_ind.begin(), point_ind.end());
    // // k = 3, d = 2 --> <0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, ...>
    // thrust::transform(point_ind.begin(), point_ind.end(), point_ind.begin(), point_ind_cent_comp_calc_functor{k, dims}); 
    // auto points_comp_begin = thrust::make_permutation_iterator(points.begin(), point_ind.begin());
    thrust::counting_iterator<int> point_ind_counter_begin(0);
    // k = 3, d = 2 --> <0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, ...>
    auto points_ind_begin = thrust::make_transform_iterator(point_ind_counter_begin, point_ind_cent_comp_calc_functor{k, dims}); 
    auto points_comp_begin = thrust::make_permutation_iterator(points.begin(), points_ind_begin);
    
    // Set up centroid ID vec for comparison
    // thrust::device_vector<int> cent_ind(k * dims * num_pts);
    // thrust::sequence(cent_ind.begin(), cent_ind.end());
    // // k = 3, d = 2 --> <0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, ...>
    // thrust::transform(cent_ind.begin(), cent_ind.end(), cent_ind.begin(), mod_by_n<int>{k * dims});
    // auto cent_comp_begin = thrust::make_permutation_iterator(centroids.begin(), cent_ind.begin());
    thrust::counting_iterator<int> cent_ind_counter_begin(0);
    // k = 3, d = 2 --> <0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, ...>
    auto cent_ind_begin = thrust::make_transform_iterator(cent_ind_counter_begin, mod_by_n<int>{k * dims});
    auto cent_comp_begin = thrust::make_permutation_iterator(centroids.begin(), cent_ind_begin);

    /****** Euclidean Distance Calc for Comparison ******/

    // Get sq diff of each point to cent comp
    auto tup_each_point_cent_comp_begin = thrust::make_zip_iterator(thrust::make_tuple(points_comp_begin, cent_comp_begin));
    auto sq_diff_each_pt_cent_comp_begin = thrust::make_transform_iterator(tup_each_point_cent_comp_begin, tup_diff_square());

    // Set up key vector for reduce by key to sum up each pt to cent sq diff
    // // Keys = cent id per point (k = 3, d = 2) --> <0, 0, 1, 1, 2, 2, 0, 0, 1, 1, ...>
    // thrust::device_vector<int> keys_pt_cent_comp(k * dims * num_pts);
    // thrust::sequence(keys_pt_cent_comp.begin(), keys_pt_cent_comp.end());
    // thrust::transform(keys_pt_cent_comp.begin(), keys_pt_cent_comp.end(), keys_pt_cent_comp.begin(), divide_by_n_mod_a<int>{dims, k});  
    // Keys = cent id per point (k = 3, d = 2) --> <0, 0, 1, 1, 2, 2, 0, 0, 1, 1, ...>
    thrust::counting_iterator<int> keys_pt_cent_comp_counter_begin(0);
    auto keys_pt_cent_comp_begin = thrust::make_transform_iterator(keys_pt_cent_comp_counter_begin, divide_by_n_mod_a<int>{dims, k});  

    // Keys = cent_id_begin, Values = sq_diff_cent..._begin --> Sum Values where keys are equal (reduce by key)
    thrust::device_vector<int> red_cent_id_dist(num_pts * k);
    //thrust::device_vector<double> red_pt_cent_sum(num_pts * k);
    thrust::device_vector<double> red_pt_cent_dist(num_pts * k);
    // Reduction k=3, d=2 --> keys<SZ_pts*k>=(0,1,2,0,1,2,...) , values=(eu00, eu01, eu02, eu10, eu11, eu12, ...)
    //auto red_key_val_sum_end = thrust::reduce_by_key(thrust::device, keys_pt_cent_comp.begin(), keys_pt_cent_comp.end(), sq_diff_each_pt_cent_comp_begin, red_cent_id.begin(), red_pt_cent_sum.begin());
    auto red_key_val_sum_end = thrust::reduce_by_key(keys_pt_cent_comp_begin, keys_pt_cent_comp_begin + k*dims*num_pts, sq_diff_each_pt_cent_comp_begin, red_cent_id_dist.begin(), red_pt_cent_dist.begin());

    // Get Euclidean Distance by square root of sum ????THE GOAL IS MIN SO DOES SQUARE ROOT ACTUALLY MATTER????
    // thrust::device_vector<double> red_pt_cent_eu_dist(num_pts * k);
    // thrust::transform(red_pt_cent_dist.begin(), red_pt_cent_dist.end(), red_pt_cent_eu_dist.begin(), square_root());

    // Get min distance for each point cent comp
    auto tup_pt_dist_per_cent_begin = thrust::make_zip_iterator(thrust::make_tuple(red_cent_id_dist.begin(), red_pt_cent_dist.begin()));

    // Set up key vector for reduce by key to get min distance per point
    // thrust::device_vector<int> point_cent_ind(num_pts * k);
    // thrust::sequence(point_cent_ind.begin(), point_cent_ind.end());
    // // red_pt_cent_dist k=3, d=2 --> <0,0,0, 1,1,1, 2,2,2, ...>
    // thrust::transform(point_cent_ind.begin(), point_cent_ind.end(), point_cent_ind.begin(), divide_by_n<int>{k});
    thrust::counting_iterator<int> point_cent_ind_counter_begin(0);
    // red_pt_cent_dist k=3, d=2 --> <0,0,0, 1,1,1, 2,2,2, ...>
    auto point_cent_ind_begin = thrust::make_transform_iterator(point_cent_ind_counter_begin, divide_by_n<int>{k});

    // Keys = point_cent_ind, Values = tuple of centroid id and eu dist for that point cent pair --> Return min dist for each pair
    thrust::device_vector<int> red_point_ind(num_pts);
    thrust::device_vector<thrust::tuple<int, double>> red_cent_ind_dist_tup(num_pts);
    thrust::device_vector<int> red_cent_ind(num_pts); // LABEL ASSIGNMENT PER POINT
    // Reduction k=3, d=2 --> keys<num_pts>=(0,1,2,3,4,...), values=(<c_id, dist>, <c_id, dist>, ...)
    auto red_key_val_labels_end = thrust::reduce_by_key(point_cent_ind_begin, point_cent_ind_begin + num_pts*k, tup_pt_dist_per_cent_begin, red_point_ind.begin(), red_cent_ind_dist_tup.begin(), thrust::equal_to<int>(), point_cent_pairs_find_min_functor());
    
    thrust::transform(red_cent_ind_dist_tup.begin(), red_cent_ind_dist_tup.end(), red_cent_ind.begin(), get_elem0_labels_tuple_functor());

    /****** New Centroid Calculation ******/

    // Set up centroid vector such that we can sum all points in the same centroid dimension
    // k=3,d=2 --> centroid 0 id=[0,1], centroid 1 id=[2,3], ... id=[centroid_num * dim, ..., centroid_num * dim + dim-1]
    // thrust::device_vector<int> labels_expand_by_dims_seq (num_pts * dims); 
    // thrust::sequence(labels_expand_by_dims_seq.begin(), labels_expand_by_dims_seq.end());
    // thrust::transform(labels_expand_by_dims_seq.begin(), labels_expand_by_dims_seq.end(), labels_expand_by_dims_seq.begin(), divide_by_n<int>{dims});
    // auto labels_expand_by_dims_begin = thrust::make_permutation_iterator(red_cent_ind.begin(), labels_expand_by_dims_seq.begin()); // each label is added dims times in the vec
    thrust::counting_iterator<int> labels_expand_by_dims_seq_counter_begin (0); 
    auto labels_expand_by_dims_seq_begin = thrust::make_transform_iterator(labels_expand_by_dims_seq_counter_begin, divide_by_n<int>{dims});
    auto labels_expand_by_dims_begin = thrust::make_permutation_iterator(red_cent_ind.begin(), labels_expand_by_dims_seq_begin); // each label is added dims times in the vec

    // thrust::device_vector<int> labels_add_dim_num(num_pts * dims); 
    // thrust::sequence(labels_add_dim_num.begin(), labels_add_dim_num.end());
    // thrust::transform(labels_add_dim_num.begin(), labels_add_dim_num.end(), labels_add_dim_num.begin(), mod_by_n<int>{dims});
    thrust::counting_iterator<int> labels_add_dim_num_counter_begin(0); 
    auto labels_add_dim_num_begin = thrust::make_transform_iterator(labels_add_dim_num_counter_begin, mod_by_n<int>{dims});

    // thrust::device_vector<int> labels_id_by_cent_dim(num_pts * dims); // IDs assigned to uniqify each centroid dimension
    // thrust::transform(labels_add_dim_num.begin(), labels_add_dim_num.end(), labels_expand_by_dims_begin, labels_id_by_cent_dim.begin(), convert_to_cent_dim_functor{dims}); 
    thrust::device_vector<int> labels_id_by_cent_dim(num_pts * dims); // IDs assigned to uniqify each centroid dimension
    thrust::transform(labels_add_dim_num_begin, labels_add_dim_num_begin + num_pts*dims, labels_expand_by_dims_begin, labels_id_by_cent_dim.begin(), convert_to_cent_dim_functor{dims}); 
    
    // Set up point vector as value for future sum reduction 
    thrust::device_vector<double> points_val = points;

    // Sort keys=cent_dim_id vector values=point vector
    thrust::stable_sort_by_key(labels_id_by_cent_dim.begin(), labels_id_by_cent_dim.end(), points_val.begin(), thrust::less<int>());

    // Reduce by key to get a vector of sums
    thrust::device_vector<int> red_label_cent_unique_id(k * dims); 
    thrust::device_vector<double> red_label_sum_point_by_dim(k * dims); // Vector of the sum of points per dimension of centroid vector
    auto red_key_val_newcent_sum_end = thrust::reduce_by_key(labels_id_by_cent_dim.begin(), labels_id_by_cent_dim.end(), points_val.begin(), red_label_cent_unique_id.begin(), red_label_sum_point_by_dim.begin());

    // Get count of each centroid in red_cent_ind
    thrust::device_vector<int> label_cent_ind_sort = red_cent_ind;
    thrust::stable_sort(label_cent_ind_sort.begin(), label_cent_ind_sort.end());

    thrust::device_vector<int> label_cent_count(num_pts, 1); // set 1 for each cent label --> reduce by key will sum each 1 up
    thrust::device_vector<int> red_label_cent_id_count(k); // 0,1,2,3,4...,k
    thrust::device_vector<int> red_label_cent_count(k); // count 0, count 1, count 2, ....
    auto red_key_val_count_end = thrust::reduce_by_key(label_cent_ind_sort.begin(), label_cent_ind_sort.end(), label_cent_count.begin(), red_label_cent_id_count.begin(), red_label_cent_count.begin());

    // Expand count of each centroid into k * dims vec
    thrust::device_vector<int> expand_cent_count_for_div (k * dims); 
    thrust::sequence(expand_cent_count_for_div.begin(), expand_cent_count_for_div.end());
    thrust::transform(expand_cent_count_for_div.begin(), expand_cent_count_for_div.end(), expand_cent_count_for_div.begin(), divide_by_n<int>{dims});
    auto labels_count_expand_by_dims_begin = thrust::make_permutation_iterator(red_label_cent_count.begin(), expand_cent_count_for_div.begin()); // each count is added dims times in the vec

    // Divide each value in new centroid vec by the num of points per centroid
    thrust::device_vector<double> new_centroids (k * dims); 
    thrust::transform(red_label_sum_point_by_dim.begin(), red_label_sum_point_by_dim.end(), labels_count_expand_by_dims_begin, new_centroids.begin(), thrust::divides<double>());

    centroids = new_centroids;

    return red_cent_ind;
}

// Checks if old and cur centroids have converged
bool check_converged_thrust(int k, int dims, thrust::device_vector<double> &old_centroids, thrust::device_vector<double> &centroids, double threshold)
{
    // Set up centroid ID vec
    thrust::device_vector<int> cent_id(k * dims);
    thrust::sequence(cent_id.begin(), cent_id.end());
    thrust::transform(cent_id.begin(), cent_id.end(), cent_id.begin(), divide_by_n<int>{dims});

    // Get sq diff of each point
    auto tup_cent_oldcent_begin = thrust::make_zip_iterator(thrust::make_tuple(centroids.begin(), old_centroids.begin()));
    auto sq_diff_cent_oldcent_begin = thrust::make_transform_iterator(tup_cent_oldcent_begin, tup_diff_square());

    // Keys = cent_id_begin, Values = sq_diff_cent..._begin --> Sum Values where keys are equal (reduce by key)
    thrust::device_vector<int> red_cent_id(k);
    thrust::device_vector<double> red_eu_dist(k);
    auto red_key_val_end = thrust::reduce_by_key(cent_id.begin(), cent_id.end(), sq_diff_cent_oldcent_begin, red_cent_id.begin(), red_eu_dist.begin());

    // Get Euclidean Distance by square root of sum
    thrust::transform(red_eu_dist.begin(), red_eu_dist.end(), red_eu_dist.begin(), square_root());

    // Check if everything is less than threshold
    thrust::stable_sort(red_eu_dist.begin(), red_eu_dist.end());
    auto converge_check = thrust::upper_bound(red_eu_dist.begin(), red_eu_dist.end(), threshold);

    return converge_check == red_eu_dist.end();
}

void compute_kmeans_thrust(struct options_t *args, int num_pts, double* points, double* sol_cent)
{
    // Declare HOST Vars
    int iter = 0;
    double ms_per_iter = 0;

    // Declare GPU Vars
    thrust::device_vector<double> points_g (points, points + (num_pts * args->dims));
    thrust::device_vector<double> old_centroids_g (args->k * args->dims);
    thrust::device_vector<double> centroids_g (args->k * args->dims);
    thrust::device_vector<int> labels (num_pts);

    // Set Random Seed
    kmeans_srand(args->rseed); 

    // Sequential Implementation of Init k random centers
    for (int i = 0; i < args->k; ++i)
    {
        int index = kmeans_rand() % num_pts;

        // copy point to centroids array
        for (int j = 0; j < args->dims; ++j)
            centroids_g[i*args->dims + j] = points_g[index*args->dims + j];
    }

    // LOOP UNTILL CONVERGENCE
    do {
        auto start = std::chrono::high_resolution_clock::now();

        old_centroids_g = centroids_g;

        // Assign New Labels
        labels = assign_labels_and_recalc_centroids(args->k, args->dims, num_pts, points_g, centroids_g);

        //printf("Iter: %d, THRUST ms: %lf\n", iter, milliseconds);

        auto end = std::chrono::high_resolution_clock::now();
        ms_per_iter += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // Update Vars
        iter++;
    } while (!check_converged_thrust(args->k, args->dims, old_centroids_g, centroids_g, args->threshold) && iter < args->max_iter); // NOT COVERGED OR NOT MAX ITER

    /************ PRINT OUTPUT ************/
    // Iterations and Elapsed Time
    printf("%d,%lf\n", iter, ms_per_iter / iter);

    // -c Option Given
    if (args->print_center) {
        for (int clusterId = 0; clusterId < args->k; ++clusterId){
            printf("%d ", clusterId);
            for (int d = 0; d < args->dims; ++d)
                std::cout << centroids_g[clusterId*args->dims + d] << " ";
            printf("\n");
        }
    } else {
    // -c Option Not Given
        printf("clusters:");
        for (int p = 0; p < num_pts; ++p)
            std::cout << " " << labels[p];
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
                    if (abs(sol_cent[j*(args->dims+1) + k] - centroids_g[i*args->dims + k-1]) > EPSILON) {
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
}