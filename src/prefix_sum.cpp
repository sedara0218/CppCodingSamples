#include "prefix_sum.h"
#include "helpers.h"
#include <cmath>
#include <chrono>

using namespace std;

pthread_barrier_t barrier;
spin_barrier s_barrier;

void* compute_prefix_sum(void *a)
{

    prefix_sum_args_t *args = (prefix_sum_args_t *)a;

    /************************
     * Your code here...    *
     * or wherever you like *
     ************************/
    
    // Up-Sweep

    int max_depth = ceil(log2(args->n_vals));
    int wrk_nxt = 2; // 2^d+1
    int wrk_cur = 1; // 2^d
    int op = 0;

    for (int d=0; d<max_depth; d++) {
        
        //auto time1 = std::chrono::high_resolution_clock::now();
        //cout << args->t_id << ": " << d << endl;

        int wrk_p_th = (args->n_vals / wrk_nxt) / args->n_threads;
        int extra = (args->n_vals / wrk_nxt) % args->n_threads;

        int cur_work = wrk_p_th + (args->t_id < extra ? 1 : 0);
        int prev_work = (wrk_p_th * args->t_id) + (args->t_id < extra ? args->t_id : extra);
        if (cur_work > 0) {
            //cout << "up:" << d << " " << args->t_id << " " << prev_work*wrk_nxt << " " << min(args->n_vals, (prev_work + cur_work)*wrk_nxt) << " " << wrk_nxt << endl;
            for (int i=prev_work*wrk_nxt; i<min(args->n_vals, (prev_work + cur_work)*wrk_nxt); i+=wrk_nxt) {
                //if (i + wrk_nxt - 1 >= min(args->n_vals, (prev_work + cur_work)*wrk_nxt) ) cout << "up ERROR" << endl;
                args->input_vals[i + wrk_nxt - 1] = args->op(args->input_vals[i + wrk_nxt - 1], args->input_vals[i + wrk_cur - 1], args->n_loops);
                op++;
                //for (int ii=0; ii<args->n_loops; ii++) args->input_vals[i + wrk_nxt - 1] += args->input_vals[i + wrk_cur - 1];
            }
        }
        wrk_nxt *= 2;
        wrk_cur *= 2;

        //cout << args->t_id << ": " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - time1).count()<< endl;

        // barrier
        if (args->spin) {
            s_barrier.wait(args->t_id);
        } else {
            pthread_barrier_wait(&barrier);
        }
    }
    
    //cout << args->t_id << " up " << op << endl;
    max_depth -= 2;
    wrk_nxt = (int) pow(2.0, max_depth+1);
    wrk_cur = (int) pow(2.0, max_depth);

    // Down-Sweep
    for (int d=max_depth; d>=0; d--) {

        int wrk_p_th = ((int) ceil(args->n_vals / (float) wrk_nxt) - 1) / args->n_threads;
        int extra = ((int) ceil(args->n_vals / (float) wrk_nxt) - 1) % args->n_threads;

        int cur_work = wrk_p_th + (args->t_id < extra ? 1 : 0);
        int prev_work = (wrk_p_th * args->t_id) + (args->t_id < extra ? args->t_id : extra);
        if (cur_work > 0) {
            //cout << "dwn:" << d << " " << args->t_id << " " << prev_work*wrk_nxt << " " << min(args->n_vals, (prev_work + cur_work)*wrk_nxt) << " " << wrk_nxt << endl;
            for (int i=prev_work*wrk_nxt; i<min(args->n_vals, (prev_work + cur_work)*wrk_nxt); i+=wrk_nxt) {
                if (i + wrk_nxt - 1 + wrk_cur >= args->n_vals) break;
                args->input_vals[i + wrk_nxt - 1 + wrk_cur] = args->op(args->input_vals[i + wrk_nxt - 1 + wrk_cur], args->input_vals[i + wrk_nxt - 1], args->n_loops);
                op++;
                //args->input_vals[i + wrk_nxt - 1 + wrk_cur] += args->input_vals[i + wrk_nxt - 1];
            }
        }
        // int st_thread = 0;
        // int thr_work = 0;
        // for (int wrk_i=0; wrk_i<args->n_vals-wrk_nxt; wrk_i+=wrk_nxt) {
        //     if (args->t_id == st_thread) {
        //         //cout << d << " " << wrk_i << " " << wrk_i + wrk_nxt - 1 + wrk_cur << " " << wrk_i + wrk_nxt - 1 << endl;
        //         args->input_vals[wrk_i + wrk_nxt - 1 + wrk_cur] += args->input_vals[wrk_i + wrk_nxt - 1];
        //     }

        //     thr_work++;
        //     if (thr_work == wrkPth) {
        //         if (extra <= 0) {
        //             st_thread = (st_thread+1) % args->n_threads;
        //         } else {
        //             extra--;
        //         }
        //     } else if (thr_work > wrkPth) {
        //         st_thread = (st_thread+1) % args->n_threads;
        //     }
        // }

        wrk_nxt /= 2;
        wrk_cur /= 2;

        //barrier
        if (args->spin) {
            s_barrier.wait(args->t_id);
        } else {
            pthread_barrier_wait(&barrier);
        }

        // DEBUG
        // if (args->t_id == 0) {
        //     for (int i=0; i<args->n_vals; i++) {
        //         cout << args->input_vals[i] << " ";
        //     }
        //     cout << endl;    
            
        //     if (d==0) {
        //     for (int i=0; i<args->n_vals; i++) {
        //         cout << "OUT: " << args->output_vals[i] << " ";
        //     }
        //     cout << endl;         
        //     }
        // }
        // if (args->spin) {
        //     s_barrier.wait(args->t_id);
        // } else {
        //     pthread_barrier_wait(&barrier);
        // }    
    }

    for (int out_i=args->t_id*args->n_vals/args->n_threads; out_i<(args->t_id+1)*args->n_vals/args->n_threads; out_i++) {
        args->output_vals[out_i] = args->input_vals[out_i];
    }

    //cout << args->t_id << " " << op <<endl;
    return 0;
}
