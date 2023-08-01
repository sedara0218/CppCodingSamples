#include <spin_barrier.h>

using namespace std;

/************************
 * Your code here...    *
 * or wherever you like *
 ************************/

void spin_barrier::init(int num_threads)
{
    n_threads = num_threads;
    COUNTER = 0;
    
    //pthread_spin_init(&fetch_inc_l, PTHREAD_PROCESS_PRIVATE);

    // await_GO_l = (pthread_spinlock_t*) malloc(n_threads * sizeof(pthread_spinlock_t));
    // for (int i=0; i<n_threads; i++) {
    //     pthread_spin_init(&await_GO_l[i], PTHREAD_PROCESS_PRIVATE);
    //     pthread_spin_lock(&await_GO_l[i]);
    // }

    pthread_mutex_init(&fetch_inc_l, NULL);

    await_GO_l = (pthread_mutex_t*) malloc(n_threads * sizeof(pthread_mutex_t));
    for (int i=0; i<n_threads; i++) {
        pthread_mutex_init(&await_GO_l[i], NULL);
        pthread_mutex_lock(&await_GO_l[i]);
    }
}

void spin_barrier::destroy()
{
    free((void *) await_GO_l);
}

int spin_barrier::wait(int id, bool debug)
{
    int counter;

    if (id < 0 || id >= n_threads) return 1; // error

    // fetch and increment
    //pthread_spin_lock(&fetch_inc_l);
    pthread_mutex_lock(&fetch_inc_l);
    counter = COUNTER;
    COUNTER += 1;
    //pthread_spin_unlock(&fetch_inc_l);
    pthread_mutex_unlock(&fetch_inc_l);

    if (counter + 1 == n_threads) {
        if (debug) cout << "if s_b: " << id << " : " << counter << endl;
        COUNTER = 0;
        for (int j=0; j<n_threads; j++) {
            if (j != id) {
                //pthread_spin_unlock(&await_GO_l[j]);
                pthread_mutex_unlock(&await_GO_l[j]);
            }
        }
    } else {
        if (debug) cout << "else s_b: " << id << " : " << counter << endl;
        //pthread_spin_lock(&await_GO_l[id]);
        pthread_mutex_lock(&await_GO_l[id]);
    }

    return 0;
}