#ifndef _SPIN_BARRIER_H
#define _SPIN_BARRIER_H

#include <pthread.h>
#include <iostream>

class spin_barrier {
	private:
		int n_threads;
		int COUNTER;
		//pthread_spinlock_t fetch_inc_l;
		//pthread_spinlock_t* await_GO_l;
		pthread_mutex_t fetch_inc_l;
		pthread_mutex_t* await_GO_l;

	public:
		void init(int num_threads);

		int wait(int id, bool debug = false);

		void destroy();
};

#endif
