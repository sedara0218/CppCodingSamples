#pragma once

#include <stdlib.h>
#include <pthread.h>
#include <spin_barrier.h>
#include <iostream>

extern pthread_barrier_t barrier;
extern spin_barrier s_barrier;

void* compute_prefix_sum(void* a);
