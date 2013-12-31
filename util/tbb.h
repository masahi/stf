#ifndef TBB_H
#define TBB_H

#include <tbb/mutex.h>

template <typename Func>
void criticalSection(tbb::mutex& mutex, Func f)
{
    tbb::mutex::scoped_lock lock;
    lock.acquire(mutex);
    f();
    lock.release();
}

#endif // TBB_H
