#ifndef FEATURE_H
#define FEATURE_H

#include <util.h>

class IdentityFeature
{
public:
   IdentityFeature(int i) : index(i){}

   template <typename T>
   double operator()(const std::vector<T>& v) const {return v[index];}

private:
   int index;
};

IdentityFeature* createFeature(int dim)
{
    return new IdentityFeature(randInt(0, dim));
}

#endif // FEATURE_H
