#include <vector>
#include <forest/general.h>
#include <forest/eigen.h>

class IdentityFeature
{
public:
    IdentityFeature(int i,int dim) : 
       index(i),
       feature_dim(dim)
   {
   }

   template <typename T>
   double operator()(const std::vector<T>& v) const { return v[index];}

   template <typename T>
   double operator()(const Vector<T>& v) const { return v(index);}

   const int getFeatureDim() const { return feature_dim; }

private:
   int index;
   int feature_dim;
};


IdentityFeature createFeature(int dim)
{
    return  IdentityFeature(randInt(0,dim), dim);
}
