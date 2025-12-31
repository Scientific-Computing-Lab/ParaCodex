#include "distance.h"

void distance_device(const double4* loc, double* dist, const int n, const int iteration) {
  
    {
    for (int i = 0; i < iteration; i++) {
            for (int p = 0; p < n; p++) {
        auto ay = loc[p].x * DEGREE_TO_RADIAN;  

        auto ax = loc[p].y * DEGREE_TO_RADIAN;  

        auto by = loc[p].z * DEGREE_TO_RADIAN;  

        auto bx = loc[p].w * DEGREE_TO_RADIAN;  


        

        auto x        = (bx - ax) / 2.0;
        auto y        = (by - ay) / 2.0;
        auto sinysqrd = sin(y) * sin(y);
        auto sinxsqrd = sin(x) * sin(x);
        auto scale    = cos(ay) * cos(by);
        dist[p] = 2.0 * EARTH_RADIUS_KM * asin(sqrt(sinysqrd + sinxsqrd * scale));
      }
    }
  }
}