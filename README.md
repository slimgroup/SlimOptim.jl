# SLimOptim

PAckage of optimizations functions forl arge scale inversion. In these implementation, the algorithm itself is 
not optimized fot speed as this oackage is designed for inverse problems where the function evaluation is the main cost (~hours for a single function + gradient evaluation) making the algorithm speed minimal.

# References

This package implements adapatations of `minConf_SPG` and `minConf_PQN` from the matlab implementation of M. Schmidt [1].