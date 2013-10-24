
SET(CUDA_NVCC_FLAGS  "-arch=sm_20" "--use_fast_math" "-O3"
                   "--ptxas-options=--verbose" "-keep"  )

ADD_DEFINITIONS(-DCUDA_BUILD -DBT_USE_DOUBLE_PRECISION -DSCAVISLAM_CUDA_SUPPORT)
