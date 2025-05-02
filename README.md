# HPP-Workshop
Code for our High Performance Programming Workshop 

## Times for different runs
### Intel i7-12700H (14 cores, 20 logical) and 32 GB RAM
50 runs, 300 nodes:

! SEQUENTIAL PYTHON ! Average time: 743.8 ms  
! SEQUENTIAL PYTHON ! G_1 time: 5.14 ms  
! SEQUENTIAL PYTHON ! G_2 time: 738.65 ms  
! SEQUENTIAL NUMBA  ! Average time: 123.25 ms  
! SEQUENTIAL NUMBA  ! G_1 time: 0.62 ms  
! SEQUENTIAL NUMBA  ! G_2 time: 122.63 ms  
! PARALLEL NUMBA    ! Average time: 16.62 ms  
! PARALLEL NUMBA    ! G_1 time: 0.59 ms  
! PARALLEL NUMBA    ! G_2 time: 16.03 ms  
! MULTIPROCESSING   ! Average time: 490857.56 ms  
! MULTIPROCESSING   ! G_1 time: 3233.47 ms  
! MULTIPROCESSING   ! G_2 time: 487624.09 ms  

1 run, 2000 nodes (NUMBA only! BEWARE: Took me 7 min to generate graphs and 31 GB RAM):

! SEQUENTIAL NUMBA  ! Average time: 5487.17 ms  
! SEQUENTIAL NUMBA  ! G_1 time: 30.47 ms  
! SEQUENTIAL NUMBA  ! G_2 time: 5456.7 ms  
! PARALLEL NUMBA    ! Average time: 588.55 ms  
! PARALLEL NUMBA    ! G_1 time: 4.63 ms  
! PARALLEL NUMBA    ! G_2 time: 583.93 ms  

50 runs, 1500 nodes (NUMBA only!):

! SEQUENTIAL NUMBA  ! Average time: 3127.46 ms  
! SEQUENTIAL NUMBA  ! G_1 time: 16.76 ms  
! SEQUENTIAL NUMBA  ! G_2 time: 3110.7 ms  
! PARALLEL NUMBA    ! Average time: 306.77 ms  
! PARALLEL NUMBA    ! G_1 time: 2.69 ms  
! PARALLEL NUMBA    ! G_2 time: 304.08 ms  

### Intel i7-11700K (8 cores, 16 logical) and 64 GB RAM
1 run, 3000 nodes (NUMBA only! BEWARE: Took me 18 minutes to generate graphs and 48 GB RAM):

! SEQUENTIAL NUMBA  ! Average time: 12754.15 ms  
! SEQUENTIAL NUMBA  ! G_1 time: 82.52 ms  
! SEQUENTIAL NUMBA  ! G_2 time: 12671.63 ms  
! PARALLEL NUMBA    ! Average time: 1436.34 ms  
! PARALLEL NUMBA    ! G_1 time: 7.64 ms  
! PARALLEL NUMBA    ! G_2 time: 1428.7 ms  

50 run, 4500 nodes (NUMBA only! BEWARE: Took me 42 minutes to generate graphs and 63 GB RAM):

! SEQUENTIAL NUMBA  ! Average time: 28704.17 ms  
! SEQUENTIAL NUMBA  ! G_1 time: 155.71 ms  
! SEQUENTIAL NUMBA  ! G_2 time: 28548.46 ms  
! PARALLEL NUMBA    ! Average time: 3399.49 ms  
! PARALLEL NUMBA    ! G_1 time: 20.72 ms  
! PARALLEL NUMBA    ! G_2 time: 3378.76 ms  