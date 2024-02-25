/**
 * @brief Different execution modes for convolution.
 */
enum class ExecutionMode {
    CONSTANT,   /* Parallel execution mode, with the kernel filter on constant memory and the image on shared memory. */
    GLOBAL,     /* Parallel execution mode, with the kernel filter and the image on shared memory. */
    SHARED,     /* Parallel execution mode, with the kernel filter on constant memory and tiling of the image on shared memory. */
    
    SEQUENTIAL  /* Sequential execution mode. */
};