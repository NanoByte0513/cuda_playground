#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.blobal, L2::128B[%0], [%1], %2;\n"::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n"::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n"::"n"(N))
