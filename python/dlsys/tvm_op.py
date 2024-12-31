from __future__ import absolute_import, print_function
import tvm
from tvm import te, topi
import numpy as np
# import topi

# Global declarations of environment.

# llvm
tgt_host="llvm"
# llvm, cuda, opencl, metal
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt="llvm"


def make_elemwise_add(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = te.placeholder(shape, dtype=dtype, name="A")
    B = te.placeholder(shape, dtype=dtype, name="B")
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i))

    s = te.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    A = te.placeholder(shape, dtype=dtype, name='A')
    B = te.placeholder(shape, dtype=dtype, name='B')
    C = te.compute(A.shape, lambda *i: A(*i) * B(*i))

    s = te.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_add_by_const(shape, const_k, tgt, tgt_host, func_name,
                               dtype="float32"):
    """TODO: Your code here"""
    A = te.placeholder(shape, dtype=dtype, name='A')
    C = te.compute(A.shape, lambda *i: A(*i) + const_k)

    s = te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul_by_const(shape, const_k, tgt, tgt_host, func_name,
                            dtype="float32"):
    """TODO: Your code here"""
    A = te.placeholder(shape, dtype=dtype, name='A')
    C = te.compute(A.shape, lambda *i: A(*i) * const_k)

    s = te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f

def make_relu(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.max, tvm.const(0, A.dtype)"""
    A = te.placeholder(shape, dtype=dtype, name='A')
    C = te.compute(A.shape, lambda *i: te.max(A(*i), te.const(0, A.dtype)) )

    s = te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f



def make_relu_gradient(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.select"""
    A = te.placeholder(shape, dtype=dtype, name='A')
    B = te.placeholder(shape, dtype=dtype, name='B')
    C = te.compute(A.shape, lambda *i: te.if_then_else(A(*i)>0, te.const(1,A.dtype) * B[i], te.const(0, A.dtype)))

    s = te.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f



def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: treat 4 cases of transposeA, transposeB separately"""
    """Hint: for tvm schedule, use split, reorder, vectorize, parallel"""
    """Hint: debug tvm schedule using tvm.lower"""

    A = te.placeholder(shapeA, dtype=dtype, name='A')
    B = te.placeholder(shapeB, dtype=dtype, name='B')

    if transposeA:
        shapeA = (shapeA[1], shapeA[0])
    if transposeB:
        shapeB = (shapeB[1], shapeB[0])

    assert shapeA[1] == shapeB[0]
    print("shape a 1",shapeA[1], "shapeB 0", shapeB[0])

    k = te.reduce_axis((0, shapeA[1]), name='k')
    if transposeA and transposeB:
        C = te.compute(
                (shapeA[0], shapeB[1]),
                lambda i, j: te.sum(A[k, i] * B(j, k), axis=k),
                name='C'
                )
    elif transposeA  and (transposeB is False):
        C = te.compute(
                (shapeA[0], shapeB[1]),
                lambda i, j: te.sum(A[k, i] * B[k, j], axis=k),
                name='C'
                )
    elif (transposeA is False) and transposeB :
        # print('a shape ', A.shape, 'b shape', B.shape)
        assert(A.shape[1] == B.shape[1])
        C = te.compute(
                (shapeA[0], shapeB[1]),
                lambda i, j: te.sum(A[i, k] * B[j, k], axis=k),
                name='C'
                )
    else:
        C = te.compute(
                (shapeA[0], shapeB[1]),
                lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
                name='C'
                )


    s = te.create_schedule(C.op)

    # here to speed up matrix multiplication
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


    
def make_conv2d(shapeX, shapeF, tgt, tgt_host, func_name, dtype="float32"):
    assert(shapeX[1] == shapeF[1])
    N, C, H, W = shapeX
    M, C, R, S = shapeF

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: go by conv2d definition. Treat stride=1, padding=0 case only."""
    """For a challenge, treat the general case for stride and padding."""
    X = te.placeholder((N, C, H, W), dtype=dtype, name='X')
    F = te.placeholder((M, C, R, S), dtype=dtype, name='F')

    rc = te.reduce_axis((0, C), name='rc')
    rr = te.reduce_axis((0, R), name='rr')
    rs = te.reduce_axis((0, S), name='rs')

    Y = te.compute(
            (N, M, H-R+1, W-S+1),
            lambda n, m, h, w: te.sum(X[n, rc, h+rr, w+rs] * F[m, rc, rr, rs], axis=[rc, rr, rs]),
            name='Y'
            )
    s= te.create_schedule(Y.op)
    # optimize matrix multiplication here. 
    f = tvm.build(s, [X, F, Y], tgt, target_host=tgt_host, name=func_name)
    return f


def make_matrix_softmax(shape, tgt, tgt_host, func_name, dtype="float32"):

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum, tvm.max, tvm.exp"""
    """Hint: do not reuse the same reduction axis j."""
    """Hint: implement the following version for better stability
        e_x = np.exp(x - np.max(x))
        softmax(x)= e_x / e_x.sum()
    """
    input = te.placeholder(shape, dtype=dtype, name='input')
    # output  = te.placeholder(shape, dtype=dtype, name='output')
    _, col_count = shape

    axis_j = te.reduce_axis((0, col_count), name='j')
    axis_k = te.reduce_axis((0, col_count), name='k')
    max_x = te.compute(
            (shape[0], ), 
           lambda i: te.max(input[i,axis_j],axis=axis_j), 
           name= 'max_x')

    e_x = te.compute(
            shape,
            lambda i,j: te.exp(input[i,j ] - max_x[i]),
            name="e_x"
            )
    ex_sum = te.compute(
            (shape[0], ),
            lambda i: te.sum(e_x[i, axis_k], axis=axis_k),
            name='ex_sm'
                        )
    softmax_x = te.compute(
            shape,
            lambda i,j: e_x[i, j] / ex_sum[i],
            name='softmax_x'
            )
    s = te.create_schedule(softmax_x.op)
    f = tvm.build(s, [input,softmax_x], tgt, target_host=tgt_host, name=func_name)
    return f


def make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, func_name,
                                      dtype="float32"):
    """TODO: Your code here"""
    """Hint: output shape should be (1,)"""

    A = te.placeholder(shape, dtype=dtype, name='A')
    A_ = te.placeholder(shape, dtype=dtype, name='A_')

    row, col = shape
    softmax_axis_j = te.reduce_axis((0, col), name='softmax_j')
    softmax_axis_k = te.reduce_axis((0, col), name='softmax_k')
    max_x = te.compute(
            (shape[0], ), 
           lambda i: te.max(A[i,softmax_axis_j],axis=softmax_axis_j), 
           name= 'max_x')

    e_x = te.compute(
            shape,
            lambda i,j: te.exp(A[i,j ] - max_x[i]),
            name="e_x"
            )
    ex_sum = te.compute(
            (shape[0], ),
            lambda i: te.sum(e_x[i, softmax_axis_k], axis=softmax_axis_k),
            name='ex_sm')
    softmax = te.compute(
            shape,
            lambda i,j: e_x[i, j] / ex_sum[i],
            name='softmax_x')
 

    axis_j = te.reduce_axis((0, col), name='j')
    axis_k = te.reduce_axis((0,row), name='k')

    log = te.compute(
            shape,
            lambda i,j: te.log(softmax[i,j]),
            name='log'
            )
    sum_cross_entropy = te.compute(
            (row,),
            # lambda i: te.sum(B[i, axis_j], axis=axis_j ),
            lambda i: te.sum(-A_[i, axis_j] * log[i, axis_j], axis=axis_j),
            name='sum_cross_entropy'
            )

    C = te.compute(
        (1,),
        lambda _: te.sum(sum_cross_entropy[axis_k]/row, axis=axis_k ) ,
        name='C'
        )

    s = te.create_schedule(C.op)
    f = tvm.build(s, [A, A_, C], tgt, target_host=tgt_host, name=func_name)
    return f




def make_reduce_sum_axis_zero(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = te.placeholder(shape, dtype=dtype, name="A")
    C = topi.sum(A, axis=0, keepdims=False)

    # i = te.reduce_axis((0,shape[0]), name='i')
    # Compute sum over the first axis (axis=0) 
    # C = te.compute((shape[1],), lambda j: te.sum(A[i, j], axis=i ), name="C")

    s = te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_broadcast_to(shape, to_shape, tgt, tgt_host, func_name,
                      dtype="float32"):
    A = te.placeholder(shape, dtype=dtype, name="A")
    C = topi.broadcast_to(A, to_shape)

    s = te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_sgd_update(shape, learning_rate, tgt, tgt_host, func_name,
                    dtype="float32"):
    X = tvm.placeholder(shape, dtype=dtype, name="A")
    grad = tvm.placeholder(shape, dtype=dtype, name="grad")
    Y = tvm.compute(shape, lambda *i: X(*i) - learning_rate * grad(*i))

    s = tvm.create_schedule(Y.op)
    f = tvm.build(s, [X, grad, Y], tgt, target_host=tgt_host, name=func_name)
    return f
