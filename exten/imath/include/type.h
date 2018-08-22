#ifndef type_h
#define type_h

enum MatType {
    L=0, //下三角矩阵
    U, //上三角矩阵
    S, //对称矩阵
    G, //其它矩阵
};

enum SparseType {
    CSR=0, //
    CSC, //
    BSR, //
    COO, //
};

#endif
