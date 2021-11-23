# -*- coding:utf-8 -*-
# Author : yxguo
# Data : 2021/5/18 8:41
import time

import numpy as np


def gen_tabel(interval):
    table = dict()
    for i in range(65535):
        table[i] = -1+i*interval
    return table


def compress4(mat, interval):
    q_matrix = np.around((mat + 1) / interval).astype(np.uint8)
    y = q_matrix.shape[1]
    q_matrix1 = q_matrix[:, :int(y/2)]
    q_matrix2 = q_matrix[:, int(y/2):]
    s_q_matrix1 = q_matrix1 << 4
    com = s_q_matrix1 + q_matrix2
    return com


def compress(mat, interval):
    if interval == 2/65535:
        q_matrix = np.around((mat + 1) / interval).astype(np.uint16)
    elif interval == 2/255:
        q_matrix = np.around((mat + 1)/interval).astype(np.uint8)
    elif interval == 2/15:   # 4bit
        q_matrix = np.around((mat + 1) / interval).astype(np.uint8)
        y = q_matrix.shape[1]
        q_matrix1 = q_matrix[:, :int(y / 2)]
        q_matrix2 = q_matrix[:, int(y / 2):]
        s_q_matrix1 = q_matrix1 << 4
        q_matrix = s_q_matrix1 + q_matrix2
    elif interval == 2/3:    #2bit
        q_matrix = np.around((mat + 1) / interval).astype(np.uint8)
        y = q_matrix.shape[1]
        q_matrix1 = q_matrix[:, :int(y / 4)]
        q_matrix2 = q_matrix[:, int(y / 4):int(y / 2)]
        q_matrix3 = q_matrix[:, int(y / 2):int(y*3 / 4)]
        q_matrix4 = q_matrix[:, int(y*3 / 4):]
        s_q_matrix1 = q_matrix1 << 6
        s_q_matrix2 = q_matrix2 << 4
        s_q_matrix3 = q_matrix3 << 2
        q_matrix = s_q_matrix1 + s_q_matrix2 + s_q_matrix3 + q_matrix4
    elif interval == 2:
        q_matrix = np.around((mat + 1) / interval).astype(np.uint8)
    else:
        q_matrix = np.around((mat + 1) / interval).astype(np.uint8)###
    return q_matrix


def decompress(mat, interval):
    if interval == 2/15:
        mat2 = mat & 0xf
        mat1 = mat >> 4
        m = np.hstack((mat1, mat2))
        de_matrix = (-1 + interval * m).astype(np.float32)
    elif interval == 2/3:
        mat1 = mat >> 6
        mat2 = mat & 0x3f
        mat2 = mat2 >>4
        mat3 = mat & 0xf
        mat3 = mat3 >>2
        mat4 = mat & 0x3
        m = np.hstack((mat1, mat2, mat3, mat4))
        de_matrix = (-1 + interval * m).astype(np.float32)
    else:
        de_matrix = (-1 + interval*mat).astype(np.float32)
    return de_matrix
