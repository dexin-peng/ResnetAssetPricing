"""Optional CUDA library version probes; safe to import on CPU-only machines."""
import ctypes

def get_lib_version(lib_names,get_version_fn):
    for name in lib_names:
        try:
            lib=ctypes.CDLL(name);version=ctypes.c_int()
            get_version_fn(lib,version)
            return version.value
        except (OSError,AttributeError,RuntimeError):
            continue
    return None

def _get_cublas_version(lib,out):
    handle=ctypes.c_void_p()
    create=lib.cublasCreate_v2;create.argtypes=[ctypes.POINTER(ctypes.c_void_p)];create.restype=ctypes.c_int
    destroy=lib.cublasDestroy_v2;destroy.argtypes=[ctypes.c_void_p];destroy.restype=ctypes.c_int
    version=lib.cublasGetVersion_v2;version.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)];version.restype=ctypes.c_int
    if create(ctypes.byref(handle))!=0:raise RuntimeError('cuBLAS initialization unavailable')
    try:
        if version(handle,ctypes.byref(out))!=0:raise RuntimeError('cuBLAS version unavailable')
    finally:destroy(handle)

def _get_curand_version(lib,out):
    fn=lib.curandGetVersion;fn.argtypes=[ctypes.POINTER(ctypes.c_int)];fn.restype=ctypes.c_int
    if fn(ctypes.byref(out))!=0:raise RuntimeError('cuRAND version unavailable')

cublas_ver=get_lib_version(['libcublas.so','libcublas.so.12','libcublas.so.11'],_get_cublas_version)
curand_ver=get_lib_version(['libcurand.so','libcurand.so.12','libcurand.so.11'],_get_curand_version)
