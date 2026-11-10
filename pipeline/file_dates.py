"""Exact Windows creation/access/write FILETIME capture and restoration."""

import ctypes
from ctypes import wintypes
import os
from pathlib import Path


def windows_file_times(path: Path, times: tuple[int, int, int] | None = None) -> tuple[int, int, int]:
    if os.name != "nt":
        raise RuntimeError("Preserving creation dates for folder batches requires Windows.")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    get = kernel.GetFileTime
    get.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 3
    get.restype = wintypes.BOOL
    set_time = kernel.SetFileTime
    set_time.argtypes = get.argtypes
    set_time.restype = wintypes.BOOL
    # FILE_READ_ATTRIBUTES / FILE_WRITE_ATTRIBUTES; share read/write/delete.
    handle = create(str(path.resolve()), 0x80 if times is None else 0x100, 7, None, 3, 0, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    values = [wintypes.FILETIME() for _ in range(3)]
    try:
        if times is None:
            success = get(handle, *(ctypes.byref(value) for value in values))
        else:
            for value, ticks in zip(values, times):
                value.dwLowDateTime = ticks & 0xFFFFFFFF
                value.dwHighDateTime = ticks >> 32
            success = set_time(handle, *(ctypes.byref(value) for value in values))
        if not success:
            raise ctypes.WinError(ctypes.get_last_error())
        return tuple((value.dwHighDateTime << 32) | value.dwLowDateTime for value in values)
    finally:
        close(handle)