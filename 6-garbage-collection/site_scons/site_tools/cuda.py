"""
SCons.Tool.cuda

CUDA Tool for SCons

"""

import os
import sys
import SCons.Tool
import SCons.Scanner.C
import SCons.Defaults

CUDAScanner = SCons.Scanner.C.CScanner()


def CUDANVCCStaticObjectEmitter(target, source, env):
    tgt, src = SCons.Defaults.StaticObjectEmitter(target, source, env)
    for file in tgt:
        lifile = os.path.splitext(file.rstr())[0] + ".linkinfo"
        env.SideEffect(lifile, file)
        env.Clean(file, lifile)
    return tgt, src


def CUDANVCCSharedObjectEmitter(target, source, env):
    tgt, src = SCons.Defaults.SharedObjectEmitter(target, source, env)
    for file in tgt:
        lifile = os.path.splitext(file.rstr())[0] + ".linkinfo"
        env.SideEffect(lifile, file)
        env.Clean(file, lifile)
    return tgt, src


def generate(env):
    staticObjBuilder, sharedObjBuilder = SCons.Tool.createObjBuilders(env)
    staticObjBuilder.add_action(".cu", "$STATICNVCCCMD")
    staticObjBuilder.add_emitter(".cu", CUDANVCCStaticObjectEmitter)
    sharedObjBuilder.add_action(".cu", "$SHAREDNVCCCMD")
    sharedObjBuilder.add_emitter(".cu", CUDANVCCSharedObjectEmitter)
    SCons.Tool.SourceFileScanner.add_scanner(".cu", CUDAScanner)

    # default compiler
    env["NVCC"] = "nvcc"

    # default flags for the NVCC compiler
    #env["NVCCFLAGS"] = ""
    #env["STATICNVCCFLAGS"] = ""
    #env["SHAREDNVCCFLAGS"] = ""
    # env["ENABLESHAREDNVCCFLAG"] = "-shared"

    # default NVCC commands
    env["STATICNVCCCMD"] = "$NVCC -o $TARGET -c $NVCCFLAGS $STATICNVCCFLAGS $SOURCES"
    env[
        "SHAREDNVCCCMD"
    ] = "$NVCC $CPPPATH -o $TARGET -c $NVCCFLAGS $SHAREDNVCCFLAGS $ENABLESHAREDNVCCFLAG $SOURCES -Xcompiler '$CXXFLAGS'"

    # helpers
    home = os.environ.get("HOME", "")
    programfiles = os.environ.get("PROGRAMFILES", "")
    homedrive = os.environ.get("HOMEDRIVE", "")

    # find CUDA Toolkit path and set CUDA_TOOLKIT_PATH
    try:
        cudaToolkitPath = env["CUDA_TOOLKIT_PATH"]
    except:
        paths = [
            home + "/NVIDIA_CUDA_TOOLKIT",
            home + "/Apps/NVIDIA_CUDA_TOOLKIT",
            home + "/Apps/NVIDIA_CUDA_TOOLKIT",
            home + "/Apps/CudaToolkit",
            home + "/Apps/CudaTK",
            "/opt/cuda",
            "/usr/local/NVIDIA_CUDA_TOOLKIT",
            "/usr/local/CUDA_TOOLKIT",
            "/usr/local/cuda_toolkit",
            "/usr/local/CUDA",
            "/usr/local/cuda",
            "/Developer/NVIDIA CUDA TOOLKIT",
            "/Developer/CUDA TOOLKIT",
            "/Developer/CUDA",
            programfiles + "NVIDIA Corporation/NVIDIA CUDA TOOLKIT",
            programfiles + "NVIDIA Corporation/NVIDIA CUDA",
            programfiles + "NVIDIA Corporation/CUDA TOOLKIT",
            programfiles + "NVIDIA Corporation/CUDA",
            programfiles + "NVIDIA/NVIDIA CUDA TOOLKIT",
            programfiles + "NVIDIA/NVIDIA CUDA",
            programfiles + "NVIDIA/CUDA TOOLKIT",
            programfiles + "NVIDIA/CUDA",
            programfiles + "CUDA TOOLKIT",
            programfiles + "CUDA",
            homedrive + "/CUDA TOOLKIT",
            homedrive + "/CUDA",
        ]
        pathFound = False
        for path in paths:
            if os.path.isdir(path):
                pathFound = True
                print("scons: CUDA Toolkit found in " + path)
                cudaToolkitPath = path
                break
        if not pathFound:
            sys.exit(
                "Cannot find the CUDA Toolkit path. Please modify your SConscript or add the path in cudaenv.py"
            )
    env["CUDA_TOOLKIT_PATH"] = cudaToolkitPath

    # add nvcc to PATH
    env.PrependENVPath("PATH", cudaToolkitPath + "/bin")

    # add required libraries
    # env.Append(CPPPATH=[cudaToolkitPath + "/include"])
    env.Append(
        LIBPATH=[
            cudaToolkitPath + "/lib",
            cudaToolkitPath + "/lib64",
        ]
    )
    env.Append(LIBS=["cudart"])


def exists(env):
    return env.Detect("nvcc")