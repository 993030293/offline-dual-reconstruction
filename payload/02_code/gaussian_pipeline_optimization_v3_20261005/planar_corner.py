"""Own an isolated planar corner response and preserve the original API fallback."""

from contextlib import contextmanager
import ctypes
from pathlib import Path
import threading

import cv2
import numpy as np

ORIGINAL_CORNER=cv2.goodFeaturesToTrack
PARAMETERS={"maxCorners":800,"qualityLevel":.01,"minDistance":5,"blockSize":5}


class PlanarCornerWorkspace:
    def __init__(self,library):
        self.dll=ctypes.CDLL(str(Path(library).resolve(strict=True)))
        self.dll.planar_corner_avx2_available.argtypes=[]
        self.dll.planar_corner_avx2_available.restype=ctypes.c_int
        self.available=bool(self.dll.planar_corner_avx2_available())
        self.covariance=self.dll.planar_corner_covariance
        self.covariance.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]
        self.covariance.restype=ctypes.c_int
        self.eigen=self.dll.planar_corner_eigen
        self.eigen.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]
        self.eigen.restype=ctypes.c_int
        self.select=self.dll.planar_corner_select
        self.select.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int]*2+[ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)]
        self.select.restype=ctypes.c_int
        self.local=threading.local()
        self.states=[]
        self.lock=threading.Lock()
        self.active_calls=self.native_calls=self.fallback_calls=0
        self.closed=False

    def supported(self,image):
        return (not self.closed and self.available and cv2.__version__=="4.13.0" and cv2.useOptimized()
            and isinstance(image,np.ndarray) and image.ndim==2 and image.dtype==np.uint8
            and 0<image.size<=2**24 and image.flags.c_contiguous)

    def state(self,image):
        state=getattr(self.local,"state",None)
        if state is None:
            state={"shape":None,"calls":0,"planes":{}}
            self.local.state=state
            with self.lock:
                self.states.append(state)
        if state["shape"]!=image.shape:
            state["planes"]={name:np.empty(image.shape,np.float32) for name in ("dx","dy","xx","xy","yy","response","dilated")}
            state["shape"]=image.shape
        return state

    def enter(self):
        with self.lock:
            if self.closed:
                raise RuntimeError("A closed planar corner workspace cannot execute")
            self.active_calls+=1

    def leave(self):
        with self.lock:
            self.active_calls-=1

    def response(self,image):
        if not self.supported(image):
            raise ValueError("The private planar response received an unsupported input")
        self.enter()
        try:
            state=self.state(image)
            planes=state["planes"]
            scale=1./(4*5*255)
            cv2.Sobel(image,cv2.CV_32F,1,0,dst=planes["dx"],ksize=3,scale=scale,borderType=cv2.BORDER_DEFAULT)
            cv2.Sobel(image,cv2.CV_32F,0,1,dst=planes["dy"],ksize=3,scale=scale,borderType=cv2.BORDER_DEFAULT)
            code=self.covariance(*[planes[name].ctypes.data for name in ("dx","dy","xx","xy","yy")],image.size)
            if code:
                raise ValueError("The private planar covariance failed with code "+str(code))
            for name in ("xx","xy","yy"):
                cv2.boxFilter(planes[name],-1,(5,5),dst=planes[name],normalize=False,borderType=cv2.BORDER_DEFAULT)
            code=self.eigen(*[planes[name].ctypes.data for name in ("xx","xy","yy","response")],image.size)
            if code:
                raise ValueError("The private planar eigen response failed with code "+str(code))
            state["calls"]+=1
            return planes["response"]
        finally:
            self.leave()

    def corners(self,image):
        self.enter()
        try:
            eigen=self.response(image)
            _,maximum,_,_=cv2.minMaxLoc(eigen)
            cv2.threshold(eigen,maximum*.01,0,cv2.THRESH_TOZERO,dst=eigen)
            dilated=self.state(image)["planes"]["dilated"]
            cv2.dilate(eigen,None,dst=dilated)
            output=np.empty((800,1,2),np.float32)
            count=ctypes.c_int()
            code=self.select(eigen.ctypes.data,dilated.ctypes.data,image.shape[0],image.shape[1],output.ctypes.data,ctypes.byref(count))
            if code or count.value<0 or count.value>800:
                raise ValueError("The private planar corner selection failed with code "+str(code))
            with self.lock:
                self.native_calls+=1
            return output[:count.value].copy() if count.value else None
        finally:
            self.leave()

    def goodFeaturesToTrack(self,image,*arguments,**keywords):
        if self.closed:
            raise RuntimeError("A closed planar corner interface cannot execute")
        if not arguments and keywords==PARAMETERS and self.supported(image):
            return self.corners(image)
        with self.lock:
            self.fallback_calls+=1
        return ORIGINAL_CORNER(image,*arguments,**keywords)

    def receipt(self):
        return {"closed":self.closed,"DLL_unloaded":self.closed,"AVX2_available":self.available,
            "active_calls":self.active_calls,"native_calls":self.native_calls,"fallback_calls":self.fallback_calls,
            "private_states":len(self.states),"retained_plane_bytes":sum(plane.nbytes for state in self.states for plane in state["planes"].values())}

    def close(self):
        with self.lock:
            if self.closed:
                return
            if self.active_calls:
                raise RuntimeError("The private planar DLL cannot unload before callers drain")
            import _ctypes
            for state in self.states:
                state["planes"].clear()
            self.covariance=self.eigen=self.select=None
            _ctypes.FreeLibrary(self.dll._handle)
            self.dll._handle=0
            self.dll=None
            self.closed=True


class OwnedCVProxy:
    def __init__(self,workspace):
        self.workspace=workspace

    def __getattr__(self,name):
        return self.workspace.goodFeaturesToTrack if name=="goodFeaturesToTrack" else getattr(cv2,name)


@contextmanager
def planar_corner_execution(library):
    import exact_bayer_gray as accepted
    original=accepted.compile_quality
    workspace=PlanarCornerWorkspace(library)
    proxy=OwnedCVProxy(workspace)

    def compile_quality(*parameters):
        implementation=original(*parameters)
        if implementation.__globals__["cv2"] is not cv2:
            raise ValueError("The original isolated quality namespace owner changed")
        implementation.__globals__["cv2"]=proxy
        return implementation

    accepted.compile_quality=compile_quality
    trace={"counted":False}
    try:
        yield trace
    finally:
        changed=accepted.compile_quality is not compile_quality
        accepted.compile_quality=original
        workspace.close()
        trace.update(original_factory_restored=accepted.compile_quality is original,
            original_cv2_corner_unchanged=cv2.goodFeaturesToTrack is ORIGINAL_CORNER,workspace=workspace.receipt())
        if changed:
            raise RuntimeError("Another writer changed the owned planar quality compiler")
