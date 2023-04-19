import os
import rasterio
import numpy as np
from tqdm import tqdm
import time
import scipy.ndimage as ndimage

def fullPython_uniformFilter(imgPath: str,
                             half_size: int,
                             outputImgPath: str) -> np.ndarray:
    """ """

    with rasterio.open( imgPath, "r" ) as imgDataset:
        
        # Read the buffer
        colorBuffer = imgDataset.read()

        start = time.perf_counter()

        # Allocate the output filtered buffer
        outBuffer = np.zeros( colorBuffer.shape, dtype=np.float32 )


        for row in range(colorBuffer.shape[1]):
            for col in range(colorBuffer.shape[2]):
                
                # Define the window
                startX: int = max(0, col - half_size)
                startY : int = max(0, row - half_size)
                endX: int = min(colorBuffer.shape[2] - 1, col + half_size)
                endY: int = min(colorBuffer.shape[1] - 1, row + half_size)
                nbPixels: float = (endY - startY + 1) * (endX - startX + 1)
                
                for b in range(colorBuffer.shape[0]):
                    meanValue: float = 0.0
                    for rr in range(startY, endY + 1):
                        for cc in range(startX, endX + 1):
                            meanValue += colorBuffer[b, rr, cc]
                    outBuffer[b, row, col] = meanValue / nbPixels
        
        end = time.perf_counter()

        exeTime: float = end - start

        outProfile = imgDataset.profile
        outProfile['dtype'] = np.float32

        with rasterio.open(outputImgPath, "w",**outProfile) as out:
            out.write(outBuffer)

        return exeTime
        
import demogreen.cyuniform as cu

def cython_uniformFilter(imgPath: str,
                         half_size: int,
                         outputImgPath: str) -> np.ndarray:
    """ """
    with rasterio.open( imgPath, "r" ) as imgDataset:

        # Read the buffer
        colorBuffer = imgDataset.read()

        start = time.perf_counter()
        outBuffer = cu.cyUniformFilter(colorBuffer,
                                       half_size)
        end = time.perf_counter()

        outProfile = imgDataset.profile
        outProfile['dtype'] = np.float32

        with rasterio.open(outputImgPath, "w",**outProfile) as out:
            out.write(outBuffer)

        return end - start

import pbuniform as pb

def pybind_uniformFilter(imgPath: str,
                         half_size: int,
                         outputImgPath: str) -> np.ndarray:
    """ """
    with rasterio.open( imgPath, "r" ) as imgDataset:

        # Read the buffer
        colorBuffer = (imgDataset.read()).flatten().astype(np.float32)

        start = time.perf_counter()

        # Allocate the output filtered buffer
        outBuffer = np.zeros( colorBuffer.shape, dtype=np.float32 )

        pb.pbApplyUniform(colorBuffer, outBuffer, half_size, imgDataset.height, imgDataset.width, imgDataset.count)

        outBuffer = np.resize(outBuffer, (imgDataset.count, imgDataset.height, imgDataset.width))

        end = time.perf_counter()

        outProfile = imgDataset.profile
        outProfile['dtype'] = np.float32

        with rasterio.open(outputImgPath, "w",**outProfile) as out:
            out.write(outBuffer)
        
        return end - start

def scipy_uniformFilter(imgPath: str,
                        size: int,
                        outputImgPath: str) -> np.ndarray:

    with rasterio.open( imgPath, "r" ) as imgDataset:

        # Read the buffer
        colorBuffer = imgDataset.read().astype(np.float32)

        start = time.perf_counter()

        outBuffer = ndimage.uniform_filter(colorBuffer, size = size)

        end = time.perf_counter()

        outProfile = imgDataset.profile
        outProfile['dtype'] = np.float32

        with rasterio.open(outputImgPath, "w",**outProfile) as out:
            out.write(outBuffer)
        
        return end - start

if __name__ == "__main__":


    imgPath: str = "./data/GF2_Barcelona_41.4158_2.1975.tif"
    outputDir: str = "/work/scratch/lassalp/GreenIT/FormationCythonPyBind/output/"
    half_size: int = 3
    
    nbTries: int = 1

    doFullPython: bool = True
    doCython: bool = True
    doPyBind11: bool = True
    doScipy: bool = True

    if doFullPython:
        print("Full python execution...")
        accTime: float = 0.0
        for i in tqdm(range(nbTries), desc="execution..."):
            accTime += fullPython_uniformFilter(imgPath = imgPath, 
                                                half_size = half_size,
                                                outputImgPath=os.path.join(outputDir, "fullPython_GF2_Barcelona_41.4158_2.1975.tif"))
        accTime /= nbTries

        print("Full python execution " + str(accTime) + " secondes")
    
    if doCython:
        print("Cython execution...")
        accTime: float = 0.0
        for i in tqdm(range(nbTries), desc="execution..."):
            accTime += cython_uniformFilter(imgPath = imgPath, 
                                            half_size = half_size,
                                            outputImgPath=os.path.join(outputDir, "cython_GF2_Barcelona_41.4158_2.1975.tif"))
        accTime /= nbTries

        print("Cython execution " + str(accTime) + " secondes")
    
    if doPyBind11:
        
        print("PyBind11 execution...")
        accTime: float = 0.0
        for i in tqdm(range(nbTries), desc="execution..."):
            accTime += pybind_uniformFilter(imgPath = imgPath, 
                                            half_size = half_size,
                                            outputImgPath=os.path.join(outputDir, "pybind_GF2_Barcelona_41.4158_2.1975.tif"))
        accTime /= nbTries

        print("Pybind11 execution " + str(accTime) + " secondes")
    
    if doScipy:
        print("Scipy uniform filter execution...")
        accTime: float = 0.0
        for i in tqdm(range(nbTries), desc="execution..."):
            accTime += scipy_uniformFilter(imgPath = imgPath, 
                                           size =  2 * half_size + 1,
                                           outputImgPath=os.path.join(outputDir, "scipy_GF2_Barcelona_41.4158_2.1975.tif"))
        accTime /= nbTries

        print("Scipy execution " + str(accTime) + " secondes")

# Benchmark
# [lassalp@node087 democythonpybind11]$ python src/demogreen/demo_green.py
# Full python execution...
# execution...: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [02:52<00:00, 17.22s/it]
# Full python execution 17.201697303168476 secondes
# Cython execution...
# execution...: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 19.37it/s]
# Cython execution 0.04051504284143448 secondes
# PyBind11 execution...
# execution...: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 19.10it/s]
# Pybind11 execution 0.03993625547736883 secondes
# Scipy uniform filter execution...
# execution...: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 30.97it/s]
# Scipy execution 0.018980666436254977 secondes

        