# KoTorch-Inference

[![Packagist](https://img.shields.io/badge/PyTorch-1.7.0-red.svg)]()
[![Packagist](https://img.shields.io/badge/pytorch_android-1.7.0-red.svg)]()
[![Packagist](https://img.shields.io/badge/pytorch_android_torchvision-1.7.0-red.svg)]()
[![Packagist](https://img.shields.io/badge/Python-3.6.13-blue.svg)]()

KoTorch-Inference is a pure Kotlin script that provides tensor (up to rank of 8) computation with PyTorch syntax in inference stage. 

For deep learning, several frameworks provide inference mechanism. The developer can deploy the model on the target mobile device. However, most frameworks focus on how to build the AI model, while few consider the flexibility of tensor computation. Inspired the flexibility and convanience of PyTorch, we decide to write the program to tackle this problem. Note that this project is not completed and still under development.

More About KoTorch-Inference
---

At a granular level, KoTorch-Inference is a library that consists of the following features:

### PyTorch First

We inherit most of the API usage from PyTorch. For PyTorch user, you can easily learn, use and migrate KoTorch-Inference to your project. Our goal is provide flexibility when building various deep learning applications. 

### Cross Platform & Cross language

KoTorch-Inference is not designed and target on single language and single platform. Inherit the property of Kotlin, you can use this script in your Java project with no obstacle. For different platform, you can comment the android-specific part and run the code on desktop JVM too.

KoTorch vs PyTorch
---
Unlike PyTorch, we've decided that we won't rewrite the automatic differentiation mechanism (like `torch.autograd`) in this project since we think that training on device is unrealistic currently and bring heavy burden for phone running. For the desktop user, we recommend you to use known popular deep learning framework and train the model in python since both of them have much mature community and maintenance.

You should note that KoTorch-Inference cannot be used independently since it's just tensor computation program with no neural network inference ability. You still need the deep learning framework (e.g. PyTorch Android API) to help you proceeding NN computation.  

Installation
---
Currently we do not have plan to upload Kotlin-Inference to online platform (e.g. Maven Central). You can just copy-and-paste the program under your project. For desktop user, just search and comment the function and related library which contains `Android-specific` note. 

Example
---
```kotlin
import com.project.kotorch as torch
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor

/**
 *  Use KoTorch-Inference with PyTorch
 */
fun classify(bitmap: Bitmap) {
    // (1) Use KoTorch to perform preprocess
    var inputTensor = torch.from_bitmap(bitmap).float() / 255.0f
    inputTensor = torch.stack(arrayListOf(inputTensor[1], inputTensor[2], inputTensor[3]), 0)
    inputTensor = torch.flip(inputTensor, arrayOf(0))   // RGB -> BGR if model is trained via BGR

    // (2) Cast as PyTorch tensor
    var inputBuffer: FloatBuffer = Tensor.allocateFloatBuffer(inputTensor.numel())
    inputBuffer = torch.toBuffer(inputTensor, inputBuffer)
    val input = Tensor.fromBlob(inputBuffer, LongArray(inputTensor.dim()) { i -> inputTensor.shape[i].toLong() })

    // (3) Model inference
    val module: Module = Module.load("THE_PATH_OF_YOUR_MODEL")
    val output = module.forward(IValue.from(input)).toTensor()

    // (4) Get output and perform post-process
    var outputTensor = torch.from_array(output.dataAsFloatArray, arrayOf(1, 1000))
    outputTensor = torch.softmax(outputTensor, 1)      // one-line perform softmax if the op not in model
}
```