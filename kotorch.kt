package KoTorch

/**
 *  The source code of Kotlin-Torch Inference (KoTorch Inference)
 *  This package can provide PyTorch-like (PyTorchic) style to perform tensor computation
 *  You can simply define tensor via your multi-dimensional array direcly,
 *  and perform fundamental operator just like PyTorch
 *  Moreover, this package can be mutually used with TensorFlow lite and PyTorch torchscript model
 *  We current support up to 8-dimensional tensor computation
 *
 *  Note:
 *      (1) In this package, we don't provide neural network API such as convolution or recurrent layer
 *          The goal of KoTorch Inference is making tensor computation more easiler "in pre- or post-process"
 *          We believe that using TensorFlow or PyTorch to construct NN is much better choice
 *          (Since the community of both framework are much well and complete)
 *      (2) Another similar library called multik which is also a multi-dimensional array library
 *          We also agree multik is a great library to perform array computation
 *          We aim to provide PyTorch-like API while multik provide numpy-like API
 *      (3) Similar to PyTorch, most API are perform in-place mechanism
 *          You should call torch.copy to remain the tensor manually
 *          However, the memory complexity is not optimized and we left the memory optimization in the future
 *      (4) We don't provide doc and detail tutorial to introduce how to use KoTorch inference
 *          since the usage is almost the same as PyTorch
 *          Feel free to use PyTorch coding style to construct KoTorch tensor :-)
 *
 *  @author SunnerLi
 */

import android.graphics.Bitmap          // [Android-specific]   Comment if if you want to use in non-android environment
import org.pytorch.Tensor               // [Android-specific]   Comment if if you want to use in non-android environment
import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.io.BufferedReader
import java.io.File
import java.io.PrintWriter
import java.lang.Math.min
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.util.*

class KoTorch {
    class RuntimeError(message: String): Exception(message)
    class IndexError(message: String): Exception(message)
    class ValueError(message: String): Exception(message)

    enum class TensorType {
        FloatTensor, DoubleTensor, LongTensor
    }

    enum class BufferType {
        FloatBuffer, ByteBuffer
    }

    /**
     *  Define Tensor object
     *  Ref: https://github.com/pytorch/pytorch/blob/v0.1.1/torch/lib/TH/generic/THTensor.h#L11-L15
     *
     *  Rather than split as THStorage and THTensor to perform memory menagement,
     *  We direcly use single class to deal with the tensor since pointer is not support in Kotlin
     *  Similarly, we follow the idea of PyTorch and use 1-dimentional array to stimulate multi-dimentional tensor
     */
    class TorchTensor {
        /**
         *  Define tensor fundamental elements
         *  Rather than original variable, we define additional two parameters: nElements and dtype
         *  nElements can help use skip computing the number of elements in single tensor repeatly
         *  dtype can perform generic-like coding style in different operator
         */
        private var storage: MutableList<Number>
        private var ndimension: Int = 0
        private var nElements: Int = 0
        var shape: MutableList<Int>
        var stride: MutableList<Int>
        var dtype = TensorType.DoubleTensor

        /**
         *  ================================ Constructors (Number) ==============================
         *
         *  The reason that we don't use generic mechanism to write this part is:
         *  The common format of TensorFlow-lite or PyTorch Android inference is simply multi-dimenaional array
         *  And don't want the user to transform as generic array manually
         */
        constructor(array: Array<Number>, size: IntArray) {
            this.shape = size.toMutableList()
            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                val aIdx = idx1
                this.storage.add(aIdx, array[idx1])
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
        }

        constructor(array: Array<Number>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                val aIdx = idx1
                this.storage.add(aIdx, array[idx1])
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
        }

        constructor(array: Array<Array<Number>>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    val aIdx = idx1 * this.shape[1] + idx2
                    this.storage.add(aIdx, array[idx1][idx2])
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
        }

        constructor(array: Array<Array<Array<Number>>>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)
            this.shape.add(array[0][0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    for (idx3 in 0 until this.shape[2]) {
                        val aIdx = idx1 * this.shape[1] * this.shape[2]
                        + idx2 * this.shape[2]
                        + idx3
                        this.storage.add(aIdx, array[idx1][idx2][idx3])
                    }
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
        }

        constructor(array: Array<Array<Array<Array<Number>>>>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)
            this.shape.add(array[0][0].size)
            this.shape.add(array[0][0][0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    for (idx3 in 0 until this.shape[2]) {
                        for (idx4 in 0 until this.shape[3]) {
                            val aIdx = idx1 * this.shape[1] * this.shape[2] * this.shape[3]
                            + idx2 * this.shape[2] * this.shape[3]
                            + idx3 * this.shape[3]
                            + idx4
                            this.storage.add(aIdx, array[idx1][idx2][idx3][idx4])
                        }
                    }
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
        }

        constructor(array: Array<Array<Array<Array<Array<Number>>>>>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)
            this.shape.add(array[0][0].size)
            this.shape.add(array[0][0][0].size)
            this.shape.add(array[0][0][0][0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    for (idx3 in 0 until this.shape[2]) {
                        for (idx4 in 0 until this.shape[3]) {
                            for (idx5 in 0 until this.shape[4]) {
                                val aIdx = idx1 * this.shape[1] * this.shape[2] * this.shape[3] * this.shape[4]
                                + idx2 * this.shape[2] * this.shape[3] * this.shape[4]
                                + idx3 * this.shape[3] * this.shape[4]
                                + idx4 * this.shape[4]
                                + idx5
                                this.storage.add(aIdx, array[idx1][idx2][idx3][idx4][idx5])
                            }
                        }
                    }
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
        }

        constructor(array: Array<Array<Array<Array<Array<Array<Number>>>>>>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)
            this.shape.add(array[0][0].size)
            this.shape.add(array[0][0][0].size)
            this.shape.add(array[0][0][0][0].size)
            this.shape.add(array[0][0][0][0][0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    for (idx3 in 0 until this.shape[2]) {
                        for (idx4 in 0 until this.shape[3]) {
                            for (idx5 in 0 until this.shape[4]) {
                                for (idx6 in 0 until this.shape[5]) {
                                    val aIdx = idx1 * this.shape[1] * this.shape[2] * this.shape[3] * this.shape[4] * this.shape[5]
                                    + idx2 * this.shape[2] * this.shape[3] * this.shape[4] * this.shape[5]
                                    + idx3 * this.shape[3] * this.shape[4] * this.shape[5]
                                    + idx4 * this.shape[4] * this.shape[5]
                                    + idx5 * this.shape[5]
                                    + idx6
                                    this.storage.add(aIdx, array[idx1][idx2][idx3][idx4][idx5][idx6])
                                }
                            }
                        }
                    }
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
        }

        constructor(array: Array<Array<Array<Array<Array<Array<Array<Number>>>>>>>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)
            this.shape.add(array[0][0].size)
            this.shape.add(array[0][0][0].size)
            this.shape.add(array[0][0][0][0].size)
            this.shape.add(array[0][0][0][0][0].size)
            this.shape.add(array[0][0][0][0][0][0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    for (idx3 in 0 until this.shape[2]) {
                        for (idx4 in 0 until this.shape[3]) {
                            for (idx5 in 0 until this.shape[4]) {
                                for (idx6 in 0 until this.shape[5]) {
                                    for (idx7 in 0 until this.shape[6]) {
                                        val aIdx = idx1 * this.shape[1] * this.shape[2] * this.shape[3] * this.shape[4] * this.shape[5] * this.shape[6]
                                        + idx2 * this.shape[2] * this.shape[3] * this.shape[4] * this.shape[5] * this.shape[6]
                                        + idx3 * this.shape[3] * this.shape[4] * this.shape[5] * this.shape[6]
                                        + idx4 * this.shape[4] * this.shape[5] * this.shape[6]
                                        + idx5 * this.shape[5] * this.shape[6]
                                        + idx6 * this.shape[6]
                                        + idx7
                                        this.storage.add(aIdx, array[idx1][idx2][idx3][idx4][idx5][idx6][idx7])
                                    }
                                }
                            }
                        }
                    }
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
        }

        constructor(array: Array<Array<Array<Array<Array<Array<Array<Array<Number>>>>>>>>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)
            this.shape.add(array[0][0].size)
            this.shape.add(array[0][0][0].size)
            this.shape.add(array[0][0][0][0].size)
            this.shape.add(array[0][0][0][0][0].size)
            this.shape.add(array[0][0][0][0][0][0].size)
            this.shape.add(array[0][0][0][0][0][0][0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    for (idx3 in 0 until this.shape[2]) {
                        for (idx4 in 0 until this.shape[3]) {
                            for (idx5 in 0 until this.shape[4]) {
                                for (idx6 in 0 until this.shape[5]) {
                                    for (idx7 in 0 until this.shape[6]) {
                                        for (idx8 in 0 until this.shape[7]) {
                                            val aIdx = idx1 * this.shape[1] * this.shape[2] * this.shape[3] * this.shape[4] * this.shape[5] * this.shape[6] * this.shape[7]
                                            + idx2 * this.shape[2] * this.shape[3] * this.shape[4] * this.shape[5] * this.shape[6] * this.shape[7]
                                            + idx3 * this.shape[3] * this.shape[4] * this.shape[5] * this.shape[6] * this.shape[7]
                                            + idx4 * this.shape[4] * this.shape[5] * this.shape[6] * this.shape[7]
                                            + idx5 * this.shape[5] * this.shape[6] * this.shape[7]
                                            + idx6 * this.shape[6] * this.shape[7]
                                            + idx7 * this.shape[7]
                                            + idx8
                                            this.storage.add(aIdx, array[idx1][idx2][idx3][idx4][idx5][idx6][idx7][idx8])
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
        }

        /**
         *  ================================ Constructors (Float) ==============================
         */
        constructor(array: FloatArray, size: IntArray) {
            this.shape = size.toMutableList()
            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                val aIdx = idx1
                this.storage.add(aIdx, array[idx1])
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.FloatTensor
        }

        constructor(array: FloatArray) {
            this.shape = mutableListOf()
            this.shape.add(array.size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                val aIdx = idx1
                this.storage.add(aIdx, array[idx1])
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.FloatTensor
        }

        constructor(array: Array<FloatArray>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    val aIdx = idx1 * this.shape[1] + idx2
                    this.storage.add(aIdx, array[idx1][idx2])
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.FloatTensor
        }

        constructor(array: Array<Array<FloatArray>>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)
            this.shape.add(array[0][0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    for (idx3 in 0 until this.shape[2]) {
                        val aIdx = idx1 * this.shape[1] * this.shape[2] +
                                idx2 * this.shape[2] +
                                idx3
                        this.storage.add(aIdx, array[idx1][idx2][idx3])
                    }
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.FloatTensor
        }

        constructor(array: Array<Array<Array<FloatArray>>>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)
            this.shape.add(array[0][0].size)
            this.shape.add(array[0][0][0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    for (idx3 in 0 until this.shape[2]) {
                        for (idx4 in 0 until this.shape[3]) {
                            val aIdx = idx1 * this.shape[1] * this.shape[2] * this.shape[3] +
                                    idx2 * this.shape[2] * this.shape[3] +
                                    idx3 * this.shape[3] +
                                    idx4
                            this.storage.add(aIdx, array[idx1][idx2][idx3][idx4])
                        }
                    }
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.FloatTensor
        }

        constructor(array: Array<Array<Array<Array<FloatArray>>>>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)
            this.shape.add(array[0][0].size)
            this.shape.add(array[0][0][0].size)
            this.shape.add(array[0][0][0][0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    for (idx3 in 0 until this.shape[2]) {
                        for (idx4 in 0 until this.shape[3]) {
                            for (idx5 in 0 until this.shape[4]) {
                                val aIdx = idx1 * this.shape[1] * this.shape[2] * this.shape[3] * this.shape[4] +
                                        idx2 * this.shape[2] * this.shape[3] * this.shape[4] +
                                        idx3 * this.shape[3] * this.shape[4] +
                                        idx4 * this.shape[4] +
                                        idx5
                                this.storage.add(aIdx, array[idx1][idx2][idx3][idx4][idx5])
                            }
                        }
                    }
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.FloatTensor
        }

        constructor(array: Array<Array<Array<Array<Array<FloatArray>>>>>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)
            this.shape.add(array[0][0].size)
            this.shape.add(array[0][0][0].size)
            this.shape.add(array[0][0][0][0].size)
            this.shape.add(array[0][0][0][0][0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    for (idx3 in 0 until this.shape[2]) {
                        for (idx4 in 0 until this.shape[3]) {
                            for (idx5 in 0 until this.shape[4]) {
                                for (idx6 in 0 until this.shape[5]) {
                                    val aIdx = idx1 * this.shape[1] * this.shape[2] * this.shape[3] * this.shape[4] * this.shape[5] +
                                            idx2 * this.shape[2] * this.shape[3] * this.shape[4] * this.shape[5] +
                                            idx3 * this.shape[3] * this.shape[4] * this.shape[5] +
                                            idx4 * this.shape[4] * this.shape[5] +
                                            idx5 * this.shape[5] +
                                            idx6
                                    this.storage.add(aIdx, array[idx1][idx2][idx3][idx4][idx5][idx6])
                                }
                            }
                        }
                    }
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.FloatTensor
        }

        constructor(array: Array<Array<Array<Array<Array<Array<FloatArray>>>>>>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)
            this.shape.add(array[0][0].size)
            this.shape.add(array[0][0][0].size)
            this.shape.add(array[0][0][0][0].size)
            this.shape.add(array[0][0][0][0][0].size)
            this.shape.add(array[0][0][0][0][0][0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    for (idx3 in 0 until this.shape[2]) {
                        for (idx4 in 0 until this.shape[3]) {
                            for (idx5 in 0 until this.shape[4]) {
                                for (idx6 in 0 until this.shape[5]) {
                                    for (idx7 in 0 until this.shape[6]) {
                                        val aIdx = idx1 * this.shape[1] * this.shape[2] * this.shape[3] * this.shape[4] * this.shape[5] * this.shape[6] +
                                                idx2 * this.shape[2] * this.shape[3] * this.shape[4] * this.shape[5] * this.shape[6] +
                                                idx3 * this.shape[3] * this.shape[4] * this.shape[5] * this.shape[6] +
                                                idx4 * this.shape[4] * this.shape[5] * this.shape[6] +
                                                idx5 * this.shape[5] * this.shape[6] +
                                                idx6 * this.shape[6] +
                                                idx7
                                        this.storage.add(aIdx, array[idx1][idx2][idx3][idx4][idx5][idx6][idx7])
                                    }
                                }
                            }
                        }
                    }
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.FloatTensor
        }

        constructor(array: Array<Array<Array<Array<Array<Array<Array<FloatArray>>>>>>>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)
            this.shape.add(array[0][0].size)
            this.shape.add(array[0][0][0].size)
            this.shape.add(array[0][0][0][0].size)
            this.shape.add(array[0][0][0][0][0].size)
            this.shape.add(array[0][0][0][0][0][0].size)
            this.shape.add(array[0][0][0][0][0][0][0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    for (idx3 in 0 until this.shape[2]) {
                        for (idx4 in 0 until this.shape[3]) {
                            for (idx5 in 0 until this.shape[4]) {
                                for (idx6 in 0 until this.shape[5]) {
                                    for (idx7 in 0 until this.shape[6]) {
                                        for (idx8 in 0 until this.shape[7]) {
                                            val aIdx = idx1 * this.shape[1] * this.shape[2] * this.shape[3] * this.shape[4] * this.shape[5] * this.shape[6] * this.shape[7] +
                                                    idx2 * this.shape[2] * this.shape[3] * this.shape[4] * this.shape[5] * this.shape[6] * this.shape[7] +
                                                    idx3 * this.shape[3] * this.shape[4] * this.shape[5] * this.shape[6] * this.shape[7] +
                                                    idx4 * this.shape[4] * this.shape[5] * this.shape[6] * this.shape[7] +
                                                    idx5 * this.shape[5] * this.shape[6] * this.shape[7] +
                                                    idx6 * this.shape[6] * this.shape[7] +
                                                    idx7 * this.shape[7] +
                                                    idx8
                                            this.storage.add(aIdx, array[idx1][idx2][idx3][idx4][idx5][idx6][idx7][idx8])
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.FloatTensor
        }

        /**
         *  ================================ Constructors (Double) ==============================
         */
        constructor(array: DoubleArray, size: IntArray) {
            this.shape = size.toMutableList()
            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                val aIdx = idx1
                this.storage.add(aIdx, array[idx1])
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.DoubleTensor
        }

        constructor(array: DoubleArray) {
            this.shape = mutableListOf()
            this.shape.add(array.size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                val aIdx = idx1
                this.storage.add(aIdx, array[idx1])
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.DoubleTensor
        }

        constructor(array: Array<DoubleArray>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    val aIdx = idx1 * this.shape[1] + idx2
                    this.storage.add(aIdx, array[idx1][idx2])
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.DoubleTensor
        }

        constructor(array: Array<Array<DoubleArray>>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)
            this.shape.add(array[0][0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    for (idx3 in 0 until this.shape[2]) {
                        val aIdx = idx1 * this.shape[1] * this.shape[2] +
                                idx2 * this.shape[2] +
                                idx3
                        this.storage.add(aIdx, array[idx1][idx2][idx3])
                    }
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.DoubleTensor
        }

        constructor(array: Array<Array<Array<DoubleArray>>>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)
            this.shape.add(array[0][0].size)
            this.shape.add(array[0][0][0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    for (idx3 in 0 until this.shape[2]) {
                        for (idx4 in 0 until this.shape[3]) {
                            val aIdx = idx1 * this.shape[1] * this.shape[2] * this.shape[3] +
                                    idx2 * this.shape[2] * this.shape[3] +
                                    idx3 * this.shape[3] +
                                    idx4
                            this.storage.add(aIdx, array[idx1][idx2][idx3][idx4])
                        }
                    }
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.DoubleTensor
        }

        constructor(array: Array<Array<Array<Array<DoubleArray>>>>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)
            this.shape.add(array[0][0].size)
            this.shape.add(array[0][0][0].size)
            this.shape.add(array[0][0][0][0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    for (idx3 in 0 until this.shape[2]) {
                        for (idx4 in 0 until this.shape[3]) {
                            for (idx5 in 0 until this.shape[4]) {
                                val aIdx = idx1 * this.shape[1] * this.shape[2] * this.shape[3] * this.shape[4] +
                                        idx2 * this.shape[2] * this.shape[3] * this.shape[4] +
                                        idx3 * this.shape[3] * this.shape[4] +
                                        idx4 * this.shape[4] +
                                        idx5
                                this.storage.add(aIdx, array[idx1][idx2][idx3][idx4][idx5])
                            }
                        }
                    }
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.DoubleTensor
        }

        constructor(array: Array<Array<Array<Array<Array<DoubleArray>>>>>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)
            this.shape.add(array[0][0].size)
            this.shape.add(array[0][0][0].size)
            this.shape.add(array[0][0][0][0].size)
            this.shape.add(array[0][0][0][0][0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    for (idx3 in 0 until this.shape[2]) {
                        for (idx4 in 0 until this.shape[3]) {
                            for (idx5 in 0 until this.shape[4]) {
                                for (idx6 in 0 until this.shape[5]) {
                                    val aIdx = idx1 * this.shape[1] * this.shape[2] * this.shape[3] * this.shape[4] * this.shape[5] +
                                            idx2 * this.shape[2] * this.shape[3] * this.shape[4] * this.shape[5] +
                                            idx3 * this.shape[3] * this.shape[4] * this.shape[5] +
                                            idx4 * this.shape[4] * this.shape[5] +
                                            idx5 * this.shape[5] +
                                            idx6
                                    this.storage.add(aIdx, array[idx1][idx2][idx3][idx4][idx5][idx6])
                                }
                            }
                        }
                    }
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.DoubleTensor
        }

        constructor(array: Array<Array<Array<Array<Array<Array<DoubleArray>>>>>>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)
            this.shape.add(array[0][0].size)
            this.shape.add(array[0][0][0].size)
            this.shape.add(array[0][0][0][0].size)
            this.shape.add(array[0][0][0][0][0].size)
            this.shape.add(array[0][0][0][0][0][0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    for (idx3 in 0 until this.shape[2]) {
                        for (idx4 in 0 until this.shape[3]) {
                            for (idx5 in 0 until this.shape[4]) {
                                for (idx6 in 0 until this.shape[5]) {
                                    for (idx7 in 0 until this.shape[6]) {
                                        val aIdx = idx1 * this.shape[1] * this.shape[2] * this.shape[3] * this.shape[4] * this.shape[5] * this.shape[6] +
                                                idx2 * this.shape[2] * this.shape[3] * this.shape[4] * this.shape[5] * this.shape[6] +
                                                idx3 * this.shape[3] * this.shape[4] * this.shape[5] * this.shape[6] +
                                                idx4 * this.shape[4] * this.shape[5] * this.shape[6] +
                                                idx5 * this.shape[5] * this.shape[6] +
                                                idx6 * this.shape[6] +
                                                idx7
                                        this.storage.add(aIdx, array[idx1][idx2][idx3][idx4][idx5][idx6][idx7])
                                    }
                                }
                            }
                        }
                    }
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.DoubleTensor
        }

        constructor(array: Array<Array<Array<Array<Array<Array<Array<DoubleArray>>>>>>>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)
            this.shape.add(array[0][0].size)
            this.shape.add(array[0][0][0].size)
            this.shape.add(array[0][0][0][0].size)
            this.shape.add(array[0][0][0][0][0].size)
            this.shape.add(array[0][0][0][0][0][0].size)
            this.shape.add(array[0][0][0][0][0][0][0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    for (idx3 in 0 until this.shape[2]) {
                        for (idx4 in 0 until this.shape[3]) {
                            for (idx5 in 0 until this.shape[4]) {
                                for (idx6 in 0 until this.shape[5]) {
                                    for (idx7 in 0 until this.shape[6]) {
                                        for (idx8 in 0 until this.shape[7]) {
                                            val aIdx = idx1 * this.shape[1] * this.shape[2] * this.shape[3] * this.shape[4] * this.shape[5] * this.shape[6] * this.shape[7] +
                                                    idx2 * this.shape[2] * this.shape[3] * this.shape[4] * this.shape[5] * this.shape[6] * this.shape[7] +
                                                    idx3 * this.shape[3] * this.shape[4] * this.shape[5] * this.shape[6] * this.shape[7] +
                                                    idx4 * this.shape[4] * this.shape[5] * this.shape[6] * this.shape[7] +
                                                    idx5 * this.shape[5] * this.shape[6] * this.shape[7] +
                                                    idx6 * this.shape[6] * this.shape[7] +
                                                    idx7 * this.shape[7] +
                                                    idx8
                                            this.storage.add(aIdx, array[idx1][idx2][idx3][idx4][idx5][idx6][idx7][idx8])
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.DoubleTensor
        }

        /**
         *  ================================ Constructors (Long) ==============================
         */
        constructor(array: LongArray, size: IntArray) {
            this.shape = size.toMutableList()
            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                val aIdx = idx1
                this.storage.add(aIdx, array[idx1])
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.LongTensor
        }

        constructor(array: LongArray) {
            this.shape = mutableListOf()
            this.shape.add(array.size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                val aIdx = idx1
                this.storage.add(aIdx, array[idx1])
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.LongTensor
        }

        constructor(array: Array<LongArray>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    val aIdx = idx1 * this.shape[1] + idx2
                    this.storage.add(aIdx, array[idx1][idx2])
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.LongTensor
        }

        constructor(array: Array<Array<LongArray>>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)
            this.shape.add(array[0][0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    for (idx3 in 0 until this.shape[2]) {
                        val aIdx = idx1 * this.shape[1] * this.shape[2] +
                                idx2 * this.shape[2] +
                                idx3
                        this.storage.add(aIdx, array[idx1][idx2][idx3])
                    }
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.LongTensor
        }

        constructor(array: Array<Array<Array<LongArray>>>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)
            this.shape.add(array[0][0].size)
            this.shape.add(array[0][0][0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    for (idx3 in 0 until this.shape[2]) {
                        for (idx4 in 0 until this.shape[3]) {
                            val aIdx = idx1 * this.shape[1] * this.shape[2] * this.shape[3] +
                                    idx2 * this.shape[2] * this.shape[3] +
                                    idx3 * this.shape[3] +
                                    idx4
                            this.storage.add(aIdx, array[idx1][idx2][idx3][idx4])
                        }
                    }
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.LongTensor
        }

        constructor(array: Array<Array<Array<Array<LongArray>>>>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)
            this.shape.add(array[0][0].size)
            this.shape.add(array[0][0][0].size)
            this.shape.add(array[0][0][0][0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    for (idx3 in 0 until this.shape[2]) {
                        for (idx4 in 0 until this.shape[3]) {
                            for (idx5 in 0 until this.shape[4]) {
                                val aIdx = idx1 * this.shape[1] * this.shape[2] * this.shape[3] * this.shape[4] +
                                        idx2 * this.shape[2] * this.shape[3] * this.shape[4] +
                                        idx3 * this.shape[3] * this.shape[4] +
                                        idx4 * this.shape[4] +
                                        idx5
                                this.storage.add(aIdx, array[idx1][idx2][idx3][idx4][idx5])
                            }
                        }
                    }
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.LongTensor
        }

        constructor(array: Array<Array<Array<Array<Array<LongArray>>>>>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)
            this.shape.add(array[0][0].size)
            this.shape.add(array[0][0][0].size)
            this.shape.add(array[0][0][0][0].size)
            this.shape.add(array[0][0][0][0][0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    for (idx3 in 0 until this.shape[2]) {
                        for (idx4 in 0 until this.shape[3]) {
                            for (idx5 in 0 until this.shape[4]) {
                                for (idx6 in 0 until this.shape[5]) {
                                    val aIdx = idx1 * this.shape[1] * this.shape[2] * this.shape[3] * this.shape[4] * this.shape[5] +
                                            idx2 * this.shape[2] * this.shape[3] * this.shape[4] * this.shape[5] +
                                            idx3 * this.shape[3] * this.shape[4] * this.shape[5] +
                                            idx4 * this.shape[4] * this.shape[5] +
                                            idx5 * this.shape[5] +
                                            idx6
                                    this.storage.add(aIdx, array[idx1][idx2][idx3][idx4][idx5][idx6])
                                }
                            }
                        }
                    }
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.LongTensor
        }

        constructor(array: Array<Array<Array<Array<Array<Array<LongArray>>>>>>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)
            this.shape.add(array[0][0].size)
            this.shape.add(array[0][0][0].size)
            this.shape.add(array[0][0][0][0].size)
            this.shape.add(array[0][0][0][0][0].size)
            this.shape.add(array[0][0][0][0][0][0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    for (idx3 in 0 until this.shape[2]) {
                        for (idx4 in 0 until this.shape[3]) {
                            for (idx5 in 0 until this.shape[4]) {
                                for (idx6 in 0 until this.shape[5]) {
                                    for (idx7 in 0 until this.shape[6]) {
                                        val aIdx = idx1 * this.shape[1] * this.shape[2] * this.shape[3] * this.shape[4] * this.shape[5] * this.shape[6] +
                                                idx2 * this.shape[2] * this.shape[3] * this.shape[4] * this.shape[5] * this.shape[6] +
                                                idx3 * this.shape[3] * this.shape[4] * this.shape[5] * this.shape[6] +
                                                idx4 * this.shape[4] * this.shape[5] * this.shape[6] +
                                                idx5 * this.shape[5] * this.shape[6] +
                                                idx6 * this.shape[6] +
                                                idx7
                                        this.storage.add(aIdx, array[idx1][idx2][idx3][idx4][idx5][idx6][idx7])
                                    }
                                }
                            }
                        }
                    }
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.LongTensor
        }

        constructor(array: Array<Array<Array<Array<Array<Array<Array<LongArray>>>>>>>) {
            this.shape = mutableListOf()
            this.shape.add(array.size)
            this.shape.add(array[0].size)
            this.shape.add(array[0][0].size)
            this.shape.add(array[0][0][0].size)
            this.shape.add(array[0][0][0][0].size)
            this.shape.add(array[0][0][0][0][0].size)
            this.shape.add(array[0][0][0][0][0][0].size)
            this.shape.add(array[0][0][0][0][0][0][0].size)

            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = mutableListOf<Number>()
            for (idx1 in 0 until this.shape[0]) {
                for (idx2 in 0 until this.shape[1]) {
                    for (idx3 in 0 until this.shape[2]) {
                        for (idx4 in 0 until this.shape[3]) {
                            for (idx5 in 0 until this.shape[4]) {
                                for (idx6 in 0 until this.shape[5]) {
                                    for (idx7 in 0 until this.shape[6]) {
                                        for (idx8 in 0 until this.shape[7]) {
                                            val aIdx = idx1 * this.shape[1] * this.shape[2] * this.shape[3] * this.shape[4] * this.shape[5] * this.shape[6] * this.shape[7] +
                                                    idx2 * this.shape[2] * this.shape[3] * this.shape[4] * this.shape[5] * this.shape[6] * this.shape[7] +
                                                    idx3 * this.shape[3] * this.shape[4] * this.shape[5] * this.shape[6] * this.shape[7] +
                                                    idx4 * this.shape[4] * this.shape[5] * this.shape[6] * this.shape[7] +
                                                    idx5 * this.shape[5] * this.shape[6] * this.shape[7] +
                                                    idx6 * this.shape[6] * this.shape[7] +
                                                    idx7 * this.shape[7] +
                                                    idx8
                                            this.storage.add(aIdx, array[idx1][idx2][idx3][idx4][idx5][idx6][idx7][idx8])
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            this.ndimension = this.shape.size
            this.stride = computeStride(this.shape)
            this.dtype = TensorType.LongTensor
        }

        /**
         *  ================================ Constructors (others) ==============================
         */
        constructor(input: TorchTensor) {
            this.shape = input.shape.toMutableList()
            this.storage = input.storage.toMutableList()
            this.ndimension = input.ndimension
            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.stride = computeStride(this.shape)
        }

        constructor(size: MutableList<Int>, initValue: Float) {
            this.shape = size.toMutableList()
            this.ndimension = this.shape.size
            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.stride = computeStride(this.shape)
            this.storage = Array(this.nElements){initValue}.toMutableList()
        }

        constructor(size: MutableList<Int>, storage: MutableList<Number>, stride: MutableList<Int>) {
            this.shape = size.toMutableList()
            this.ndimension = this.shape.size
            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = storage.toMutableList()
            this.stride = stride.toMutableList()
        }

        constructor(size: MutableList<Int>, storage: MutableList<Number>) {
            this.shape = size.toMutableList()
            this.ndimension = this.shape.size
            this.nElements = this.shape.reduce { acc, i -> acc * i }
            this.storage = storage.toMutableList()
            this.stride = computeStride(this.shape)
        }

        /**
         *  Compute strides for given size
         *
         *  @param  size    The size mutable list
         *  @return         The stride mutable list
         */
        private fun computeStride(size: MutableList<Int>): MutableList<Int> {
            val newStrides = mutableListOf<Int>()
            val nDim = size.size
            for (i in 0 until nDim) {
                val subList = size.subList(min(i + 1, nDim), nDim)
                if (subList.size == 0) {
                    newStrides.add(1)
                } else {
                    val stride = subList.reduce { acc, i -> acc * i }
                    newStrides.add(stride)
                }
            }
            return newStrides
        }

        /**
         *  Returns the number of dimensions of this tensor
         *
         *  @return     The number of dimensions
         */
        fun dim(): Int {
            return this.ndimension
        }

        /**
         *  Returns the total number of elements in this tensor
         *
         *  @return     The total number of elements
         */
        fun numel(): Int {
            return this.nElements
        }

        /**
         *  Return the size mutable list
         *
         *  @return     The size mutable list
         */
        fun size(): MutableList<Int> {
            return this.shape
        }

        /**
         *  Return the total number of element for given tensor by recursive
         *  Note that this function is a private function
         *
         *  @param  array   The array you want to observe
         *  @return         The total number of element in given tensor
         */
        private fun getNElement(array: Any): Int {
            if (array is Int || array is Float || array is Double) {
                return 1;
            } else if (array is Array<*>) {
                return array.size * getNElement(array[0]!!)
            } else {
                throw NotImplementedError("We only support float, double and int array currently")
            }
        }

        /**
         *  Permute the dimensions of this tensor.
         *
         *  Note:
         *      (1) This function should be improved since it use bubble-sort to find repeat order
         *          (Use hash map is more wise!)
         *
         *  @param  order   The desired ordering of dimensions
         *  @return         The permuted tensor
         */
        fun permute(order: Array<Int>): TorchTensor {
            if (order.size != this.ndimension) {
                throw RuntimeError("number of dims don't match in permute")
            }
            if (order.maxOrNull()!! >= this.ndimension) {
                throw IndexError("Dimension out of range (expected to be in range of [" +
                        -this.dim() + ", " + (this.dim() - 1) + "], but got " +
                        order.maxOrNull() + ")\n")
            }

            // Adjust as positive order list
            var posOrder = order
            for (idx in 0 until posOrder.size) {
                if (posOrder[idx] < 0) {
                    posOrder[idx] += this.dim()
                }
            }

            // Check if repeat order
            // TODO: speed-up from O(n^2)
            for (i in 0 until posOrder.size) {
                if (i != posOrder.size - 1) {
                    for (j in i + 1 until posOrder.size) {
                        if (posOrder[i] == posOrder[j]) {
                            throw RuntimeError("repeated dim in permute")
                        }
                    }
                }
            }

            // do!
            val newSize = Array(this.ndimension) {0}
            val newStride = Array(this.ndimension) {0}
            for (idx in 0 until posOrder.size) {
                newSize[idx] = this.shape[posOrder[idx]]
                newStride[idx] = this.stride[posOrder[idx]]
            }

            // in-place alter
            val output = TorchTensor(size = newSize.toMutableList(), storage = this.storage, stride = newStride.toMutableList())
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        /**
         *  Fills self tensor with the specified value
         *
         *  @param  value   The value you want to fill in
         */
        fun fill_(value: Number): TorchTensor {
            for (idx in 0 until this.nElements) {
                this.storage[idx] = value
            }
            return this
        }

        /**
         *  ================================ Operation overloading (single value) ==============================
         *
         *  @param  a   The number you want to compute
         *  @return     The computed tensor result
         */
        operator fun plus(a: Number): TorchTensor {
            val output = TorchTensor(this)
            when (this.dtype) {
                TensorType.DoubleTensor -> {
                    for (i in 0 until this.nElements) {
                        output.storage[i] = output.storage[i].toDouble() + a.toDouble()
                    }
                }
                TensorType.FloatTensor -> {
                    for (i in 0 until this.nElements) {
                        output.storage[i] = output.storage[i].toFloat() + a.toFloat()
                    }
                }
                TensorType.LongTensor -> {
                    for (i in 0 until this.nElements) {
                        output.storage[i] = output.storage[i].toLong() + a.toLong()
                    }
                }
            }

            // in-place alter
            this.stride = output.stride
            this.shape = output.shape
            this.storage = output.storage
            return this
        }

        operator fun minus(a: Number): TorchTensor {
            val output = TorchTensor(this)
            when (this.dtype) {
                TensorType.DoubleTensor -> {
                    for (i in 0 until this.nElements) {
                        output.storage[i] = output.storage[i].toDouble() - a.toDouble()
                    }
                }
                TensorType.FloatTensor -> {
                    for (i in 0 until this.nElements) {
                        output.storage[i] = output.storage[i].toFloat() - a.toFloat()
                    }
                }
                TensorType.LongTensor -> {
                    for (i in 0 until this.nElements) {
                        output.storage[i] = output.storage[i].toLong() - a.toLong()
                    }
                }
            }

            // in-place alter
            this.stride = output.stride
            this.shape = output.shape
            this.storage = output.storage
            return this
        }

        operator fun times(a: Number): TorchTensor {
            val output = TorchTensor(this)
            when (this.dtype) {
                TensorType.DoubleTensor -> {
                    for (i in 0 until this.nElements) {
                        output.storage[i] = output.storage[i].toDouble() * a.toDouble()
                    }
                }
                TensorType.FloatTensor -> {
                    for (i in 0 until this.nElements) {
                        output.storage[i] = output.storage[i].toFloat() * a.toFloat()
                    }
                }
                TensorType.LongTensor -> {
                    for (i in 0 until this.nElements) {
                        output.storage[i] = output.storage[i].toLong() * a.toLong()
                    }
                }
            }

            // in-place alter
            this.stride = output.stride
            this.shape = output.shape
            this.storage = output.storage
            return this
        }

        operator fun div(a: Number): TorchTensor {
            val output = TorchTensor(this)
            when (this.dtype) {
                TensorType.DoubleTensor -> {
                    for (i in 0 until this.nElements) {
                        output.storage[i] = output.storage[i].toDouble() / a.toDouble()
                    }
                }
                TensorType.FloatTensor -> {
                    for (i in 0 until this.nElements) {
                        output.storage[i] = output.storage[i].toFloat() / a.toFloat()
                    }
                }
                TensorType.LongTensor -> {
                    for (i in 0 until this.nElements) {
                        output.storage[i] = output.storage[i].toLong() / a.toLong()
                    }
                }
            }

            // in-place alter
            this.stride = output.stride
            this.shape = output.shape
            this.storage = output.storage
            return this
        }

        /**
         *  ================================ Operation overloading (tensor) ==============================
         *  Note:
         *      (1) Differ to PyTorch, we don't support broadcasting currectly.
         *          You should make sure both tensor have equal size
         *
         *  TODO: support boradcasting!!
         *
         *  @param  a   The tensor you want to compute
         *  @return     The computed tensor result
         */

        operator fun plus(a: TorchTensor): TorchTensor {
            if (this.shape != a.shape) {
                throw RuntimeError("operator plus only work on equal size. We don't support broadcasting currectly. Left size: " +
                        this.shape + "\tRight size: " + a.shape)
            }
            val output = TorchTensor(this)
            when (this.dtype) {
                TensorType.DoubleTensor -> {
                    for (i in 0 until this.nElements) {
                        output.storage[i] = output.storage[i].toDouble() + a.storage[i].toDouble()
                    }
                }
                TensorType.FloatTensor -> {
                    for (i in 0 until this.nElements) {
                        output.storage[i] = output.storage[i].toFloat() + a.storage[i].toFloat()
                    }
                }
                TensorType.LongTensor -> {
                    for (i in 0 until this.nElements) {
                        output.storage[i] = output.storage[i].toLong() + a.storage[i].toLong()
                    }
                }
            }

            // in-place alter
            this.stride = output.stride
            this.shape = output.shape
            this.storage = output.storage
            return this
        }

        operator fun minus(a: TorchTensor): TorchTensor {
            if (this.shape != a.shape) {
                throw RuntimeError("operator plus only work on equal size. We don't support broadcasting currectly. Left size: " +
                        this.shape + "\tRight size: " + a.shape)
            }
            val output = TorchTensor(this)
            when (this.dtype) {
                TensorType.DoubleTensor -> {
                    for (i in 0 until this.nElements) {
                        output.storage[i] = output.storage[i].toDouble() - a.storage[i].toDouble()
                    }
                }
                TensorType.FloatTensor -> {
                    for (i in 0 until this.nElements) {
                        output.storage[i] = output.storage[i].toFloat() - a.storage[i].toFloat()
                    }
                }
                TensorType.LongTensor -> {
                    for (i in 0 until this.nElements) {
                        output.storage[i] = output.storage[i].toLong() - a.storage[i].toLong()
                    }
                }
            }

            // in-place alter
            this.stride = output.stride
            this.shape = output.shape
            this.storage = output.storage
            return this
        }

        operator fun times(a: TorchTensor): TorchTensor {
            if (this.shape != a.shape) {
                throw RuntimeError("operator plus only work on equal size. We don't support broadcasting currectly. Left size: " +
                        this.shape + "\tRight size: " + a.shape)
            }
            val output = TorchTensor(this)
            when (this.dtype) {
                TensorType.DoubleTensor -> {
                    for (i in 0 until this.nElements) {
                        output.storage[i] = output.storage[i].toDouble() * a.storage[i].toDouble()
                    }
                }
                TensorType.FloatTensor -> {
                    for (i in 0 until this.nElements) {
                        output.storage[i] = output.storage[i].toFloat() * a.storage[i].toFloat()
                    }
                }
                TensorType.LongTensor -> {
                    for (i in 0 until this.nElements) {
                        output.storage[i] = output.storage[i].toLong() * a.storage[i].toLong()
                    }
                }
            }

            // in-place alter
            this.stride = output.stride
            this.shape = output.shape
            this.storage = output.storage
            return this
        }

        operator fun div(a: TorchTensor): TorchTensor {
            if (this.shape != a.shape) {
                throw RuntimeError("operator plus only work on equal size. We don't support broadcasting currectly. Left size: " +
                        this.shape + "\tRight size: " + a.shape)
            }
            val output = TorchTensor(this)
            when (this.dtype) {
                TensorType.DoubleTensor -> {
                    for (i in 0 until this.nElements) {
                        output.storage[i] = output.storage[i].toDouble() / a.storage[i].toDouble()
                    }
                }
                TensorType.FloatTensor -> {
                    for (i in 0 until this.nElements) {
                        output.storage[i] = output.storage[i].toFloat() / a.storage[i].toFloat()
                    }
                }
                TensorType.LongTensor -> {
                    for (i in 0 until this.nElements) {
                        output.storage[i] = output.storage[i].toLong() / a.storage[i].toLong()
                    }
                }
            }

            // in-place alter
            this.stride = output.stride
            this.shape = output.shape
            this.storage = output.storage
            return this
        }

        /**
         *  ========================================= Get value =========================================
         */
        operator fun get(i: Int): TorchTensor {
            // Check correctness call or return value directly
            val minNDim = 1
            if (this.ndimension < minNDim) {
                throw Exception("Assess error. Tensor ndim: " +
                        this.ndimension + "\tMinNDim: $minNDim")
            } else if (this.ndimension == minNDim) {
                return TorchTensor(Array<Number>(1) { this.storage[i * this.stride[0]] })
            }

            // Re-create storage to deal with multi-dimensional array
            val newSize = this.shape.subList(minNDim, this.ndimension)
            // val newStride = this.stride.subList(minNDim, this.ndimension)
            val newStride = computeStride(newSize)
            val N = newSize.reduce { acc, i -> acc * i }
            val newStorage = Array(N) {0 as Number}.toMutableList()
            // val newStorage = emptyArray<Number>().toMutableList()
            when (this.ndimension) {
                2 -> {
                    for (idx1 in 0 until this.shape[1]) {
                        val aIdx = idx1
                        val bIdx = i * this.stride[0] + idx1 * this.stride[1]
                        newStorage.add(aIdx, this.storage[bIdx])
                    }
                }
                3 -> {
                    for (idx1 in 0 until this.shape[1]) {
                        for (idx2 in 0 until this.shape[2]) {
                            val aIdx = idx1 * newStride[0] +
                                    idx2 * newStride[1]
                            val bIdx = i * this.stride[0] +
                                    idx1 * this.stride[1] +
                                    idx2 * this.stride[2]
                            newStorage.add(aIdx, this.storage[bIdx])
                        }
                    }
                }
                4 -> {
                    for (idx1 in 0 until this.shape[1]) {
                        for (idx2 in 0 until this.shape[2]) {
                            for (idx3 in 0 until this.shape[3]) {
                                val aIdx = idx1 * newStride[0] +
                                        idx2 * newStride[1] +
                                        idx3 * newStride[2]
                                val bIdx = i * this.stride[0] +
                                        idx1 * this.stride[1] +
                                        idx2 * this.stride[2] +
                                        idx3 * this.stride[3]
                                newStorage.add(aIdx, this.storage[bIdx])
                            }
                        }
                    }
                }
                5 -> {
                    for (idx1 in 0 until this.shape[1]) {
                        for (idx2 in 0 until this.shape[2]) {
                            for (idx3 in 0 until this.shape[3]) {
                                for (idx4 in 0 until this.shape[4]) {
                                    val aIdx = idx1 * newStride[0] +
                                            idx2 * newStride[1] +
                                            idx3 * newStride[2] +
                                            idx4 * newStride[3]
                                    val bIdx = i * this.stride[0] +
                                            idx1 * this.stride[1] +
                                            idx2 * this.stride[2] +
                                            idx3 * this.stride[3] +
                                            idx4 * this.stride[4]
                                    newStorage.add(aIdx, this.storage[bIdx])
                                }
                            }
                        }
                    }
                }
                6 -> {
                    for (idx1 in 0 until this.shape[1]) {
                        for (idx2 in 0 until this.shape[2]) {
                            for (idx3 in 0 until this.shape[3]) {
                                for (idx4 in 0 until this.shape[4]) {
                                    for (idx5 in 0 until this.shape[5]) {
                                        val aIdx = idx1 * newStride[0] +
                                                idx2 * newStride[1] +
                                                idx3 * newStride[2] +
                                                idx4 * newStride[3] +
                                                idx5 * newStride[4]
                                        val bIdx = i * this.stride[0] +
                                                idx1 * this.stride[1] +
                                                idx2 * this.stride[2] +
                                                idx3 * this.stride[3] +
                                                idx4 * this.stride[4] +
                                                idx5 * this.stride[5]
                                        newStorage.add(aIdx, this.storage[bIdx])
                                    }
                                }
                            }
                        }
                    }
                }
                7 -> {
                    for (idx1 in 0 until this.shape[1]) {
                        for (idx2 in 0 until this.shape[2]) {
                            for (idx3 in 0 until this.shape[3]) {
                                for (idx4 in 0 until this.shape[4]) {
                                    for (idx5 in 0 until this.shape[5]) {
                                        for (idx6 in 0 until this.shape[6]) {
                                            val aIdx = idx1 * newStride[0] +
                                                    idx2 * newStride[1] +
                                                    idx3 * newStride[2] +
                                                    idx4 * newStride[3] +
                                                    idx5 * newStride[4] +
                                                    idx6 * newStride[5]
                                            val bIdx = i * this.stride[0] +
                                                    idx1 * this.stride[1] +
                                                    idx2 * this.stride[2] +
                                                    idx3 * this.stride[3] +
                                                    idx4 * this.stride[4] +
                                                    idx5 * this.stride[5] +
                                                    idx6 * this.stride[6]
                                            newStorage.add(aIdx, this.storage[bIdx])
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                8 -> {
                    for (idx1 in 0 until this.shape[1]) {
                        for (idx2 in 0 until this.shape[2]) {
                            for (idx3 in 0 until this.shape[3]) {
                                for (idx4 in 0 until this.shape[4]) {
                                    for (idx5 in 0 until this.shape[5]) {
                                        for (idx6 in 0 until this.shape[6]) {
                                            for (idx7 in 0 until this.shape[7]) {
                                                val aIdx = idx1 * newStride[0] +
                                                        idx2 * newStride[1] +
                                                        idx3 * newStride[2] +
                                                        idx4 * newStride[3] +
                                                        idx5 * newStride[4] +
                                                        idx6 * newStride[5] +
                                                        idx7 * newStride[6]
                                                val bIdx = i * this.stride[0] +
                                                        idx1 * this.stride[1] +
                                                        idx2 * this.stride[2] +
                                                        idx3 * this.stride[3] +
                                                        idx4 * this.stride[4] +
                                                        idx5 * this.stride[5] +
                                                        idx6 * this.stride[6] +
                                                        idx7 * this.stride[7]
                                                newStorage.add(aIdx, this.storage[bIdx])
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                else -> {
                    throw NotImplementedError();
                }
            }
            var output = TorchTensor(size = newSize, storage = newStorage, stride = newStride)
            output.dtype = this.dtype
            return output
        }

        operator fun get(i: Int, j: Int): TorchTensor {
            // Check correctness call or return value directly
            val minNDim = 2
            if (this.ndimension < minNDim) {
                throw Exception("Assess error. Tensor ndim: " +
                        this.ndimension + "\tMinNDim: $minNDim")
            } else if (this.ndimension == minNDim) {
                val idx = i * this.stride[0] +
                        j * this.stride[1]
                return TorchTensor(Array<Number>(1) { this.storage[idx] })
            }

            // Re-create storage to deal with multi-dimensional array
            val newSize = this.shape.subList(minNDim, this.ndimension)
            // val newStride = this.stride.subList(minNDim, this.ndimension)
            val newStride = computeStride(newSize)
            val N = newSize.reduce { acc, i -> acc * i }
            val newStorage = Array(N) {0 as Number}.toMutableList()
            // val newStorage = emptyArray<Number>().toMutableList()
            when (this.ndimension) {
                3 -> {
                    for (idx1 in 0 until this.shape[2]) {
                        val aIdx = idx1 * newStride[0]
                        val bIdx = i * this.stride[0] + j * this.stride[1] +
                                idx1 * this.stride[2]
                        newStorage.add(aIdx, this.storage[bIdx])
                    }
                }
                4 -> {
                    for (idx1 in 0 until this.shape[2]) {
                        for (idx2 in 0 until this.shape[3]) {
                            val aIdx = idx1 * newStride[0] +
                                    idx2 * newStride[1]
                            val bIdx = i * this.stride[0] + j * this.stride[1] +
                                    idx1 * this.stride[2] +
                                    idx2 * this.stride[3]
                            newStorage.add(aIdx, this.storage[bIdx])
                        }
                    }
                }
                5 -> {
                    for (idx1 in 0 until this.shape[2]) {
                        for (idx2 in 0 until this.shape[3]) {
                            for (idx3 in 0 until this.shape[4]) {
                                val aIdx = idx1 * newStride[0] +
                                        idx2 * newStride[1] +
                                        idx3 * newStride[2]
                                val bIdx = i * this.stride[0] + j * this.stride[1] +
                                        idx1 * this.stride[2] +
                                        idx2 * this.stride[3] +
                                        idx3 * this.stride[4]
                                // newStorage[aIdx] = this.storage[bIdx]
                                newStorage.add(aIdx, this.storage[bIdx])
                            }
                        }
                    }
                }
                6 -> {
                    for (idx1 in 0 until this.shape[2]) {
                        for (idx2 in 0 until this.shape[3]) {
                            for (idx3 in 0 until this.shape[4]) {
                                for (idx4 in 0 until this.shape[5]) {
                                    val aIdx = idx1 * newStride[0] +
                                            idx2 * newStride[1] +
                                            idx3 * newStride[2] +
                                            idx4 * newStride[3]
                                    val bIdx = i * this.stride[0] + j * this.stride[1] +
                                            idx1 * this.stride[2] +
                                            idx2 * this.stride[3] +
                                            idx3 * this.stride[4] +
                                            idx4 * this.stride[5]
                                    // newStorage[aIdx] = this.storage[bIdx]
                                    newStorage.add(aIdx, this.storage[bIdx])
                                }
                            }
                        }
                    }
                }
                7 -> {
                    for (idx1 in 0 until this.shape[2]) {
                        for (idx2 in 0 until this.shape[3]) {
                            for (idx3 in 0 until this.shape[4]) {
                                for (idx4 in 0 until this.shape[5]) {
                                    for (idx5 in 0 until this.shape[6]) {
                                        val aIdx = idx1 * newStride[0] +
                                                idx2 * newStride[1] +
                                                idx3 * newStride[2] +
                                                idx4 * newStride[3] +
                                                idx5 * newStride[4]
                                        val bIdx = i * this.stride[0] + j * this.stride[1] +
                                                idx1 * this.stride[2] +
                                                idx2 * this.stride[3] +
                                                idx3 * this.stride[4] +
                                                idx4 * this.stride[5] +
                                                idx5 * this.stride[6]
                                        // newStorage[aIdx] = this.storage[bIdx]
                                        newStorage.add(aIdx, this.storage[bIdx])
                                    }
                                }
                            }
                        }
                    }
                }
                8 -> {
                    for (idx1 in 0 until this.shape[2]) {
                        for (idx2 in 0 until this.shape[3]) {
                            for (idx3 in 0 until this.shape[4]) {
                                for (idx4 in 0 until this.shape[5]) {
                                    for (idx5 in 0 until this.shape[6]) {
                                        for (idx6 in 0 until this.shape[7]) {
                                            val aIdx = idx1 * newStride[0] +
                                                    idx2 * newStride[1] +
                                                    idx3 * newStride[2] +
                                                    idx4 * newStride[3] +
                                                    idx5 * newStride[4] +
                                                    idx6 * newStride[5]
                                            val bIdx = i * this.stride[0] + j * this.stride[1] +
                                                    idx1 * this.stride[2] +
                                                    idx2 * this.stride[3] +
                                                    idx3 * this.stride[4] +
                                                    idx4 * this.stride[5] +
                                                    idx5 * this.stride[6] +
                                                    idx6 * this.stride[7]
                                            // newStorage[aIdx] = this.storage[bIdx]
                                            newStorage.add(aIdx, this.storage[bIdx])
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                else -> {
                    throw NotImplementedError();
                }
            }
            var output = TorchTensor(size = newSize, storage = newStorage, stride = newStride)
            output.dtype = this.dtype
            return output
        }

        operator fun get(i: Int, j: Int, k: Int): TorchTensor {
            // Check correctness call or return value directly
            val minNDim = 3
            if (this.ndimension < minNDim) {
                throw Exception("Assess error. Tensor ndim: " +
                        this.ndimension + "\tMinNDim: $minNDim")
            } else if (this.ndimension == minNDim) {
                val idx = i * this.stride[0] +
                        j * this.stride[1] +
                        k * this.stride[2]
                return TorchTensor(Array<Number>(1) { this.storage[idx] })
            }

            // Re-create storage to deal with multi-dimensional array
            val newSize = this.shape.subList(minNDim, this.ndimension)
            // val newStride = this.stride.subList(minNDim, this.ndimension)
            val newStride = computeStride(newSize)
            val N = newSize.reduce { acc, i -> acc * i }
            val newStorage = Array(N) {0 as Number}.toMutableList()
            // val newStorage = emptyArray<Number>().toMutableList()
            when (this.ndimension) {
                4 -> {
                    for (idx1 in 0 until this.shape[3]) {
                        val aIdx = idx1 * newStride[0]
                        val bIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] +
                                idx1 * this.stride[3]
                        // newStorage[aIdx] = this.storage[bIdx]
                        newStorage.add(aIdx, this.storage[bIdx])
                    }
                }
                5 -> {
                    for (idx1 in 0 until this.shape[3]) {
                        for (idx2 in 0 until this.shape[4]) {
                            val aIdx = idx1 * newStride[0] +
                                    idx2 * newStride[1]
                            val bIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] +
                                    idx1 * this.stride[3] +
                                    idx2 * this.stride[4]
                            // newStorage[aIdx] = this.storage[bIdx]
                            newStorage.add(aIdx, this.storage[bIdx])
                        }
                    }
                }
                6 -> {
                    for (idx1 in 0 until this.shape[3]) {
                        for (idx2 in 0 until this.shape[4]) {
                            for (idx3 in 0 until this.shape[5]) {
                                val aIdx = idx1 * newStride[0] +
                                        idx2 * newStride[1] +
                                        idx3 * newStride[2]
                                val bIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] +
                                        idx1 * this.stride[3] +
                                        idx2 * this.stride[4] +
                                        idx3 * this.stride[5]
                                // newStorage[aIdx] = this.storage[bIdx]
                                newStorage.add(aIdx, this.storage[bIdx])
                            }
                        }
                    }
                }
                7 -> {
                    for (idx1 in 0 until this.shape[3]) {
                        for (idx2 in 0 until this.shape[4]) {
                            for (idx3 in 0 until this.shape[5]) {
                                for (idx4 in 0 until this.shape[6]) {
                                    val aIdx = idx1 * newStride[0] +
                                            idx2 * newStride[1] +
                                            idx3 * newStride[2] +
                                            idx4 * newStride[3]
                                    val bIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] +
                                            idx1 * this.stride[3] +
                                            idx2 * this.stride[4] +
                                            idx3 * this.stride[5] +
                                            idx4 * this.stride[6]
                                    // newStorage[aIdx] = this.storage[bIdx]
                                    newStorage.add(aIdx, this.storage[bIdx])
                                }
                            }
                        }
                    }
                }
                8 -> {
                    for (idx1 in 0 until this.shape[3]) {
                        for (idx2 in 0 until this.shape[4]) {
                            for (idx3 in 0 until this.shape[5]) {
                                for (idx4 in 0 until this.shape[6]) {
                                    for (idx5 in 0 until this.shape[7]) {
                                        val aIdx = idx1 * newStride[0] +
                                                idx2 * newStride[1] +
                                                idx3 * newStride[2] +
                                                idx4 * newStride[3] +
                                                idx5 * newStride[4]
                                        val bIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] +
                                                idx1 * this.stride[3] +
                                                idx2 * this.stride[4] +
                                                idx3 * this.stride[5] +
                                                idx4 * this.stride[6] +
                                                idx5 * this.stride[7]
                                        // newStorage[aIdx] = this.storage[bIdx]
                                        newStorage.add(aIdx, this.storage[bIdx])
                                    }
                                }
                            }
                        }
                    }
                }
                else -> {
                    throw NotImplementedError()
                }
            }
            var output = TorchTensor(size = newSize, storage = newStorage, stride = newStride)
            output.dtype = this.dtype
            return output
        }

        operator fun get(i: Int, j: Int, k: Int, l: Int): TorchTensor {
            // Check correctness call or return value directly
            val minNDim = 4
            if (this.ndimension < minNDim) {
                throw Exception("Assess error. Tensor ndim: " +
                        this.ndimension + "\tMinNDim: $minNDim")
            } else if (this.ndimension == minNDim) {
                val idx = i * this.stride[0] +
                        j * this.stride[1] +
                        k * this.stride[2] +
                        l * this.stride[3]
                return TorchTensor(Array<Number>(1) { this.storage[idx] })
            }

            // Re-create storage to deal with multi-dimensional array
            val newSize = this.shape.subList(minNDim, this.ndimension)
            // val newStride = this.stride.subList(minNDim, this.ndimension)
            val newStride = computeStride(newSize)
            val N = newSize.reduce { acc, i -> acc * i }
            val newStorage = Array(N) {0 as Number}.toMutableList()
            // val newStorage = emptyArray<Number>().toMutableList()
            when (this.ndimension) {
                5 -> {
                    for (idx1 in 0 until this.shape[4]) {
                        val aIdx = idx1 * newStride[0]
                        val bIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] + l * this.stride[3] +
                                idx1 * this.stride[4]
                        newStorage.add(aIdx, this.storage[bIdx])
                    }
                }
                6 -> {
                    for (idx1 in 0 until this.shape[4]) {
                        for (idx2 in 0 until this.shape[5]) {
                            val aIdx = idx1 * newStride[0] +
                                    idx2 * newStride[1]
                            val bIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] + l * this.stride[3] +
                                    idx1 * this.stride[4] +
                                    idx2 * this.stride[5]
                            newStorage.add(aIdx, this.storage[bIdx])
                        }
                    }
                }
                7 -> {
                    for (idx1 in 0 until this.shape[4]) {
                        for (idx2 in 0 until this.shape[5]) {
                            for (idx3 in 0 until this.shape[6]) {
                                val aIdx = idx1 * newStride[0] +
                                        idx2 * newStride[1] +
                                        idx3 * newStride[2]
                                val bIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] + l * this.stride[3] +
                                        idx1 * this.stride[4] +
                                        idx2 * this.stride[5] +
                                        idx3 * this.stride[6]
                                newStorage.add(aIdx, this.storage[bIdx])
                            }
                        }
                    }
                }
                8 -> {
                    for (idx1 in 0 until this.shape[4]) {
                        for (idx2 in 0 until this.shape[5]) {
                            for (idx3 in 0 until this.shape[6]) {
                                for (idx4 in 0 until this.shape[7]) {
                                    val aIdx = idx1 * newStride[0] +
                                            idx2 * newStride[1] +
                                            idx3 * newStride[2] +
                                            idx4 * newStride[3]
                                    val bIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] + l * this.stride[3] +
                                            idx1 * this.stride[4] +
                                            idx2 * this.stride[5] +
                                            idx3 * this.stride[6] +
                                            idx4 * this.stride[7]
                                    newStorage.add(aIdx, this.storage[bIdx])
                                }
                            }
                        }
                    }
                }
                else -> {
                    throw NotImplementedError()
                }
            }
            var output = TorchTensor(size = newSize, storage = newStorage, stride = newStride)
            output.dtype = this.dtype
            return output
        }

        operator fun get(i: Int, j: Int, k: Int, l: Int, m: Int): TorchTensor {
            // Check correctness call or return value directly
            val minNDim = 5
            if (this.ndimension < minNDim) {
                throw Exception("Assess error. Tensor ndim: " +
                        this.ndimension + "\tMinNDim: $minNDim")
            } else if (this.ndimension == minNDim) {
                val idx = i * this.stride[0] +
                        j * this.stride[1] +
                        k * this.stride[2] +
                        l * this.stride[3] +
                        m * this.stride[4]
                return TorchTensor(Array<Number>(1) { this.storage[idx] })
            }

            // Re-create storage to deal with multi-dimensional array
            val newSize = this.shape.subList(minNDim, this.ndimension)
            // val newStride = this.stride.subList(minNDim, this.ndimension)
            val newStride = computeStride(newSize)
            val N = newSize.reduce { acc, i -> acc * i }
            val newStorage = Array(N) {0 as Number}.toMutableList()
            // val newStorage = emptyArray<Number>().toMutableList()
            when (this.ndimension) {
                6 -> {
                    for (idx1 in 0 until this.shape[5]) {
                        val aIdx = idx1 * newStride[0]
                        val bIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] + l * this.stride[3] + m * this.stride[4] +
                                idx1 * this.stride[5]
                        newStorage.add(aIdx, this.storage[bIdx])
                    }
                }
                7 -> {
                    for (idx1 in 0 until this.shape[5]) {
                        for (idx2 in 0 until this.shape[6]) {
                            val aIdx = idx1 * newStride[0] +
                                    idx2 * newStride[1]
                            val bIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] + l * this.stride[3] + m * this.stride[4] +
                                    idx1 * this.stride[5] +
                                    idx2 * this.stride[6]
                            newStorage.add(aIdx, this.storage[bIdx])
                        }
                    }
                }
                8 -> {
                    for (idx1 in 0 until this.shape[5]) {
                        for (idx2 in 0 until this.shape[6]) {
                            for (idx3 in 0 until this.shape[7]) {
                                val aIdx = idx1 * newStride[0] +
                                        idx2 * newStride[1] +
                                        idx3 * newStride[2]
                                val bIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] + l * this.stride[3] + m * this.stride[4] +
                                        idx1 * this.stride[5] +
                                        idx2 * this.stride[6] +
                                        idx3 * this.stride[7]
                                newStorage.add(aIdx, this.storage[bIdx])
                            }
                        }
                    }
                }
                else -> {
                    throw NotImplementedError()
                }
            }
            var output = TorchTensor(size = newSize, storage = newStorage, stride = newStride)
            output.dtype = this.dtype
            return output
        }

        operator fun get(i: Int, j: Int, k: Int, l: Int, m: Int, n: Int): TorchTensor {
            // Check correctness call or return value directly
            val minNDim = 6
            if (this.ndimension < minNDim) {
                throw Exception("Assess error. Tensor ndim: " +
                        this.ndimension + "\tMinNDim: $minNDim")
            } else if (this.ndimension == minNDim) {
                val idx = i * this.stride[0] +
                        j * this.stride[1] +
                        k * this.stride[2] +
                        l * this.stride[3] +
                        m * this.stride[4] +
                        n * this.stride[5]
                return TorchTensor(Array<Number>(1) { this.storage[idx] })
            }

            // Re-create storage to deal with multi-dimensional array
            val newSize = this.shape.subList(minNDim, this.ndimension)
            // val newStride = this.stride.subList(minNDim, this.ndimension)
            val newStride = computeStride(newSize)
            val N = newSize.reduce { acc, i -> acc * i }
            val newStorage = Array(N) {0 as Number}.toMutableList()
            // val newStorage = emptyArray<Number>().toMutableList()
            when (this.ndimension) {
                7 -> {
                    for (idx1 in 0 until this.shape[6]) {
                        val aIdx = idx1 * newStride[0]
                        val bIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] + l * this.stride[3] + m * this.stride[4] + n * this.stride[5] +
                                idx1 * this.stride[6]
                        // newStorage[aIdx] = this.storage[bIdx]
                        newStorage.add(aIdx, this.storage[bIdx])
                    }
                }
                8 -> {
                    for (idx1 in 0 until this.shape[6]) {
                        for (idx2 in 0 until this.shape[7]) {
                            val aIdx = idx1 * newStride[0] +
                                    idx2 * newStride[1]
                            val bIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] + l * this.stride[3] + m * this.stride[4] + n * this.stride[5] +
                                    idx1 * this.stride[6] +
                                    idx2 * this.stride[7]
                            // newStorage[aIdx] = this.storage[bIdx]
                            newStorage.add(aIdx, this.storage[bIdx])
                        }
                    }
                }
                else -> {
                    throw NotImplementedError()
                }
            }
            var output = TorchTensor(size = newSize, storage = newStorage, stride = newStride)
            output.dtype = this.dtype
            return output
        }

        operator fun get(i: Int, j: Int, k: Int, l: Int, m: Int, n: Int, o: Int): TorchTensor {
            // Check correctness call or return value directly
            val minNDim = 7
            if (this.ndimension < minNDim) {
                throw Exception("Assess error. Tensor ndim: " +
                        this.ndimension + "\tMinNDim: $minNDim")
            } else if (this.ndimension == minNDim) {
                val idx = i * this.stride[0] +
                        j * this.stride[1] +
                        k * this.stride[2] +
                        l * this.stride[3] +
                        m * this.stride[4] +
                        n * this.stride[5] +
                        o * this.stride[6]
                return TorchTensor(Array<Number>(1) { this.storage[idx] })
            }

            // Re-create storage to deal with multi-dimensional array
            val newSize = this.shape.subList(minNDim, this.ndimension)
            // val newStride = this.stride.subList(minNDim, this.ndimension)
            val newStride = computeStride(newSize)
            val N = newSize.reduce { acc, i -> acc * i }
            val newStorage = Array(N) {0 as Number}.toMutableList()
            // val newStorage = emptyArray<Number>().toMutableList()
            when (this.ndimension) {
                8 -> {
                    for (idx1 in 0 until this.shape[7]) {
                        val aIdx = idx1 * newStride[0]
                        val bIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] + l * this.stride[3] + m * this.stride[4] + n * this.stride[5] + o * this.stride[6] +
                                idx1 * this.stride[7]
                        // newStorage[aIdx] = this.storage[bIdx]
                        newStorage.add(aIdx, this.storage[bIdx])
                    }
                }
                else -> {
                    throw NotImplementedError()
                }
            }
            var output = TorchTensor(size = newSize, storage = newStorage, stride = newStride)
            output.dtype = this.dtype
            return output
        }

        operator fun get(i: Int, j: Int, k: Int, l: Int, m: Int, n: Int, o: Int, p: Int): TorchTensor {
            // Check correctness call or return value directly
            val minNDim = 8
            if (this.ndimension < minNDim) {
                throw Exception("Assess error. Tensor ndim: " +
                        this.ndimension + "\tMinNDim: $minNDim")
            } else if (this.ndimension == minNDim) {
                val idx = i * this.stride[0] +
                        j * this.stride[1] +
                        k * this.stride[2] +
                        l * this.stride[3] +
                        m * this.stride[4] +
                        n * this.stride[5] +
                        o * this.stride[6] +
                        p * this.stride[7]
                return TorchTensor(Array<Number>(1) { this.storage[idx] })
            } else {
                throw NotImplementedError()
            }
        }

        /**
         *  You should not call this function directly!
         *  Get the value from storage directly for given index
         *
         *  @param  idx     The index of element you want to get
         *  @return         The corresponding value
         */
        fun _get(idx: Int): Number {
            return this.storage[idx]
        }

        /**
         *  You should not call this function directly!
         *  Get the storage array directly
         *
         *  @return         The storage array
         */
        public fun _array(): MutableList<Number> {
            return this.storage
        }

        /**
         *  Alias of PyTorch dataAsTypeArray
         *  Actually, the format of 1D array is same as this.storage
         *  So, we just cast it and return
         *  Ref: https://github.com/pytorch/pytorch/blob/master/android/pytorch_android/src/main/java/org/pytorch/Tensor.java#L458-L479
         */
        public fun dataAsFloatArray(): FloatArray {
            var output = FloatArray(this.storage.size)
            for (i in 0 until this.storage.size) {
                output[i] = this.storage[i].toFloat()
            }
            return output
        }

        public fun dataAsDoubleArray(): DoubleArray {
            var output = DoubleArray(this.storage.size)
            for (i in 0 until this.storage.size) {
                output[i] = this.storage[i].toDouble()
            }
            return output
        }

        public fun dataAsLongArray(): LongArray {
            var output = LongArray(this.storage.size)
            for (i in 0 until this.storage.size) {
                output[i] = this.storage[i].toLong()
            }
            return output
        }

        /**
         *  You should not call this function directly!
         *  Get the total number of elements in the storage array
         *
         *  @return         Total number of elements
         */
        public fun _nElement(): Int {
            return this.nElements
        }

        /**
         *  Returns the value of this tensor as a standard Kotlin number.
         *  This only works for tensors with one element.
         *
         *  @return         The number in the tensor
         */
        public fun item(): Number {
            if (this.shape == mutableListOf(1)) {
                return this.storage[0]
            } else {
                throw ValueError("ValueError: only one element tensors can be converted to Kotlin scalars")
            }
        }

        /**
         *  ========================================= Set value (Single value) =========================================
         */
        operator fun set(i: Int, value: Number): TorchTensor {
            // Check correctness call or set value directly
            val minNDim = 1
            val newStorage = this._array()
            if (this.ndimension < minNDim) {
                throw Exception("Assess error. Tensor ndim: " +
                        this.ndimension + "\tMinNDim: $minNDim")
            } else if (this.ndimension == minNDim) {
                newStorage[i] = value
            } else {
                // Deal with multi-dimensional cases
                when (this.ndimension) {
                    2 -> {
                        for (idx1 in 0 until this.shape[1]) {
                            val aIdx = i * this.stride[0] +
                                    idx1 * this.stride[1]
                            newStorage[aIdx] = value
                        }
                    }
                    3 -> {
                        for (idx1 in 0 until this.shape[1]) {
                            for (idx2 in 0 until this.shape[2]) {
                                val aIdx = i * this.stride[0] +
                                        idx1 * this.stride[1] +
                                        idx2 * this.stride[2]
                                newStorage[aIdx] = value
                            }
                        }
                    }
                    4 -> {
                        for (idx1 in 0 until this.shape[1]) {
                            for (idx2 in 0 until this.shape[2]) {
                                for (idx3 in 0 until this.shape[3]) {
                                    val aIdx = i * this.stride[0] +
                                            idx1 * this.stride[1] +
                                            idx2 * this.stride[2] +
                                            idx3 * this.stride[3]
                                    newStorage[aIdx] = value
                                }
                            }
                        }
                    }
                    5 -> {
                        for (idx1 in 0 until this.shape[1]) {
                            for (idx2 in 0 until this.shape[2]) {
                                for (idx3 in 0 until this.shape[3]) {
                                    for (idx4 in 0 until this.shape[4]) {
                                        val aIdx = i * this.stride[0] +
                                                idx1 * this.stride[1] +
                                                idx2 * this.stride[2] +
                                                idx3 * this.stride[3] +
                                                idx4 * this.stride[4]
                                        newStorage[aIdx] = value
                                    }
                                }
                            }
                        }
                    }
                    6 -> {
                        for (idx1 in 0 until this.shape[1]) {
                            for (idx2 in 0 until this.shape[2]) {
                                for (idx3 in 0 until this.shape[3]) {
                                    for (idx4 in 0 until this.shape[4]) {
                                        for (idx5 in 0 until this.shape[5]) {
                                            val aIdx = i * this.stride[0] +
                                                    idx1 * this.stride[1] +
                                                    idx2 * this.stride[2] +
                                                    idx3 * this.stride[3] +
                                                    idx4 * this.stride[4] +
                                                    idx5 * this.stride[5]
                                            newStorage[aIdx] = value
                                        }
                                    }
                                }
                            }
                        }
                    }
                    7 -> {
                        for (idx1 in 0 until this.shape[1]) {
                            for (idx2 in 0 until this.shape[2]) {
                                for (idx3 in 0 until this.shape[3]) {
                                    for (idx4 in 0 until this.shape[4]) {
                                        for (idx5 in 0 until this.shape[5]) {
                                            for (idx6 in 0 until this.shape[6]) {
                                                val aIdx = i * this.stride[0] +
                                                        idx1 * this.stride[1] +
                                                        idx2 * this.stride[2] +
                                                        idx3 * this.stride[3] +
                                                        idx4 * this.stride[4] +
                                                        idx5 * this.stride[5] +
                                                        idx6 * this.stride[6]
                                                newStorage[aIdx] = value
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    8 -> {
                        for (idx1 in 0 until this.shape[1]) {
                            for (idx2 in 0 until this.shape[2]) {
                                for (idx3 in 0 until this.shape[3]) {
                                    for (idx4 in 0 until this.shape[4]) {
                                        for (idx5 in 0 until this.shape[5]) {
                                            for (idx6 in 0 until this.shape[6]) {
                                                for (idx7 in 0 until this.shape[7]) {
                                                    val aIdx = i * this.stride[0] +
                                                            idx1 * this.stride[1] +
                                                            idx2 * this.stride[2] +
                                                            idx3 * this.stride[3] +
                                                            idx4 * this.stride[4] +
                                                            idx5 * this.stride[5] +
                                                            idx6 * this.stride[6] +
                                                            idx7 * this.stride[7]
                                                    newStorage[aIdx] = value
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else -> {
                        throw NotImplementedError()
                    }
                }
            }

            // in-place alter
            // return TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            val output = TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        operator fun set(i: Int, j: Int, value: Number): TorchTensor {
            // Check correctness call or set value directly
            val minNDim = 2
            val newStorage = this._array()
            if (this.ndimension < minNDim) {
                throw Exception("Assess error. Tensor ndim: " +
                        this.ndimension + "\tMinNDim: $minNDim")
            } else if (this.ndimension == minNDim) {
                val idx = i * this.stride[0] + j
                newStorage[idx] = value
            } else {
                // Deal with multi-dimensional cases
                when (this.ndimension) {
                    3 -> {
                        for (idx1 in 0 until this.shape[2]) {
                            val aIdx = i * this.stride[0] + j * this.stride[1] +
                                    idx1 * this.stride[2]
                            newStorage[aIdx] = value
                        }
                    }
                    4 -> {
                        for (idx1 in 0 until this.shape[2]) {
                            for (idx2 in 0 until this.shape[3]) {
                                val aIdx = i * this.stride[0] + j * this.stride[1] +
                                        idx1 * this.stride[2] +
                                        idx2 * this.stride[3]
                                newStorage[aIdx] = value
                            }
                        }
                    }
                    5 -> {
                        for (idx1 in 0 until this.shape[2]) {
                            for (idx2 in 0 until this.shape[3]) {
                                for (idx3 in 0 until this.shape[4]) {
                                    val aIdx = i * this.stride[0] + j * this.stride[1] +
                                            idx1 * this.stride[2] +
                                            idx2 * this.stride[3] +
                                            idx3 * this.stride[4]
                                    newStorage[aIdx] = value
                                }
                            }
                        }
                    }
                    6 -> {
                        for (idx1 in 0 until this.shape[2]) {
                            for (idx2 in 0 until this.shape[3]) {
                                for (idx3 in 0 until this.shape[4]) {
                                    for (idx4 in 0 until this.shape[5]) {
                                        val aIdx = i * this.stride[0] + j * this.stride[1] +
                                                idx1 * this.stride[2] +
                                                idx2 * this.stride[3] +
                                                idx3 * this.stride[4] +
                                                idx4 * this.stride[5]
                                        newStorage[aIdx] = value
                                    }
                                }
                            }
                        }
                    }
                    7 -> {
                        for (idx1 in 0 until this.shape[2]) {
                            for (idx2 in 0 until this.shape[3]) {
                                for (idx3 in 0 until this.shape[4]) {
                                    for (idx4 in 0 until this.shape[5]) {
                                        for (idx5 in 0 until this.shape[6]) {
                                            val aIdx = i * this.stride[0] + j * this.stride[1] +
                                                    idx1 * this.stride[2] +
                                                    idx2 * this.stride[3] +
                                                    idx3 * this.stride[4] +
                                                    idx4 * this.stride[5] +
                                                    idx5 * this.stride[6]
                                            newStorage[aIdx] = value
                                        }
                                    }
                                }
                            }
                        }
                    }
                    8 -> {
                        for (idx1 in 0 until this.shape[2]) {
                            for (idx2 in 0 until this.shape[3]) {
                                for (idx3 in 0 until this.shape[4]) {
                                    for (idx4 in 0 until this.shape[5]) {
                                        for (idx5 in 0 until this.shape[6]) {
                                            for (idx6 in 0 until this.shape[7]) {
                                                val aIdx = i * this.stride[0] + j * this.stride[1] +
                                                        idx1 * this.stride[2] +
                                                        idx2 * this.stride[3] +
                                                        idx3 * this.stride[4] +
                                                        idx4 * this.stride[5] +
                                                        idx5 * this.stride[6] +
                                                        idx6 * this.stride[7]
                                                newStorage[aIdx] = value
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else -> {
                        throw NotImplementedError()
                    }
                }
            }

            // in-place alter
            // return TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            val output = TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        operator fun set(i: Int, j: Int, k: Int, value: Number): TorchTensor {
            // Check correctness call or set value directly
            val minNDim = 3
            val newStorage = this._array()
            if (this.ndimension < minNDim) {
                throw Exception("Assess error. Tensor ndim: " +
                        this.ndimension + "\tMinNDim: $minNDim")
            } else if (this.ndimension == minNDim) {
                val idx = i * this.stride[0] + j * this.stride[1] + k
                newStorage[idx] = value
            } else {
                // Deal with multi-dimensional cases
                when (this.ndimension) {
                    4 -> {
                        for (idx1 in 0 until this.shape[3]) {
                            val aIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] +
                                    idx1 * this.stride[3]
                            newStorage[aIdx] = value
                        }
                    }
                    5 -> {
                        for (idx1 in 0 until this.shape[3]) {
                            for (idx2 in 0 until this.shape[4]) {
                                val aIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] +
                                        idx1 * this.stride[3] +
                                        idx2 * this.stride[4]
                                newStorage[aIdx] = value
                            }
                        }
                    }
                    6 -> {
                        for (idx1 in 0 until this.shape[3]) {
                            for (idx2 in 0 until this.shape[4]) {
                                for (idx3 in 0 until this.shape[5]) {
                                    val aIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] +
                                            idx1 * this.stride[3] +
                                            idx2 * this.stride[4] +
                                            idx3 * this.stride[5]
                                    newStorage[aIdx] = value
                                }
                            }
                        }
                    }
                    7 -> {
                        for (idx1 in 0 until this.shape[3]) {
                            for (idx2 in 0 until this.shape[4]) {
                                for (idx3 in 0 until this.shape[5]) {
                                    for (idx4 in 0 until this.shape[6]) {
                                        val aIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] +
                                                idx1 * this.stride[3] +
                                                idx2 * this.stride[4] +
                                                idx3 * this.stride[5] +
                                                idx4 * this.stride[6]
                                        newStorage[aIdx] = value
                                    }
                                }
                            }
                        }
                    }
                    8 -> {
                        for (idx1 in 0 until this.shape[3]) {
                            for (idx2 in 0 until this.shape[4]) {
                                for (idx3 in 0 until this.shape[5]) {
                                    for (idx4 in 0 until this.shape[6]) {
                                        for (idx5 in 0 until this.shape[7]) {
                                            val aIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] +
                                                    idx1 * this.stride[3] +
                                                    idx2 * this.stride[4] +
                                                    idx3 * this.stride[5] +
                                                    idx4 * this.stride[6] +
                                                    idx5 * this.stride[7]
                                            newStorage[aIdx] = value
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else -> {
                        throw NotImplementedError()
                    }
                }
            }

            // in-place alter
            // return TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            val output = TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        operator fun set(i: Int, j: Int, k: Int, l: Int, value: Number): TorchTensor {
            // Check correctness call or set value directly
            val minNDim = 4
            val newStorage = this._array()
            if (this.ndimension < minNDim) {
                throw Exception("Assess error. Tensor ndim: " +
                        this.ndimension + "\tMinNDim: $minNDim")
            } else if (this.ndimension == minNDim) {
                val idx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] + l
                newStorage[idx] = value
            } else {
                // Deal with multi-dimensional cases
                when (this.ndimension) {
                    5 -> {
                        for (idx1 in 0 until this.shape[4]) {
                            val aIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] + l * this.stride[3] +
                                    idx1 * this.stride[4]
                            newStorage[aIdx] = value
                        }
                    }
                    6 -> {
                        for (idx1 in 0 until this.shape[4]) {
                            for (idx2 in 0 until this.shape[5]) {
                                val aIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] + l * this.stride[3] +
                                        idx1 * this.stride[4] +
                                        idx2 * this.stride[5]
                                newStorage[aIdx] = value
                            }
                        }
                    }
                    7 -> {
                        for (idx1 in 0 until this.shape[4]) {
                            for (idx2 in 0 until this.shape[5]) {
                                for (idx3 in 0 until this.shape[6]) {
                                    val aIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] + l * this.stride[3] +
                                            idx1 * this.stride[4] +
                                            idx2 * this.stride[5] +
                                            idx3 * this.stride[6]
                                    newStorage[aIdx] = value
                                }
                            }
                        }
                    }
                    8 -> {
                        for (idx1 in 0 until this.shape[4]) {
                            for (idx2 in 0 until this.shape[5]) {
                                for (idx3 in 0 until this.shape[6]) {
                                    for (idx4 in 0 until this.shape[7]) {
                                        val aIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] + l * this.stride[3] +
                                                idx1 * this.stride[4] +
                                                idx2 * this.stride[5] +
                                                idx3 * this.stride[6] +
                                                idx4 * this.stride[7]
                                        newStorage[aIdx] = value
                                    }
                                }
                            }
                        }
                    }
                    else -> {
                        throw NotImplementedError()
                    }
                }
            }

            // in-place alter
            // return TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            val output = TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        operator fun set(i: Int, j: Int, k: Int, l: Int, m: Int, value: Number): TorchTensor {
            // Check correctness call or set value directly
            val minNDim = 5
            val newStorage = this._array()
            if (this.ndimension < minNDim) {
                throw Exception("Assess error. Tensor ndim: " +
                        this.ndimension + "\tMinNDim: $minNDim")
            } else if (this.ndimension == minNDim) {
                val idx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] + l * this.stride[3] + m
                newStorage[idx] = value
            } else {
                // Deal with multi-dimensional cases
                when (this.ndimension) {
                    6 -> {
                        for (idx1 in 0 until this.shape[5]) {
                            val aIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] + l * this.stride[3] + m * this.stride[4] +
                                    idx1 * this.stride[5]
                            newStorage[aIdx] = value
                        }
                    }
                    7 -> {
                        for (idx1 in 0 until this.shape[5]) {
                            for (idx2 in 0 until this.shape[6]) {
                                val aIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] + l * this.stride[3] + m * this.stride[4] +
                                        idx1 * this.stride[5] +
                                        idx2 * this.stride[6]
                                newStorage[aIdx] = value
                            }
                        }
                    }
                    8 -> {
                        for (idx1 in 0 until this.shape[5]) {
                            for (idx2 in 0 until this.shape[6]) {
                                for (idx3 in 0 until this.shape[7]) {
                                    val aIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] + l * this.stride[3] + m * this.stride[4] +
                                            idx1 * this.stride[5] +
                                            idx2 * this.stride[6] +
                                            idx3 * this.stride[7]
                                    newStorage[aIdx] = value
                                }
                            }
                        }
                    }
                    else -> {
                        throw NotImplementedError()
                    }
                }
            }

            // in-place alter
            // return TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            val output = TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        operator fun set(i: Int, j: Int, k: Int, l: Int, m: Int, n: Int, value: Number): TorchTensor {
            // Check correctness call or set value directly
            val minNDim = 6
            val newStorage = this._array()
            if (this.ndimension < minNDim) {
                throw Exception("Assess error. Tensor ndim: " +
                        this.ndimension + "\tMinNDim: $minNDim")
            } else if (this.ndimension == minNDim) {
                val idx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] + l * this.stride[3] + m * this.stride[4] + n
                newStorage[idx] = value
            } else {
                // Deal with multi-dimensional cases
                when (this.ndimension) {
                    7 -> {
                        for (idx1 in 0 until this.shape[6]) {
                            val aIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] + l * this.stride[3] + m * this.stride[4] + n * this.stride[5] +
                                    idx1 * this.stride[6]
                            newStorage[aIdx] = value
                        }
                    }
                    8 -> {
                        for (idx1 in 0 until this.shape[6]) {
                            for (idx2 in 0 until this.shape[7]) {
                                val aIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] + l * this.stride[3] + m * this.stride[4] + n * this.stride[5] +
                                        idx1 * this.stride[6] +
                                        idx2 * this.stride[7]
                                newStorage[aIdx] = value
                            }
                        }
                    }
                    else -> {
                        throw NotImplementedError()
                    }
                }
            }

            // in-place alter
            // return TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            val output = TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        operator fun set(i: Int, j: Int, k: Int, l: Int, m: Int, n: Int, o: Int, value: Number): TorchTensor {
            // Check correctness call or set value directly
            val minNDim = 7
            val newStorage = this._array()
            if (this.ndimension < minNDim) {
                throw Exception("Assess error. Tensor ndim: " +
                        this.ndimension + "\tMinNDim: $minNDim")
            } else if (this.ndimension == minNDim) {
                val idx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] + l * this.stride[3] + m * this.stride[4] + n * this.stride[5] + o
                newStorage[idx] = value
            } else {
                // Deal with multi-dimensional cases
                when (this.ndimension) {
                    8 -> {
                        for (idx1 in 0 until this.shape[7]) {
                            val aIdx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] + l * this.stride[3] + m * this.stride[4] + n * this.stride[5] + o * this.stride[6] +
                                    idx1 * this.stride[7]
                            newStorage[aIdx] = value
                        }
                    }
                    else -> {
                        throw NotImplementedError()
                    }
                }
            }

            // in-place alter
            // return TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            val output = TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        operator fun set(i: Int, j: Int, k: Int, l: Int, m: Int, n: Int, o: Int, p: Int, value: Number): TorchTensor {
            // Check correctness call or set value directly
            val minNDim = 8
            val newStorage = this._array()
            if (this.ndimension < minNDim) {
                throw Exception("Assess error. Tensor ndim: " +
                        this.ndimension + "\tMinNDim: $minNDim")
            } else if (this.ndimension == minNDim) {
                val idx = i * this.stride[0] + j * this.stride[1] + k * this.stride[2] + l * this.stride[3] + m * this.stride[4] + n * this.stride[5] + o * this.stride[6] + p
                newStorage[idx] = value
            }

            // in-place alter
            // return TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            val output = TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        /**
         *  ========================================= Set value (Tensor) =========================================
         */
        operator fun set(i: Int, tensor: TorchTensor): TorchTensor {
            if (tensor.shape != this.shape.subList(1, this.ndimension)) {
                throw Exception("shape not match. ")
            }
            val newStorage = this.storage
            val aSize = this.shape
            val aStride = this.stride
            val bStride = tensor.stride
            when (tensor.ndimension) {
                1 -> {
                    for (idx1 in 0 until aSize[1]) {
                        val aIdx = i * aStride[0] + idx1
                        val bIdx = idx1
                        newStorage[aIdx] = tensor._array()[bIdx]
                    }
                }
                2 -> {
                    for (idx1 in 0 until aSize[1]) {
                        for (idx2 in 0 until aSize[2]) {
                            val aIdx = i * aStride[0] +
                                    idx1 * aStride[1] +
                                    idx2
                            val bIdx = idx1 * bStride[0] +
                                    idx2
                            newStorage[aIdx] = tensor._array()[bIdx]
                        }
                    }
                }
                3 -> {
                    for (idx1 in 0 until aSize[1]) {
                        for (idx2 in 0 until aSize[2]) {
                            for (idx3 in 0 until aSize[3]) {
                                val aIdx = i * aStride[0] +
                                        idx1 * aStride[1] +
                                        idx2 * aStride[2] +
                                        idx3
                                val bIdx = idx1 * bStride[0] +
                                        idx2 * bStride[1] +
                                        idx3
                                newStorage[aIdx] = tensor._array()[bIdx]
                            }
                        }
                    }
                }
                4 -> {
                    for (idx1 in 0 until aSize[1]) {
                        for (idx2 in 0 until aSize[2]) {
                            for (idx3 in 0 until aSize[3]) {
                                for (idx4 in 0 until aSize[4]) {
                                    val aIdx = i * aStride[0] +
                                            idx1 * aStride[1] +
                                            idx2 * aStride[2] +
                                            idx3 * aStride[3] +
                                            idx4
                                    val bIdx = idx1 * bStride[0] +
                                            idx2 * bStride[1] +
                                            idx3 * bStride[2] +
                                            idx4
                                    newStorage[aIdx] = tensor._array()[bIdx]
                                }
                            }
                        }
                    }
                }
                5 -> {
                    for (idx1 in 0 until aSize[1]) {
                        for (idx2 in 0 until aSize[2]) {
                            for (idx3 in 0 until aSize[3]) {
                                for (idx4 in 0 until aSize[4]) {
                                    for (idx5 in 0 until aSize[5]) {
                                        val aIdx = i * aStride[0] +
                                                idx1 * aStride[1] +
                                                idx2 * aStride[2] +
                                                idx3 * aStride[3] +
                                                idx4 * aStride[4] +
                                                idx5
                                        val bIdx = idx1 * bStride[0] +
                                                idx2 * bStride[1] +
                                                idx3 * bStride[2] +
                                                idx4 * bStride[3] +
                                                idx5
                                        newStorage[aIdx] = tensor._array()[bIdx]
                                    }
                                }
                            }
                        }
                    }
                }
                6 -> {
                    for (idx1 in 0 until aSize[1]) {
                        for (idx2 in 0 until aSize[2]) {
                            for (idx3 in 0 until aSize[3]) {
                                for (idx4 in 0 until aSize[4]) {
                                    for (idx5 in 0 until aSize[5]) {
                                        for (idx6 in 0 until aSize[6]) {
                                            val aIdx = i * aStride[0] +
                                                    idx1 * aStride[1] +
                                                    idx2 * aStride[2] +
                                                    idx3 * aStride[3] +
                                                    idx4 * aStride[4] +
                                                    idx5 * aStride[5] +
                                                    idx6
                                            val bIdx = idx1 * bStride[0] +
                                                    idx2 * bStride[1] +
                                                    idx3 * bStride[2] +
                                                    idx4 * bStride[3] +
                                                    idx5 * bStride[4] +
                                                    idx6
                                            newStorage[aIdx] = tensor._array()[bIdx]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                7 -> {
                    for (idx1 in 0 until aSize[1]) {
                        for (idx2 in 0 until aSize[2]) {
                            for (idx3 in 0 until aSize[3]) {
                                for (idx4 in 0 until aSize[4]) {
                                    for (idx5 in 0 until aSize[5]) {
                                        for (idx6 in 0 until aSize[6]) {
                                            for (idx7 in 0 until aSize[7]) {
                                                val aIdx = i * aStride[0] +
                                                        idx1 * aStride[1] +
                                                        idx2 * aStride[2] +
                                                        idx3 * aStride[3] +
                                                        idx4 * aStride[4] +
                                                        idx5 * aStride[5] +
                                                        idx6 * aStride[6] +
                                                        idx7
                                                val bIdx = idx1 * bStride[0] +
                                                        idx2 * bStride[1] +
                                                        idx3 * bStride[2] +
                                                        idx4 * bStride[3] +
                                                        idx5 * bStride[4] +
                                                        idx6 * bStride[5] +
                                                        idx7
                                                newStorage[aIdx] = tensor._array()[bIdx]
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                else -> {
                    throw NotImplementedError()
                }
            }

            // in-place alter
            // return TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            val output = TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        operator fun set(i: Int, j: Int, tensor: TorchTensor): TorchTensor {
            if (tensor.shape != this.shape.subList(2, this.ndimension)) {
                throw Exception("shape not match. ")
            }
            val newStorage = this.storage
            val aSize = this.shape
            val aStride = this.stride
            val bStride = tensor.stride
            when (tensor.ndimension) {
                1 -> {
                    for (idx1 in 0 until aSize[2]) {
                        val aIdx = i * aStride[0] + j * aStride[1] +
                                idx1
                        val bIdx = idx1 * bStride[0]
                        newStorage[aIdx] = tensor._array()[bIdx]
                    }
                }
                2 -> {
                    for (idx1 in 0 until aSize[2]) {
                        for (idx2 in 0 until aSize[3]) {
                            val aIdx = i * aStride[0] + j * aStride[1] +
                                    idx1 * aStride[2] +
                                    idx2
                            val bIdx = idx1 * bStride[0] +
                                    idx2 * bStride[1]
                            newStorage[aIdx] = tensor._array()[bIdx]
                        }
                    }
                }
                3 -> {
                    for (idx1 in 0 until aSize[2]) {
                        for (idx2 in 0 until aSize[3]) {
                            for (idx3 in 0 until aSize[4]) {
                                val aIdx = i * aStride[0] + j * aStride[1] +
                                        idx1 * aStride[2] +
                                        idx2 * aStride[3] +
                                        idx3
                                val bIdx = idx1 * bStride[0] +
                                        idx2 * bStride[1] +
                                        idx3 * bStride[2]
                                newStorage[aIdx] = tensor._array()[bIdx]
                            }
                        }
                    }
                }
                4 -> {
                    for (idx1 in 0 until aSize[2]) {
                        for (idx2 in 0 until aSize[3]) {
                            for (idx3 in 0 until aSize[4]) {
                                for (idx4 in 0 until aSize[5]) {
                                    val aIdx = i * aStride[0] + j * aStride[1] +
                                            idx1 * aStride[2] +
                                            idx2 * aStride[3] +
                                            idx3 * aStride[4] +
                                            idx4
                                    val bIdx = idx1 * bStride[0] +
                                            idx2 * bStride[1] +
                                            idx3 * bStride[2] +
                                            idx4 * bStride[3]
                                    newStorage[aIdx] = tensor._array()[bIdx]
                                }
                            }
                        }
                    }
                }
                5 -> {
                    for (idx1 in 0 until aSize[2]) {
                        for (idx2 in 0 until aSize[3]) {
                            for (idx3 in 0 until aSize[4]) {
                                for (idx4 in 0 until aSize[5]) {
                                    for (idx5 in 0 until aSize[6]) {
                                        val aIdx = i * aStride[0] + j * aStride[1] +
                                                idx1 * aStride[2] +
                                                idx2 * aStride[3] +
                                                idx3 * aStride[4] +
                                                idx4 * aStride[5] +
                                                idx5
                                        val bIdx = idx1 * bStride[0] +
                                                idx2 * bStride[1] +
                                                idx3 * bStride[2] +
                                                idx4 * bStride[3] +
                                                idx5 * bStride[4]
                                        newStorage[aIdx] = tensor._array()[bIdx]
                                    }
                                }
                            }
                        }
                    }
                }
                6 -> {
                    for (idx1 in 0 until aSize[2]) {
                        for (idx2 in 0 until aSize[3]) {
                            for (idx3 in 0 until aSize[4]) {
                                for (idx4 in 0 until aSize[5]) {
                                    for (idx5 in 0 until aSize[6]) {
                                        for (idx6 in 0 until aSize[7]) {
                                            val aIdx = i * aStride[0] + j * aStride[1] +
                                                    idx1 * aStride[2] +
                                                    idx2 * aStride[3] +
                                                    idx3 * aStride[4] +
                                                    idx4 * aStride[5] +
                                                    idx5 * aStride[6] +
                                                    idx6
                                            val bIdx = idx1 * bStride[0] +
                                                    idx2 * bStride[1] +
                                                    idx3 * bStride[2] +
                                                    idx4 * bStride[3] +
                                                    idx5 * bStride[4] +
                                                    idx6 * bStride[5]
                                            newStorage[aIdx] = tensor._array()[bIdx]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                else -> {
                    throw NotImplementedError()
                }
            }

            // in-place alter
            // return TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            val output = TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        operator fun set(i: Int, j: Int, k: Int, tensor: TorchTensor): TorchTensor {
            if (tensor.shape != this.shape.subList(3, this.ndimension)) {
                throw Exception("shape not match. ")
            }
            val newStorage = this.storage
            val aSize = this.shape
            val aStride = this.stride
            val bStride = tensor.stride
            when (tensor.ndimension) {
                1 -> {
                    for (idx1 in 0 until aSize[3]) {
                        val aIdx = i * aStride[0] + j * aStride[1] + k * aStride[2] +
                                idx1
                        val bIdx = idx1 * bStride[0]
                        newStorage[aIdx] = tensor._array()[bIdx]
                    }
                }
                2 -> {
                    for (idx1 in 0 until aSize[3]) {
                        for (idx2 in 0 until aSize[4]) {
                            val aIdx = i * aStride[0] + j * aStride[1] + k * aStride[2] +
                                    idx1 * aStride[3] +
                                    idx2
                            val bIdx = idx1 * bStride[0] +
                                    idx2 * bStride[1]
                            newStorage[aIdx] = tensor._array()[bIdx]
                        }
                    }
                }
                3 -> {
                    for (idx1 in 0 until aSize[3]) {
                        for (idx2 in 0 until aSize[4]) {
                            for (idx3 in 0 until aSize[5]) {
                                val aIdx = i * aStride[0] + j * aStride[1] + k * aStride[2] +
                                        idx1 * aStride[3] +
                                        idx2 * aStride[4] +
                                        idx3
                                val bIdx = idx1 * bStride[0] +
                                        idx2 * bStride[1] +
                                        idx3 * bStride[2]
                                newStorage[aIdx] = tensor._array()[bIdx]
                            }
                        }
                    }
                }
                4 -> {
                    for (idx1 in 0 until aSize[3]) {
                        for (idx2 in 0 until aSize[4]) {
                            for (idx3 in 0 until aSize[5]) {
                                for (idx4 in 0 until aSize[6]) {
                                    val aIdx = i * aStride[0] + j * aStride[1] + k * aStride[2] +
                                            idx1 * aStride[3] +
                                            idx2 * aStride[4] +
                                            idx3 * aStride[5] +
                                            idx4
                                    val bIdx = idx1 * bStride[0] +
                                            idx2 * bStride[1] +
                                            idx3 * bStride[2] +
                                            idx4 * bStride[3]
                                    newStorage[aIdx] = tensor._array()[bIdx]
                                }
                            }
                        }
                    }
                }
                5 -> {
                    for (idx1 in 0 until aSize[3]) {
                        for (idx2 in 0 until aSize[4]) {
                            for (idx3 in 0 until aSize[5]) {
                                for (idx4 in 0 until aSize[6]) {
                                    for (idx5 in 0 until aSize[7]) {
                                        val aIdx = i * aStride[0] + j * aStride[1] + k * aStride[2] +
                                                idx1 * aStride[3] +
                                                idx2 * aStride[4] +
                                                idx3 * aStride[5] +
                                                idx4 * aStride[6] +
                                                idx5
                                        val bIdx = idx1 * bStride[0] +
                                                idx2 * bStride[1] +
                                                idx3 * bStride[2] +
                                                idx4 * bStride[3] +
                                                idx5 * bStride[4]
                                        newStorage[aIdx] = tensor._array()[bIdx]
                                    }
                                }
                            }
                        }
                    }
                }
                else -> {
                    throw NotImplementedError()
                }
            }

            // in-place alter
            // return TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            val output = TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        operator fun set(i: Int, j: Int, k: Int, l: Int, tensor: TorchTensor): TorchTensor {
            if (tensor.shape != this.shape.subList(4, this.ndimension)) {
                throw Exception("shape not match. ")
            }
            val newStorage = this.storage
            val aSize = this.shape
            val aStride = this.stride
            val bStride = tensor.stride
            when (tensor.ndimension) {
                1 -> {
                    for (idx1 in 0 until aSize[4]) {
                        val aIdx = i * aStride[0] + j * aStride[1] + k * aStride[2] + l * aStride[3] +
                                idx1
                        val bIdx = idx1 * bStride[0]
                        newStorage[aIdx] = tensor._array()[bIdx]
                    }
                }
                2 -> {
                    for (idx1 in 0 until aSize[4]) {
                        for (idx2 in 0 until aSize[5]) {
                            val aIdx = i * aStride[0] + j * aStride[1] + k * aStride[2] + l * aStride[3] +
                                    idx1 * aStride[4] +
                                    idx2
                            val bIdx = idx1 * bStride[0] +
                                    idx2 * bStride[1]
                            newStorage[aIdx] = tensor._array()[bIdx]
                        }
                    }
                }
                3 -> {
                    for (idx1 in 0 until aSize[4]) {
                        for (idx2 in 0 until aSize[5]) {
                            for (idx3 in 0 until aSize[6]) {
                                val aIdx = i * aStride[0] + j * aStride[1] + k * aStride[2] + l * aStride[3] +
                                        idx1 * aStride[4] +
                                        idx2 * aStride[5] +
                                        idx3
                                val bIdx = idx1 * bStride[0] +
                                        idx2 * bStride[1] +
                                        idx3 * bStride[2]
                                newStorage[aIdx] = tensor._array()[bIdx]
                            }
                        }
                    }
                }
                4 -> {
                    for (idx1 in 0 until aSize[4]) {
                        for (idx2 in 0 until aSize[5]) {
                            for (idx3 in 0 until aSize[6]) {
                                for (idx4 in 0 until aSize[7]) {
                                    val aIdx = i * aStride[0] + j * aStride[1] + k * aStride[2] + l * aStride[3] +
                                            idx1 * aStride[4] +
                                            idx2 * aStride[5] +
                                            idx3 * aStride[6] +
                                            idx4
                                    val bIdx = idx1 * bStride[0] +
                                            idx2 * bStride[1] +
                                            idx3 * bStride[2] +
                                            idx4 * bStride[3]
                                    newStorage[aIdx] = tensor._array()[bIdx]
                                }
                            }
                        }
                    }
                }
                else -> {
                    throw NotImplementedError()
                }
            }

            // in-place alter
            // return TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            val output = TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        operator fun set(i: Int, j: Int, k: Int, l: Int, m: Int, tensor: TorchTensor): TorchTensor {
            if (tensor.shape != this.shape.subList(5, this.ndimension)) {
                throw Exception("shape not match. ")
            }
            val newStorage = this.storage
            val aSize = this.shape
            val aStride = this.stride
            val bStride = tensor.stride
            when (tensor.ndimension) {
                1 -> {
                    for (idx1 in 0 until aSize[5]) {
                        val aIdx = i * aStride[0] + j * aStride[1] + k * aStride[2] + l * aStride[3] + m * aStride[4] +
                                idx1
                        val bIdx = idx1 * bStride[0]
                        newStorage[aIdx] = tensor._array()[bIdx]
                    }
                }
                2 -> {
                    for (idx1 in 0 until aSize[5]) {
                        for (idx2 in 0 until aSize[6]) {
                            val aIdx = i * aStride[0] + j * aStride[1] + k * aStride[2] + l * aStride[3] + m * aStride[4] +
                                    idx1 * aStride[5] +
                                    idx2
                            val bIdx = idx1 * bStride[0] +
                                    idx2 * bStride[1]
                            newStorage[aIdx] = tensor._array()[bIdx]
                        }
                    }
                }
                3 -> {
                    for (idx1 in 0 until aSize[5]) {
                        for (idx2 in 0 until aSize[6]) {
                            for (idx3 in 0 until aSize[7]) {
                                val aIdx = i * aStride[0] + j * aStride[1] + k * aStride[2] + l * aStride[3] + m * aStride[4] +
                                        idx1 * aStride[5] +
                                        idx2 * aStride[6] +
                                        idx3
                                val bIdx = idx1 * bStride[0] +
                                        idx2 * bStride[1] +
                                        idx3 * bStride[2]
                                newStorage[aIdx] = tensor._array()[bIdx]
                            }
                        }
                    }
                }
                else -> {
                    throw NotImplementedError()
                }
            }

            // in-place alter
            // return TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            val output = TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        operator fun set(i: Int, j: Int, k: Int, l: Int, m: Int, n: Int, tensor: TorchTensor): TorchTensor {
            if (tensor.shape != this.shape.subList(6, this.ndimension)) {
                throw Exception("shape not match. ")
            }
            val newStorage = this.storage
            val aSize = this.shape
            val aStride = this.stride
            val bStride = tensor.stride
            when (tensor.ndimension) {
                1 -> {
                    for (idx1 in 0 until aSize[6]) {
                        val aIdx = i * aStride[0] + j * aStride[1] + k * aStride[2] + l * aStride[3] + m * aStride[4] + n * aStride[5] +
                                idx1
                        val bIdx = idx1 * bStride[0]
                        newStorage[aIdx] = tensor._array()[bIdx]
                    }
                }
                2 -> {
                    for (idx1 in 0 until aSize[6]) {
                        for (idx2 in 0 until aSize[7]) {
                            val aIdx = i * aStride[0] + j * aStride[1] + k * aStride[2] + l * aStride[3] + m * aStride[4] + n * aStride[5] +
                                    idx1 * aStride[6] +
                                    idx2
                            val bIdx = idx1 * bStride[0] +
                                    idx2 * bStride[1]
                            newStorage[aIdx] = tensor._array()[bIdx]
                        }
                    }
                }
                else -> {
                    throw NotImplementedError()
                }
            }

            // in-place alter
            // return TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            val output = TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        operator fun set(i: Int, j: Int, k: Int, l: Int, m: Int, n: Int, o: Int, tensor: TorchTensor): TorchTensor {
            if (tensor.shape != this.shape.subList(7, this.ndimension)) {
                throw Exception("shape not match. ")
            }
            val newStorage = this.storage
            val aSize = this.shape
            val aStride = this.stride
            val bStride = tensor.stride
            when (tensor.ndimension) {
                1 -> {
                    for (idx1 in 0 until aSize[7]) {
                        val aIdx = i * aStride[0] + j * aStride[1] + k * aStride[2] + l * aStride[3] + m * aStride[4] + n * aStride[5] + o * aStride[6] +
                                idx1
                        val bIdx = idx1 * bStride[0]
                        newStorage[aIdx] = tensor._array()[bIdx]
                    }
                }
                else -> {
                    throw NotImplementedError()
                }
            }

            // in-place alter
            // return TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            val output = TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        operator fun set(i: Int, j: Int, k: Int, l: Int, m: Int, n: Int, o: Int, p: Int, tensor: TorchTensor): TorchTensor {
            if (tensor.shape != this.shape.subList(8, this.ndimension)) {
                throw Exception("shape not match. ")
            }
            val newStorage = this.storage
            val aSize = this.shape
            val aStride = this.stride
            val bStride = tensor.stride
            when (tensor.ndimension) {
                1 -> {
                    val aIdx = i * aStride[0] + j * aStride[1] + k * aStride[2] + l * aStride[3] + m * aStride[4] + n * aStride[5] + o * aStride[6] + p * aStride[7]
                    newStorage[aIdx] = tensor._array()[0]
                }
                else -> {
                    throw NotImplementedError()
                }
            }

            // in-place alter
            // return TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            val output = TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        /**
         *  You should not call this function directly!
         *  Set the value to the storage directly
         *
         *  @param  idx     The position you want to insert in
         *  @param  value   The value you want to replace as
         */
        fun _set(idx: Int, value: Number) {
            if (idx < this.storage.size) {
                this.storage[idx] = value
            } else {
                throw IndexError("index $idx is out of bounds for dimension 0 with size " +
                        this.storage.size)
            }
        }

        /**
         *  Returns a new tensor with the reciprocal of the elements of input
         *
         *  Note:
         *      (1) Differ from PyTorch, we only support type of double or float
         *
         *  @param  epsilon The dummy term to avoid NaN
         *  @return         The computed result
         */
        fun reciprocal(epsilon: Float = 1e-10f): TorchTensor {
            val newStorage = this.storage;
            when (this.dtype) {
                TensorType.DoubleTensor -> {
                    for (i in 0 until this.nElements) {
                        newStorage[i] = 1 / (newStorage[i].toDouble() + epsilon)
                    }
                }
                TensorType.FloatTensor -> {
                    for (i in 0 until this.nElements) {
                        newStorage[i] = 1 / (newStorage[i].toFloat() + epsilon)
                    }
                }
                else -> throw NotImplementedError("Reciprocal currently only supports type of double or float")
            }

            // in-place alter
            val output = TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        /**
         *  Computes the expit (also known as the logistic sigmoid function) of the elements of input
         *
         *  @param  input   The input tensor
         *  @return         The computed result
         */
        fun sigmoid(): TorchTensor {
            fun _sigmoid(x: Float) = (1.0f / (1.0f + Math.exp((-x).toDouble()))).toFloat()
            val newStorage = this.storage;
            for (i in 0 until this.nElements) {
                newStorage[i] = _sigmoid(newStorage[i].toFloat())
            }

            // in-place alter
            val output = TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        /**
         *  Returns a contiguous in memory tensor containing the same data as self tensor
         *
         *  @return     The contiguous tensor
         */
        fun contiguous(): TorchTensor {
            //var newStorage = Array(this.nElements) { 0f }.toMutableList()
            var newStorage = emptyArray<Number>().toMutableList()
            var newStride = computeStride(this.shape.toMutableList())
            when (dim()) {
                1 -> {
                    newStorage = this.storage
                }
                2 -> {
                    for (idx1 in 0 until this.shape[0]) {
                        for (idx2 in 0 until this.shape[1]) {
                            val aIdx = idx1 * newStride[0] +
                                    idx2
                            val bIdx = idx1 * this.stride[0] +
                                    idx2 * this.stride[1]
                            newStorage.add(aIdx, this.storage[bIdx])
                        }
                    }
                }
                3 -> {
                    for (idx1 in 0 until this.shape[0]) {
                        for (idx2 in 0 until this.shape[1]) {
                            for (idx3 in 0 until this.shape[2]) {
                                val aIdx = idx1 * newStride[0] +
                                        idx2 * newStride[1] +
                                        idx3
                                val bIdx = idx1 * this.stride[0] +
                                        idx2 * this.stride[1] +
                                        idx3 * this.stride[2]
                                newStorage.add(aIdx, this.storage[bIdx])
                            }
                        }
                    }
                }
                4 -> {
                    for (idx1 in 0 until this.shape[0]) {
                        for (idx2 in 0 until this.shape[1]) {
                            for (idx3 in 0 until this.shape[2]) {
                                for (idx4 in 0 until this.shape[3]) {
                                    val aIdx = idx1 * newStride[0] +
                                            idx2 * newStride[1] +
                                            idx3 * newStride[2] +
                                            idx4
                                    val bIdx = idx1 * this.stride[0] +
                                            idx2 * this.stride[1] +
                                            idx3 * this.stride[2] +
                                            idx4 * this.stride[3]
                                    newStorage.add(aIdx, this.storage[bIdx])
                                }
                            }
                        }
                    }
                }
                5 -> {
                    for (idx1 in 0 until this.shape[0]) {
                        for (idx2 in 0 until this.shape[1]) {
                            for (idx3 in 0 until this.shape[2]) {
                                for (idx4 in 0 until this.shape[3]) {
                                    for (idx5 in 0 until this.shape[4]) {
                                        val aIdx = idx1 * newStride[0] +
                                                idx2 * newStride[1] +
                                                idx3 * newStride[2] +
                                                idx4 * newStride[3] +
                                                idx5
                                        val bIdx = idx1 * this.stride[0] +
                                                idx2 * this.stride[1] +
                                                idx3 * this.stride[2] +
                                                idx4 * this.stride[3] +
                                                idx5 * this.stride[4]
                                        newStorage.add(aIdx, this.storage[bIdx])
                                    }
                                }
                            }
                        }
                    }
                }
                6 -> {
                    for (idx1 in 0 until this.shape[0]) {
                        for (idx2 in 0 until this.shape[1]) {
                            for (idx3 in 0 until this.shape[2]) {
                                for (idx4 in 0 until this.shape[3]) {
                                    for (idx5 in 0 until this.shape[4]) {
                                        for (idx6 in 0 until this.shape[5]) {
                                            val aIdx = idx1 * newStride[0] +
                                                    idx2 * newStride[1] +
                                                    idx3 * newStride[2] +
                                                    idx4 * newStride[3] +
                                                    idx5 * newStride[4] +
                                                    idx6
                                            val bIdx = idx1 * this.stride[0] +
                                                    idx2 * this.stride[1] +
                                                    idx3 * this.stride[2] +
                                                    idx4 * this.stride[3] +
                                                    idx5 * this.stride[4] +
                                                    idx6 * this.stride[5]
                                            newStorage.add(aIdx, this.storage[bIdx])
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                7 -> {
                    for (idx1 in 0 until this.shape[0]) {
                        for (idx2 in 0 until this.shape[1]) {
                            for (idx3 in 0 until this.shape[2]) {
                                for (idx4 in 0 until this.shape[3]) {
                                    for (idx5 in 0 until this.shape[4]) {
                                        for (idx6 in 0 until this.shape[5]) {
                                            for (idx7 in 0 until this.shape[6]) {
                                                val aIdx = idx1 * newStride[0] +
                                                        idx2 * newStride[1] +
                                                        idx3 * newStride[2] +
                                                        idx4 * newStride[3] +
                                                        idx5 * newStride[4] +
                                                        idx6 * newStride[5] +
                                                        idx7
                                                val bIdx = idx1 * this.stride[0] +
                                                        idx2 * this.stride[1] +
                                                        idx3 * this.stride[2] +
                                                        idx4 * this.stride[3] +
                                                        idx5 * this.stride[4] +
                                                        idx6 * this.stride[5] +
                                                        idx7 * this.stride[6]
                                                newStorage.add(aIdx, this.storage[bIdx])
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                8 -> {
                    for (idx1 in 0 until this.shape[0]) {
                        for (idx2 in 0 until this.shape[1]) {
                            for (idx3 in 0 until this.shape[2]) {
                                for (idx4 in 0 until this.shape[3]) {
                                    for (idx5 in 0 until this.shape[4]) {
                                        for (idx6 in 0 until this.shape[5]) {
                                            for (idx7 in 0 until this.shape[6]) {
                                                for (idx8 in 0 until this.shape[7]) {
                                                    val aIdx = idx1 * newStride[0] +
                                                            idx2 * newStride[1] +
                                                            idx3 * newStride[2] +
                                                            idx4 * newStride[3] +
                                                            idx5 * newStride[4] +
                                                            idx6 * newStride[5] +
                                                            idx7 * newStride[6] +
                                                            idx8
                                                    val bIdx = idx1 * this.stride[0] +
                                                            idx2 * this.stride[1] +
                                                            idx3 * this.stride[2] +
                                                            idx4 * this.stride[3] +
                                                            idx5 * this.stride[4] +
                                                            idx6 * this.stride[5] +
                                                            idx7 * this.stride[6] +
                                                            idx8 * this.stride[7]
                                                    newStorage.add(aIdx, this.storage[bIdx])
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                else -> throw NotImplementedError()
            }

            // in-place alter
            val output = TorchTensor(size = this.shape, storage = newStorage, stride = newStride)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        /**
         *  Returns a new tensor that is a narrowed version of input tensor
         *  The dimension dim is input from start to start + length
         *
         *  @param  dim     The dimension along which to narrow
         *  @param  start   The starting dimension
         *  @param  length  The distance to the ending dimension
         *  @return         Narrowed version of input tensor
         */
        fun narrow(dim: Int, start: Int, length: Int): TorchTensor {
            // Check if the parameters are valid
            if (dim >= this.dim()) {
                throw IndexError("Dimension out of range (expected to be in range of [" +
                        (this.ndimension * -1) + ", " +
                        (this.ndimension - 1) + "], but got " + dim + ")")
            }
            if (!(start >= 0 && length >= 0 && start < this.shape[dim] && start + length <= this.shape[dim])) {
                throw RuntimeError("start ($start) + length ($length) exceeds dimension size (" +
                        this.shape[dim] + ").")
            }

            // Update size
            var newSize = this.shape.toMutableList()
            newSize[dim] = length

            // Re-create new tensor
            var newStorage = emptyArray<Number>().toMutableList()
            when (this.dim()) {
                1 -> {
                    var newPtr = 0
                    for (idx1 in 0 until this.shape[0]) {
                        val idxList = listOf(idx1)
                        if (idxList[dim] >= start && idxList[dim] < start + length) {
                            val bIdx = idx1
                            newStorage.add(newPtr, this.storage[bIdx])
                            newPtr += 1
                        }
                    }
                }
                2 -> {
                    var newPtr = 0
                    for (idx1 in 0 until this.shape[0]) {
                        for (idx2 in 0 until this.shape[1]) {
                            val idxList = listOf(idx1, idx2)
                            if (idxList[dim] >= start && idxList[dim] < start + length) {
                                val bIdx = idx1 * this.stride[0] +
                                        idx2 * this.stride[1]
                                newStorage.add(newPtr, this.storage[bIdx])
                                newPtr += 1
                            }
                        }
                    }
                }
                3 -> {
                    var newPtr = 0
                    for (idx1 in 0 until this.shape[0]) {
                        for (idx2 in 0 until this.shape[1]) {
                            for (idx3 in 0 until this.shape[2]) {
                                val idxList = listOf(idx1, idx2, idx3)
                                if (idxList[dim] >= start && idxList[dim] < start + length) {
                                    val bIdx = idx1 * this.stride[0] +
                                            idx2 * this.stride[1] +
                                            idx3 * this.stride[2]
                                    newStorage.add(newPtr, this.storage[bIdx])
                                    newPtr += 1
                                }
                            }
                        }
                    }
                }
                4 -> {
                    var newPtr = 0
                    for (idx1 in 0 until this.shape[0]) {
                        for (idx2 in 0 until this.shape[1]) {
                            for (idx3 in 0 until this.shape[2]) {
                                for (idx4 in 0 until this.shape[3]) {
                                    val idxList = listOf(idx1, idx2, idx3, idx4)
                                    if (idxList[dim] >= start && idxList[dim] < start + length) {
                                        val bIdx = idx1 * this.stride[0] +
                                                idx2 * this.stride[1] +
                                                idx3 * this.stride[2] +
                                                idx4 * this.stride[3]
                                        newStorage.add(newPtr, this.storage[bIdx])
                                        newPtr += 1
                                    }
                                }
                            }
                        }
                    }
                }
                5 -> {
                    var newPtr = 0
                    for (idx1 in 0 until this.shape[0]) {
                        for (idx2 in 0 until this.shape[1]) {
                            for (idx3 in 0 until this.shape[2]) {
                                for (idx4 in 0 until this.shape[3]) {
                                    for (idx5 in 0 until this.shape[4]) {
                                        val idxList = listOf(idx1, idx2, idx3, idx4, idx5)
                                        if (idxList[dim] >= start && idxList[dim] < start + length) {
                                            val bIdx = idx1 * this.stride[0] +
                                                    idx2 * this.stride[1] +
                                                    idx3 * this.stride[2] +
                                                    idx4 * this.stride[3] +
                                                    idx5 * this.stride[4]
                                            newStorage.add(newPtr, this.storage[bIdx])
                                            newPtr += 1
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                6 -> {
                    var newPtr = 0
                    for (idx1 in 0 until this.shape[0]) {
                        for (idx2 in 0 until this.shape[1]) {
                            for (idx3 in 0 until this.shape[2]) {
                                for (idx4 in 0 until this.shape[3]) {
                                    for (idx5 in 0 until this.shape[4]) {
                                        for (idx6 in 0 until this.shape[5]) {
                                            val idxList = listOf(idx1, idx2, idx3, idx4, idx5, idx6)
                                            if (idxList[dim] >= start && idxList[dim] < start + length) {
                                                val bIdx = idx1 * this.stride[0] +
                                                        idx2 * this.stride[1] +
                                                        idx3 * this.stride[2] +
                                                        idx4 * this.stride[3] +
                                                        idx5 * this.stride[4] +
                                                        idx6 * this.stride[5]
                                                newStorage.add(newPtr, this.storage[bIdx])
                                                newPtr += 1
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                7 -> {
                    var newPtr = 0
                    for (idx1 in 0 until this.shape[0]) {
                        for (idx2 in 0 until this.shape[1]) {
                            for (idx3 in 0 until this.shape[2]) {
                                for (idx4 in 0 until this.shape[3]) {
                                    for (idx5 in 0 until this.shape[4]) {
                                        for (idx6 in 0 until this.shape[5]) {
                                            for (idx7 in 0 until this.shape[6]) {
                                                val idxList = listOf(idx1, idx2, idx3, idx4, idx5, idx6, idx7)
                                                if (idxList[dim] >= start && idxList[dim] < start + length) {
                                                    val bIdx = idx1 * this.stride[0] +
                                                            idx2 * this.stride[1] +
                                                            idx3 * this.stride[2] +
                                                            idx4 * this.stride[3] +
                                                            idx5 * this.stride[4] +
                                                            idx6 * this.stride[5] +
                                                            idx7 * this.stride[6]
                                                    newStorage.add(newPtr, this.storage[bIdx])
                                                    newPtr += 1
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                8 -> {
                    var newPtr = 0
                    for (idx1 in 0 until this.shape[0]) {
                        for (idx2 in 0 until this.shape[1]) {
                            for (idx3 in 0 until this.shape[2]) {
                                for (idx4 in 0 until this.shape[3]) {
                                    for (idx5 in 0 until this.shape[4]) {
                                        for (idx6 in 0 until this.shape[5]) {
                                            for (idx7 in 0 until this.shape[6]) {
                                                for (idx8 in 0 until this.shape[7]) {
                                                    val idxList = listOf(idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8)
                                                    if (idxList[dim] >= start && idxList[dim] < start + length) {
                                                        val bIdx = idx1 * this.stride[0] +
                                                                idx2 * this.stride[1] +
                                                                idx3 * this.stride[2] +
                                                                idx4 * this.stride[3] +
                                                                idx5 * this.stride[4] +
                                                                idx6 * this.stride[5] +
                                                                idx7 * this.stride[6] +
                                                                idx8 * this.stride[7]
                                                        newStorage.add(newPtr, this.storage[bIdx])
                                                        newPtr += 1
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                else -> throw NotImplementedError()
            }

            // in-place alter
            val output = TorchTensor(size = newSize, storage = newStorage)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            this.nElements = output.nElements
            this.ndimension = output.ndimension
            return this
        }

        /**
         *  Returns a new tensor with the same data as the self tensor but of a different shape
         *
         *  @param  size    The desired size. Can be list or int
         *  @return         The reshaped tensor
         */
        fun view(size: MutableList<Int>): TorchTensor {
            // Determine if there is only up to one dimension should be inferred
            var posSize = size.toMutableList()
            var hasNegative = false
            for (idx in 0 until posSize.size) {
                if (posSize[idx] < 0 && !hasNegative) {
                    if (posSize[idx] == -1) {
                        hasNegative = true
                    } else {
                        throw RuntimeError("RuntimeError: invalid shape dimension " + posSize[idx])
                    }
                } else if (posSize[idx] < 0 && hasNegative) {
                    throw RuntimeError("RuntimeError: only one dimension can be inferred")
                }
            }

            // Infer negative dimension
            if (hasNegative) {
                for (idx in 0 until posSize.size) {
                    if (posSize[idx] < 0) {
                        posSize[idx] = this.nElements / (-1 * posSize.reduce { acc, i -> acc * i })
                        break
                    }
                }
            }

            // Reshape the tensor
            val N = posSize.reduce { acc, i -> acc * i }
            if (N == this.nElements) {
                val output = TorchTensor(size = posSize, storage = this.storage)
                this.shape = output.shape
                this.storage = output.storage
                this.stride = output.stride
                this.ndimension = output.ndimension
                return this
            } else {
                throw RuntimeError("shape '$size' is invalid for input of size " + this.nElements)
            }
        }

        fun view(size: Array<Int>): TorchTensor {
            return view(size.toMutableList())
        }

        fun view(size: Int): TorchTensor {
            if (size < 0) {
                if (size == -1) {
                    return view(arrayOf(this.nElements))
                } else {
                    throw RuntimeError("RuntimeError: invalid shape dimension $size")
                }
            } else {
                if (size == this.nElements) {
                    return view(arrayOf(this.nElements))
                } else {
                    throw RuntimeError("RuntimeError: shape '[$size]' is invalid for input of size " +
                            this.nElements)
                }
            }
        }

        /**
         *  Repeats this tensor along the specified dimensions
         *
         *  @param  reps    The number of times to repeat this tensor along each dimension
         *  @return         The repeated tensor
         */
        fun repeat(reps: MutableList<Int>): TorchTensor {
            // Broadcasting the tensor if needed
            var unsqueezeTensor = clone()
            if (this.dim() > reps.size) {
                throw RuntimeError("RuntimeError: Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor")
            } else if (this.dim() < reps.size) {
                for (idx in 0 until reps.size - this.dim()) {
                    unsqueezeTensor = unsqueeze(unsqueezeTensor, 0)
                }
            }
            this.shape = unsqueezeTensor.shape
            this.storage = unsqueezeTensor.storage
            this.stride = unsqueezeTensor.stride
            this.ndimension = this.shape.size

            // Compute the updated size and stride
            val newSize = this.shape.toMutableList()
            val newStorage = mutableListOf<Number>()
            for (idx in 0 until reps.size) {
                newSize[idx] *= reps[idx]
            }
            val newStride = computeStride(newSize)

            // Copy the tensor
            when (this.dim()) {
                1 -> {
                    for (idx1 in 0 until newSize[0]) {
                        val aIdx = idx1 * newStride[0]
                        val bIdx = (idx1 % this.shape[0]) * this.stride[0]
                        newStorage.add(aIdx, this.storage[bIdx])
                    }
                }
                2 -> {
                    for (idx1 in 0 until newSize[0]) {
                        for (idx2 in 0 until newSize[1]) {
                            val aIdx = idx1 * newStride[0] +
                                    idx2 * newStride[1]
                            val bIdx = (idx1 % this.shape[0]) * this.stride[0] +
                                    (idx2 % this.shape[1]) * this.stride[1]
                            newStorage.add(aIdx, this.storage[bIdx])
                        }
                    }
                }
                3 -> {
                    for (idx1 in 0 until newSize[0]) {
                        for (idx2 in 0 until newSize[1]) {
                            for (idx3 in 0 until newSize[2]) {
                                val aIdx = idx1 * newStride[0] +
                                        idx2 * newStride[1] +
                                        idx3 * newStride[2]
                                val bIdx = (idx1 % this.shape[0]) * this.stride[0] +
                                        (idx2 % this.shape[1]) * this.stride[1] +
                                        (idx3 % this.shape[2]) * this.stride[2]
                                newStorage.add(aIdx, this.storage[bIdx])
                            }
                        }
                    }
                }
                4 -> {
                    for (idx1 in 0 until newSize[0]) {
                        for (idx2 in 0 until newSize[1]) {
                            for (idx3 in 0 until newSize[2]) {
                                for (idx4 in 0 until newSize[3]) {
                                    val aIdx = idx1 * newStride[0] +
                                            idx2 * newStride[1] +
                                            idx3 * newStride[2] +
                                            idx4 * newStride[3]
                                    val bIdx = (idx1 % this.shape[0]) * this.stride[0] +
                                            (idx2 % this.shape[1]) * this.stride[1] +
                                            (idx3 % this.shape[2]) * this.stride[2] +
                                            (idx4 % this.shape[3]) * this.stride[3]
                                    newStorage.add(aIdx, this.storage[bIdx])
                                }
                            }
                        }
                    }
                }
                5 -> {
                    for (idx1 in 0 until newSize[0]) {
                        for (idx2 in 0 until newSize[1]) {
                            for (idx3 in 0 until newSize[2]) {
                                for (idx4 in 0 until newSize[3]) {
                                    for (idx5 in 0 until newSize[4]) {
                                        val aIdx = idx1 * newStride[0] +
                                                idx2 * newStride[1] +
                                                idx3 * newStride[2] +
                                                idx4 * newStride[3] +
                                                idx5 * newStride[4]
                                        val bIdx = (idx1 % this.shape[0]) * this.stride[0] +
                                                (idx2 % this.shape[1]) * this.stride[1] +
                                                (idx3 % this.shape[2]) * this.stride[2] +
                                                (idx4 % this.shape[3]) * this.stride[3] +
                                                (idx5 % this.shape[4]) * this.stride[4]
                                        newStorage.add(aIdx, this.storage[bIdx])
                                    }
                                }
                            }
                        }
                    }
                }
                6 -> {
                    for (idx1 in 0 until newSize[0]) {
                        for (idx2 in 0 until newSize[1]) {
                            for (idx3 in 0 until newSize[2]) {
                                for (idx4 in 0 until newSize[3]) {
                                    for (idx5 in 0 until newSize[4]) {
                                        for (idx6 in 0 until newSize[5]) {
                                            val aIdx = idx1 * newStride[0] +
                                                    idx2 * newStride[1] +
                                                    idx3 * newStride[2] +
                                                    idx4 * newStride[3] +
                                                    idx5 * newStride[4] +
                                                    idx6 * newStride[5]
                                            val bIdx = (idx1 % this.shape[0]) * this.stride[0] +
                                                    (idx2 % this.shape[1]) * this.stride[1] +
                                                    (idx3 % this.shape[2]) * this.stride[2] +
                                                    (idx4 % this.shape[3]) * this.stride[3] +
                                                    (idx5 % this.shape[4]) * this.stride[4] +
                                                    (idx6 % this.shape[5]) * this.stride[5]
                                            newStorage.add(aIdx, this.storage[bIdx])
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                7 -> {
                    for (idx1 in 0 until newSize[0]) {
                        for (idx2 in 0 until newSize[1]) {
                            for (idx3 in 0 until newSize[2]) {
                                for (idx4 in 0 until newSize[3]) {
                                    for (idx5 in 0 until newSize[4]) {
                                        for (idx6 in 0 until newSize[5]) {
                                            for (idx7 in 0 until newSize[6]) {
                                                val aIdx = idx1 * newStride[0] +
                                                        idx2 * newStride[1] +
                                                        idx3 * newStride[2] +
                                                        idx4 * newStride[3] +
                                                        idx5 * newStride[4] +
                                                        idx6 * newStride[5] +
                                                        idx7 * newStride[6]
                                                val bIdx = (idx1 % this.shape[0]) * this.stride[0] +
                                                        (idx2 % this.shape[1]) * this.stride[1] +
                                                        (idx3 % this.shape[2]) * this.stride[2] +
                                                        (idx4 % this.shape[3]) * this.stride[3] +
                                                        (idx5 % this.shape[4]) * this.stride[4] +
                                                        (idx6 % this.shape[5]) * this.stride[5] +
                                                        (idx7 % this.shape[6]) * this.stride[6]
                                                newStorage.add(aIdx, this.storage[bIdx])
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                8 -> {
                    for (idx1 in 0 until newSize[0]) {
                        for (idx2 in 0 until newSize[1]) {
                            for (idx3 in 0 until newSize[2]) {
                                for (idx4 in 0 until newSize[3]) {
                                    for (idx5 in 0 until newSize[4]) {
                                        for (idx6 in 0 until newSize[5]) {
                                            for (idx7 in 0 until newSize[6]) {
                                                for (idx8 in 0 until newSize[7]) {
                                                    val aIdx = idx1 * newStride[0] +
                                                            idx2 * newStride[1] +
                                                            idx3 * newStride[2] +
                                                            idx4 * newStride[3] +
                                                            idx5 * newStride[4] +
                                                            idx6 * newStride[5] +
                                                            idx7 * newStride[6] +
                                                            idx8 * newStride[7]
                                                    val bIdx = (idx1 % this.shape[0]) * this.stride[0] +
                                                            (idx2 % this.shape[1]) * this.stride[1] +
                                                            (idx3 % this.shape[2]) * this.stride[2] +
                                                            (idx4 % this.shape[3]) * this.stride[3] +
                                                            (idx5 % this.shape[4]) * this.stride[4] +
                                                            (idx6 % this.shape[5]) * this.stride[5] +
                                                            (idx7 % this.shape[6]) * this.stride[6] +
                                                            (idx8 % this.shape[7]) * this.stride[7]
                                                    newStorage.add(aIdx, this.storage[bIdx])
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                else -> throw NotImplementedError()
            }

            // in-place alter
            val output = TorchTensor(size = newSize, storage = newStorage)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        /**
         *  Expand this tensor to the same size as other
         *
         *  @param  input   Given template tensor
         *  @return         The result tensor which has the same size as input
         */
        fun expand_as(input: TorchTensor): TorchTensor {
            if (input.dim() != this.dim()) {
                throw Exception("Two tensor should have the same ndim, but get " +
                        (input.dim()) + " and " + input.dim())
            }
            val reps: MutableList<Int> = input.shape
            for (idx in 0 until reps.size) {
                if (reps[idx] == this.shape[idx] || this.shape[idx] == 1) {
                    reps[idx] = reps[idx] / this.shape[idx]
                } else {
                    throw RuntimeError("The expanded size of the tensor (" + reps[idx] +
                            ") must match the existing size (" + this.shape[idx] +
                            ") at non-singleton dimension " + idx +
                            ". Target sizes: " + input.shape +
                            ". Tensor sizes: " + this.shape)
                }
            }
            return this.repeat(reps)
        }

        /**
         *  Computes the absolute value of each element
         *
         *  @return         The tensor with all-positive elements
         */
        fun abs(): TorchTensor {
            val newStorage = this.storage.toMutableList()
            when (this.dtype) {
                TensorType.DoubleTensor -> {
                    for (idx in 0 until this.nElements) {
                        newStorage[idx] = Math.abs(newStorage[idx].toDouble())
                    }
                }
                TensorType.FloatTensor -> {
                    for (idx in 0 until this.nElements) {
                        newStorage[idx] = Math.abs(newStorage[idx].toFloat())
                    }
                }
                TensorType.LongTensor -> {
                    for (idx in 0 until this.nElements) {
                        newStorage[idx] = Math.abs(newStorage[idx].toLong())
                    }
                }
            }

            // in-place alter
            val output = TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        /**
         *  Returns a new tensor with the sine of the elements
         *
         *  @return         The output result
         */
        fun sin(): TorchTensor {
            val newStorage = this.storage.toMutableList()
            when (this.dtype) {
                TensorType.DoubleTensor -> {
                    for (idx in 0 until this.nElements) {
                        newStorage[idx] = Math.sin(newStorage[idx].toDouble())
                    }
                }
                TensorType.FloatTensor -> {
                    for (idx in 0 until this.nElements) {
                        newStorage[idx] = Math.sin(newStorage[idx].toDouble()).toFloat()
                    }
                }
                TensorType.LongTensor -> {
                    for (idx in 0 until this.nElements) {
                        newStorage[idx] = Math.sin(newStorage[idx].toDouble()).toLong()
                    }
                }
            }

            // in-place alter
            val output = TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        /**
         *  Returns a new tensor with the cosine of the elements
         *
         *  @return         The output result
         */
        fun cos(): TorchTensor {
            val newStorage = this.storage.toMutableList()
            when (this.dtype) {
                TensorType.DoubleTensor -> {
                    for (idx in 0 until this.nElements) {
                        newStorage[idx] = Math.cos(newStorage[idx].toDouble())
                    }
                }
                TensorType.FloatTensor -> {
                    for (idx in 0 until this.nElements) {
                        newStorage[idx] = Math.cos(newStorage[idx].toDouble()).toFloat()
                    }
                }
                TensorType.LongTensor -> {
                    for (idx in 0 until this.nElements) {
                        newStorage[idx] = Math.cos(newStorage[idx].toDouble()).toLong()
                    }
                }
            }

            // in-place alter
            val output = TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        /**
         *  Returns a new tensor with the tangent of the elements of input
         *
         *  @return         The output result
         */
        fun tan(): TorchTensor {
            val newStorage = this.storage.toMutableList()
            when (this.dtype) {
                TensorType.DoubleTensor -> {
                    for (idx in 0 until this.nElements) {
                        newStorage[idx] = Math.tan(newStorage[idx].toDouble())
                    }
                }
                TensorType.FloatTensor -> {
                    for (idx in 0 until this.nElements) {
                        newStorage[idx] = Math.tan(newStorage[idx].toDouble()).toFloat()
                    }
                }
                TensorType.LongTensor -> {
                    for (idx in 0 until this.nElements) {
                        newStorage[idx] = Math.tan(newStorage[idx].toDouble()).toLong()
                    }
                }
            }

            // in-place alter
            val output = TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        /**
         *  Returns a new tensor with the exponential of the elements of the input tensor input
         *
         *  @return         The output result
         */
        fun exp(): TorchTensor {
            val newStorage = this.storage.toMutableList()
            when (this.dtype) {
                TensorType.DoubleTensor -> {
                    for (idx in 0 until this.nElements) {
                        newStorage[idx] = Math.exp(newStorage[idx].toDouble())
                    }
                }
                TensorType.FloatTensor -> {
                    for (idx in 0 until this.nElements) {
                        newStorage[idx] = Math.exp(newStorage[idx].toDouble()).toFloat()
                    }
                }
                TensorType.LongTensor -> {
                    for (idx in 0 until this.nElements) {
                        newStorage[idx] = Math.exp(newStorage[idx].toDouble()).toLong()
                    }
                }
            }

            // in-place alter
            val output = TorchTensor(size = this.shape, storage = newStorage, stride = this.stride)
            this.shape = output.shape
            this.storage = output.storage
            this.stride = output.stride
            return this
        }

        /**
         *  Returns the minimum value of all elements in this tensor.
         *
         *  Note:
         *      (1) We do not support return indice as PyTorch
         *
         *  @return         The tensor which contains minimum value
         */
        fun min(): TorchTensor {
            if (this.storage.size >= 1) {
                when (this.dtype) {
                    TensorType.DoubleTensor -> {
                        var min = this.storage[0].toDouble()
                        for (n in this.storage) {
                            if (n.toDouble() < min) {
                                min = n.toDouble()
                            }
                        }
                        return TorchTensor(Array<Number>(1) { min })
                    }
                    TensorType.FloatTensor -> {
                        var min = this.storage[0].toFloat()
                        for (n in this.storage) {
                            if (n.toDouble() < min) {
                                min = n.toFloat()
                            }
                        }
                        return TorchTensor(Array<Number>(1) { min }).float()
                    }
                    TensorType.LongTensor -> {
                        var min = this.storage[0].toLong()
                        for (n in this.storage) {
                            if (n.toDouble() < min) {
                                min = n.toLong()
                            }
                        }
                        return TorchTensor(Array<Number>(1) { min }).long()
                    }
                }
            } else {
                throw RuntimeError("operation does not have an identity.\n")
            }
        }

        fun min(dim: Int): TorchTensor {
            if (this.storage.size >= 1) {
                // Determine order of auto-contiguous and reverse
                val posDim = if (dim < 0) dim + dim() else dim
                fun getReverseOrder(order: Array<Int>, N: Int): Array<Int> {
                    val out = mutableListOf<Int>()
                    for (idx in 0 until N) {
                        out.add(order.indexOf(idx))
                    }
                    return out.toTypedArray()
                }
                var tensor = clone()
                val autoOrder = (mutableListOf(posDim) + (0 until posDim) + (posDim+1 until tensor.dim())).toTypedArray()
                val autoReverseOrder = getReverseOrder(autoOrder, N = tensor.dim())

                // Permute and construct output tensor
                permute(autoOrder)
                var newSize = this.shape.toMutableList().subList(1, this.ndimension)
                var newStride = computeStride(newSize)
                val N = newSize.reduce { acc, i -> acc * i }
                val newStorage = Array(N) {0 as Number}.toMutableList()
                var output = TorchTensor(size = newSize, storage = newStorage, stride = newStride)
                output.dtype = this.dtype

                // Compute mean by different ndim
                when (this.ndimension) {
                    2 -> {
                        for (idx1 in 0 until this.shape[1]) {
                            var minV: Double = get(0, idx1).item().toDouble()
                            for (idx0 in 0 until this.shape[0]) {
                                minV = Math.min(get(idx0, idx1).item().toDouble(), minV)
                            }
                            output[idx1] = minV
                        }
                    }
                    3 -> {
                        for (idx1 in 0 until this.shape[1]) {
                            for (idx2 in 0 until this.shape[2]) {
                                var minV: Double = get(0, idx1, idx2).item().toDouble()
                                for (idx0 in 0 until this.shape[0]) {
                                    minV = Math.min(get(idx0, idx1, idx2).item().toDouble(), minV)
                                }
                                output[idx1, idx2] = minV
                            }
                        }
                    }
                    4 -> {
                        for (idx1 in 0 until this.shape[1]) {
                            for (idx2 in 0 until this.shape[2]) {
                                for (idx3 in 0 until this.shape[3]) {
                                    var minV: Double = get(0, idx1, idx2, idx3).item().toDouble()
                                    for (idx0 in 0 until this.shape[0]) {
                                        minV = Math.min(get(idx0, idx1, idx2, idx3).item().toDouble(), minV)
                                    }
                                    output[idx1, idx2, idx3] = minV
                                }
                            }
                        }
                    }
                    5 -> {
                        for (idx1 in 0 until this.shape[1]) {
                            for (idx2 in 0 until this.shape[2]) {
                                for (idx3 in 0 until this.shape[3]) {
                                    for (idx4 in 0 until this.shape[4]) {
                                        var minV: Double = get(0, idx1, idx2, idx3, idx4).item().toDouble()
                                        for (idx0 in 0 until this.shape[0]) {
                                            minV = Math.min(get(idx0, idx1, idx2, idx3, idx4).item().toDouble(), minV)
                                        }
                                        output[idx1, idx2, idx3, idx4] = minV
                                    }
                                }
                            }
                        }
                    }
                    6 -> {
                        for (idx1 in 0 until this.shape[1]) {
                            for (idx2 in 0 until this.shape[2]) {
                                for (idx3 in 0 until this.shape[3]) {
                                    for (idx4 in 0 until this.shape[4]) {
                                        for (idx5 in 0 until this.shape[5]) {
                                            var minV: Double = get(0, idx1, idx2, idx3, idx4, idx5).item().toDouble()
                                            for (idx0 in 0 until this.shape[0]) {
                                                minV = Math.min(get(idx0, idx1, idx2, idx3, idx4, idx5).item().toDouble(), minV)
                                            }
                                            output[idx1, idx2, idx3, idx4, idx5] = minV
                                        }
                                    }
                                }
                            }
                        }
                    }
                    7 -> {
                        for (idx1 in 0 until this.shape[1]) {
                            for (idx2 in 0 until this.shape[2]) {
                                for (idx3 in 0 until this.shape[3]) {
                                    for (idx4 in 0 until this.shape[4]) {
                                        for (idx5 in 0 until this.shape[5]) {
                                            for (idx6 in 0 until this.shape[6]) {
                                                var minV: Double = get(0, idx1, idx2, idx3, idx4, idx5, idx6).item().toDouble()
                                                for (idx0 in 0 until this.shape[0]) {
                                                    minV = Math.min(get(idx0, idx1, idx2, idx3, idx4, idx5, idx6).item().toDouble(), minV)
                                                }
                                                output[idx1, idx2, idx3, idx4, idx5, idx6] = minV
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    8 -> {
                        for (idx1 in 0 until this.shape[1]) {
                            for (idx2 in 0 until this.shape[2]) {
                                for (idx3 in 0 until this.shape[3]) {
                                    for (idx4 in 0 until this.shape[4]) {
                                        for (idx5 in 0 until this.shape[5]) {
                                            for (idx6 in 0 until this.shape[6]) {
                                                for (idx7 in 0 until this.shape[7]) {
                                                    var minV: Double = get(0, idx1, idx2, idx3, idx4, idx5, idx6, idx7).item().toDouble()
                                                    for (idx0 in 0 until this.shape[0]) {
                                                        minV = Math.min(get(idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7).item().toDouble(), minV)
                                                    }
                                                    output[idx1, idx2, idx3, idx4, idx5, idx6, idx7] = minV
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else -> throw NotImplementedError()
                }
                // output = output.permute(autoReverseOrder)
                output.contiguous()

                // in-place alter
                this.shape = output.shape
                this.storage = output.storage
                this.stride = output.stride
                this.nElements = output.nElements
                this.ndimension = output.ndimension
                return this
            } else {
                throw RuntimeError("operation does not have an identity.\n")
            }
        }

        /**
         *  Returns the maximum value of all elements in this tensor.
         *
         *  Note:
         *      (1) We do not support return indice as PyTorch
         *
         *  @return         The tensor which contains maximum value
         */
        fun max(): TorchTensor {
            if (this.storage.size >= 1) {
                when (this.dtype) {
                    TensorType.DoubleTensor -> {
                        var max = this.storage[0].toDouble()
                        for (n in this.storage) {
                            if (n.toDouble() > max) {
                                max = n.toDouble()
                            }
                        }
                        return TorchTensor(Array<Number>(1) { max })
                    }
                    TensorType.FloatTensor -> {
                        var max = this.storage[0].toFloat()
                        for (n in this.storage) {
                            if (n.toDouble() > max) {
                                max = n.toFloat()
                            }
                        }
                        return TorchTensor(Array<Number>(1) { max }).float()
                    }
                    TensorType.LongTensor -> {
                        var max = this.storage[0].toLong()
                        for (n in this.storage) {
                            if (n.toDouble() > max) {
                                max = n.toLong()
                            }
                        }
                        return TorchTensor(Array<Number>(1) { max }).long()
                    }
                }
            } else {
                throw RuntimeError("operation does not have an identity.\n")
            }
        }

        fun max(dim: Int): TorchTensor {
            if (this.storage.size >= 1) {
                // Determine order of auto-contiguous and reverse
                val posDim = if (dim < 0) dim + dim() else dim
                fun getReverseOrder(order: Array<Int>, N: Int): Array<Int> {
                    val out = mutableListOf<Int>()
                    for (idx in 0 until N) {
                        out.add(order.indexOf(idx))
                    }
                    return out.toTypedArray()
                }
                var tensor = clone()
                val autoOrder = (mutableListOf(posDim) + (0 until posDim) + (posDim+1 until tensor.dim())).toTypedArray()
                val autoReverseOrder = getReverseOrder(autoOrder, N = tensor.dim())

                // Permute and construct output tensor
                permute(autoOrder)
                var newSize = this.shape.toMutableList().subList(1, this.ndimension)
                var newStride = computeStride(newSize)
                val N = newSize.reduce { acc, i -> acc * i }
                val newStorage = Array(N) {0 as Number}.toMutableList()
                var output = TorchTensor(size = newSize, storage = newStorage, stride = newStride)
                output.dtype = this.dtype

                // Compute mean by different ndim
                when (this.ndimension) {
                    2 -> {
                        for (idx1 in 0 until this.shape[1]) {
                            var maxV: Double = get(0, idx1).item().toDouble()
                            println("Idx1: $idx1   maxV: $maxV")
                            for (idx0 in 0 until this.shape[0]) {
                                maxV = Math.max(get(idx0, idx1).item().toDouble(), maxV)
                            }
                            output[idx1] = maxV
                        }
                    }
                    3 -> {
                        for (idx1 in 0 until this.shape[1]) {
                            for (idx2 in 0 until this.shape[2]) {
                                var maxV: Double = get(0, idx1, idx2).item().toDouble()
                                for (idx0 in 0 until this.shape[0]) {
                                    maxV = Math.max(get(idx0, idx1, idx2).item().toDouble(), maxV)
                                }
                                output[idx1, idx2] = maxV
                            }
                        }
                    }
                    4 -> {
                        for (idx1 in 0 until this.shape[1]) {
                            for (idx2 in 0 until this.shape[2]) {
                                for (idx3 in 0 until this.shape[3]) {
                                    var maxV: Double = get(0, idx1, idx2, idx3).item().toDouble()
                                    for (idx0 in 0 until this.shape[0]) {
                                        maxV = Math.max(get(idx0, idx1, idx2, idx3).item().toDouble(), maxV)
                                    }
                                    output[idx1, idx2, idx3] = maxV
                                }
                            }
                        }
                    }
                    5 -> {
                        for (idx1 in 0 until this.shape[1]) {
                            for (idx2 in 0 until this.shape[2]) {
                                for (idx3 in 0 until this.shape[3]) {
                                    for (idx4 in 0 until this.shape[4]) {
                                        var maxV: Double = get(0, idx1, idx2, idx3, idx4).item().toDouble()
                                        for (idx0 in 0 until this.shape[0]) {
                                            maxV = Math.max(get(idx0, idx1, idx2, idx3, idx4).item().toDouble(), maxV)
                                        }
                                        output[idx1, idx2, idx3, idx4] = maxV
                                    }
                                }
                            }
                        }
                    }
                    6 -> {
                        for (idx1 in 0 until this.shape[1]) {
                            for (idx2 in 0 until this.shape[2]) {
                                for (idx3 in 0 until this.shape[3]) {
                                    for (idx4 in 0 until this.shape[4]) {
                                        for (idx5 in 0 until this.shape[5]) {
                                            var maxV: Double = get(0, idx1, idx2, idx3, idx4, idx5).item().toDouble()
                                            for (idx0 in 0 until this.shape[0]) {
                                                maxV = Math.max(get(idx0, idx1, idx2, idx3, idx4, idx5).item().toDouble(), maxV)
                                            }
                                            output[idx1, idx2, idx3, idx4, idx5] = maxV
                                        }
                                    }
                                }
                            }
                        }
                    }
                    7 -> {
                        for (idx1 in 0 until this.shape[1]) {
                            for (idx2 in 0 until this.shape[2]) {
                                for (idx3 in 0 until this.shape[3]) {
                                    for (idx4 in 0 until this.shape[4]) {
                                        for (idx5 in 0 until this.shape[5]) {
                                            for (idx6 in 0 until this.shape[6]) {
                                                var maxV: Double = get(0, idx1, idx2, idx3, idx4, idx5, idx6).item().toDouble()
                                                for (idx0 in 0 until this.shape[0]) {
                                                    maxV = Math.max(get(idx0, idx1, idx2, idx3, idx4, idx5, idx6).item().toDouble(), maxV)
                                                }
                                                output[idx1, idx2, idx3, idx4, idx5, idx6] = maxV
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    8 -> {
                        for (idx1 in 0 until this.shape[1]) {
                            for (idx2 in 0 until this.shape[2]) {
                                for (idx3 in 0 until this.shape[3]) {
                                    for (idx4 in 0 until this.shape[4]) {
                                        for (idx5 in 0 until this.shape[5]) {
                                            for (idx6 in 0 until this.shape[6]) {
                                                for (idx7 in 0 until this.shape[7]) {
                                                    var maxV: Double = get(0, idx1, idx2, idx3, idx4, idx5, idx6, idx7).item().toDouble()
                                                    for (idx0 in 0 until this.shape[0]) {
                                                        maxV = Math.max(get(idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7).item().toDouble(), maxV)
                                                    }
                                                    output[idx1, idx2, idx3, idx4, idx5, idx6, idx7] = maxV
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else -> throw NotImplementedError()
                }
                // output = output.permute(autoReverseOrder)
                output.contiguous()

                // in-place alter
                this.shape = output.shape
                this.storage = output.storage
                this.stride = output.stride
                this.nElements = output.nElements
                this.ndimension = output.ndimension
                return this
            } else {
                throw RuntimeError("operation does not have an identity.\n")
            }
        }

        /**
         *  Returns the median of the values in input
         *
         *  Note:
         *      (1) We only support global maximum currently
         *
         *  @return         The tensor which contains median value
         */
        fun median(): TorchTensor {
            if (this.storage.size >= 1) {
                var numArray = this.storage.toMutableList().toTypedArray()
                Arrays.sort(numArray, object : Comparator<Number> {
                    override fun compare(o1: Number, o2: Number): Int {
                        return o1.toDouble().compareTo(o2.toDouble())
                    }
                })
                var median: Number
                if (numArray.size % 2 == 0) {
                    median = numArray[numArray.size / 2 - 1]
                } else {
                    median = numArray[numArray.size / 2]
                }
                when (this.dtype) {
                    TensorType.DoubleTensor -> return TorchTensor(Array<Number>(1) { median })
                    TensorType.FloatTensor -> return TorchTensor(Array<Number>(1) { median }).float()
                    TensorType.LongTensor -> return TorchTensor(Array<Number>(1) { median }).long()
                }
            } else {
                throw RuntimeError("operation does not have an identity.\n")
            }
        }

        /**
         *  Returns the sum of all elements in the input tensor
         *
         *  @param  dim         The dimension(s) or dimensions to reduce
         *  @param  keepdim     Whether the output tensor has dim retained or not
         *  @return             The tensor which contains median value
         */
        fun sum(): TorchTensor {
            if (this.storage.size >= 1) {
                when (this.dtype) {
                    TensorType.DoubleTensor -> {
                        var sum = 0.0
                        for (num in this.storage) {
                            sum += num.toDouble()
                        }
                        return TorchTensor(Array<Number>(1) { sum })
                    }
                    TensorType.FloatTensor -> {
                        var sum = 0.0
                        for (num in this.storage) {
                            sum += num.toFloat()
                        }
                        return TorchTensor(Array<Number>(1) { sum }).float()
                    }
                    TensorType.LongTensor -> {
                        var sum = 0.0
                        for (num in this.storage) {
                            sum += num.toLong()
                        }
                        return TorchTensor(Array<Number>(1) { sum }).long()
                    }
                }
            } else {
                throw RuntimeError("operation does not have an identity.\n")
            }
        }

        fun sum(dim: Int, keepdim: Boolean = false): TorchTensor {
            if (this.storage.size >= 1) {
                // Determine order of auto-contiguous and reverse
                val posDim = if (dim < 0) dim + dim() else dim
                fun getReverseOrder(order: Array<Int>, N: Int): Array<Int> {
                    val out = mutableListOf<Int>()
                    for (idx in 0 until N) {
                        out.add(order.indexOf(idx))
                    }
                    return out.toTypedArray()
                }
                var tensor = clone()
                val autoOrder = (mutableListOf(posDim) + (0 until posDim) + (posDim+1 until tensor.dim())).toTypedArray()
                val autoReverseOrder = getReverseOrder(autoOrder, N = tensor.dim())

                // Permute and compute sum manually
                tensor = tensor.permute(autoOrder)
                var sum = tensor[0]
                for (idx in 1 until tensor.shape[0]) {
                    sum += tensor[idx]
                }

                // Compute sum and permute back
                if (keepdim) {
                    sum = unsqueeze(sum, 0)
                    sum = sum.permute(autoReverseOrder)
                }
                sum.contiguous()
                this.shape = sum.shape
                this.storage = sum.storage
                this.stride = sum.stride
                this.ndimension = sum.ndimension
                return this
            } else {
                throw RuntimeError("operation does not have an identity.\n")
            }
        }

        fun sum(dim: Array<Int>, keepdim: Boolean = false): TorchTensor {
            for (idx in 0 until dim.size) {
                sum(dim[idx], keepdim)
            }
            return this
        }

        /**
         *  Returns the product of all elements in the input tensor
         *
         *  Note:
         *      (1) We only support global product currently
         *
         *  @return         The tensor which contains median value
         */
        fun prod(): TorchTensor {
            if (this.storage.size >= 1) {
                when (this.dtype) {
                    TensorType.DoubleTensor -> {
                        var sum = 1.0
                        for (num in this.storage) {
                            sum *= num.toDouble()
                        }
                        return TorchTensor(Array<Number>(1) { sum })
                    }
                    TensorType.FloatTensor -> {
                        var sum = 1.0
                        for (num in this.storage) {
                            sum *= num.toFloat()
                        }
                        return TorchTensor(Array<Number>(1) { sum }).float()
                    }
                    TensorType.LongTensor -> {
                        var sum: Long = 1
                        for (num in this.storage) {
                            sum *= num.toLong()
                        }
                        return TorchTensor(Array<Number>(1) { sum }).long()
                    }
                }
            } else {
                throw RuntimeError("operation does not have an identity.\n")
            }
        }

        /**
         *  Returns the mean value of all elements
         *
         *  @param  dim         The dimension(s) or dimensions to reduce
         *  @param  keepdim     Whether the output tensor has dim retained or not
         *  @return             Mean tensor
         */
        fun mean(): TorchTensor {
            if (this.storage.size >= 1) {
                when (this.dtype) {
                    TensorType.DoubleTensor -> {
                        var sum = 0.0
                        for (num in this.storage) {
                            sum += num.toDouble()
                        }
                        val mean = sum / this.nElements
                        return TorchTensor(Array<Number>(1) { mean })
                    }
                    TensorType.FloatTensor -> {
                        var sum = 0.0
                        for (num in this.storage) {
                            sum += num.toFloat()
                        }
                        val mean = sum / this.nElements
                        return TorchTensor(Array<Number>(1) { mean })
                    }
                    else -> throw RuntimeError("RuntimeError: Can only calculate the mean of floating types. Got " +
                            this.dtype + " instead.")
                }
            } else {
                throw RuntimeError("operation does not have an identity.\n")
            }
        }

        fun mean(dim: Int, keepdim: Boolean = false): TorchTensor {
            if (this.storage.size >= 1) {
                when (this.dtype) {
                    TensorType.DoubleTensor, TensorType.FloatTensor -> {
                        // Determine order of auto-contiguous and reverse
                        val posDim = if (dim < 0) dim + dim() else dim
                        fun getReverseOrder(order: Array<Int>, N: Int): Array<Int> {
                            val out = mutableListOf<Int>()
                            for (idx in 0 until N) {
                                out.add(order.indexOf(idx))
                            }
                            return out.toTypedArray()
                        }

                        var tensor = clone()
                        val autoOrder = (mutableListOf(posDim) + (0 until posDim) + (posDim + 1 until tensor.dim())).toTypedArray()
                        val autoReverseOrder = getReverseOrder(autoOrder, N = tensor.dim())

                        // Permute and compute sum manually
                        tensor = tensor.permute(autoOrder)
                        var sum = tensor[0]
                        for (idx in 1 until tensor.shape[0]) {
                            sum += tensor[idx]
                        }

                        // Compute mean and permute back
                        var mean = sum / tensor.shape[0]
                        if (keepdim) {
                            mean = unsqueeze(mean, 0)
                            mean = mean.permute(autoReverseOrder)
                        }
                        mean.contiguous()
                        this.shape = mean.shape
                        this.storage = mean.storage
                        this.stride = mean.stride
                        this.ndimension = mean.ndimension
                        return this
                    }
                    else -> throw RuntimeError("RuntimeError: Can only calculate the mean of floating types. Got " +
                            this.dtype + " instead.")
                }
            } else {
                throw RuntimeError("operation does not have an identity.\n")
            }
        }

        @JvmName("mean_Array")
        fun mean(dim: Array<Int>, keepdim: Boolean = false): TorchTensor {
            for (idx in 0 until dim.size) {
                mean(dim[idx], keepdim)
            }
            return this
        }

        @JvmName("mean_MutableList")
        fun mean(dim: MutableList<Int>, keepdim: Boolean = false): TorchTensor {
            return mean(dim.toTypedArray(), keepdim)
        }

        @JvmName("mean_ArrayList")
        fun mean(dim: ArrayList<Int>, keepdim: Boolean = false): TorchTensor {
            return mean(dim.toTypedArray(), keepdim)
        }

        @JvmName("mean_List")
        fun mean(dim: List<Int>, keepdim: Boolean = false): TorchTensor {
            return mean(dim.toTypedArray(), keepdim)
        }

        /**
         *  Returns the standard-deviation of all elements in the input Tensor.
         *  (Unbias estimation)
         *
         *  @param  dim         The dimension(s) or dimensions to reduce
         *  @param  keepdim     Whether the output tensor has dim retained or not
         *  @return             Std tensor
         */
        fun std(): TorchTensor {
            if (this.storage.size >= 1) {
                when (this.dtype) {
                    TensorType.DoubleTensor -> {
                        var sum = 0.0
                        var standardDeviation = 0.0
                        for (num in this.storage) {
                            sum += num.toDouble()
                        }
                        val mean = sum / this.nElements
                        for (num in this.storage) {
                            standardDeviation += Math.pow(num.toDouble() - mean, 2.0)
                        }
                        standardDeviation = Math.sqrt(standardDeviation / (this.nElements - 1))

                        val output = TorchTensor(Array<Number>(1) { standardDeviation })
                        this.shape = output.shape
                        this.storage = output.storage
                        this.stride = output.stride
                        this.ndimension = output.ndimension
                        this.nElements = output.nElements
                        return this
                    }
                    TensorType.FloatTensor -> {
                        var sum = 0.0
                        var standardDeviation = 0.0
                        for (num in this.storage) {
                            sum += num.toFloat()
                        }
                        val mean = sum / this.nElements
                        for (num in this.storage) {
                            standardDeviation += Math.pow(num.toFloat() - mean, 2.0)
                        }
                        standardDeviation = Math.sqrt(standardDeviation / (this.nElements - 1))

                        val output = TorchTensor(Array<Number>(1) { standardDeviation })
                        this.shape = output.shape
                        this.storage = output.storage
                        this.stride = output.stride
                        this.ndimension = output.ndimension
                        this.nElements = output.nElements
                        return this
                    }
                    else -> throw RuntimeError("std only supports floating-point dtypes")
                }
            } else {
                throw RuntimeError("operation does not have an identity.\n")
            }
        }

        fun std(dim: Int, keepdim: Boolean = false): TorchTensor {
            if (this.storage.size >= 1) {
                if (this.ndimension == 1) {
                    return std()
                }
                when (this.dtype) {
                    TensorType.DoubleTensor, TensorType.FloatTensor -> {
                        // Determine order of auto-contiguous and reverse
                        val posDim = if (dim < 0) dim + dim() else dim
                        fun getReverseOrder(order: Array<Int>, N: Int): Array<Int> {
                            val out = mutableListOf<Int>()
                            for (idx in 0 until N) {
                                out.add(order.indexOf(idx))
                            }
                            return out.toTypedArray()
                        }

                        var tensor = clone()
                        val autoOrder = (mutableListOf(posDim) + (0 until posDim) + (posDim + 1 until tensor.dim())).toTypedArray()
                        val autoReverseOrder = getReverseOrder(autoOrder, N = tensor.dim())

                        // Permute and compute sum manually
                        println("Shape: " + this.shape + "\tOrder: " + autoOrder.toMutableList())
                        tensor = tensor.permute(autoOrder)
                        var sum = tensor.clone()[0]
                        for (idx in 1 until tensor.shape[0]) {
                            sum += tensor[idx]
                        }

                        // Compute mean and std
                        var mean = sum / tensor.shape[0]
                        var standardDeviation = zeros_like(mean)
                        val N = mean.shape.toMutableList().reduce { acc, i -> acc * i }
                        for (idx in 0 until tensor.shape[0]) {
                            standardDeviation += pow(tensor[idx] - mean, 2.0)
                        }
                        standardDeviation = pow(standardDeviation.clone() / (tensor.shape[0] - 1), 0.5)

                        // permute back and alter
                        if (keepdim) {
                            standardDeviation = unsqueeze(standardDeviation, 0)
                            standardDeviation = standardDeviation.permute(autoReverseOrder)
                        }
                        standardDeviation.contiguous()
                        this.shape = standardDeviation.shape
                        this.storage = standardDeviation.storage
                        this.stride = standardDeviation.stride
                        this.ndimension = standardDeviation.ndimension
                        this.nElements = standardDeviation.nElements
                        return this
                    }
                    else -> throw RuntimeError("RuntimeError: Can only calculate the mean of floating types. Got " +
                            this.dtype + " instead.")
                }
            } else {
                throw RuntimeError("operation does not have an identity.\n")
            }
        }

        @JvmName("std_Array")
        fun std(dim: Array<Int>, keepdim: Boolean = false): TorchTensor {
            var unitDims: MutableList<Int> = (0 until this.ndimension).toMutableList()
            var restShape: MutableList<Int> = this.shape.toMutableList()
            for (idx in 0 until dim.size) {
                unitDims.remove(dim[idx])
                restShape.removeAt(dim[idx] - idx)
            }
            unitDims = (dim.toMutableList() + unitDims).toMutableList()
            restShape = (mutableListOf(-1) + restShape).toMutableList()
            permute(unitDims.toTypedArray())
            contiguous()
            view(restShape)
            var output = std(0)
            if (keepdim) {
                for (idx in 0 until dim.size) {
                    output = unsqueeze(output, dim[idx])
                }
            }
            return output
        }

        @JvmName("std_MutableList")
        fun std(dim: MutableList<Int>, keepdim: Boolean = false): TorchTensor {
            return std(dim.toTypedArray(), keepdim)
        }

        @JvmName("std_ArrayList")
        fun std(dim: ArrayList<Int>, keepdim: Boolean = false): TorchTensor {
            return std(dim.toTypedArray(), keepdim)
        }

        @JvmName("std_List")
        fun std(dim: List<Int>, keepdim: Boolean = false): TorchTensor {
            return std(dim.toTypedArray(), keepdim)
        }

        /**
         *  Returns a copy of input
         *
         *  @return     The cloned tensor
         */
        fun clone(): TorchTensor {
            val newSize = this.shape.toMutableList()
            val newStorage = this.storage.toMutableList()
            val newStride = this.stride.toMutableList()
            val output = TorchTensor(size = newSize, storage = newStorage, stride = newStride)
            when (this.dtype) {
                TensorType.DoubleTensor -> output.double()
                TensorType.FloatTensor -> output.float()
                TensorType.LongTensor -> output.long()
            }
            output.nElements = this.nElements
            return output
        }

        /**
         *  ================================ Type conversion ===================================
         */
        fun float(): TorchTensor {
            if (this.dtype != TensorType.FloatTensor) {
                this.dtype = TensorType.FloatTensor
                for (idx in 0 until this.nElements) {
                    this.storage[idx].toFloat()
                }
            }
            return this
        }

        fun double(): TorchTensor {
            if (this.dtype != TensorType.DoubleTensor) {
                this.dtype = TensorType.DoubleTensor
                for (idx in 0 until this.nElements) {
                    this.storage[idx].toDouble()
                }
            }
            return this
        }

        fun long(): TorchTensor {
            if (this.dtype != TensorType.LongTensor) {
                this.dtype = TensorType.LongTensor
                for (idx in 0 until this.nElements) {
                    this.storage[idx].toLong()
                }
            }
            return this
        }

        /**
         *  Perform transpose for the 2D matrix
         *  This function acts as same as torch.t()
         *
         *  @return         The transposed tensor
         */
        fun t(): TorchTensor {
            return t(this)
        }

        /**
         * ================================ Print tensor ===================================
         */

        /**
         *  Determine the format string
         *  This function is refer to TensorPrinting.py
         *  ref: https://github.com/pytorch/pytorch/blob/v0.1.1/torch/TensorPrinting.py
         *
         *  @param  tensor  The tensor part you want to serialize
         *  @return         format string
         */
        private fun printformat(tensor: TorchTensor): String {
            var int_mode = true
            tensor.double()
            val doubleTensor = tensor.clone().abs()
            doubleTensor.double()
            for (value in doubleTensor.storage) {
                if (value != Math.ceil(value.toDouble())) {
                    int_mode = false
                    break
                }
            }

            var exp_min = doubleTensor.min().item().toDouble()
            exp_min = if (exp_min != 0.0) {
                Math.floor(Math.log10(exp_min)) + 1
            } else {
                1.0
            }
            var exp_max = tensor.max().item().toDouble()
            exp_max = if (exp_max != 0.0) {
                Math.floor(Math.log10(exp_max)) + 1
            } else {
                1.0
            }

            var scale = 1
            var format: String = ""
            var sz: Int
            if (int_mode) {
                if (exp_max > 9) {
                    format = "%11.4e"
                    sz = 11
                } else {
                    sz = exp_max.toInt() + 1
                    sz = Math.max(sz.toInt(), 1)
                    format = "%$sz.0f"
                }
            } else {
                if (exp_max - exp_min > 4) {
                    sz = 11
                    if (Math.abs(exp_max) > 99 || Math.abs(exp_min) > 99) {
                        sz = sz + 1
                    }
                    format = "%" + sz.toString() + ".4e"
                } else {
                    if (exp_max > 5 || exp_max < -2) {
                        sz = 11
                        scale = Math.pow(10.0, exp_max - 1).toInt()
                        format = "%" + sz.toString() + ".4e"
                    } else {
                        // if (exp_max == 0.0) {
                        //     sz = 7
                        // } else {
                        //     sz = (exp_max + 6).toInt()
                        // }
                        sz = 7
                        sz = Math.max(sz.toInt(), 1)
                        format = "%" + sz.toString() + ".4f"
                    }
                }
            }
            return format
        }

        private fun printVector(tensor: TorchTensor, indent: String = " ", nColumn: Int = 8, skip: Boolean = false): String {
            val fmt = printformat(tensor)
            // println(fmt)
            var strt = ""
            if (skip) {
                for (idx in 0 until tensor.shape[0]) {
                    val t = tensor[idx].item()
                    if (idx > 2 && idx < tensor.shape[0] - 3) {
                        if (strt.substring(strt.length - 4) != "...,") {
                            strt += "  ...,"
                        }
                    } else {
                        if (idx != 0 && idx % nColumn == 0) {
                            strt += ("\n" + indent + fmt.format(t))
                        } else if (idx != 0) {
                            strt += (" " + fmt.format(t))
                        } else {
                            strt += fmt.format(t)
                        }
                        if (idx == tensor.shape[0] - 1) {
                            strt += ""
                        } else {
                            strt += ","
                        }
                    }
                }
            } else {
                for (idx in 0 until tensor.shape[0]) {
                    val t = tensor[idx].item()
                    if (idx != 0 && idx % nColumn == 0) {
                        strt += ("\n" + indent + fmt.format(t))
                    } else if (idx != 0) {
                        strt += (" " + fmt.format(t))
                    } else {
                        strt += fmt.format(t)
                    }
                    if (idx == tensor.shape[0] - 1) {
                        strt += ""
                    } else {
                        strt += ","
                    }
                }
            }
            return strt
        }

        private fun printMatrix(tensor: TorchTensor, indent: String = " ", nColumn: Int = 8, skip: Boolean = false): String {
            var strt = ""
            if (skip) {
                // for idx, t in enumerate(tensor):
                for (idx in 0 until tensor.shape[0]) {
                    val t = tensor[idx]
                    if (idx > 2 && idx < tensor.shape[0] - 3) {
                        if (idx == 3) {
                            strt += ("...,\n" + indent)
                        }
                    } else {
                        strt += ('[' + printVector(t, indent, nColumn = nColumn, skip = skip) + ']')
                        if (idx == tensor.shape[0] - 1) {
                            strt += ""
                        } else {
                            strt += ('\n' + indent)
                        }
                    }
                }
            } else {
                // for idx, t in enumerate(tensor):
                for (idx in 0 until tensor.shape[0]) {
                    val t = tensor[idx]
                    strt += ('[' + printVector(t, indent, skip = skip) + ']')
                    if (idx == tensor.shape[0] - 1) {
                        strt += ""
                    } else {
                        strt += ('\n' + indent)
                    }
                }
            }
            return strt
        }

        private fun printTensor3D(tensor: TorchTensor, indent: String = " ", nColumn: Int = 8, skip: Boolean = true): String {
            var strt = ""
            // for idx, t in enumerate(tensor):
            for (idx in 0 until tensor.shape[0]) {
                val t = tensor[idx]
                strt += ('[' + printMatrix(t, "$indent ", nColumn = nColumn, skip = skip) + ']')
                strt += if (idx == tensor.shape[0] - 1) {
                    ""
                } else {
                    (",\n\n$indent")
                }
            }
            return strt
        }

        private fun printTensor4D(tensor: TorchTensor, indent: String = " ", nColumn: Int = 8, skip: Boolean = true): String {
            var strt = ""
            // for idx, t in enumerate(tensor):
            for (idx in 0 until tensor.shape[0]) {
                val t = tensor[idx]
                strt += ('[' + printTensor3D(t, "$indent ", nColumn = nColumn, skip = skip) + ']')
                strt += if (idx == tensor.shape[0] - 1) {
                    ""
                } else {
                    (",\n\n$indent")
                }
            }
            return strt
        }

        private fun printTensor5D(tensor: TorchTensor, indent: String = " ", nColumn: Int = 8, skip: Boolean = true): String {
            var strt = ""
            // for idx, t in enumerate(tensor):
            for (idx in 0 until tensor.shape[0]) {
                val t = tensor[idx]
                strt += ('[' + printTensor4D(t, "$indent ", nColumn = nColumn, skip = skip) + ']')
                strt += if (idx == tensor.shape[0] - 1) {
                    ""
                } else {
                    (",\n\n$indent")
                }
            }
            return strt
        }

        private fun printTensor6D(tensor: TorchTensor, indent: String = " ", nColumn: Int = 8, skip: Boolean = true): String {
            var strt = ""
            // for idx, t in enumerate(tensor):
            for (idx in 0 until tensor.shape[0]) {
                val t = tensor[idx]
                strt += ('[' + printTensor5D(t, "$indent ", nColumn = nColumn, skip = skip) + ']')
                strt += if (idx == tensor.shape[0] - 1) {
                    ""
                } else {
                    (",\n\n$indent")
                }
            }
            return strt
        }

        private fun printTensor7D(tensor: TorchTensor, indent: String = " ", nColumn: Int = 8, skip: Boolean = true): String {
            var strt = ""
            // for idx, t in enumerate(tensor):
            for (idx in 0 until tensor.shape[0]) {
                val t = tensor[idx]
                strt += ('[' + printTensor6D(t, "$indent ", nColumn = nColumn, skip = skip) + ']')
                strt += if (idx == tensor.shape[0] - 1) {
                    ""
                } else {
                    (",\n\n$indent")
                }
            }
            return strt
        }

        private fun printTensor8D(tensor: TorchTensor, indent: String = " ", nColumn: Int = 8, skip: Boolean = true): String {
            var strt = ""
            // for idx, t in enumerate(tensor):
            for (idx in 0 until tensor.shape[0]) {
                val t = tensor[idx]
                strt += ('[' + printTensor7D(t, "$indent ", nColumn = nColumn, skip = skip) + ']')
                strt += if (idx == tensor.shape[0] - 1) {
                    ""
                } else {
                    (",\n\n$indent")
                }
            }
            return strt
        }

        /**
         *  Serialize the tensor object
         *  Currently we only support ndim less than eight
         *  This function need to be improved in the future since we align the style of TensorPrinting
         *  However, the format string of different vector might be different
         *  (Since the formatting string of different vector are determined individually)
         *
         *  @return     The serialized string
         */
        override fun toString(): String {
            var strt = "tensor(["

            // Generate indent
            val indent = mutableListOf<Char>()
            for (i in 0 until "tensor([".length) {
                indent.add(' ')
            }

            // Add string via ndim
            strt += when (this.dim()) {
                1 -> printVector(this, indent = indent.joinToString(separator = ""))
                2 -> printMatrix(this, indent = indent.joinToString(separator = ""))
                3 -> printTensor3D(this, indent = indent.joinToString(separator = ""))
                4 -> printTensor4D(this, indent = indent.joinToString(separator = ""))
                5 -> printTensor5D(this, indent = indent.joinToString(separator = ""))
                6 -> printTensor6D(this, indent = indent.joinToString(separator = ""))
                7 -> printTensor7D(this, indent = indent.joinToString(separator = ""))
                8 -> printTensor8D(this, indent = indent.joinToString(separator = ""))
                else -> throw NotImplementedError()
            }
            strt += "])\n"
            return strt
        }

        /**
         *  Applies the Softmax function to an n-dimensional input Tensor rescaling them
         *  so that the elements of the n-dimensional output Tensor lie in the range [0,1] and sum to 1.
         *
         *  @param  dim     A dimension along which Softmax will be computed
         *                  (so every slice along dim will sum to 1)
         *  @return         Output result
         */
        fun softmax(dim: Int? = null): TorchTensor {
            if (dim == null) {
                val expTensor = clone().exp()
                val sum: Double = expTensor.sum().item().toDouble()
                for (idx in expTensor.storage.indices) {
                    expTensor.storage[idx] = expTensor.storage[idx].toDouble() / sum
                }

                this.shape = expTensor.shape
                this.storage = expTensor.storage
                this.stride = expTensor.stride
                return this
            } else {
                // Determine order of auto-contiguous and reverse
                val posDim = if (dim < 0) dim + dim() else dim
                fun getReverseOrder(order: Array<Int>, N: Int): Array<Int> {
                    val out = mutableListOf<Int>()
                    for (idx in 0 until N) {
                        out.add(order.indexOf(idx))
                    }
                    return out.toTypedArray()
                }
                var tensor = clone()
                val autoOrder = (mutableListOf(posDim) + (0 until posDim) + (posDim+1 until tensor.dim())).toTypedArray()
                val autoReverseOrder = getReverseOrder(autoOrder, N = tensor.dim())

                // Make each tensor continuous and get exponential
                tensor = tensor.permute(autoOrder)
                tensor = tensor.contiguous()
                tensor = tensor.exp()

                // Compute repeat order
                val reps = mutableListOf(tensor.shape[0])
                for (idx in 1 until tensor.shape.size) {
                    reps.add(1)
                }

                // Sum up for each slice manually
                var sumTensor: TorchTensor = zeros_like(tensor[0])
                for (idx in 0 until tensor.shape[0]) {
                    var t = tensor[idx]
                    sumTensor += t
                }
                sumTensor = unsqueeze(sumTensor, 0)
                sumTensor = sumTensor.repeat(reps)

                // Divide as equation and transpose back to original shape
                var output = tensor / sumTensor
                output = output.permute(autoReverseOrder)
                output.contiguous()
                return output
            }
        }
    }

    companion object {
        /**
         * ================================ from_array ===================================
         *
         *  Creates a Tensor from a Java ordinary array
         *
         *  @param  input   The input array
         */
        fun from_array(input: DoubleArray, size: Array<Int>): TorchTensor {
            val newStorage = mutableListOf<Number>()
            for (i in 0 until input.size) {
                newStorage.add(i, input[i] as Number)
            }
            return TorchTensor(size = size.toMutableList(), storage = newStorage)
        }

        fun from_array(input: DoubleArray): TorchTensor {
            return TorchTensor(input)
        }

        fun from_array(input: Array<DoubleArray>): TorchTensor {
            return TorchTensor(input)
        }

        fun from_array(input: Array<Array<DoubleArray>>): TorchTensor {
            return TorchTensor(input)
        }

        fun from_array(input: Array<Array<Array<DoubleArray>>>): TorchTensor {
            return TorchTensor(input)
        }

        fun from_array(input: Array<Array<Array<Array<DoubleArray>>>>): TorchTensor {
            return TorchTensor(input)
        }

        fun from_array(input: Array<Array<Array<Array<Array<DoubleArray>>>>>): TorchTensor {
            return TorchTensor(input)
        }

        fun from_array(input: Array<Array<Array<Array<Array<Array<DoubleArray>>>>>>): TorchTensor {
            return TorchTensor(input)
        }

        fun from_array(input: Array<Array<Array<Array<Array<Array<Array<DoubleArray>>>>>>>): TorchTensor {
            return TorchTensor(input)
        }

        fun from_array(input: FloatArray, size: Array<Int>): TorchTensor {
            val newStorage = mutableListOf<Number>()
            for (i in 0 until input.size) {
                newStorage.add(i, input[i] as Number)
            }
            return TorchTensor(size = size.toMutableList(), storage = newStorage)
        }

        fun from_array(input: FloatArray): TorchTensor {
            return TorchTensor(input)
        }

        fun from_array(input: Array<FloatArray>): TorchTensor {
            return TorchTensor(input)
        }

        fun from_array(input: Array<Array<FloatArray>>): TorchTensor {
            return TorchTensor(input)
        }

        fun from_array(input: Array<Array<Array<FloatArray>>>): TorchTensor {
            return TorchTensor(input)
        }

        fun from_array(input: Array<Array<Array<Array<FloatArray>>>>): TorchTensor {
            return TorchTensor(input)
        }

        fun from_array(input: Array<Array<Array<Array<Array<FloatArray>>>>>): TorchTensor {
            return TorchTensor(input)
        }

        fun from_array(input: Array<Array<Array<Array<Array<Array<FloatArray>>>>>>): TorchTensor {
            return TorchTensor(input)
        }

        fun from_array(input: Array<Array<Array<Array<Array<Array<Array<FloatArray>>>>>>>): TorchTensor {
            return TorchTensor(input)
        }

        fun from_array(input: LongArray, size: Array<Int>): TorchTensor {
            val newStorage = mutableListOf<Number>()
            for (i in 0 until input.size) {
                newStorage.add(i, input[i] as Number)
            }
            return TorchTensor(size = size.toMutableList(), storage = newStorage)
        }

        fun from_array(input: LongArray): TorchTensor {
            return TorchTensor(input)
        }

        fun from_array(input: Array<LongArray>): TorchTensor {
            return TorchTensor(input)
        }

        fun from_array(input: Array<Array<LongArray>>): TorchTensor {
            return TorchTensor(input)
        }

        fun from_array(input: Array<Array<Array<LongArray>>>): TorchTensor {
            return TorchTensor(input)
        }

        fun from_array(input: Array<Array<Array<Array<LongArray>>>>): TorchTensor {
            return TorchTensor(input)
        }

        fun from_array(input: Array<Array<Array<Array<Array<LongArray>>>>>): TorchTensor {
            return TorchTensor(input)
        }

        fun from_array(input: Array<Array<Array<Array<Array<Array<LongArray>>>>>>): TorchTensor {
            return TorchTensor(input)
        }

        fun from_array(input: Array<Array<Array<Array<Array<Array<Array<LongArray>>>>>>>): TorchTensor {
            return TorchTensor(input)
        }

        /**
         *  [Android-specific]
         *  Creates a Tensor from android bitmap
         *
         *  @param  input       The input bitmap
         *  @return             Corresponding tensor with shape [C, H, W]
         *                      Color-order is ARGB
         */
        fun from_bitmap(input: Bitmap): TorchTensor {
            var height = input.getHeight()
            var width = input.getWidth()
            val pixelsCount = height * width
            val pixels = IntArray(pixelsCount)
            input.getPixels(pixels, 0, width, 0, 0, width, height)

            var output = zeros(arrayOf(4, height, width))
            for (y in 0 until height) {
                for (x in 0 until width) {
                    output[0, y, x] = (pixels[y * width + x] shr 24) and 0xff // A
                    output[1, y, x] = (pixels[y * width + x] shr 16) and 0xff // R
                    output[2, y, x] = (pixels[y * width + x] shr 8) and 0xff  // G
                    output[3, y, x] = pixels[y * width + x] and 0xff          // B
                }
            }
            output.long()
            return output
        }

        /**
         *  [Android-specific]
         *  Wrapper to create PyTorch tensor for given KoTorch tensor
         *
         *  @param  input       The input KoTorch tensor
         *  @return             The output PyTorch tensor
         */
        fun toPyTorchFloat32Tensor(input: TorchTensor, channelCheck: Boolean = true): Tensor? {
            val tensor = input.contiguous()
            if (input.dtype != TensorType.FloatTensor) {
                throw NotImplementedError("We only support transform float type currently")
            }
            if (channelCheck) {
                if (tensor.shape.size >= 2 && tensor.shape[0] > tensor.shape[1]) {
                    println("[Warning] toPyTorchFloat32Tensor handle tensor with channel-first format. But detect channel-0 larger than channel-1")
                }
            }
            var outBuffer: FloatBuffer = Tensor.allocateFloatBuffer(tensor.numel())
            outBuffer = toBuffer(tensor, outBuffer)

            // Construct PyTorch tensor via float buffer
            val shape = mutableListOf<Long>()
            shape.add(1)
            for (i in 0 until tensor.shape.size) {
                shape.add(tensor.shape[i].toLong())
            }
            return Tensor.fromBlob(outBuffer, shape.toLongArray())
        }

        /**
         *  Encode KoTorch tensor into bytebuffer (for TensorFlow inference)
         *  Note:
         *      (1) Unlike usual original, we perform NCHW format encoding toward buffer
         *      (2) We have only test correctness on tflite which is transformed from ONNX
         *          We have not test the tflite which is transformed from tensorflow directly
         *
         *  @param  input       The input KoTorch tensor
         *  @return             The encode byte buffer
         */
        fun toTensorFlowByteBuffer(input: TorchTensor): ByteBuffer {
            val tensor = input.contiguous()
            if (input.dtype != TensorType.FloatTensor) {
                throw NotImplementedError("We only support transform float type currently")
            }
            var outBuffer: ByteBuffer = ByteBuffer.allocateDirect(4 * input.numel())
            outBuffer = toBuffer(tensor, outBuffer)
            return outBuffer
        }

        /**
         * ================================ toBuffer ===================================
         *  Encode tensor and put into given buffer
         *  We share the same function name for both PyTorch and TensorFlow
         *  For PyTorch, you should provide floatbuffer
         *  For TensorFlow, you should provide bytebuffer
         *  We only test and validate the correctness on 3-dimension (image)
         *
         *  Note:
         *      (1) We have only test correctness on tflite which is transformed from ONNX
         *          We have not test the tflite which is transformed from tensorflow directly
         *
         *  @param  input       The tensor you want to encode
         *  @return             filled buffer
         */
        fun toBuffer(input: TorchTensor, buffer: FloatBuffer): FloatBuffer {
            when (input.dim()) {
                1 -> {
                    var ptr = 0
                    for (idx0 in 0 until input.shape[0]) {
                        buffer.put(ptr, input[idx0].item().toFloat())
                        ptr += 1
                    }
                }
                2 -> {
                    var ptr = 0
                    for (idx0 in 0 until input.shape[0]) {
                        for (idx1 in 0 until input.shape[1]) {
                            buffer.put(ptr, input[idx0, idx1].item().toFloat())
                            ptr += 1
                        }
                    }
                }
                3 -> {
                    var ptr = 0
                    for (idx0 in 0 until input.shape[0]) {
                        for (idx1 in 0 until input.shape[1]) {
                            for (idx2 in 0 until input.shape[2]) {
                                buffer.put(ptr, input[idx0, idx1, idx2].item().toFloat())
                                ptr += 1
                            }
                        }
                    }
                }
                4 -> {
                    var ptr = 0
                    for (idx0 in 0 until input.shape[0]) {
                        for (idx1 in 0 until input.shape[1]) {
                            for (idx2 in 0 until input.shape[2]) {
                                for (idx3 in 0 until input.shape[3]) {
                                    buffer.put(ptr, input[idx0, idx1, idx2, idx3].item().toFloat())
                                    ptr += 1
                                }
                            }
                        }
                    }
                }
                5 -> {
                    var ptr = 0
                    for (idx0 in 0 until input.shape[0]) {
                        for (idx1 in 0 until input.shape[1]) {
                            for (idx2 in 0 until input.shape[2]) {
                                for (idx3 in 0 until input.shape[3]) {
                                    for (idx4 in 0 until input.shape[4]) {
                                        buffer.put(ptr, input[idx0, idx1, idx2, idx3, idx4].item().toFloat())
                                        ptr += 1
                                    }
                                }
                            }
                        }
                    }
                }
                6 -> {
                    var ptr = 0
                    for (idx0 in 0 until input.shape[0]) {
                        for (idx1 in 0 until input.shape[1]) {
                            for (idx2 in 0 until input.shape[2]) {
                                for (idx3 in 0 until input.shape[3]) {
                                    for (idx4 in 0 until input.shape[4]) {
                                        for (idx5 in 0 until input.shape[5]) {
                                            buffer.put(ptr, input[idx0, idx1, idx2, idx3, idx4, idx5].item().toFloat())
                                            ptr += 1
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                7 -> {
                    var ptr = 0
                    for (idx0 in 0 until input.shape[0]) {
                        for (idx1 in 0 until input.shape[1]) {
                            for (idx2 in 0 until input.shape[2]) {
                                for (idx3 in 0 until input.shape[3]) {
                                    for (idx4 in 0 until input.shape[4]) {
                                        for (idx5 in 0 until input.shape[5]) {
                                            for (idx6 in 0 until input.shape[6]) {
                                                buffer.put(ptr, input[idx0, idx1, idx2, idx3, idx4, idx5, idx6].item().toFloat())
                                                ptr += 1
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                8 -> {
                    var ptr = 0
                    for (idx0 in 0 until input.shape[0]) {
                        for (idx1 in 0 until input.shape[1]) {
                            for (idx2 in 0 until input.shape[2]) {
                                for (idx3 in 0 until input.shape[3]) {
                                    for (idx4 in 0 until input.shape[4]) {
                                        for (idx5 in 0 until input.shape[5]) {
                                            for (idx6 in 0 until input.shape[6]) {
                                                for (idx7 in 0 until input.shape[7]) {
                                                    buffer.put(ptr, input[idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7].item().toFloat())
                                                    ptr += 1
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return buffer
        }

        fun toBuffer(input: TorchTensor, buffer: ByteBuffer, unitSize: Int = 4): ByteBuffer {
            when (input.dim()) {
                1 -> {
                    var ptr = 0
                    for (idx0 in 0 until input.shape[0]) {
                        for (idx1 in 0 until input.shape[1]) {
                            buffer.putFloat(ptr, input[idx0, idx1].item().toFloat())
                            ptr += unitSize
                        }
                    }
                }
                2 -> {
                    var ptr = 0
                    for (idx0 in 0 until input.shape[0]) {
                        for (idx1 in 0 until input.shape[1]) {
                            buffer.putFloat(ptr, input[idx0, idx1].item().toFloat())
                            ptr += unitSize
                        }
                    }
                }
                3 -> {
                    var ptr = 0
                    for (idx0 in 0 until input.shape[0]) {
                        for (idx1 in 0 until input.shape[1]) {
                            for (idx2 in 0 until input.shape[2]) {
                                buffer.putFloat(ptr, input[idx0, idx1, idx2].item().toFloat())
                                ptr += unitSize
                            }
                        }
                    }
                }
                4 -> {
                    var ptr = 0
                    for (idx0 in 0 until input.shape[0]) {
                        for (idx1 in 0 until input.shape[1]) {
                            for (idx2 in 0 until input.shape[2]) {
                                for (idx3 in 0 until input.shape[3]) {
                                    buffer.putFloat(ptr, input[idx0, idx1, idx2, idx3].item().toFloat())
                                    ptr += unitSize
                                }
                            }
                        }
                    }
                }
                5 -> {
                    var ptr = 0
                    for (idx0 in 0 until input.shape[0]) {
                        for (idx1 in 0 until input.shape[1]) {
                            for (idx2 in 0 until input.shape[2]) {
                                for (idx3 in 0 until input.shape[3]) {
                                    for (idx4 in 0 until input.shape[4]) {
                                        buffer.putFloat(ptr, input[idx0, idx1, idx2, idx3, idx4].item().toFloat())
                                        ptr += unitSize
                                    }
                                }
                            }
                        }
                    }
                }
                6 -> {
                    var ptr = 0
                    for (idx0 in 0 until input.shape[0]) {
                        for (idx1 in 0 until input.shape[1]) {
                            for (idx2 in 0 until input.shape[2]) {
                                for (idx3 in 0 until input.shape[3]) {
                                    for (idx4 in 0 until input.shape[4]) {
                                        for (idx5 in 0 until input.shape[5]) {
                                            buffer.putFloat(ptr, input[idx0, idx1, idx2, idx3, idx4, idx5].item().toFloat())
                                            ptr += unitSize
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                7 -> {
                    var ptr = 0
                    for (idx0 in 0 until input.shape[0]) {
                        for (idx1 in 0 until input.shape[1]) {
                            for (idx2 in 0 until input.shape[2]) {
                                for (idx3 in 0 until input.shape[3]) {
                                    for (idx4 in 0 until input.shape[4]) {
                                        for (idx5 in 0 until input.shape[5]) {
                                            for (idx6 in 0 until input.shape[6]) {
                                                buffer.putFloat(ptr, input[idx0, idx1, idx2, idx3, idx4, idx5, idx6].item().toFloat())
                                                ptr += unitSize
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                8 -> {
                    var ptr = 0
                    for (idx0 in 0 until input.shape[0]) {
                        for (idx1 in 0 until input.shape[1]) {
                            for (idx2 in 0 until input.shape[2]) {
                                for (idx3 in 0 until input.shape[3]) {
                                    for (idx4 in 0 until input.shape[4]) {
                                        for (idx5 in 0 until input.shape[5]) {
                                            for (idx6 in 0 until input.shape[6]) {
                                                for (idx7 in 0 until input.shape[7]) {
                                                    buffer.putFloat(ptr, input[idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7].item().toFloat())
                                                    ptr += unitSize
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return buffer
        }

        /**
         *  Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size
         *
         *  @param  size        A sequence of integers defining the shape of the output tensor
         *  @return             The all-zero tensor
         */
        @JvmName("zerosWithInputArray")
        fun zeros(size: Array<Int>): TorchTensor {
            return TorchTensor(size.toMutableList(), 0f)
        }

        @JvmName("zerosWithInputMutableList")
        fun zeros(size: MutableList<Int>): TorchTensor {
            return TorchTensor(size, 0f)
        }

        @JvmName("zerosWithInputArrayList")
        fun zeros(size: ArrayList<Int>): TorchTensor {
            return TorchTensor(size, 0f)
        }

        @JvmName("zerosWithInputList")
        fun zeros(size: List<Int>): TorchTensor {
            return TorchTensor(size.toMutableList(), 0f)
        }

        /**
         *  Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument size
         *
         *  @param  size        A sequence of integers defining the shape of the output tensor
         *  @return             The all-one tensor
         */
        @JvmName("onesWithInputArray")
        fun ones(size: Array<Int>): TorchTensor {
            return TorchTensor(size.toMutableList(), 1f)
        }

        @JvmName("onesWithInputMutableList")
        fun ones(size: MutableList<Int>): TorchTensor {
            return TorchTensor(size, 1f)
        }

        @JvmName("onesWithInputArrayList")
        fun ones(size: ArrayList<Int>): TorchTensor {
            return TorchTensor(size, 1f)
        }

        @JvmName("onesWithInputList")
        fun ones(size: List<Int>): TorchTensor {
            return TorchTensor(size.toMutableList(), 1f)
        }

        /**
         *  Returns a tensor filled with the scalar value 0, with the same size as input
         *
         *  @param  input       The size of input will determine size of the output tensor
         *  @return             The all-zero tensor
         */
        fun zeros_like(input: TorchTensor): TorchTensor {
            return TorchTensor(input.shape.toMutableList(), 0f)
        }

        /**
         *  Returns a tensor filled with the scalar value 1, with the same size as input
         *
         *  @param  input       The size of input will determine size of the output tensor
         *  @return             The all-one tensor
         */
        fun ones_like(input: TorchTensor): TorchTensor {
            return TorchTensor(input.shape.toMutableList(), 1f)
        }

        /**
         *  Returns the total number of elements in the input tensor
         *
         *  @return             The total number of elements
         */
        fun numel(input: TorchTensor): Int {
            return input.numel()
        }

        /**
         *  Concatenates the given sequence of seq Tensors in the given dimension.
         *
         *  @param  inputs      List of Tensor of the same type.
         *  @param  dim         The dimension over which the tensors are concatenated
         *  @return             The concatenated tensor
         */
        fun cat(inputs: List<TorchTensor>, dim: Int): TorchTensor {
            // Check if list of tensor is more than two
            when (inputs.size) {
                0 -> throw Exception("Empty list")
                1 -> return inputs[0]
                else -> {}
            }

            // Check if shapes are valid
            val posDim = if (dim < 0) dim + inputs[0].dim() else dim
            val exampleSize = inputs[0].shape.toMutableList()
            exampleSize.removeAt(posDim)
            for (input in inputs) {
                val size = input.shape.toMutableList()
                size.removeAt(posDim)
                if (size != exampleSize) {
                    throw Exception("Invalid shape. aSize: $exampleSize   bSize: $size")
                }
            }

            // Determine order of auto-contiguous and reverse
            fun getReverseOrder(order: Array<Int>, N: Int): Array<Int> {
                val out = mutableListOf<Int>()
                for (idx in 0 until N) {
                    out.add(order.indexOf(idx))
                }
                return out.toTypedArray()
            }
            val tensor = inputs[0]
            val autoOrder = (mutableListOf(posDim) + (0 until posDim) + (posDim+1 until tensor.dim())).toTypedArray()
            val autoReverseOrder = getReverseOrder(autoOrder, N = tensor.dim())

            // Make each tensor continuous first
            val contiguousInputs = mutableListOf<TorchTensor>()
            for (input in inputs) {
                val contiguousInput = input.permute(autoOrder)
                contiguousInputs.add(contiguousInput.contiguous())
            }

            // Compute new shape
            val newSize = contiguousInputs[0].shape
            for (tensor_idx in 1 until contiguousInputs.size) {
                newSize[0] += contiguousInputs[tensor_idx].shape[0]
            }

            // Generate new array
            val newStorage = mutableListOf<Number>()
            for (input in contiguousInputs) {
                newStorage += input._array()
            }

            // Transpose back to original shape
            var output = TorchTensor(size = newSize, storage = newStorage)
            output.dtype = inputs[0].dtype
            output = output.permute(autoReverseOrder)
            output.contiguous()
            return output
        }

        /**
         *  Reverse the order of a n-D tensor along given axis in dims
         *
         *  @param  input       The input tensor
         *  @param  dims        Axis to flip on
         *  @return             The flip result
         */
        private fun flip(input: TorchTensor, dim: Int): TorchTensor {
            // Determine order of auto-contiguous and reverse
            fun getReverseOrder(order: Array<Int>, N: Int): Array<Int> {
                val out = mutableListOf<Int>()
                for (idx in 0 until N) {
                    out.add(order.indexOf(idx))
                }
                return out.toTypedArray()
            }
            val posDim = if (dim < 0) dim + input[0].dim() else dim
            var tensor = input.clone()
            val autoOrder = (mutableListOf(posDim) + (0 until posDim) + (posDim+1 until tensor.dim())).toTypedArray()
            val autoReverseOrder = getReverseOrder(autoOrder, N = tensor.dim())

            // permute
            tensor.permute(autoOrder)
            tensor.contiguous()

            // flip via ndim
            var output = tensor.clone()
            when (tensor.dim()) {
                1 -> {
                    for (idx0 in 0 until tensor.shape[0]) {
                        output[idx0] = tensor[tensor.shape[0] - idx0 - 1].item()
                    }
                }
                2 -> {
                    for (idx1 in 0 until tensor.shape[1]) {
                        for (idx0 in 0 until tensor.shape[0]) {
                            output[idx0, idx1] = tensor[tensor.shape[0] - idx0 - 1, idx1].item()
                        }
                    }
                }
                3 -> {
                    for (idx1 in 0 until tensor.shape[1]) {
                        for (idx2 in 0 until tensor.shape[2]) {
                            for (idx0 in 0 until tensor.shape[0]) {
                                output[idx0, idx1, idx2] = tensor[tensor.shape[0] - idx0 - 1, idx1, idx2].item()
                            }
                        }
                    }
                }
                4 -> {
                    for (idx1 in 0 until tensor.shape[1]) {
                        for (idx2 in 0 until tensor.shape[2]) {
                            for (idx3 in 0 until tensor.shape[3]) {
                                for (idx0 in 0 until tensor.shape[0]) {
                                    output[idx0, idx1, idx2, idx3] = tensor[tensor.shape[0] - idx0 - 1, idx1, idx2, idx3].item()
                                }
                            }
                        }
                    }
                }
                5 -> {
                    for (idx1 in 0 until tensor.shape[1]) {
                        for (idx2 in 0 until tensor.shape[2]) {
                            for (idx3 in 0 until tensor.shape[3]) {
                                for (idx4 in 0 until tensor.shape[4]) {
                                    for (idx0 in 0 until tensor.shape[0]) {
                                        output[idx0, idx1, idx2, idx3, idx4] = tensor[tensor.shape[0] - idx0 - 1, idx1, idx2, idx3, idx4].item()
                                    }
                                }
                            }
                        }
                    }
                }
                6 -> {
                    for (idx1 in 0 until tensor.shape[1]) {
                        for (idx2 in 0 until tensor.shape[2]) {
                            for (idx3 in 0 until tensor.shape[3]) {
                                for (idx4 in 0 until tensor.shape[4]) {
                                    for (idx5 in 0 until tensor.shape[5]) {
                                        for (idx0 in 0 until tensor.shape[0]) {
                                            output[idx0, idx1, idx2, idx3, idx4, idx5] = tensor[tensor.shape[0] - idx0 - 1, idx1, idx2, idx3, idx4, idx5].item()
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                7 -> {
                    for (idx1 in 0 until tensor.shape[1]) {
                        for (idx2 in 0 until tensor.shape[2]) {
                            for (idx3 in 0 until tensor.shape[3]) {
                                for (idx4 in 0 until tensor.shape[4]) {
                                    for (idx5 in 0 until tensor.shape[5]) {
                                        for (idx6 in 0 until tensor.shape[6]) {
                                            for (idx0 in 0 until tensor.shape[0]) {
                                                output[idx0, idx1, idx2, idx3, idx4, idx5, idx6] = tensor[tensor.shape[0] - idx0 - 1, idx1, idx2, idx3, idx4, idx5, idx6].item()
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                8 -> {
                    for (idx1 in 0 until tensor.shape[1]) {
                        for (idx2 in 0 until tensor.shape[2]) {
                            for (idx3 in 0 until tensor.shape[3]) {
                                for (idx4 in 0 until tensor.shape[4]) {
                                    for (idx5 in 0 until tensor.shape[5]) {
                                        for (idx6 in 0 until tensor.shape[6]) {
                                            for (idx7 in 0 until tensor.shape[7]) {
                                                for (idx0 in 0 until tensor.shape[0]) {
                                                    output[idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7] = tensor[tensor.shape[0] - idx0 - 1, idx1, idx2, idx3, idx4, idx5, idx6, idx7].item()
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Transpose back to original shape
            output = output.permute(autoReverseOrder)
            output.contiguous()
            return output
        }

        @JvmName("torch_flip_Array")
        fun flip(input: TorchTensor, dims: Array<Int>): TorchTensor {
            var output = input
            for (i in 0 until dims.size) {
                output = flip(output, dims[i])
            }
            return output
        }

        @JvmName("torch_flip_MutableList")
        fun flip(input: TorchTensor, dims: MutableList<Int>): TorchTensor {
            return flip(input, dims.toTypedArray())
        }

        @JvmName("torch_flip_ArrayList")
        fun flip(input: TorchTensor, dims: ArrayList<Int>): TorchTensor {
            return flip(input, dims.toTypedArray())
        }

        @JvmName("torch_flip_List")
        fun flip(input: TorchTensor, dims: List<Int>): TorchTensor {
            return flip(input, dims.toTypedArray())
        }

        /**
         *  Returns a tensor that is a transposed version of input
         *  The given dimensions dim0 and dim1 are swapped
         *
         *  @param  input       The input tensor
         *  @param  dim0        The first dimension to be transposed
         *  @param  dim1        The second dimension to be transposed
         *  @return             The transposed tensor
         */
        fun transpose(input: TorchTensor, dim0: Int, dim1: Int): TorchTensor {
            val posDim0 = if (dim0 < 0) dim0 + input.dim() else dim0
            val posDim1 = if (dim1 < 0) dim1 + input.dim() else dim1

            if (posDim0 >= input.dim() || posDim0 < 0)  {
                throw IndexError("Dimension out of range (expected to be in range of [" +
                        (input.dim()) * -1 + ", " + (input.dim() - 1) +
                        "], but got " + posDim0 + ")")
            } else if (posDim1 >= input.dim() || posDim1 < 0)  {
                throw IndexError("Dimension out of range (expected to be in range of [" +
                        (input.dim()) * -1 + ", " + (input.dim() - 1) +
                        "], but got " + posDim1 + ")")
            } else {
                val newSize = input.shape.toMutableList()
                val newStride = input.stride.toMutableList()
                newSize[posDim0] = input.shape[posDim1]
                newSize[posDim1] = input.shape[posDim0]
                newStride[posDim0] = input.stride[posDim1]
                newStride[posDim1] = input.stride[posDim0]
                return TorchTensor(size = newSize, storage = input._array(), stride = newStride)
            }
        }

        /**
         *  Returns a new tensor with a dimension of size one inserted at the specified position
         *  A dim value within the range [-input.dim() - 1, input.dim() + 1) can be used
         *  Negative dim will correspond to unsqueeze() applied at dim = dim + input.dim() + 1
         *
         *  @param  input       The input tensor
         *  @param  dim         The index at which to insert the singleton dimension
         *  @return             The flat tensor
         */
        fun unsqueeze(input: TorchTensor, dim: Int): TorchTensor {
            val posDim = if (dim < 0) dim + input.dim() + 1 else dim
            if (posDim <= input.dim() && posDim >= 0) {
                val newSize = input.shape.toMutableList()
                val newStride = input.stride.toMutableList()
                newSize.add(posDim, 1)
                newStride.add(posDim, (if (posDim == input.dim()) 1 else input.stride[posDim]))

                val output = TorchTensor(size = newSize, storage = input._array(), stride = newStride)
                output.dtype = input.dtype
                return output
            } else {
                throw IndexError("Dimension out of range (expected to be in range of [" +
                        ((input.dim() + 1) * -1) + ", " +
                        input.dim() + "], but got " + dim + ")")
            }
        }

        /**
         *  Returns a tensor with all the dimensions of input of size 1 removed
         *  When dim is given, a squeeze operation is done only in the given dimension
         *
         *  @param  input       The input tensor
         *  @param  dim         If given, the input will be squeezed only in this dimension
         *  @return             The output tensor
         */
        fun squeeze(input: TorchTensor, dim: Int): TorchTensor {
            val posDim = if (dim < 0) dim + input.dim() else dim
            if (posDim < input.dim() && posDim >= 0) {
                if (input.size()[posDim] == 1) {
                    val newSize = input.shape.toMutableList()
                    val newStride = input.stride.toMutableList()
                    newSize.removeAt(posDim)
                    newStride.removeAt(posDim)
                    return TorchTensor(size = newSize, storage = input._array(), stride = newStride)
                } else {
                    return input
                }
            } else {
                throw IndexError("Dimension out of range (expected to be in range of [" +
                        (input.dim() * -1) + ", " +
                        (input.dim() - 1) + "], but got " + dim + ")")
            }
        }

        fun squeeze(input: TorchTensor): TorchTensor {
            var output = input
            while (true) {
                var hasOne = false
                for (idx in 0 until output.size().size) {
                    if (output.size()[idx] == 1) {
                        hasOne = true
                        output = squeeze(output, idx)
                        break
                    }
                }
                if (!hasOne) {
                    break
                }
            }
            return output
        }

        /**
         *  Concatenates a sequence of tensors along a new dimension
         *  All tensors need to be of the same size
         *
         *  @param  tensors     Sequence of tensors to concatenate
         *  @param  dim         Dimension to insert.
         *                      Has to be between 0 and the number of dimensions of concatenated tensors
         *  @return             The output tensor
         */
        @JvmName("torch_stack_Array")
        fun stack(tensors: Array<TorchTensor>, dim: Int): TorchTensor {
            return stack(tensors.toList(), dim)
        }

        @JvmName("torch_stack_MutableList")
        fun stack(tensors: MutableList<TorchTensor>, dim: Int): TorchTensor {
            return stack(tensors.toList(), dim)
        }

        @JvmName("torch_stack_ArrayList")
        fun stack(tensors: ArrayList<TorchTensor>, dim: Int): TorchTensor {
            return stack(tensors.toList(), dim)
        }

        @JvmName("torch_stack_List")
        fun stack(tensors: List<TorchTensor>, dim: Int): TorchTensor {
            val posDim = if (dim < 0) dim + tensors[0].dim() + 1 else dim
            if (tensors.isEmpty()) {
                throw Exception("stack expects a non-empty sequence of tensors")
            }
            val input = mutableListOf<TorchTensor>()
            for (t in tensors) {
                input.add(unsqueeze(t, posDim))
            }
            return cat(input, posDim)
        }

        /**
         *  Splits a tensor into a number of chunks along a given dimension.
         *  Last chunk will be smaller if the tensor size along the given dimension dim is not divisible by chunks
         *  Ref: https://pytorch.org/docs/0.1.12/_modules/torch/functional.html#split
         *
         *  @param  tensor      Tensor to split
         *  @param  chunks      Number of chunks to return
         *  @param  dim         Dimension along which to split the tensor
         *  @return             The chunks of tensor list
         */
        fun chunk(tensor: TorchTensor, chunks: Int, dim: Int): MutableList<TorchTensor> {
            var posDim = dim
            if (dim < 0) {
                posDim += tensor.dim()
            }
            val split_size = ((tensor.shape[posDim] + chunks - 1) / chunks).toInt()
            return split(tensor, split_size, posDim)
        }

        /**
         *  Splits the tensor into equally sized chunks (if possible).
         *  Last chunk will be smaller if the tensor size along a given dimension
         *  is not divisible by ``split_size``.
         *  Ref: https://pytorch.org/docs/0.1.12/_modules/torch/functional.html#split
         *
         *  @param  tensor      Tensor to split
         *  @param  split_size  Size of a single chunk
         *  @param  dim         Dimension along which to split the tensor
         *  @return             The splitted tensor list
         */
        fun split(tensor: TorchTensor, split_size: Int, dim: Int): MutableList<TorchTensor> {
            var posDim = dim
            if (dim < 0) {
                posDim += tensor.dim()
            }
            val dim_size = tensor.shape[dim]
            val num_splits = ((dim_size + split_size - 1) / split_size).toInt()
            val last_split_size = split_size - (split_size * num_splits - dim_size)
            fun get_split_size(i: Int): Int {
                return if (i < num_splits - 1) {
                    split_size
                } else {
                    last_split_size
                }
            }
            val outputs = mutableListOf<TorchTensor>()
            for (i in 0 until num_splits) {
                var output = tensor.clone()
                output = output.narrow(dim, (i * split_size), get_split_size(i))
                outputs.add(output)
            }
            return outputs
        }

        /**
         *  Perform transpose for the 2D matrix
         *  Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.
         *
         *  @param  tensor      The input tensor.
         *  @return             The transposed tensor
         */
        fun t(tensor: TorchTensor): TorchTensor {
            return when (tensor.dim()) {
                1 -> tensor
                2 -> transpose(tensor, 0, 1)
                else -> {
                    throw RuntimeError("t() expects a tensor with <= 2 dimensions, but self is "
                            + tensor.dim() + "D\n")
                }
            }
        }

        /**
         *  Returns a tensor with the same data and number of elements as input,
         *  but with the specified shape
         *
         *  @param  tensor      The tensor to be reshaped
         *  @param  shape       The new shape
         *  @return             The reshaped tensor
         */
        @JvmName("torch_reshape_Array")
        fun reshape(tensor: TorchTensor, shape: Array<Int>): TorchTensor {
            return tensor.view(shape.toMutableList())
        }

        @JvmName("torch_reshape_MutableList")
        fun reshape(tensor: TorchTensor, shape: MutableList<Int>): TorchTensor {
            return tensor.view(shape)
        }

        @JvmName("torch_reshape_ArrayList")
        fun reshape(tensor: TorchTensor, shape: ArrayList<Int>): TorchTensor {
            return tensor.view(shape.toMutableList())
        }

        @JvmName("torch_reshape_List")
        fun reshape(tensor: TorchTensor, shape: List<Int>): TorchTensor {
            return tensor.view(shape.toMutableList())
        }

        /**
         *  Constructs a tensor by repeating the elements of input.
         *  The reps argument specifies the number of repetitions in each dimension.
         *
         *  If reps specifies fewer dimensions than input has,
         *  then ones are prepended to reps until all dimensions are specified.
         *  For example, if input has shape (8, 6, 4, 2) and reps is (2, 2),
         *  then reps is treated as (1, 1, 2, 2).
         *
         *  Analogously, if input has fewer dimensions than reps specifies,
         *  then input is treated as if it were unsqueezed at dimension zero
         *  until it has as many dimensions as reps specifies.
         *  For example, if input has shape (4, 2) and reps is (3, 3, 2, 2),
         *  then input is treated as if it had the shape (1, 1, 4, 2).
         *  Ref: https://pytorch.org/docs/stable/generated/torch.tile.html
         *
         *  @param  tensor      The tensor whose elements to repeat.
         *  @param  reps        The number of repetitions per dimension
         *  @return             Tiling result
         */
        @JvmName("torch_tile_Array")
        fun tile(tensor: TorchTensor, reps: Array<Int>): TorchTensor {
            return tile(tensor, reps.toMutableList())
        }

        @JvmName("torch_tile_MutableList")
        fun tile(tensor: TorchTensor, reps: MutableList<Int>): TorchTensor {
            var inputTensor = tensor
            var inputReps = reps
            while (true) {
                if (inputTensor.shape.size > inputReps.size) {
                    inputReps.add(0, 1)
                } else if (inputTensor.shape.size < inputReps.size) {
                    inputTensor = unsqueeze(inputTensor, 0)
                } else {
                    break
                }
            }
            return inputTensor.repeat(inputReps)
        }

        @JvmName("torch_tile_ArrayList")
        fun tile(tensor: TorchTensor, reps: ArrayList<Int>): TorchTensor {
            return tile(tensor, reps.toMutableList())
        }

        @JvmName("torch_tile_Array")
        fun tile(tensor: TorchTensor, reps: List<Int>): TorchTensor {
            return tile(tensor, reps.toMutableList())
        }

        /**
         *  Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
         *  Ref: https://pytorch.org/docs/0.1.12/torch.html#torch.eye
         *
         *  @param  n           Number of rows
         *  @param  m           Number of columns. If no specify, defaults to n
         *  @return             A 2-D tensor with ones on the diagonal and zeros elsewhere
         */
        fun eye(n: Int, m: Int = -1): TorchTensor {
            val output = if (m == -1) {
                zeros(arrayOf(n, n))
            } else {
                zeros(arrayOf(n, m))
            }
            println(output.dim())
            for (idx in 0 until n) {
                output[idx, idx] = 1.0
            }
            return output
        }

        /**
         *  Returns a Tensor filled with random numbers from a normal distribution with zero mean and variance of one
         *  The shape of the Tensor is defined by the varargs sizes
         *
         *  @param  size        A set of ints defining the shape of the output Tensor
         *  @return             A tensor which form Isochopic Gaussian distribution
         */
        @JvmName("torch_randn_Array")
        fun randn(size: Array<Int>): TorchTensor {
            val newStorage = mutableListOf<Number>()
            val N = size.reduce { acc, i -> acc * i }
            for (idx in 0 until N) {
                newStorage.add(java.util.Random().nextGaussian().toFloat())
            }
            return TorchTensor(size = size.toMutableList(), storage = newStorage)
        }

        @JvmName("torch_randn_MutableList")
        fun randn(size: MutableList<Int>): TorchTensor {
            return randn(size.toTypedArray())
        }

        @JvmName("torch_randn_ArrayList")
        fun randn(size: ArrayList<Int>): TorchTensor {
            return randn(size.toTypedArray())
        }

        @JvmName("torch_randn_List")
        fun randn(size: List<Int>): TorchTensor {
            return randn(size.toTypedArray())
        }

        @JvmName("torch_randn_mean_std_float_Array")
        fun randn(size: Array<Int>, mean: Float, std: Float): TorchTensor {
            var out = randn(size = size)
            out = (out * std) + mean
            return out
        }

        @JvmName("torch_randn_mean_std_float_MutableList")
        fun randn(size: MutableList<Int>, mean: Float, std: Float): TorchTensor {
            return randn(size.toTypedArray(), mean, std)
        }

        @JvmName("torch_randn_mean_std_float_ArrayList")
        fun randn(size: ArrayList<Int>, mean: Float, std: Float): TorchTensor {
            return randn(size.toTypedArray(), mean, std)
        }

        @JvmName("torch_randn_mean_std_float_List")
        fun randn(size: List<Int>, mean: Float, std: Float): TorchTensor {
            return randn(size.toTypedArray(), mean, std)
        }

        @JvmName("torch_randn_mean_std_tensor_Array")
        fun randn(size: Array<Int>, mean: TorchTensor, std: TorchTensor): TorchTensor {
            if (mean.dim() == 1 && std.dim() == 1) {
                if (mean._array().size == size.size && std._array().size == size.size) {
                    val meanShape = mutableListOf<Int>()
                    for (idx in 0 until size.size) {
                        meanShape.add(1)
                    }
                    val meanT = reshape(mean, meanShape).expand_as(zeros(size))
                    val stdT = reshape(std, meanShape).expand_as(zeros(size))
                    var output = randn(size)
                    output = (output + meanT) * stdT
                    return output
                } else {
                    throw Exception("The size of mean and std tensor should be equal to size list. Size length" +
                            (size.size) + "\tMean length: " + (mean._array().size) +
                            "\tStd ndim: " + (std._array().size))
                }
            } else if (mean.dim() == size.size && std.dim() == size.size) {
                val meanT: TorchTensor = if (mean.shape != size) {
                    mean.expand_as(zeros(size))
                } else {
                    mean
                }
                val stdT: TorchTensor = if (std.shape != size) {
                    std.expand_as(zeros(size))
                } else {
                    std
                }
                var output = randn(size)
                output = (output + meanT) * stdT
                return output
            } else {
                throw Exception("The ndim of mean and std tensor should equal to size length or 1. Size length: " +
                        (size.size).toString() + "\tMean ndim: " + (mean.dim()).toString() +
                        "\tStd ndim: " + std.dim())
            }
        }

        @JvmName("torch_randn_mean_std_tensor_MutableList")
        fun randn(size: MutableList<Int>, mean: TorchTensor, std: TorchTensor): TorchTensor {
            return randn(size.toTypedArray(), mean, std)
        }

        @JvmName("torch_randn_mean_std_tensor_ArrayList")
        fun randn(size: ArrayList<Int>, mean: TorchTensor, std: TorchTensor): TorchTensor {
            return randn(size.toTypedArray(), mean, std)
        }

        @JvmName("torch_randn_mean_std_tensor_List")
        fun randn(size: List<Int>, mean: TorchTensor, std: TorchTensor): TorchTensor {
            return randn(size.toTypedArray(), mean, std)
        }

        /**
         *  Returns a Tensor filled with random numbers from a uniform distribution on the interval [0,1)
         *  The shape of the Tensor is defined by the varargs sizes
         *
         *  @param  size        A set of ints defining the shape of the output Tensor
         *  @return             A tensor which form Uniform distribution
         */
        @JvmName("torch_rand_Array")
        fun rand(size: Array<Int>): TorchTensor {
            return rand(size.toMutableList())
        }

        @JvmName("torch_rand_MutableList")
        fun rand(size: MutableList<Int>): TorchTensor {
            val newStorage = mutableListOf<Number>()
            val N = size.reduce { acc, i -> acc * i }
            for (idx in 0 until N) {
                newStorage.add(java.util.Random().nextFloat())
            }
            return TorchTensor(size = size, storage = newStorage)
        }

        @JvmName("torch_rand_ArrayList")
        fun rand(size: ArrayList<Int>): TorchTensor {
            return rand(size.toMutableList())
        }

        @JvmName("torch_rand_List")
        fun rand(size: List<Int>): TorchTensor {
            return rand(size.toMutableList())
        }

        /**
         *  Returns a new Tensor with the ceil of the elements of input,
         *  the smallest integer greater than or equal to each element.
         *
         *  @param  input       The input Tensor
         *  @return             The result Tensor
         */
        fun ceil(input: TorchTensor): TorchTensor {
            // in-place function
            for (idx in 0 until input._nElement()) {
                input._set(idx, kotlin.math.ceil(input._get(idx).toDouble()).toFloat())
            }
            return input
        }

        /**
         *  Clamp all elements in input into the range [min, max] and return a resulting Tensor.
         *  The detail can be refer in: https://pytorch.org/docs/0.1.12/torch.html?#torch.clamp
         *
         *  @param  input       The input Tensor
         *  @param  min         Lower-bound of the range to be clamped to
         *  @param  max         Upper-bound of the range to be clamped to
         *  @return             The result Tensor
         */
        fun clamp(input: TorchTensor, min: Float, max: Float): TorchTensor {
            // in-place function
            when (input.dtype) {
                TensorType.FloatTensor -> {
                    for (idx in 0 until input._nElement()) {
                        val a = input._get(idx).toFloat()
                        val b = Math.max(min, Math.min(max, a))
                        input._set(idx, b)
                    }
                }
                else -> throw NotImplementedError()
            }
            return input
        }

        /**
         *  Returns a new Tensor with the floor of the elements of input,
         *  the largest integer less than or equal to each element.
         *
         *  @param  input       The input Tensor
         *  @return             The result Tensor
         */
        fun floor(input: TorchTensor): TorchTensor {
            // in-place function
            for (idx in 0 until input._nElement()) {
                input._set(idx, kotlin.math.floor(input._get(idx).toDouble()).toFloat())
            }
            return input
        }

        /**
         *  Returns a new Tensor with the natural logarithm of the elements of input
         *
         *  @param  input       The input Tensor
         *  @return             The result Tensor
         */
        fun log(input: TorchTensor): TorchTensor {
            // in-place function
            for (idx in 0 until input._nElement()) {
                val a = Math.log(input._get(idx).toDouble())
                input._set(idx, a.toFloat())
            }
            return input
        }

        /**
         *  Returns a new tensor with the sine of the elements of input
         *
         *  @param  input       The input Tensor
         *  @return             The result Tensor
         */
        fun sin(input: TorchTensor): TorchTensor {
            return input.sin()
        }

        /**
         *  Returns a new tensor with the cosine of the elements of input
         *
         *  @param  input       The input Tensor
         *  @return             The result Tensor
         */
        fun cos(input: TorchTensor): TorchTensor {
            return input.cos()
        }

        /**
         *  Returns a new tensor with the tangent of the elements of input
         *
         *  @param  input       The input Tensor
         *  @return             The result Tensor
         */
        fun tan(input: TorchTensor): TorchTensor {
            return input.tan()
        }

        /**
         *  Returns a new tensor with the exponential of the elements of the input tensor input
         *
         *  @param  input       The input Tensor
         *  @return             The result Tensor
         */
        fun exp(input: TorchTensor): TorchTensor {
            return input.exp()
        }

        /**
         *  Returns a new Tensor with the negative of the elements of input
         *
         *  @param  input       The input Tensor
         *  @return             The result Tensor
         */
        fun neg(input: TorchTensor): TorchTensor {
            // in-place function
            return input * -1f
        }

        /**
         *  Returns a new Tensor with the reciprocal of the elements of input
         *
         *  @param  input   The input tensor
         *  @param  epsilon The dummy term to avoid NaN
         *  @return         The computed result
         */
        fun reciprocal(input: TorchTensor, epsilon: Float = 1e-10f): TorchTensor {
            return input.reciprocal(epsilon)
        }

        /**
         *  Returns a new Tensor with each of the elements of input rounded to the closest integer.
         *
         *  @param  input       The input Tensor
         *  @return             The result Tensor
         */
        fun round(input: TorchTensor): TorchTensor {
            // in-place function
            for (idx in 0 until input._nElement()) {
                input._set(idx, kotlin.math.round(input._get(idx).toDouble()).toFloat())
            }
            return input
        }

        /**
         *  Computes the expit (also known as the logistic sigmoid function) of the elements of input
         *
         *  @param  input   The input tensor
         *  @return         The computed result
         */
        fun sigmoid(input: TorchTensor): TorchTensor {
            return input.sigmoid()
        }

        /**
         *  Takes the power of each element in input with exponent and returns a tensor with the result.
         *  Currently, we only support exponent as number (not support tensor)
         *
         *  @param  input       The input tensor
         *  @param  exponent    The exponent term.
         *                      This value will be cast to double directly to do further computation
         *  @return             The exponented tensor
         */
        fun pow(input: TorchTensor, exponent: Number): TorchTensor {
            // in-place function
            for (idx in 0 until input._nElement()) {
                val a = Math.pow(input._get(idx).toDouble(), exponent.toDouble())
                input._set(idx, a.toFloat())
            }
            return input
        }

        /**
         *  Returns the minimum value of all elements in the input Tensor
         *
         *  @param  input       The input tensor
         *  @return             The computed result
         */
        fun min(input: TorchTensor): TorchTensor {
            return input.min()
        }

        /**
         *  Returns the maximum value of all elements in the input Tensor
         *
         *  @param  input       The input tensor
         *  @return             The computed result
         */
        fun max(input: TorchTensor): TorchTensor {
            return input.max()
        }

        /**
         *  Returns the mean value of all elements in the input Tensor
         *
         *  @param  input       The input tensor
         *  @return             The computed result
         */
        fun mean(input: TorchTensor): TorchTensor {
            return input.mean()
        }

        /**
         *  Returns the median value of all elements in the input Tensor
         *
         *  @param  input       The input tensor
         *  @return             The computed result
         */
        fun median(input: TorchTensor): TorchTensor {
            return input.median()
        }

        /**
         *  Returns the product value of all elements in the input Tensor
         *
         *  @param  input       The input tensor
         *  @return             The computed result
         */
        fun prod(input: TorchTensor): TorchTensor {
            return input.prod()
        }

        /**
         *  Returns the standard-deviation of all elements in the input Tensor
         *
         *  @param  input       The input tensor
         *  @return             The computed result
         */
        fun std(input: TorchTensor): TorchTensor {
            return input.std()
        }

        /**
         *  Save the tensor into file
         *  Currently we only support save tensor into JSON format
         *  (Since Kotlin does not have pickle-related package)
         */
        fun save(input: TorchTensor, fileName: String) {
            var arrayStorage = JsonArray()
            for (v in input._array()) {
                arrayStorage.add(v)
            }
            var arraySize = JsonArray()
            for (v in input.shape) {
                arraySize.add(v)
            }
            var dtypeString: String
            dtypeString = when (input.dtype) {
                TensorType.DoubleTensor -> "torch.DoubleTensor"
                TensorType.FloatTensor -> "torch.FloatTensor"
                TensorType.LongTensor -> "torch.LongTensor"
            }

            val savedObject = JsonObject()
            savedObject.add("size", arraySize)
            savedObject.addProperty("dtype", dtypeString)
            savedObject.add("storage", arrayStorage)
            val objectString = savedObject.toString()

            val writer = PrintWriter(fileName, "UTF-8")
            writer.println(objectString)
            writer.close()
        }

        /**
         *  Load the JSON file and generate corresponding tensor
         *
         *  Note & TODO:
         *  (1) There is but while casting JsonPrimitive to Number
         *      To avoid the error, we simply cast as Double
         *      This way should be fix since it is illegal for other sub-type
         *      (e.g. complex number)
         *
         *  @param  fileName    The fileName of JSON file
         *  @return             Corresponding torch tensor
         */
        fun load(fileName: String): TorchTensor {
            // Read contain of JSON file
            val bufferedReader: BufferedReader = File(fileName).bufferedReader()
            var inputString = bufferedReader.use { it.readText() }

            // Parse the string into JSON object
            val savedObject = JsonParser.parseString(inputString).asJsonObject
            var arraySize = savedObject.getAsJsonArray("size")
            var arrayStorage = savedObject.getAsJsonArray("storage")
            var dtypeString = savedObject.get("dtype")

            // Transform JsonArray as mutable list
            var newStorage = mutableListOf<Number>()
            for (v in arrayStorage) {
                var v2 = v.asDouble as Number
                newStorage.add(v2)

            }
            var newSize = mutableListOf<Int>()
            for (v in arraySize) {
                newSize.add(v.asInt)
            }

            // Form output tensor
            var output = TorchTensor(size = newSize, storage = newStorage)
            when (dtypeString.asString) {
                "torch.FloatTensor" -> output.float()
                "torch.DoubleTensor" -> output.double()
                "torch.LongTensor" -> output.long()
            }

            return output
        }

        /**
         *  Applies the Softmax function to an n-dimensional input Tensor rescaling them
         *  so that the elements of the n-dimensional output Tensor lie in the range [0,1] and sum to 1.
         *
         *  @param  input   The input tensor
         *  @param  dim     A dimension along which Softmax will be computed
         *                  (so every slice along dim will sum to 1)
         *  @return         Output result
         */
        fun softmax(input: TorchTensor, dim: Int? = null): TorchTensor {
            return input.softmax(dim)
        }
    }
}