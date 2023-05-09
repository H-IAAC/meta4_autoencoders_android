/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.pytorch.digitautoencoder

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.TaskCompletionSource
import org.pytorch.*
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


class Decoder(private val context: Context) {
    // TODO: Add a TF Lite interpreter as a field.
    // private var interpreter: Interpreter? = null
    var isInitialized = false
        private set

    /** Executor to run inference task in the background. */
    private val executorService: ExecutorService = Executors.newCachedThreadPool()

    private var inputImageWidth: Int = 28 // will be inferred from TF Lite model.
    private var inputImageHeight: Int = 28 // will be inferred from TF Lite model.
    private var modelInputSize: Int = 28 * 28 // will be inferred from TF Lite model.

    fun initialize(laten_dim: Int): Task<Void?> {
        val task = TaskCompletionSource<Void?>()
        executorService.execute {
            try {
                initializeInterpreter(laten_dim)
                task.setResult(null)
            } catch (e: IOException) {
                task.setException(e)
            }
        }
        return task.task
    }

    var module: Module? = null

    @Throws(IOException::class)
    private fun initializeInterpreter(laten_dim: Int) {
        // loading serialized torchscript module from packaged into app android asset model.pt,
        // app/src/model/assets/model.pt

        try {
            // creating bitmap from packaged into app android asset 'image.jpg',
            // app/src/main/assets/image.jpg

            // loading serialized torchscript module from packaged into app android asset model.pt,
            // app/src/model/assets/model.pt
            //module 1= LiteModuleLoader.load(assetFilePath(context, "deeplabv3_scripted_optimized.ptl"))
            module = LiteModuleLoader.load(
                assetFilePath(
                    context,
                    "model_pytorch_decoder_" + laten_dim.toString() + ".ptl"
                )
            )
            isInitialized = true;
        } catch (e: IOException) {
            Log.e("PytorchHelloWorld", "Error reading assets", e)
            // finish()
        }
        ///https://medium.com/mlearning-ai/integrating-custom-pytorch-models-into-an-android-app-a2cdfce14fe8


    }


    fun decoder(vector: FloatArray): Tensor? {
        check(isInitialized) { "TF Lite Interpreter is not initialized yet." }
        val inputTensor: Tensor = generateTensor(vector)
// running the model

        // running the model
        val outputTensor = module!!.forward(IValue.from(inputTensor)).toTensor()





        return outputTensor
    }

    // Generate a tensor of random numbers given the size of that tensor.
    fun generateTensor(vector: FloatArray): Tensor {

        var size = vector.size
        Log.d("size dec", size.toString())
        //val rand = Random()
        val arr = DoubleArray(size)
        for (i in 0 until size) {
            arr[i] = vector[i].toDouble()
        }


        val s = longArrayOf(1, vector.size.toLong())
        // Create the tensor and return it
        return Tensor.fromBlob(vector, s)
    }

    fun close() {
        executorService.execute {
            /// interpreter?.close()
            Log.d(TAG, "Closed TFLite interpreter.")
        }
    }


    fun floatArrayToGrayscaleBitmap(
        outputTensor: Tensor,
        width: Int,
        height: Int,
        alpha: Byte = (255).toByte(),
        reverseScale: Boolean = false
    ): Bitmap {


        // getting tensor content as java array of floats
        val floatArray = outputTensor.dataAsFloatArray

        Log.d("result_dec", floatArray.size.toString());

        // Create empty bitmap in RGBA format (even though it says ARGB but channels are RGBA)
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val byteBuffer = ByteBuffer.allocate(width * height * 4)

        // mapping smallest value to 0 and largest value to 255
        val maxValue = floatArray.maxOrNull() ?: 1.0f
        val minValue = floatArray.minOrNull() ?: 0.0f
        val delta = maxValue - minValue
        var tempValue: Byte

        // Define if float min..max will be mapped to 0..255 or 255..0
        val conversion = when (reverseScale) {
            false -> { v: Float -> ((v - minValue) / delta * 255).toInt().toByte() }
            true -> { v: Float -> (255 - (v - minValue) / delta * 255).toInt().toByte() }
        }

        // copy each value from float array to RGB channels and set alpha channel
        floatArray.forEachIndexed { i, value ->
            tempValue = conversion(value)
            byteBuffer.put(4 * i, tempValue)
            byteBuffer.put(4 * i + 1, tempValue)
            byteBuffer.put(4 * i + 2, tempValue)
            byteBuffer.put(4 * i + 3, alpha)
        }

        bmp.copyPixelsFromBuffer(byteBuffer)

        return bmp
    }

    /* private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
       val byteBuffer = ByteBuffer.allocateDirect(modelInputSize)
       byteBuffer.order(ByteOrder.nativeOrder())

       val pixels = IntArray(inputImageWidth * inputImageHeight)
       bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

       for (pixelValue in pixels) {
         val r = (pixelValue shr 16 and 0xFF)
         val g = (pixelValue shr 8 and 0xFF)
         val b = (pixelValue and 0xFF)

         // Convert RGB to grayscale and normalize pixel value to [0..1].
         val normalizedPixelValue = (r + g + b) / 3.0f / 255.0f
         byteBuffer.putFloat(normalizedPixelValue)
       }

       return byteBuffer
     }
   */
    companion object {
        private const val TAG = "DigitClassifier"

        private const val FLOAT_TYPE_SIZE = 4
        private const val PIXEL_SIZE = 1

        private const val OUTPUT_CLASSES_COUNT = 32
    }


    @Throws(IOException::class)
    fun assetFilePath(context: Context, assetName: String?): String? {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        context.assets.open(assetName!!).use { `is` ->
            FileOutputStream(file).use { os ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (`is`.read(buffer).also { read = it } != -1) {
                    os.write(buffer, 0, read)
                }
                os.flush()
            }
            return file.absolutePath
        }
    }


}
