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
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


class Encoder(private val context: Context) {
    // TODO: Add a TF Lite interpreter as a field.
    // private var interpreter: Interpreter? = null
    var isInitialized = false
        private set

    /** Executor to run inference task in the background. */
    private val executorService: ExecutorService = Executors.newCachedThreadPool()

    private var inputImageWidth: Int = 28 // will be inferred from TF Lite model.
    private var inputImageHeight: Int = 28 // will be inferred from TF Lite model.


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

    private var module: Module? = null

    @Throws(IOException::class)
    private fun initializeInterpreter(laten_dim: Int) {
        try {
            module = LiteModuleLoader.load(
                assetFilePath(
                    context,
                  "model_pytorch_encoder_$laten_dim.ptl"
                )
            )
            isInitialized = true
        } catch (e: IOException) {
            Log.e("PytorchHelloWorld", "Error reading assets", e)
            // finish()
        }
    }


    fun encoder(inputTensor: Tensor): FloatArray? {
        check(isInitialized) { "TF Lite Interpreter is not initialized yet." }
        Log.d("encoder", "hola encoder")
        // TODO: Add code to run inference with TF Lite.
        // Pre-processing: resize the input image to match the model input shape.

        // running the model
        val outputTensor = module!!.forward(IValue.from(inputTensor)).toTensor()
        return outputTensor.dataAsFloatArray
    }


    fun generateTensor(bitmap: Bitmap): Tensor {

        val resizedImage = Bitmap.createScaledBitmap(
            bitmap,
            28,
            28,
            true
        )

        val pixels = IntArray(inputImageWidth * inputImageHeight)
        resizedImage.getPixels(
            pixels,
            0,
            resizedImage.width,
            0,
            0,
            resizedImage.width,
            resizedImage.height
        )

        val size = pixels.size
        Log.d("size dec", size.toString())
        val arr = FloatArray(size)
        for (i in 0 until size) {
            val r = (pixels[i] shr 16 and 0xFF)
            val g = (pixels[i] shr 8 and 0xFF)
            val b = (pixels[i] and 0xFF)

            // Convert RGB to grayscale and normalize pixel value to [0..1].
            val normalizedPixelValue = (r + g + b) / 3.0f / 255.0f
            arr[i] = normalizedPixelValue


        }


        val s = longArrayOf(1, pixels.size.toLong())
        // Create the tensor and return it
        return Tensor.fromBlob(arr, s)
    }


    fun close() {
        executorService.execute {
            /// interpreter?.close()
            Log.d(TAG, "Closed TFLite interpreter.")
        }
    }


    companion object {
        private const val TAG = "DigitReconstruct"
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
