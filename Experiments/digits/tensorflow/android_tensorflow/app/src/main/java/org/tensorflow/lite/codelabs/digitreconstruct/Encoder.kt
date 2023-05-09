package org.tensorflow.lite.codelabs.digitreconstruct

import android.annotation.SuppressLint
import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.TaskCompletionSource
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class Encoder(private val context: Context) {
    // TODO: Add a TF Lite interpreter as a field.
    private var interpreter: Interpreter? = null
    var isInitialized = false
        private set

    /** Executor to run inference task in the background. */
    private val executorService: ExecutorService = Executors.newCachedThreadPool()

    private var inputImageWidth: Int = 0 // will be inferred from TF Lite model.
    private var inputImageHeight: Int = 0 // will be inferred from TF Lite model.
    private var modelInputSize: Int = 0 // will be inferred from TF Lite model.

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

    @Throws(IOException::class)
    private fun initializeInterpreter(laten_dim: Int) {
        // TODO: Load the TF Lite model from file and initialize an interpreter.

        val assetManager = context.assets
        val model =
            loadModelFile(assetManager, "model_tensorflow_encoder_$laten_dim.tflite")
        val interpreter = Interpreter(model)

        // TODO: Read the model input shape from model file.

        // Read input shape from model file.
        val inputShape = interpreter.getInputTensor(0).shape()
        inputImageWidth = inputShape[1]
        inputImageHeight = inputShape[2]
        modelInputSize = FLOAT_TYPE_SIZE * inputImageWidth *
                inputImageHeight * PIXEL_SIZE

        // Finish interpreter initialization.
        this.interpreter = interpreter

        isInitialized = true
        Log.d(TAG, "Initialized TFLite interpreter.")
    }

    @Throws(IOException::class)
    private fun loadModelFile(assetManager: AssetManager, filename: String): ByteBuffer {
        val fileDescriptor = assetManager.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    @SuppressLint("SuspiciousIndentation")
    fun encoder(byteBuffer: ByteBuffer, laten_dim: Int): FloatArray {
        check(isInitialized) { "TF Lite Interpreter is not initialized yet." }
        // TODO: Add code to run inference with TF Lite.
        // Pre-processing: resize the input image to match the model input shape.

        // Define an array to store the model output.
        val output = Array(1) { FloatArray(laten_dim) }
        // Run inference with the input data.
        interpreter?.run(byteBuffer, output)



        val result = output[0]
          Log.d("result_enc", result.joinToString(","))
        return result
    }

    fun close() {
        executorService.execute {
            interpreter?.close()
            Log.d(TAG, "Closed TFLite interpreter.")
        }
    }



    companion object {
        private const val TAG = "Encoder"
        const val FLOAT_TYPE_SIZE = 4
        const val PIXEL_SIZE = 1

    }
}
