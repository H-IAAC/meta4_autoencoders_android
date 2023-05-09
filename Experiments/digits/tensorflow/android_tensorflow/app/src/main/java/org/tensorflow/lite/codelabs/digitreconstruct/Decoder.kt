package org.tensorflow.lite.codelabs.digitreconstruct

import android.content.Context
import android.content.res.AssetManager
import android.graphics.*
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


class Decoder(private val context: Context) {
    // TODO: Add a TF Lite interpreter as a field.
    private var interpreter: Interpreter? = null
    private var isInitialized = false
    /** Executor to run inference task in the background. */
    private val executorService: ExecutorService = Executors.newCachedThreadPool()

    private var outputImageWidth: Int = 0 // will be inferred from TF Lite model.
    private var outputImageHeight: Int = 0 // will be inferred from TF Lite model.

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

        // Load the TF Lite model from asset folder and initialize TF Lite Interpreter with NNAPI enabled.
        val assetManager = context.assets
        val model = loadModelFile(assetManager, "model_tensorflow_decoder_$laten_dim.tflite")
        val interpreter = Interpreter(model)

        // TODO: Read the model input shape from model file.

        // Read input shape from model file.
        outputImageWidth = 28
        outputImageHeight = 28
        this.interpreter = interpreter

        isInitialized = true
        Log.d(TAG, "Initialized TFLite interpreter.")
    }

    @Throws(IOException::class)
    private fun loadModelFile(assetManager: AssetManager, filename_decoder: String): ByteBuffer {
        val fileDescriptor = assetManager.openFd(filename_decoder)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun decoder(byteBufferlatent: FloatArray, laten_dim: Int): ByteBuffer {
        check(isInitialized) { "TF Lite Interpreter is not initialized yet." }
        val outputd = ByteBuffer.allocateDirect(FLOAT_TYPE_SIZE * outputImageWidth * 28)

        // Run inference with the input data.
        interpreter?.run(byteBufferlatent, outputd.asFloatBuffer())

     return outputd
    }

    fun close() {
        executorService.execute {
            interpreter?.close()
            Log.d(TAG, "Closed TFLite interpreter.")
        }
    }

    companion object {
        private const val TAG = "Decoder"
        private const val FLOAT_TYPE_SIZE = 4
        //private const val LATENT_DIM = 32
    }
}
