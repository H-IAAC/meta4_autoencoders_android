package org.tensorflow.lite.codelabs.digitreconstruct

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.os.Debug
import android.util.Log
import android.view.MotionEvent
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.divyanshu.draw.widget.DrawView
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {

    private var drawView: DrawView? = null
    private var clearButton: Button? = null
    private var infoTextView: TextView? = null


    private var encoder = Encoder(this)
    private var decoder = Decoder(this)
    private var imageView: ImageView? = null
    private var latent_dimsTextView: TextView? = null
    val laten_dim = 16

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Setup view instances.
        drawView = findViewById(R.id.draw_view)
        drawView?.setStrokeWidth(70.0f)
        drawView?.setColor(Color.WHITE)
        drawView?.setBackgroundColor(Color.BLACK)
        clearButton = findViewById(R.id.clear_button)
        infoTextView = findViewById(R.id.predicted_text)
        latent_dimsTextView = findViewById(R.id.latent_dims)
        latent_dimsTextView?.text = "TENSORFLOW latent dim $laten_dim"
        imageView = findViewById(R.id.imageView2)

        // Setup clear drawing button.
        clearButton?.setOnClickListener {
            drawView?.clearCanvas()
            imageView?.setImageResource(0)
            infoTextView?.text = getString(R.string.prediction_text_placeholder)

        }

        drawView?.setOnTouchListener { _, event ->
            drawView?.onTouchEvent(event)

            // Then if user finished a touch event, run models
            if (event.action == MotionEvent.ACTION_UP) {
                reconstructDrawing()
            }
            true
        }

        // Setup digit reconstruct.
        encoder
            .initialize(laten_dim)
            .addOnFailureListener { e -> Log.e(TAG, "Error to setting up digit encoder.", e) }
        decoder
            .initialize(laten_dim)
            .addOnFailureListener { e -> Log.e(TAG, "Error to setting up digit decoder.", e) }
    }

    override fun onDestroy() {
        encoder.close()
        decoder.close()
        super.onDestroy()
    }

    private fun createImage(outputd: ByteBuffer): Bitmap? {
        val outputImageWidth = 28
        val outputImageHeight = 28
        val rawData = IntArray(outputImageWidth * outputImageHeight)
        outputd.rewind()
        for (i in 0 until outputImageWidth * outputImageHeight step 1) {
            val hue = (outputd.float) * 255

            val red = Color.red(hue.toInt())
            val green = Color.green(hue.toInt())
            val blue = Color.blue(hue.toInt())
            val grayscalePixel = (red + green + blue)
            val grayscaleColor = Color.rgb(grayscalePixel, grayscalePixel, grayscalePixel)
            rawData[i] = grayscaleColor
        }
        val bitmap = Bitmap.createBitmap(outputImageWidth, 28, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(rawData, 0, 28, 0, 0, outputImageWidth, outputImageHeight)
        return bitmap
        
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {

        var inputImageWidth = 28
        var inputImageHeight = 28

        val resizedImage = Bitmap.createScaledBitmap(
            bitmap,
            inputImageWidth,
            inputImageHeight,
            true
        )

        var modelInputSize = Encoder.FLOAT_TYPE_SIZE * inputImageWidth *
                inputImageHeight * Encoder.PIXEL_SIZE
        val byteBuffer = ByteBuffer.allocateDirect(modelInputSize)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputImageWidth * inputImageHeight)
        resizedImage.getPixels(pixels, 0, resizedImage.width, 0, 0, resizedImage.width, resizedImage.height)

        for (pixelValue in pixels) {
            val r = (pixelValue shr 16 and 0xFF)
            val g = (pixelValue shr 8 and 0xFF)
            val b = (pixelValue and 0xFF)

            // Convert RGB to grayscale and normalize pixel value to [0..1].
            val normalizedPixelValue = (r + g + b) / 3.0f / 255.0f
            byteBuffer.putFloat(normalizedPixelValue)
        }


       // val byteBuffer = convertBitmapToByteBuffer(resizedImage)

        return byteBuffer
    }


    private fun reconstructDrawing() {
        val bitmap = drawView?.getBitmap()
        if ((bitmap != null) && (encoder.isInitialized)) {
            val bitmapbuffer=convertBitmapToByteBuffer(bitmap)

            val startTimeCpu: Long = Debug.threadCpuTimeNanos() // Captura o tempo inicial
            val startTime = System.currentTimeMillis() //  tempo atual antes da inferência

            val memoryBefore = Debug.getNativeHeapAllocatedSize()

            val enc = encoder
                .encoder(bitmapbuffer,laten_dim)
            val outputdecoder = decoder.decoder(enc,laten_dim)

            val memoryAfter = Debug.getNativeHeapAllocatedSize()

            val memoryConsumed = memoryAfter - memoryBefore //   consumo de memória durante a inferência
            val elapsedTime = System.currentTimeMillis()  - startTime // tempo decorrido em milissegundos
            val cpuTime = Debug.threadCpuTimeNanos() - startTimeCpu // Tempo total de CPU utilizado

            // Exibição dos resultados
            //Log.d("Medição", "CPU:$cpuTime nanossegundos / memória:$memoryConsumed KB /Tempo: $elapsedTime ms")
            Log.d("Medição", "//$cpuTime //$memoryConsumed // $elapsedTime")


            imageView?.setImageBitmap(createImage(outputdecoder))

        }
    }

    companion object {
        private const val TAG = "MainActivity"
    }
    
    
    
    
    
    
    
    
}
