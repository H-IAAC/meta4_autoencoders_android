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

import android.annotation.SuppressLint
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
import org.tensorflow.lite.codelabs.digitclassifier.R



class MainActivity : AppCompatActivity() {

    private var drawView: DrawView? = null
    private var clearButton: Button? = null
    private var predictedTextView: TextView? = null
    private var imageView: ImageView? = null
    private var encoder = Encoder(this)
    private var decoder = Decoder(this)
    private var latentDimstextview: TextView? = null
    private val laten_dim = 32

    @SuppressLint("ClickableViewAccessibility", "SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Setup view instances.
        drawView = findViewById(R.id.draw_view)
        drawView?.setStrokeWidth(70.0f)
        drawView?.setColor(Color.WHITE)
        drawView?.setBackgroundColor(Color.BLACK)
        clearButton = findViewById(R.id.clear_button)
        predictedTextView = findViewById(R.id.predicted_text)
        latentDimstextview = findViewById(R.id.latent_dims)
        latentDimstextview?.text = "PYTORCH latent dim $laten_dim"

        imageView = findViewById(R.id.imageView2)


        // Setup clear drawing button.
        clearButton?.setOnClickListener {
            drawView?.clearCanvas()
            predictedTextView?.text = getString(R.string.prediction_text_placeholder)
            imageView?.setImageResource(0)
        }

        // Setup classification trigger so that it classify after every stroke drew.
        drawView?.setOnTouchListener { _, event ->
            // As we have interrupted DrawView's touch event,
            // we first need to pass touch events through to the instance for the drawing to show up.
            drawView?.onTouchEvent(event)

            // Then if user finished a touch event, run classification
            if (event.action == MotionEvent.ACTION_UP) {
                classifyDrawing()
            }

            true
        }

        // Setup digit classifier.
        encoder
            .initialize(laten_dim)
            .addOnFailureListener { e -> Log.e(TAG, "Error to setting up digit classifier.", e) }
        decoder
            .initialize(laten_dim)
            .addOnFailureListener { e -> Log.e(TAG, "Error to setting up digit classifier.", e) }
    }


    override fun onDestroy() {
        // Sync DigitClassifier instance lifecycle with MainActivity lifecycle,
        // and free up resources (e.g. TF Lite instance) once the activity is destroyed.
        encoder.close()
        decoder.close()
        super.onDestroy()
    }

    private fun classifyDrawing() {
        val bitmap = drawView?.getBitmap()
        if ((bitmap != null) && (encoder.isInitialized)) {
            val inputTensor = encoder.generateTensor(bitmap)

            val memoryBefore = Debug.getNativeHeapAllocatedSize()
            val startTimeCpu: Long = Debug.threadCpuTimeNanos() // Captura o tempo inicial
            val startTime = System.currentTimeMillis() //  tempo atual antes da inferência


            val spaceLatent = encoder.encoder(inputTensor)
            val dec = spaceLatent?.let { decoder.decoder(it) }

            val elapsedTime = (System.currentTimeMillis()  - startTime) // tempo decorrido em milissegundos
            val cpuTime = Debug.threadCpuTimeNanos() - startTimeCpu // Tempo total de CPU utilizado
            val memoryConsumed = (Debug.getNativeHeapAllocatedSize()  - memoryBefore) //   consumo de memória durante a inferência

            // Exibição dos resultados
            //Log.d("Medição", "CPU:$cpuTime nanossegundos / memória:$memoryConsumed KB /Tempo: $elapsedTime ms")
            Log.d("Medição", "//$cpuTime //$memoryConsumed // $elapsedTime")



            Log.d("Encoder",spaceLatent.contentToString())

            val output_image = dec?.let { decoder.floatArrayToGrayscaleBitmap(it, 28, 28) }

            imageView?.setImageBitmap(output_image)

        }
    }


    companion object {
        private const val TAG = "MainActivity"
    }
}
