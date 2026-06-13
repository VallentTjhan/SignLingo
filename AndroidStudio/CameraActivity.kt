package com.example.signlingo

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.signlingo.databinding.ActivityCameraBinding // AWAS: Sesuaikan dengan nama package anda jika berbeza!
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraActivity : AppCompatActivity() {

    private lateinit var binding: ActivityCameraBinding
    private lateinit var cameraExecutor: ExecutorService
    private var cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

    // AI & MediaPipe Engines
    private var tfliteInterpreter: Interpreter? = null
    private var handLandmarker: HandLandmarker? = null
    private var poseLandmarker: PoseLandmarker? = null

    // Struktur Ingatan LSTM (Mengingat 60 frame, setiap frame berisi 258 titik koordinat)
    private val maxFrames = 60
    private val numFeatures = 258
    private val frameBuffer = Array(maxFrames) { FloatArray(numFeatures) }
    private var frameCount = 0
    private var headX = 0.5f
    private var headY = 0.5f
    private var isPersonDetected = false

    // DAFTAR KATA (Sesuaikan urutan Index ini dengan label/classes semasa anda melatih AI di Python!)
    private val labels = listOf("Halo", "Terima Kasih", "Sama-Sama", "Apa Kabar?", "Senang Bertemu Denganmu", "Sampai Jumpa", "Maaf")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupAIAndMediaPipe()

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        binding.btnSwitchCamera.setOnClickListener {
            cameraSelector = if (cameraSelector == CameraSelector.DEFAULT_BACK_CAMERA) {
                CameraSelector.DEFAULT_FRONT_CAMERA
            } else {
                CameraSelector.DEFAULT_BACK_CAMERA
            }
            startCamera()
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun setupAIAndMediaPipe() {
        try {
            // 1. Muat Model Jalur TFLite
            tfliteInterpreter = Interpreter(loadModelFile("model_bisindo.tflite"))

            // 2. Konfigurasi Detektor Tangan MediaPipe
            val handOptions = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build())
                .setMinHandDetectionConfidence(0.5f)
                .setRunningMode(com.google.mediapipe.tasks.vision.core.RunningMode.LIVE_STREAM)
                .setResultListener { result, _ -> processHandResult(result) }
                .build()
            handLandmarker = HandLandmarker.createFromOptions(this, handOptions)

            // 3. Konfigurasi Detektor Badan (Pose) MediaPipe
            val poseOptions = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath("pose_landmarker_lite.task").build())
                .setMinPoseDetectionConfidence(0.5f)
                .setRunningMode(com.google.mediapipe.tasks.vision.core.RunningMode.LIVE_STREAM)
                .setResultListener { result, _ -> processPoseResult(result) }
                .build()
            poseLandmarker = PoseLandmarker.createFromOptions(this, poseOptions)

            Log.d(TAG, "Semua enjin AI berjaya dihidupkan!")
        } catch (e: Exception) {
            Log.e(TAG, "Gagal mengaktifkan AI/MediaPipe: ", e)
        }
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        analyzeImage(imageProxy)
                    }
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
            } catch (exc: Exception) {
                Log.e(TAG, "Gagal menghidupkan kamera fizikal", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private var currentFrameFeatures = FloatArray(numFeatures)

    private fun analyzeImage(imageProxy: ImageProxy) {
        try {
            // 1. Ambil gambar asli dari sensor kamera
            val originalBitmap = imageProxy.toBitmap()

            // 2. Putar gambar agar posisinya 'berdiri' (portrait) sesuai dengan layar HP Anda
            val matrix = android.graphics.Matrix()
            matrix.postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
            val rotatedBitmap = android.graphics.Bitmap.createBitmap(
                originalBitmap, 0, 0, originalBitmap.width, originalBitmap.height, matrix, true
            )

            // 3. Masukkan gambar yang sudah berdiri tegak ke MediaPipe
            val mpImage = com.google.mediapipe.framework.image.BitmapImageBuilder(rotatedBitmap).build()
            val timestamp = System.currentTimeMillis()

            handLandmarker?.detectAsync(mpImage, timestamp)
            poseLandmarker?.detectAsync(mpImage, timestamp)

            pushFrameToBuffer(currentFrameFeatures.clone())

            if (frameCount >= maxFrames) {
                runTFLiteInference()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Ralat semasa menganalisis imej", e)
        } finally {
            imageProxy.close() // Wajib ditutup agar kamera tidak lag
        }
    }

    private fun pushFrameToBuffer(frame: FloatArray) {
        for (i in 0 until maxFrames - 1) {
            frameBuffer[i] = frameBuffer[i + 1]
        }
        frameBuffer[maxFrames - 1] = frame
        if (frameCount < maxFrames) frameCount++
    }

    private fun processHandResult(result: HandLandmarkerResult) {
        var index = 132
        val landmarks = result.landmarks()

        for (i in index until index + 126) currentFrameFeatures[i] = 0f

        // PEMBETULAN PERKATAAN 'until' DI SINI
        for (handIdx in 0 until minOf(landmarks.size, 2)) {
            val handLandmarks = landmarks[handIdx]
            for (lm in handLandmarks) {
                if (index < numFeatures) currentFrameFeatures[index++] = lm.x()
                if (index < numFeatures) currentFrameFeatures[index++] = lm.y()
                if (index < numFeatures) currentFrameFeatures[index++] = lm.z()
            }
        }
    }

    private fun processPoseResult(result: PoseLandmarkerResult) {
        var index = 0
        val landmarks = result.landmarks()

        // --- TAMBAHAN LOGIKA AR TRACKING ---
        // Mengecek apakah ada badan/kepala yang terdeteksi di kamera
        if (landmarks.isNotEmpty() && landmarks[0].isNotEmpty()) {
            isPersonDetected = true
            // Ambil koordinat hidung (Landmark index 0)
            val nose = landmarks[0][0]
            headX = nose.x()
            headY = nose.y()
        } else {
            isPersonDetected = false
        }
        // -----------------------------------

        // --- KODE EKSTRAKSI LAMA ANDA YANG AKURAT ---
        for (poseIdx in 0 until minOf(landmarks.size, 1)) {
            val poseLandmarks = landmarks[poseIdx]
            for (lm in poseLandmarks) {
                if (index < 132) {
                    currentFrameFeatures[index++] = lm.x()
                    currentFrameFeatures[index++] = lm.y()
                    currentFrameFeatures[index++] = lm.z()
                }
            }
        }
    }

    private fun runTFLiteInference() {
        val inputBuffer = ByteBuffer.allocateDirect(1 * maxFrames * numFeatures * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        for (f in 0 until maxFrames) {
            for (feat in 0 until numFeatures) {
                inputBuffer.putFloat(frameBuffer[f][feat])
            }
        }
        inputBuffer.rewind()

        val outputArray = Array(1) { FloatArray(labels.size) }

        tfliteInterpreter?.run(inputBuffer, outputArray)

        val probabilities = outputArray[0]
        var maxIdx = 0
        var maxVal = 0f
        for (i in probabilities.indices) {
            if (probabilities[i] > maxVal) {
                maxVal = probabilities[i]
                maxIdx = i
            }
        }

        runOnUiThread {
            if (maxVal > 0.70f) {
                val detectedText = labels[maxIdx]
                // Teks di bawah (Opsional, bisa Anda hapus kalau tidak mau)
                binding.tvTranslationResult.text = detectedText

                // LOGIKA AR: Pindahkan teks ke atas kepala
                if (isPersonDetected) {
                    binding.tvArOverlay.text = detectedText
                    binding.tvArOverlay.visibility = android.view.View.VISIBLE

                    val screenWidth = binding.viewFinder.width.toFloat()
                    val screenHeight = binding.viewFinder.height.toFloat()

                    // Balikkan sumbu X jika pakai kamera depan (Mirror effect)
                    val mappedX = if (cameraSelector == CameraSelector.DEFAULT_FRONT_CAMERA) {
                        (1f - headX) * screenWidth
                    } else {
                        headX * screenWidth
                    }

                    val mappedY = headY * screenHeight
                    // Naikkan posisi teks agar berada di ATAS kepala, bukan menutupi muka
                    val offsetAtasKepala = screenHeight * 0.25f

                    // Set posisi teks agar tepat berada di tengah kepala
                    binding.tvArOverlay.x = mappedX - (binding.tvArOverlay.width / 2)
                    binding.tvArOverlay.y = mappedY - offsetAtasKepala
                } else {
                    binding.tvArOverlay.visibility = android.view.View.GONE
                }

            } else {
                binding.tvTranslationResult.text = "Mendeteksi Isyarat..."
                binding.tvArOverlay.visibility = android.view.View.GONE
            }
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        tfliteInterpreter?.close()
        handLandmarker?.close()
        poseLandmarker?.close()
        cameraExecutor.shutdown()
    }

    companion object {

        init {
            // Memuat OpenCV dan MediaPipe secara paksa ke memori Android
            System.loadLibrary("opencv_java4")
            System.loadLibrary("mediapipe_jni")
        }

        private const val TAG = "SignLingoCamera"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}

