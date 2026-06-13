package com.example.signlingo

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.signlingo.databinding.ActivityMainBinding
import com.unity3d.player.UnityPlayerActivity

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding




    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnStartCamera.setOnClickListener {
            // 1. Cek apakah izin kamera sudah diberikan
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
                // Jika belum, munculkan pop-up permintaan izin
                ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_CODE)
            } else {
                // Jika sudah, langsung buka Unity
                startUnityCamera()
            }
        }
    }

    // 2. Menangkap hasil dari pop-up izin (User klik "Allow" atau "Deny")
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Log.d("SignLingo", "Izin kamera diberikan!")
                startUnityCamera()
            } else {
                Toast.makeText(this, "Aplikasi membutuhkan izin kamera untuk mendeteksi bahasa isyarat", Toast.LENGTH_LONG).show()
            }
        }
    }

    // 3. Fungsi untuk mengeksekusi Unity
    private fun startUnityCamera() {
        try {
            val intent = Intent(this, UnityPlayerActivity::class.java)
            intent.flags = Intent.FLAG_ACTIVITY_SINGLE_TOP
            startActivity(intent)
        } catch (e: Exception) {
            Log.e("SignLingo", "Gagal membuka Unity: ${e.message}")
        }
    }

    companion object {
        const val CAMERA_PERMISSION_CODE = 100
        init {
            // Cukup muat dua ini, sisanya biar diurus oleh sistem internal Unity
            System.loadLibrary("opencv_java4")
            System.loadLibrary("mediapipe_jni")
        }
    }

}
