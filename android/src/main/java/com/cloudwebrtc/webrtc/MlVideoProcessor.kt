package com.cloudwebrtc.webrtc

/* CHANGES LOG:
 * 1. Added: MlVideoProcessor for background face detection on camera video frames.
 * 2. Added: Hooks for optional mask rendering using Canvas over detected faces.
 */

// NEW ADDITION START
// ADDED: Kotlin-based video frame processor that uses ML Kit Face Detection on a background thread.

import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.os.Handler
import android.os.HandlerThread
import android.util.Log

import com.cloudwebrtc.webrtc.video.LocalVideoTrack
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions

import org.webrtc.VideoFrame
import org.webrtc.YuvHelper

import java.nio.ByteBuffer
import java.util.Arrays

/**
 * NEW ADDITION:
 * Lightweight face-detection processor that plugs into the existing LocalVideoTrack
 * external processing pipeline. It runs ML Kit face detection on a separate thread so
 * it does not block the WebRTC capture thread.
 *
 * NOTE: This processor currently does NOT mutate the underlying video frame; it is
 * intended for tracking / callbacks and can be extended to apply visual effects.
 */
class MlVideoProcessor(
    context: Context,
    private val listener: FaceDetectionListener? = null
) : LocalVideoTrack.ExternalVideoFrameProcessing {

    interface FaceDetectionListener {
        fun onFacesDetected(faces: List<Face>, frame: VideoFrame)
        fun onNoFacesDetected(frame: VideoFrame)
    }

    private val applicationContext: Context = context.applicationContext

    // Background thread for ML Kit so we don't block WebRTC's capture thread.
    private val thread = HandlerThread("FlutterWebRTC-ML-FaceDetection")
    private val handler: Handler

    // ML Kit face detector instance tuned for real-time performance.
    private val faceDetector: FaceDetector
    
    // MODIFIED: Add counters for logging
    @Volatile
    private var frameCount: Long = 0
    @Volatile
    private var noFaceCount: Long = 0

    init {
        thread.start()
        handler = Handler(thread.looper)

        val realTimeOpts = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
            .setMinFaceSize(0.1f) // MODIFIED: Set minimum face size to 10% of image to detect smaller faces
            .build()

        faceDetector = FaceDetection.getClient(realTimeOpts)
        Log.i(TAG, "MlVideoProcessor initialized - Face detection will start automatically when video track is created")
    }

    /**
     * Called on the WebRTC capture thread for every frame.
     * We offload the heavy work to a background Handler thread and immediately
     * return the original frame so the media pipeline is not delayed.
     */
    override fun onFrame(frame: VideoFrame): VideoFrame {
        val buffer = frame.buffer
        val i420 = buffer.toI420()

        // MODIFIED: Added null-safety check for I420 buffer conversion.
        if (i420 == null) {
            Log.w(TAG, "Failed to convert video frame buffer to I420 format, skipping face detection")
            return frame
        }

        // Copy width/height before posting to another thread.
        val width = i420.width
        val height = i420.height
        val rotation = frame.rotation

        // MODIFIED: Add periodic logging for frame processing (every 30 frames to avoid spam)
        frameCount++
        if (frameCount % 30L == 0L) {
            Log.d(TAG, "Processing frame #$frameCount: ${width}x${height}, rotation=$rotation")
        }

        val inputImage = try {
            inputImageFromI420(i420, rotation)
        } catch (e: Throwable) {
            Log.e(TAG, "Failed to convert I420 frame to InputImage (${width}x${height}, rotation=$rotation)", e)
            i420.release()
            return frame
        }

        // We can now safely release the I420 buffer as ML Kit works on its own copy.
        i420.release()

        handler.post {
            processFrame(inputImage, frame, width, height)
        }

        // IMPORTANT: Return the original frame to keep the pipeline intact.
        return frame
    }

    private fun processFrame(
        inputImage: InputImage,
        frame: VideoFrame,
        width: Int,
        height: Int
    ) {
        faceDetector
            .process(inputImage)
            .addOnSuccessListener { faces ->
                if (faces.isNotEmpty()) {
                    // MODIFIED: Enhanced logging with detailed face information
                    Log.i(TAG, "✅ FACE DETECTED! Count: ${faces.size} face(s) in ${width}x${height} frame")
                    
                    faces.forEachIndexed { index, face ->
                        val bbox = face.boundingBox
                        val faceWidth = bbox.width()
                        val faceHeight = bbox.height()
                        val centerX = bbox.centerX()
                        val centerY = bbox.centerY()
                        
                        Log.i(TAG, "  Face #${index + 1}: " +
                                "BoundingBox(left=${bbox.left}, top=${bbox.top}, right=${bbox.right}, bottom=${bbox.bottom}), " +
                                "Size(${faceWidth}x${faceHeight}), " +
                                "Center(${centerX}, ${centerY}), " +
                                "Area=${faceWidth * faceHeight}px²")
                    }
                    
                    listener?.onFacesDetected(faces, frame)

                    // This is the hook where mask rendering can be implemented
                    // using OpenGL or Canvas with the detection results.
                    //
                    // For example (Canvas-based pseudocode):
                    //   1. Convert the YUV frame to a Bitmap.
                    //   2. Draw masks aligned to each Face.boundingBox on a Canvas.
                    //   3. Convert the Bitmap back to a VideoFrame (I420) if you want to
                    //      feed the modified frame into the WebRTC pipeline.
                    //
                    // The actual visual rendering strategy depends on your app's needs
                    // and can be implemented on top of this callback.
                    renderMasksIfNeeded(faces, width, height)
                } else {
                    // MODIFIED: Only log "no faces" periodically to reduce spam
                    noFaceCount++
                    if (noFaceCount % 30L == 0L) {
                        Log.d(TAG, "No faces detected (checked $noFaceCount frames, frame size: ${width}x${height})")
                    }
                    listener?.onNoFacesDetected(frame)
                }
            }
            .addOnFailureListener { e ->
                Log.e(TAG, "Face detection failed for ${width}x${height} frame", e)
            }
    }

    /**
     * Placeholder hook for Canvas / OpenGL mask rendering.
     * Implementers can override this class or wrap it to draw masks onto a
     * dedicated surface, texture, or overlay view.
     */
    protected open fun renderMasksIfNeeded(faces: List<Face>, width: Int, height: Int) {
        // NEW ADDITION:
        // This method is intentionally left as a no-op in the base implementation.
        // Apps can subclass MlVideoProcessor and override this to render masks
        // using Canvas or OpenGL with the provided face bounding boxes.
        if (faces.isNotEmpty()) {
            val boxes = faces.joinToString(prefix = "[", postfix = "]") { face: Face ->
                val b: Rect = face.boundingBox
                "(${b.left},${b.top})-(${b.right},${b.bottom})"
            }
            Log.d(TAG, "Mask rendering hook invoked for faces: $boxes on $width x $height")
        }
    }

    /**
     * Convert an I420 buffer into an InputImage in NV21 format for ML Kit.
     */
    private fun inputImageFromI420(i420Buffer: VideoFrame.I420Buffer, rotation: Int): InputImage {
        val y: ByteBuffer = i420Buffer.dataY
        val u: ByteBuffer = i420Buffer.dataU
        val v: ByteBuffer = i420Buffer.dataV
        val width = i420Buffer.width
        val height = i420Buffer.height

        val strideY = i420Buffer.strideY
        val strideU = i420Buffer.strideU
        val strideV = i420Buffer.strideV
        
        val strides = intArrayOf(strideY, strideU, strideV)

        val chromaWidth = (width + 1) / 2
        val chromaHeight = (height + 1) / 2
        val minSize = width * height + chromaWidth * chromaHeight * 2
        
        // MODIFIED: Log suspicious stride values (U/V should typically be half of Y for I420)
        if (frameCount % 120L == 0L) {
            val expectedUVStride = width / 2
            if (strideU == strideY || strideV == strideY) {
                Log.w(TAG, "⚠️ SUSPICIOUS STRIDES: Y=$strideY, U=$strideU, V=$strideV (expected UV stride ~$expectedUVStride)")
                Log.w(TAG, "   Frame size: ${width}x${height}, this may indicate incorrect plane data!")
            }
        }

        // MODIFIED: Use allocateDirect() like in the original isqad/flutter-webrtc commit
        // Direct buffers work better with native WebRTC YuvHelper
        val yuvBuffer = ByteBuffer.allocateDirect(minSize)

        // NOTE: NV21 is like NV12 but with U and V swapped, so we use I420ToNV12
        // helper and swap U/V buffers accordingly.
        YuvHelper.I420ToNV12(
            y, strides[0],
            v, strides[2],
            u, strides[1],
            yuvBuffer,
            width,
            height
        )

        // MODIFIED: Handle both direct and heap ByteBuffers properly
        // Original commit uses allocateDirect() which doesn't have backing array
        val cleanedArray = if (yuvBuffer.hasArray()) {
            // Heap buffer - has backing array
            Arrays.copyOfRange(
                yuvBuffer.array(), 
                yuvBuffer.arrayOffset(), 
                yuvBuffer.arrayOffset() + minSize
            )
        } else {
            // Direct buffer - must read bytes manually
            yuvBuffer.position(0)
            val array = ByteArray(minSize)
            yuvBuffer.get(array, 0, minSize)
            array
        }

        // MODIFIED: Normalize rotation to ML Kit's expected values (0, 90, 180, 270)
        // WebRTC uses 0, 90, 180, 270 degrees, which matches ML Kit's format
        val normalizedRotation = when {
            rotation == 0 || rotation == 90 || rotation == 180 || rotation == 270 -> rotation
            rotation < 0 -> ((rotation % 360) + 360) % 360
            else -> rotation % 360
        }
        
        // MODIFIED: Verify buffer size matches expected NV21 size
        val expectedSize = width * height * 3 / 2  // Y plane + UV interleaved
        if (cleanedArray.size != expectedSize) {
            Log.w(TAG, "⚠️ Buffer size mismatch! Expected: $expectedSize bytes, Got: ${cleanedArray.size} bytes for ${width}x${height}")
        }
        
        // MODIFIED: Sample the YUV data to verify it's not all zeros or corrupted
        var yNonZero = 0
        var uvNonZero = 0
        val ySampleEnd = minOf(100, width * height)
        val uvStart = width * height
        val uvSampleEnd = minOf(uvStart + 100, cleanedArray.size)
        
        for (i in 0 until ySampleEnd) {
            if (cleanedArray[i].toInt() and 0xFF != 0) yNonZero++
        }
        for (i in uvStart until uvSampleEnd) {
            if (cleanedArray[i].toInt() and 0xFF != 0) uvNonZero++
        }
        
        // Log rotation info periodically for debugging
        if (frameCount % 60L == 0L) {
            Log.d(TAG, "Image conversion: I420 ${width}x${height} -> NV21 ${width}x${height}, rotation=$rotation° (normalized=$normalizedRotation°)")
            Log.d(TAG, "  Buffer size: ${cleanedArray.size} bytes, expected: $expectedSize bytes")
            Log.d(TAG, "  Strides: Y=${strides[0]}, U=${strides[1]}, V=${strides[2]}")
            Log.d(TAG, "  YUV data validation: Y plane ${yNonZero}/${ySampleEnd} non-zero, UV plane ${uvNonZero}/${uvSampleEnd - uvStart} non-zero")
            
            if (yNonZero < ySampleEnd / 10) {
                Log.e(TAG, "❌ CRITICAL: Y plane is mostly zeros! Image data is corrupted or not being copied correctly")
            }
            if (uvNonZero < (uvSampleEnd - uvStart) / 10) {
                Log.w(TAG, "⚠️ WARNING: UV plane is mostly zeros! Color data may be missing")
            }
        }

        // MODIFIED: Try with rotation=0 first to test if rotation is the issue
        // ML Kit should handle rotation, but let's test with 0 to isolate the problem
        val testRotation = if (frameCount < 10L) {
            Log.d(TAG, "🧪 TEST MODE: Using rotation=0 for first 10 frames to test detection")
            0
        } else {
            normalizedRotation
        }

        return InputImage.fromByteArray(
            cleanedArray,
            width,
            height,
            testRotation,
            ImageFormat.NV21
        )
    }

    /**
     * Should be called when camera capture is stopped to cleanly shut down
     * the background thread and ML Kit detector.
     */
    fun dispose() {
        try {
            faceDetector.close()
        } catch (e: Throwable) {
            Log.e(TAG, "Error while closing face detector", e)
        }

        try {
            thread.quitSafely()
        } catch (e: Throwable) {
            Log.e(TAG, "Error while stopping ML thread", e)
        }
    }

    companion object {
        private const val TAG = "MlVideoProcessor"
    }
}
// NEW ADDITION END


