package com.cloudwebrtc.webrtc

/* CHANGES LOG:
 * 1. Added: MlVideoProcessor for background face detection on camera video frames.
 * 2. Added: Hooks for optional mask rendering using Canvas over detected faces.
 * 3. Added: Real-time overlay rendering on video frames sent through WebRTC.
 * 4. Added: Method to enable/disable overlay rendering (setOverlayEnabled).
 */

// NEW ADDITION START
// ADDED: Kotlin-based video frame processor that uses ML Kit Face Detection on a background thread.

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.Rect
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.util.Log

import com.cloudwebrtc.webrtc.video.LocalVideoTrack
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions

import org.webrtc.JavaI420Buffer
import org.webrtc.VideoFrame
import org.webrtc.YuvHelper

import java.nio.ByteBuffer
import java.util.Arrays
import java.util.concurrent.atomic.AtomicReference

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
    
    // NEW ADDITION START - Store latest detected faces for overlay rendering
    private val latestFaces = AtomicReference<List<Face>>(emptyList())
    private var enableOverlay = true // Toggle to enable/disable overlay rendering
    // NEW ADDITION END

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
     * return the modified frame (with overlay) to the media pipeline.
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

        handler.post {
            processFrame(inputImage, frame, width, height)
        }

        // NEW ADDITION START - Draw overlay on frame if enabled and faces detected
        val processedFrame = if (enableOverlay) {
            val faces = latestFaces.get()
            if (faces.isNotEmpty()) {
                try {
                    drawOverlayOnFrame(i420, faces, rotation, frame.timestampNs)
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to draw overlay on frame", e)
                    i420.release()
                    frame // Return original frame on error
                }
            } else {
                i420.release()
                frame // No faces, return original
            }
        } else {
            i420.release()
            frame // Overlay disabled, return original
        }
        // NEW ADDITION END

        return processedFrame
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
                    // MODIFIED: Store detected faces for overlay rendering
                    latestFaces.set(faces)
                    
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

                    // MODIFIED: Send face data to Flutter
                    sendFaceDataToFlutter(faces, width, height)

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
                    // MODIFIED: Clear faces when none detected
                    latestFaces.set(emptyList())
                    
                    // MODIFIED: Only log "no faces" periodically to reduce spam
                    noFaceCount++
                    if (noFaceCount % 30L == 0L) {
                        Log.d(TAG, "No faces detected (checked $noFaceCount frames, frame size: ${width}x${height})")
                    }
                    listener?.onNoFacesDetected(frame)
                    
                    // MODIFIED: Send "no faces" event to Flutter
                    sendNoFacesToFlutter()
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
     * NEW ADDITION: Draw overlay (bounding boxes) on the video frame
     * This converts I420 -> ARGB Bitmap, draws overlay, then converts back to I420
     */
    private fun drawOverlayOnFrame(
        i420Buffer: VideoFrame.I420Buffer,
        faces: List<Face>,
        rotation: Int,
        timestampNs: Long
    ): VideoFrame {
        val width = i420Buffer.width
        val height = i420Buffer.height

        try {
            // Step 1: Convert I420 to ARGB bitmap
            val argbBuffer = ByteBuffer.allocateDirect(width * height * 4)
            YuvHelper.I420ToABGR(
                i420Buffer.dataY, i420Buffer.strideY,
                i420Buffer.dataU, i420Buffer.strideU,
                i420Buffer.dataV, i420Buffer.strideV,
                argbBuffer,
                width,
                height
            )

            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            bitmap.copyPixelsFromBuffer(argbBuffer)

            // Step 2: Draw overlay on bitmap
            val canvas = Canvas(bitmap)
            
            val facePaint = Paint().apply {
                color = Color.GREEN
                style = Paint.Style.STROKE
                strokeWidth = 8f
                isAntiAlias = true
            }

            val cornerPaint = Paint().apply {
                color = Color.YELLOW
                style = Paint.Style.STROKE
                strokeWidth = 8f
                isAntiAlias = true
            }

            val centerPaint = Paint().apply {
                color = Color.RED
                style = Paint.Style.FILL
                isAntiAlias = true
            }

            // Draw bounding boxes for each face
            for (face in faces) {
                val bbox = face.boundingBox
                
                // Draw rectangle
                canvas.drawRect(bbox, facePaint)

                // Draw corner markers
                val cornerSize = 20f
                val left = bbox.left.toFloat()
                val top = bbox.top.toFloat()
                val right = bbox.right.toFloat()
                val bottom = bbox.bottom.toFloat()

                // Top-left corner
                canvas.drawLine(left, top, left + cornerSize, top, cornerPaint)
                canvas.drawLine(left, top, left, top + cornerSize, cornerPaint)

                // Top-right corner
                canvas.drawLine(right, top, right - cornerSize, top, cornerPaint)
                canvas.drawLine(right, top, right, top + cornerSize, cornerPaint)

                // Bottom-left corner
                canvas.drawLine(left, bottom, left + cornerSize, bottom, cornerPaint)
                canvas.drawLine(left, bottom, left, bottom - cornerSize, cornerPaint)

                // Bottom-right corner
                canvas.drawLine(right, bottom, right - cornerSize, bottom, cornerPaint)
                canvas.drawLine(right, bottom, right, bottom - cornerSize, cornerPaint)

                // Draw center point
                canvas.drawCircle(bbox.centerX().toFloat(), bbox.centerY().toFloat(), 8f, centerPaint)
            }

            // Step 3: Convert bitmap back to I420
            val pixels = IntArray(width * height)
            bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

            // Create I420 buffer from ARGB pixels
            val ySize = width * height
            val uvSize = (width / 2) * (height / 2)
            val yBuffer = ByteBuffer.allocateDirect(ySize)
            val uBuffer = ByteBuffer.allocateDirect(uvSize)
            val vBuffer = ByteBuffer.allocateDirect(uvSize)

            // Convert ARGB to I420
            for (j in 0 until height) {
                for (i in 0 until width) {
                    val pixel = pixels[j * width + i]
                    val r = (pixel shr 16) and 0xFF
                    val g = (pixel shr 8) and 0xFF
                    val b = pixel and 0xFF

                    // RGB to YUV conversion
                    val y = ((66 * r + 129 * g + 25 * b + 128) shr 8) + 16
                    yBuffer.put((y.coerceIn(0, 255)).toByte())

                    if (j % 2 == 0 && i % 2 == 0) {
                        val u = ((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128
                        val v = ((112 * r - 94 * g - 18 * b + 128) shr 8) + 128
                        uBuffer.put((u.coerceIn(0, 255)).toByte())
                        vBuffer.put((v.coerceIn(0, 255)).toByte())
                    }
                }
            }

            yBuffer.rewind()
            uBuffer.rewind()
            vBuffer.rewind()

            val newI420Buffer = JavaI420Buffer.wrap(
                width, height,
                yBuffer, width,
                uBuffer, width / 2,
                vBuffer, width / 2,
                null
            )

            val newFrame = VideoFrame(newI420Buffer, rotation, timestampNs)
            
            // Clean up
            bitmap.recycle()
            i420Buffer.release()

            return newFrame
        } catch (e: Exception) {
            Log.e(TAG, "Error drawing overlay on frame", e)
            i420Buffer.release()
            throw e
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

        // MODIFIED: Use the actual rotation from the camera frame
        // ML Kit handles rotation internally and adjusts face coordinates accordingly
        return InputImage.fromByteArray(
            cleanedArray,
            width,
            height,
            normalizedRotation,
            ImageFormat.NV21
        )
    }

    // NEW ADDITION START - Methods to send face data to Flutter via EventChannel
    /**
     * Sends face detection data to Flutter via the EventChannel.
     * Data format: { "faces": [ { "left": x, "top": y, "right": x, "bottom": y }, ... ], "frameWidth": w, "frameHeight": h }
     */
    private fun sendFaceDataToFlutter(faces: List<Face>, frameWidth: Int, frameHeight: Int) {
        val sink = FlutterWebRTCPlugin.faceDetectionEventSink
        if (sink == null) {
            // Log only occasionally to avoid spam
            if (frameCount % 120L == 0L) {
                Log.d(TAG, "Face detection event sink is null - Flutter may not be listening yet")
            }
            return
        }

        try {
            val data = mutableMapOf<String, Any>()
            data["type"] = "faces"
            data["frameWidth"] = frameWidth
            data["frameHeight"] = frameHeight
            
            val facesList = mutableListOf<Map<String, Any>>()
            for (face in faces) {
                val bbox = face.boundingBox
                val faceData = mutableMapOf<String, Any>()
                faceData["left"] = bbox.left
                faceData["top"] = bbox.top
                faceData["right"] = bbox.right
                faceData["bottom"] = bbox.bottom
                faceData["width"] = bbox.width()
                faceData["height"] = bbox.height()
                
                // Add tracking ID if available
                face.trackingId?.let { faceData["trackingId"] = it }
                
                facesList.add(faceData)
            }
            
            data["faces"] = facesList
            
            // Send event on main thread as required by Flutter
            Handler(Looper.getMainLooper()).post {
                sink.success(data)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error sending face data to Flutter", e)
        }
    }

    /**
     * Sends a "no faces" event to Flutter.
     */
    private fun sendNoFacesToFlutter() {
        val sink = FlutterWebRTCPlugin.faceDetectionEventSink ?: return

        try {
            val data = mutableMapOf<String, Any>()
            data["type"] = "noFaces"
            
            // Only send "no faces" event occasionally to avoid spam
            if (noFaceCount % 30L == 0L) {
                Handler(Looper.getMainLooper()).post {
                    sink.success(data)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error sending no-faces event to Flutter", e)
        }
    }
    // NEW ADDITION END

    /**
     * NEW ADDITION: Enable or disable overlay rendering on video frames
     */
    fun setOverlayEnabled(enabled: Boolean) {
        enableOverlay = enabled
        Log.i(TAG, "Overlay rendering ${if (enabled) "enabled" else "disabled"}")
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


