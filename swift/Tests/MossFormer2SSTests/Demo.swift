import XCTest
import AVFoundation
import MLX
import MLXNN
import AudioUtils
@testable import MossFormer2SS

final class MossFormer2Tests: XCTestCase {
    
    /// Helper function to test model with specific configuration
    private func testMossFormer2WithConfig(numSpeakers: Int, sampleRate: Int, modelType: ModelDownloader.ModelType, audioFile: String, outputPrefix: String, isWHAMR: Bool = false) throws {
        print("MossFormer2 Swift MLX \(numSpeakers)-Speaker Speech Separation Test")
        print("=" + String(repeating: "=", count: 49))
        if isWHAMR {
            print("Using WHAMR mode (skip_mask_multiplication=true)")
        }
        
        let startTime = Date()
        
        // 1. Create model
        print("Creating MossFormer2 MLX model...")
        let modelStartTime = Date()
        
        let config = MossFormer2Config(
            encoder_embedding_dim: 512,
            mossformer_sequence_dim: 512,
            num_mossformer_layer: 24,
            encoder_kernel_size: 16,
            num_spks: numSpeakers,
            skip_mask_multiplication: isWHAMR
        )
        
        let model = MossFormer2_SS_16K(config: config)
        let modelCreateTime = Date().timeIntervalSince(modelStartTime)
        print("⏱️ Model creation: \(String(format: "%.3f", modelCreateTime))s")
        
        // 2. Download and load weights
        print("\nDownloading and loading model weights...")
        let weightsStartTime = Date()

        do {
            // Download model from HuggingFace
            let downloader = ModelDownloader(modelType: modelType)
            let weightsURL = try downloader.downloadModelSync()

            print("📂 Loading weights from: \(weightsURL.path)")

            // Load the flat dictionary of weights
            let weights = try loadArrays(url: weightsURL)

            // Convert flat dictionary to nested structure using unflattened
            let nestedWeights = NestedDictionary<String, MLXArray>.unflattened(weights)

            // Update model weights (nestedWeights is already of type ModuleParameters)
            try model.update(parameters: nestedWeights, verify: .all)

            let weightsLoadTime = Date().timeIntervalSince(weightsStartTime)
            print("⏱️ Weight loading: \(String(format: "%.3f", weightsLoadTime))s")
            print("✅ Weights loaded successfully")
        } catch {
            print("⚠️ Could not download or load weights")
            print("   Error: \(error)")
            print("   Continuing with random weights for testing...")
        }
        
        // 3. Print model info
        print("\n✅ Model created successfully")
        print("   - Encoder output dim: \(config.encoder_embedding_dim)")
        print("   - MossFormer blocks: \(config.num_mossformer_layer)")
        print("   - Number of speakers: \(config.num_spks)")
        
        // 4. Load test audio
        let audioStartTime = Date()

        let audioPath = "\(URL(fileURLWithPath: #filePath).deletingLastPathComponent().path(percentEncoded: false))\(audioFile)"
        var mixedAudio: [Float]
        var samples: Int
        
        // Load actual audio file
        print("📂 Loading test audio from: \(audioPath)")
        let audioLoader = AudioLoader(config: .init(
              targetSampleRate: Double(sampleRate),
              normalizationMode: .none,
              resamplingMethod: .avAudioConverter(
                  algorithm: AVSampleRateConverterAlgorithm_Mastering,
                  quality: .max
              )
          ))
        let loadedAudio = try audioLoader.load(from: URL(fileURLWithPath: audioPath))
        mixedAudio = loadedAudio.asArray(Float.self)

        samples = mixedAudio.count
        print("✅ Loaded Audio: \(samples) samples, \(sampleRate) Hz, \(String(format: "%.2f", Double(samples) / Double(sampleRate)))s")
        
        let audioCreateTime = Date().timeIntervalSince(audioStartTime)
        print("⏱️ Audio loading: \(String(format: "%.3f", audioCreateTime))s")
        
        // 5. Prewarm the model (important for compiled functions and GPU warmup)
        /*print("\nPrewarming model...")
        let prewarmStartTime = Date()
        
        // Create a shorter dummy input for warmup (1 second of audio)
        let warmupSamples = min(sampleRate, samples)  // 1 second or less
        let warmupAudio = Array(mixedAudio[0..<warmupSamples])
        let warmupMLX = MLXArray(warmupAudio).expandedDimensions(axis: 0)
        
        // Run warmup passes
        for i in 0..<3 {
            let warmupSources = model(warmupMLX)
            MLX.eval(warmupSources)
            print("   Warmup pass \(i + 1)/3 complete")
        }
        
        let prewarmTime = Date().timeIntervalSince(prewarmStartTime)
        print("⏱️  Prewarm time: \(String(format: "%.3f", prewarmTime))s")*/
        
        // 6. Run actual inference
        print("\nRunning speech separation...")
        let inferenceStartTime = Date()
        
        // Convert to MLX array and add batch dimension
        let audioMLX = MLXArray(mixedAudio).expandedDimensions(axis: 0)  // [1, T]
        
        // Run model
        let separatedSources = model(audioMLX)
        
        // Force evaluation
        MLX.eval(separatedSources)
        
        let inferenceTime = Date().timeIntervalSince(inferenceStartTime)
        print("⏱️ Model inference: \(String(format: "%.3f", inferenceTime))s")
        print("\n✅ Separated into \(separatedSources.count) sources")
        
        // 7. Verify output shapes
        for (i, source) in separatedSources.enumerated() {
            let shape = source.shape
            // print("   Source \(i + 1) shape: \(shape)")
            let maxVal = MLX.max(MLX.abs(source)).item(Float.self)
            let meanVal = MLX.mean(MLX.abs(source)).item(Float.self)
            let stdVal = std(source).item(Float.self)
            print("Source \(i + 1) stats - Shape: \(shape), Max: \(String(format: "%.6f", maxVal)), Mean: \(String(format: "%.6f", meanVal)), Std: \(String(format: "%.6f", stdVal))")

            XCTAssertEqual(shape[0], 1, "Batch size should be 1")
            XCTAssertEqual(shape[1], samples, "Output length should match input")
        }
        
        // 8. Save separated sources
        print("\n💾 Saving separated sources...")
        let saveStartTime = Date()
        
        let outputDir = "\(URL(fileURLWithPath: #filePath).deletingLastPathComponent().path(percentEncoded: false))\(modelType)_output"

        // Create output directory if it doesn't exist
        try FileManager.default.createDirectory(
            atPath: outputDir,
            withIntermediateDirectories: true,
            attributes: nil
        )

        // Create audio saver
        let audioSaver = AudioSaver(config: .init(
             sampleRate: Double(sampleRate),
             bitDepth: .float32,
             fileFormat: .wav
         ))
        
        // Save each separated source
        for (i, source) in separatedSources.enumerated() {
            let filename = "\(outputPrefix)_speaker_\(i + 1).wav"
            let filepath = "\(outputDir)/\(filename)"
            
            // Remove batch dimension and save
            let sourceAudio = source.squeezed(axis: 0)
            //
            // try audioSaver.saveAudio(sourceAudio, to: filepath)
            // AudioUtils
            let normalizedAudio = normalizeToPeak(sourceAudio, targetPeak: 1.0)
            try audioSaver.save(normalizedAudio, to: filepath)
            
            // Calculate RMS for info (after normalization to match Python output)
            var audioData = sourceAudio.asArray(Float.self)
            let maxVal = audioData.map { abs($0) }.max() ?? 1.0
            if maxVal > 1.0 {
                audioData = audioData.map { $0 / maxVal }
            }
            let rms = sqrt(audioData.map { $0 * $0 }.reduce(0, +) / Float(audioData.count))
            
            print("   - \(filename): \(audioData.count) samples, RMS=\(String(format: "%.4f", rms))")
        }
        
        // Also save the input mix for comparison
        let mixMLX = MLXArray(mixedAudio)
        // try audioSaver.saveAudio(mixMLX, to: "\(outputDir)/input_mix.wav")
        // AudioUtils
        let normalizedMixMLX = normalizeToPeak(mixMLX, targetPeak: 1.0)
        try audioSaver.save(normalizedMixMLX, to: "\(outputDir)/input_mix.wav")

        print("   - input_mix.wav: \(mixedAudio.count) samples")
        
        let saveTime = Date().timeIntervalSince(saveStartTime)
        print("⏱️ Audio saving: \(String(format: "%.3f", saveTime))s")
        
        /*print("\n🔍 Checking separation quality...")
        for (i, source) in separatedSources.enumerated() {
            var sourceData = source.squeezed().asArray(Float.self)
            let maxVal = sourceData.map { abs($0) }.max() ?? 0
            
            // Show pre-normalized values
            let preNormRMS = sqrt(sourceData.map { $0 * $0 }.reduce(0, +) / Float(sourceData.count))
            print("   Source \(i + 1): Pre-norm RMS=\(String(format: "%.4f", preNormRMS)), Max=\(String(format: "%.4f", maxVal))")
            
            // Normalize if needed
            if maxVal > 1.0 {
                sourceData = sourceData.map { $0 / maxVal }
            }
            let postNormRMS = sqrt(sourceData.map { $0 * $0 }.reduce(0, +) / Float(sourceData.count))
            print("              Post-norm RMS=\(String(format: "%.4f", postNormRMS))")
            
            // If RMS is very low, weights might not be loaded properly
            if postNormRMS < 0.001 {
                print("   ⚠️ Warning: Source \(i + 1) has very low RMS, weights might not be loaded correctly")
            }
        }*/

        let totalTime = Date().timeIntervalSince(startTime)
        print("\n⏱️ Total execution time: \(String(format: "%.3f", totalTime))s")
        print("\nTest completed successfully")
        print("Output directory: \(outputDir)")
    }
    
    /// Test 2-speaker model (16kHz)
    func testMossFormer2Pipeline_2Speakers() throws {
        try testMossFormer2WithConfig(
            numSpeakers: 2,
            sampleRate: 16000,
            modelType: .ss_2spk_16k,
            audioFile: "mix.wav",
            outputPrefix: "2spk"
        )
    }
    
    /// Test 3-speaker model (8kHz)
    func testMossFormer2Pipeline_3Speakers() throws {
        try testMossFormer2WithConfig(
            numSpeakers: 3,
            sampleRate: 8000,
            modelType: .ss_3spk_8k,
            audioFile: "mix3_8k.wav",
            outputPrefix: "3spk"
        )
    }
    
    /// Test 2-speaker WHAMR model (8kHz)
    func testMossFormer2Pipeline_2Speakers_WHAMR() throws {
        try testMossFormer2WithConfig(
            numSpeakers: 2,
            sampleRate: 8000,
            modelType: .ss_2spk_whamr_8k,
            audioFile: "mix_8k.wav",
            outputPrefix: "2spk_whamr",
            isWHAMR: true
        )
    }
}
