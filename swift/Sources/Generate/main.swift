import Foundation
import ArgumentParser
import AVFoundation
import MLX
import MLXNN
import AudioUtils
import MossFormer2SS

struct MossFormer2CLI: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mossformer2",
        abstract: "MossFormer2 speech separation for MLX",
        discussion: """
        Separates mixed audio into individual speaker sources.

        Models:
          2spk        - 2-speaker separation (16kHz)
          3spk        - 3-speaker separation (8kHz)
          2spk-whamr  - 2-speaker WHAMR (8kHz)
        """
    )

    enum Model: String, ExpressibleByArgument {
        case twoSpeaker = "2spk"
        case threeSpeaker = "3spk"
        case twoSpeakerWHAMR = "2spk-whamr"

        var numSpeakers: Int {
            switch self {
            case .twoSpeaker: return 2
            case .threeSpeaker: return 3
            case .twoSpeakerWHAMR: return 2
            }
        }

        var sampleRate: Int {
            switch self {
            case .twoSpeaker: return 16000
            case .threeSpeaker: return 8000
            case .twoSpeakerWHAMR: return 8000
            }
        }

        var modelType: ModelDownloader.ModelType {
            switch self {
            case .twoSpeaker: return .ss_2spk_16k
            case .threeSpeaker: return .ss_3spk_8k
            case .twoSpeakerWHAMR: return .ss_2spk_whamr_8k
            }
        }

        var outputPrefix: String {
            switch self {
            case .twoSpeaker: return "2spk"
            case .threeSpeaker: return "3spk"
            case .twoSpeakerWHAMR: return "2spk_whamr"
            }
        }

        var isWHAMR: Bool {
            switch self {
            case .twoSpeakerWHAMR: return true
            default: return false
            }
        }
    }

    @Option(name: .shortAndLong, help: "Model type")
    var model: Model

    @Option(name: .shortAndLong, help: "Input audio file")
    var input: String

    @Option(name: .shortAndLong, help: "Output directory (default: <model>_output)")
    var output: String?

    func run() throws {
        let numSpeakers = model.numSpeakers
        let sampleRate = model.sampleRate
        let modelType = model.modelType
        let outputPrefix = model.outputPrefix
        let isWHAMR = model.isWHAMR

        print("MossFormer2 Swift MLX \(numSpeakers)-Speaker Speech Separation Test")
        print("=" + String(repeating: "=", count: 49))
        if isWHAMR {
            print("Using WHAMR mode (skip_mask_multiplication=true)")
        }

        let startTime = Date()

        // Create model
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
        print("Model creation: \(String(format: "%.3f", modelCreateTime))s")

        // Download and load weights
        print("\nDownloading and loading model weights...")
        let weightsStartTime = Date()

        do {
            // Download model from HuggingFace
            let downloader = ModelDownloader(modelType: modelType)
            let weightsURL = try downloader.downloadModelSync()

            print("Loading weights from: \(weightsURL.path)")

            // Load the flat dictionary of weights
            let weights = try loadArrays(url: weightsURL)

            // Convert flat dictionary to nested structure using unflattened
            let nestedWeights = NestedDictionary<String, MLXArray>.unflattened(weights)

            // Update model weights (nestedWeights is already of type ModuleParameters)
            try model.update(parameters: nestedWeights, verify: .all)

            let weightsLoadTime = Date().timeIntervalSince(weightsStartTime)
            print("Weights loaded successfully, duration: \(String(format: "%.3f", weightsLoadTime))s")
        } catch {
            print("Could not download or load weights")
            print("   Error: \(error)")
            print("   Continuing with random weights for testing...")
        }

        // Print model info
        print("\nModel created successfully")
        print("   - Encoder output dim: \(config.encoder_embedding_dim)")
        print("   - MossFormer blocks: \(config.num_mossformer_layer)")
        print("   - Number of speakers: \(config.num_spks)")

        // Load test audio
        let audioStartTime = Date()

        let audioPath = URL(fileURLWithPath: input).standardized.path
        var mixedAudio: [Float]
        var samples: Int

        // Load actual audio file
        print("\nLoading audio from: \(audioPath)")
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
        print("Loaded Audio: \(samples) samples, \(sampleRate) Hz, \(String(format: "%.2f", Double(samples) / Double(sampleRate)))s")

        let audioCreateTime = Date().timeIntervalSince(audioStartTime)
        print("Audio loading: \(String(format: "%.3f", audioCreateTime))s")

        // Prewarm the model (important for compiled functions and GPU warmup)
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
        print("Prewarm time: \(String(format: "%.3f", prewarmTime))s")*/

        // Run actual inference
        print("\nRunning speech separation...")
        let inferenceStartTime = Date()

        // Convert to MLX array and add batch dimension
        let audioMLX = MLXArray(mixedAudio).expandedDimensions(axis: 0)  // [1, T]

        // Run model
        let separatedSources = model(audioMLX)

        // Force evaluation
        MLX.eval(separatedSources)

        let inferenceTime = Date().timeIntervalSince(inferenceStartTime)
        print("Model inference: \(String(format: "%.3f", inferenceTime))s")
        print("\nSeparated into \(separatedSources.count) sources")

        // Verify output shapes
        for (i, source) in separatedSources.enumerated() {
            let shape = source.shape
            // print("   Source \(i + 1) shape: \(shape)")
            let maxVal = MLX.max(MLX.abs(source)).item(Float.self)
            let meanVal = MLX.mean(MLX.abs(source)).item(Float.self)
            let stdVal = std(source).item(Float.self)
            print("Source \(i + 1) stats - Shape: \(shape), Max: \(String(format: "%.6f", maxVal)), Mean: \(String(format: "%.6f", meanVal)), Std: \(String(format: "%.6f", stdVal))")

            guard shape[0] == 1 && shape[1] == samples else {
                throw ValidationError("Invalid output shape for source \(i + 1): expected [1, \(samples)], got \(shape)")
            }
        }

        // Save separated sources
        print("\nSaving separated sources...")
        let saveStartTime = Date()

        let outputDir = output ?? "\(modelType)_output"

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

        print("Audio saving: \(String(format: "%.3f", Date().timeIntervalSince(saveStartTime)))s")

        let totalTime = Date().timeIntervalSince(startTime)
        print("\nTotal execution time: \(String(format: "%.3f", totalTime))s")
        print("\nTest completed successfully")
        print("Output directory: \(outputDir)")
    }
}

MossFormer2CLI.main()
