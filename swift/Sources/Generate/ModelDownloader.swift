import Foundation
import Hub

/// Downloads and caches MossFormer2 model files from HuggingFace
class ModelDownloader {
    enum ModelType: String {
        case ss_2spk_16k
        case ss_2spk_whamr_8k
        case ss_3spk_8k

        var modelId: String {
            switch self {
            case .ss_2spk_16k:
                return "starkdmi/MossFormer2_SS_2SPK_16K_MLX"
            case .ss_2spk_whamr_8k:
                return "starkdmi/MossFormer2_SS_2SPK_WHAMR_8K_MLX"
            case .ss_3spk_8k:
                return "starkdmi/MossFormer2_SS_3SPK_8K_MLX"
            }
        }
    }

    private let modelType: ModelType

    init(modelType: ModelType) {
        self.modelType = modelType
    }

    /// Downloads model weights from HuggingFace
    /// - Returns: Path to weights file
    func downloadModel() async throws -> URL {
        let repo = Hub.Repo(id: modelType.modelId)

        print("Starting download from HuggingFace: \(modelType.modelId)")

        // Download model_fp32.safetensors only
        let modelDirectory = try await HubApi.shared.snapshot(
            from: repo,
            matching: ["model_fp32.safetensors"],
            progressHandler: { progress in
                print("Download progress: \(Int(progress.fractionCompleted * 100))%")
            }
        )

        print("Download completed to: \(modelDirectory.path)")

        let weightsURL = modelDirectory.appendingPathComponent("model_fp32.safetensors")

        return weightsURL
    }

    /// Synchronous wrapper for downloading model
    /// - Returns: Path to weights file
    func downloadModelSync() throws -> URL {
        var result: URL?
        var error: Error?

        let group = DispatchGroup()
        group.enter()

        Task {
            do {
                result = try await downloadModel()
            } catch let err {
                error = err
            }
            group.leave()
        }

        group.wait()

        if let error = error {
            throw error
        }

        guard let result = result else {
            throw NSError(
                domain: "ModelDownloader",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to download model"]
            )
        }

        return result
    }
}
