// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MossFormer2SS",
    platforms: [
        .macOS("13.3"),
        .iOS(.v16)
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "MossFormer2SS",
            targets: ["MossFormer2SS"]),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.18.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.0.0"),
        .package(url: "https://github.com/starkdmi/SwiftAudio", exact: "1.0.0")
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "MossFormer2SS",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift")
            ],
            path: "Sources/MossFormer2SS"
        ),
        .testTarget(
            name: "MossFormer2SSTests",
            dependencies: [
                "MossFormer2SS",
                .product(name: "AudioUtils", package: "SwiftAudio"),
                .product(name: "Hub", package: "swift-transformers")
            ],
            path: "Tests/MossFormer2SSTests"
        ),
    ]
)
