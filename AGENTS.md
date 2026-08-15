# VulkanCraft Project Rules

## Language and documentation

- Write all source code, comments, documentation, build files, tests, asset metadata, and user-facing project text in English.
- Document public classes and methods with Google-style docstrings.
- Keep documentation synchronized with implemented behavior and controls.
- Follow Semantic Versioning. Every user-visible feature or bug-fix batch must update the project version in CMake and README before completion: increment MAJOR for incompatible changes, MINOR for backward-compatible features, and PATCH for fixes. Vulkan runtime metadata must derive from the CMake project version.

## C++ and architecture

- Use C++23 or the newest broadly supported standard available to the configured toolchains.
- Follow modern C++ ownership, RAII, const-correctness, type safety, and standard-library conventions.
- Apply SOLID, KISS, DRY, YAGNI, and clean-code principles pragmatically. Do not add abstractions without a current use.
- Keep procedural world logic independent from Vulkan so it remains deterministic and unit-testable.
- Target Windows exclusively during the MVP. Linux and MinGW build support must not be present.

## Build system

- Use CMake and the existing presets. Keep precompiled headers enabled for project targets.
- The primary Windows generator is Visual Studio 18 2026 with the v180 toolset.
- Keep strict warnings enabled and do not suppress a warning instead of fixing its cause.
- Build and test Debug and Release before declaring a material change complete.
- Keep exactly two user-facing presets: `windows-debug` and `windows-release`.

## World invariants

- Chunks are 32 by 32 blocks horizontally.
- The world height is 384 blocks, with an inclusive lower limit of Y -64 and an exclusive upper limit of Y 320.
- World generation must remain deterministic and continuous across chunk boundaries.
- Rivers may carve only low terrain and must never flatten mountain-scale elevation. Lakes, ravines, and caves must use global deterministic coordinates and remain continuous at chunk boundaries.
- Streaming generation must not block the main game loop.
- Procedural generation and chunk meshing must use exactly four worker threads.
- Block edits must update only affected chunk meshes and must remain visible across streamed window replacements.
- The default draw window is exactly 32 by 32 (1024) chunks with an 832-block camera far plane. Distant world fragments must converge to the directional sky gradient before the nearest window edge.
- Local block edits must replace only affected GPU chunk batches; they must not recombine or re-upload the 32 by 32 window.
- Background streaming must precombine entering chunks and retained LOD-transition chunks into one replacement upload. Dynamic chunk geometry must use one packed GPU buffer per upload batch.
- Reuse overlapping voxel columns and chunk meshes when the streaming center moves. Generate only newly entering padded strips.
- Keep full voxel topology within eight chunks of the player and use compact deterministic heightfield surface LOD throughout the remainder of the 32 by 32 window. Surface LOD must retain deterministic trunks and cutout foliage.
- Seal every full-detail-to-heightfield transition down to the lower world boundary. Caves and deep water must never expose the missing interior of a surface-LOD chunk.
- Water simulation uses scheduled 0.25-second updates, falls one block per update without a vertical distance limit, spreads at most seven horizontal blocks beyond its source, resets horizontal depth after a fall, and prefers a downward path reachable within four horizontal blocks. Fluid edits and levels must persist across streamed window replacements.

## Rendering invariants

- Use Vulkan 1.3 dynamic rendering and synchronization2.
- Preserve the Z pre-pass, batched opaque/transparent rendering, and four cascaded shadow maps.
- Apply six-plane Vulkan frustum culling to chunk AABBs before every scene batch, and conservatively include animated cloud displacement. Use an independent clip volume for each shadow cascade.
- Use four stabilized 2048 by 2048 cascades with a caster guard band and conservative per-cascade chunk culling.
- Bound dynamic shadows to 320 blocks. Update cascade zero every frame and stabilized cascades one through three every 2, 4, and 8 frames respectively.
- Keep the indexed vertex format at 24 bytes: position, native atlas UV, and packed normal/material/wind attributes. Do not restore per-vertex colors.
- Submit visible base chunk ranges with Vulkan multi-draw indirect. Edited override buffers may use direct indexed fallback draws.
- The block sprite sheet contains an exact 4 by 3 grid of 32 by 32 sprites in a 128 by 96 image, with no global border or padding.
- Atlas UVs must cover the complete 16 by 16 cell boundaries and use integer `texelFetch` sampling. They must never omit edge texels or sample adjacent sprites.
- Generate all six atlas mip levels independently per 32 by 32 sprite. Mip filtering must never cross a sprite boundary, and color and alpha-tested depth passes must choose the same integer mip.
- Every block face samples one complete 32 by 32 sprite at its native scale and fixed orientation. Never rotate, mirror, stretch, or world-project block textures.
- Leaves use the atlas cell at zero-based origin (64, 32), render from both sides, retain one plane at every internal leaf boundary, and honor the alpha supplied by the spritesheet. Opaque faces adjacent to leaves must remain present. Log side faces use (0, 32), while their top and bottom faces use (96, 0). Water uses exactly the cell at zero-based atlas origin (32, 64), remains translucent, and must not hide opaque terrain below it. Water animation may use integer-wrapped texel offsets and symmetric edge blending only within that cell so it preserves native resolution, hides non-seamless cell borders, and never samples a neighboring cell.
- Sun and moon must use world-oriented square quads and must never derive their rotation from the camera. Their horizontal axis remains fixed to world X, while their vertical axis follows only the centered orbital plane. Their shared orbit lies in the vertical Y-Z plane: sunrise begins at the center of the default forward horizon, the path crosses directly overhead without lateral drift, and sunset ends at the opposite horizon. The active celestial body must drive directional lighting and cascaded shadows.
- Clouds must be dense, varied groups of connected rectangles exactly two blocks high rather than a continuous floating layer. Greedily merge coplanar cloud surfaces and emit only the external boundary shell, never faces between adjacent cloud voxels. Render that shell from both sides through a cloud depth pre-pass followed by equal-depth color shading so clouds remain visible from inside without blending their near and far surfaces. Render clouds before water so water cannot blend over them. Clouds are slightly translucent, cast CSM shadows, never receive shadow attenuation, and transition from near-white by day to gray at night.
- Move clouds slowly along one shared wind direction. Assign connected groups to one of four stable speeds from 0.055 to 0.13 blocks per second. Separate consecutive wind classes vertically by 2.5 blocks so two-block-high clouds with different speeds and their shadow casters never overlap. Clouds in the same altitude class must have the same velocity so they cannot drift into each other. Apply identical horizontal and vertical displacement in color and shadow passes.
- Generate cloud formations at three times the base horizontal shape scale. Use stable crisp PCF shadows; do not restore blocker-distance blur.
- Preserve the HDR post-processing foundation: depth-based SSAO, bloom, shadow-aware volumetric sunlight, god rays, water-only bounded medium-range SSR, tone mapping, PBR roughness/Fresnel, and sprite-local texture relief. Keep sample counts bounded and verify the Debug performance gate after shader changes.
- Keep daytime sky and water chroma after tone mapping, use subtle SSAO, darken water with visible depth, and prevent underwater fog from revealing distant surface-LOD interiors.
- F3 toggles pure wireframe rendering for opaque blocks, water, and clouds. Filled world-material passes must not be submitted while wireframe mode is active.
- Alpha-test leaf textures in both the scene depth and shadow depth passes. Sunlight must pass through transparent foliage texels.

## Player and controls

- `W`, `A`, `S`, and `D` move the player; Left Shift sprints and Left Ctrl crouches.
- Space jumps and moves the player upward while swimming.
- A Space double-tap toggles flight. In flight, Space ascends, Left Ctrl descends, Left Shift provides a substantial speed boost, and gravity or buoyancy must not cause drift without vertical input.
- Underwater fog depends on the camera voxel, not whether another part of the player body intersects water.
- The left mouse button breaks blocks and the right mouse button places blocks without allowing placement inside the player.
- Day and night each last twelve real-time minutes. The apostrophe key toggles the in-frame cheat console, which initially supports `/time set day` and `/time set night`. F11 toggles borderless mode at the monitor resolution.
- Keep the initial window responsive while terrain is generated. Show the native loading screen, start from a maximized 1280 by 720 window, and never initialize the complete world synchronously on the GLFW event thread.

## Quality gates

- Add or update regression tests for deterministic world behavior, atlas coordinates, mesh topology, physics, and streaming changes.
- Run CTest and relevant Vulkan smoke tests before reporting completion.
- Validate changed SPIR-V shaders and treat Vulkan validation-layer errors as failures.
- Inspect graphical fixes visually when numeric tests cannot prove the result.
- Never claim a build, test, performance result, or visual outcome that was not actually verified.
