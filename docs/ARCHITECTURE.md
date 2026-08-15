# Architecture

## Design goals

VulkanCraft favors explicit ownership, small cohesive types, and data that moves
in one direction. The initial version uses only the abstractions needed by the
working vertical slice. Public methods document their contracts in a Google-style
format, and resources are destroyed in the inverse order of creation.

## Modules

### World

`World` owns contiguous three-dimensional arrays of `BlockType` and compact
water-level values plus a 16-cubed occupancy index. It is the single authority
for bounds checking and block lookup. `WaterSimulation` advances a duplicate-free
scheduled queue every 0.25 seconds. Water falls one block per update, spreads
seven blocks horizontally, resets that distance after falling, and searches up
to four horizontal blocks for a preferred drop. `WorldGenerator`
uses global coordinates to create continental terrain, mountains, lowland river
beds, oceans, lakes, ravines, cave tunnels, strata, and deterministic trees.
Rivers are allowed to carve only terrain close to sea level with a negligible
mountain mask, preventing waterways from slicing peaks down to sea level.
`StreamingWorld` follows the camera with a moving 32 by 32 (1024-chunk) window and
rebuilds only after a chunk boundary is crossed. A replacement copies overlapping
voxel columns only within the 17 by 17 interactive area, generates its padded
entering strips, moves retained chunk meshes into their new slots, and packs the
entering edge plus retained chunks that cross an LOD boundary into one upload.
Retired dense storage, outgoing CPU chunk meshes, and already uploaded replacement
batches are handed to the next background rebuild for destruction, so the render
thread does not synchronously release large allocation groups. The
four persistent generation workers run below the render thread's normal CPU
priority so streaming load yields before frame and input work. The
inner radius of eight chunks retains caves, trees, water, and complete voxel
topology. The outer area is generated from short-lived surface-only sources and
retains only a deterministic terrain heightfield shell and mesh while preserving native
per-face atlas sampling and real deterministic tree geometry. Its opaque terrain
height and translucent water height are tracked independently, so ocean LOD keeps
a visible seabed. Inner transition edges receive downward skirts so caves cannot
open into the omitted LOD volume. This hierarchical
near/far representation follows the
same broad data-reduction principle used by distant-terrain renderers without
copying third-party implementation code.
Generation padding keeps terrain and vegetation continuous at visible borders.
Chunk columns are 32 by 32 blocks and cover 384 vertical blocks from Y -64 through
Y 319. `WorldMesher` emits opaque, transparent, and cloud-shadow index batches,
plus stable chunk draw ranges. Leaf-to-leaf boundaries retain one double-sided
plane, while solid faces touching leaves remain in the mesh so trunks and terrain
are visible through transparent texels. The alpha-tested Z pre-pass exposes the
nearest internal layer through each cutout so foliage remains dense. Two-block-high
coplanar cloud blocks are greedily merged
into one external boundary shell so translucent rasterization cannot expose a grid
at individual block edges. No face is emitted between adjacent cloud voxels.
Connected groups receive stable, slightly different wind speeds. Motion uses four
wind classes separated by 2.5 vertical blocks. Groups in one altitude class share
one velocity, while groups with different velocities cannot become coplanar as
they cross.

Generation and meshing do not know about Vulkan. Four process-wide persistent
workers replace per-operation thread creation, and section occupancy lets the
mesher skip empty 16-block vertical spans. This separation makes the code fast
to test and lets the four-worker build produce replacement windows and
precombine only their entering chunks without a graphics context.

### Core

`Camera` owns view orientation and projection state. `PlayerController` owns the
collidable body, gravity, jumping, crouching, sprinting, flight, and water movement.
Flight assigns vertical velocity directly from input and bypasses gravity and
buoyancy, leaving the player completely stationary when no movement is requested.
`Application` translates GLFW input into player and block-edit commands. Initial
terrain generation runs asynchronously while a Win32 loading surface continues
to poll events. The application also owns window-mode toggling, the Vulkan-rendered
cheat console prompt, and the 24-minute world clock.

### Rendering

`CascadeShadowCalculator` is pure CPU-side math. It splits the first 320 blocks
of the camera frustum,
fits one directional-light projection around each slice, and snaps projections to
shadow texels to reduce shimmering.

`CelestialCycle` defines a centered Y-Z orbit that begins at the default forward
horizon and crosses directly overhead. It also supplies a camera-independent
world basis: the horizontal axis remains fixed to world X and the vertical axis
depends only on orbital direction. The celestial vertex shader expands sun and
moon corners in that world basis before applying the camera view transform.

`VulkanRenderer` is the only Vulkan owner. It creates the instance, surface,
device, swapchain, streamed mesh buffers, decoded sprite atlas, shadow array,
opaque, cloud, water, wireframe, sky, opaque Z pre-pass, and cloud depth pre-pass
pipelines, descriptors, and
frame synchronization. A frame records the scheduled shadow layers, a scene-depth
pass, HDR color, transparent geometry, and a full-screen effects composite. A
persistent coordinate-keyed GPU chunk cache
retains every overlapping batch during a streamed-window move. A coordinate hash
maps old slots in linear time, and one replacement batch contains both the
entering edge and retained LOD transitions. Dynamic geometry packs vertices and every index class into one
device-local allocation and one transfer. Local edits replace the affected cache
entry immediately.
The fragment shader selects a cascade from view depth and uses stable 3 by 3 PCF.
The shadow fragment stage samples the same native atlas UV and discards transparent
leaf texels, producing foliage-shaped rather than cube-shaped silhouettes. Each
face samples one native 32 by 32 atlas cell
from boundary to boundary with integer texel fetches and a fixed orientation.
The CPU constructs six mip levels by reducing every 32 by 32 sprite independently;
the fragment and alpha-depth shaders select the matching integer mip without
allowing colors or alpha from adjacent atlas cells to bleed across boundaries.
The compact 128 by 96 atlas contains a 4 by 3 grid of 32-pixel cells. Leaves use
the bright cell at origin (64, 32) and honor its supplied alpha; the current RGB
asset is fully opaque. Logs use the bark cell at (0, 32) on their sides and the
end cell at (96, 0). Water samples only the first blue cell at (32, 64), applies an integer-wrapped texel
phase, and symmetrically blends opposite edge texels inside that cell. This
preserves the native pixel grid while hiding non-seamless repeated borders. A
depth-sensitive water tint reduces the contrast of the remaining repetition,
preserves blue chroma through tone mapping, and combines a 0.94 surface opacity
with distance and underwater fog without replacing or scaling that cell. Water-only SSR
and depth tint fade out with that same fog factor, while SSAO converges to neutral
for every material as atmospheric coverage increases. A pair of low-amplitude,
world-oriented wave slopes supplies a stable water normal without requiring a
second texture. SSR is limited to upward water faces, uses angle-correct Fresnel,
rejects misses instead of substituting the sky, and filters valid hits over five
nearby samples according to water roughness.
Clouds bypass CSM sampling, use the daylight factor for their day/night tint,
and remain in the caster batch. A double-sided cloud depth pre-pass records the
nearest boundary surface, and the color pass accepts only equal-depth fragments.
This renders the boundary shell from inside while preventing the near and far
surfaces of one translucent volume from blending over the same screen region.
Cloud color, depth, and shadow vertex stages apply the same elapsed-time wind and
altitude displacement, keeping moving silhouettes and CSM shadows aligned.
The world vertex format stores position, UV, and one packed integer containing
axis normal, material, and cloud wind class. It is 24 bytes rather than the old
52-byte colored format. Triangle indices already share the four corners within
each face, and persistent multi-draw indirect buffers collapse hundreds of
visible chunk submissions into one command per material/pass.

The HDR post-process reconstructs view-space positions from depth to apply
subtle eight-direction SSAO, local bloom, aerial fog, six-sample CSM-aware
volumetric sunlight, ten-step occlusion-aware god rays, and a bounded medium-range
water-only reflected-ray screen-space search. The world shader adds material roughness,
Schlick-style Fresnel, and sprite-local luminance relief. The deliberately small
sample budgets provide the requested effects without turning the MVP into a
full deferred renderer.

Conservative chunk-volume tests prevent off-screen geometry from being submitted
to the scene and each shadow cascade. A six-plane Vulkan frustum is extracted once
per pass and tested against each 32 by 32 full-height chunk AABB with an early-out
positive-vertex test. Bounds conservatively include cloud wind displacement, so
moving cloud geometry and its caster remain eligible as they cross a chunk edge.
The resulting scene list is shared by the Z pre-pass, opaque, water, cloud, and
wireframe draws; every shadow cascade receives its own caster list. In F3 mode,
opaque, cloud, and water batches all use the line polygon pipeline and their filled
color pipelines are skipped.

## Frame flow

1. Poll input and simulate player collision, gravity, swimming, and interactions.
2. Poll the worker result or request a background window when a boundary is crossed.
3. Rotate the sun and moon, calculate the four bounded cascade matrices, and
   schedule the far layers at 2-, 4-, and 8-frame intervals.
4. Upload the frame uniform block.
5. Render only intersecting opaque terrain and cloud casters into scheduled 2048
   by 2048 shadow layers.
6. Populate scene depth with alpha-tested opaque geometry.
7. Draw the sky, depth-tested sun and moon quads when above water, and equal-depth opaque batches.
8. Record the nearest double-sided cloud boundary into depth, then blend its
   color over the completed opaque scene and draw water afterward. In F3 mode, replace these
   world-material color passes with line-only opaque, cloud, and water draws.
   Cached chunk batches substitute only their corresponding range and remain
   reusable after the streaming window moves.
9. Sample HDR color and depth for SSAO, bloom, volumetric fog/god rays, water SSR,
   tone mapping, and gamma correction, then present.

Only one frame is in flight in version 1.6.0. This keeps shadow and depth resource
ownership unambiguous. Increasing concurrency later requires one uniform buffer,
depth target, and shadow target per in-flight frame.

## Extension points

- Replace the atlas with a texture-array material descriptor without changing
  voxel topology.
- Add collision and block interaction above `World`; rendering remains read-only.
