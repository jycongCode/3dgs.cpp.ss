#ifndef RENDERER_H
#define RENDERER_H

#define GLM_SWIZZLE

#include <atomic>
#include "3dgs.h"

#include "vulkan/Window.h"
#include "GSScene.h"
#include "vulkan/pipelines/ComputePipeline.h"
#include "vulkan/Swapchain.h"
#include <glm/gtc/quaternion.hpp>

#include "GUIManager.h"
#include "vulkan/ImguiManager.h"
#include "vulkan/QueryManager.h"

class Renderer {
public:
    struct alignas(16) UniformBuffer {
        glm::vec4 camera_position;
        glm::mat4 proj_mat;
        glm::mat4 view_mat;
        uint32_t width;
        uint32_t height;
        float tan_fovx;
        float tan_fovy;
    };

    struct VertexAttributeBuffer {
        glm::vec4 conic_opacity;
        glm::vec4 color_radii;
        glm::uvec4 aabb;
        glm::vec2 uv;
        float depth;
        uint32_t __padding[1];
    };

    struct Camera {
        glm::vec3 position;
        glm::quat rotation;
        float fov;
        float nearPlane;
        float farPlane;

        void translate(glm::vec3 translation) {
            position += rotation * translation;
        }
    };

    struct RadixSortPushConstants {
        uint32_t g_num_elements; // == NUM_ELEMENTS
        uint32_t g_shift; // (*)
        uint32_t g_num_workgroups; // == NUMBER_OF_WORKGROUPS as defined in the section above
        uint32_t g_num_blocks_per_workgroup; // == NUM_BLOCKS_PER_WORKGROUP
    };

    enum class RendererType {
        NormalSplatting,
        StochasticSplatting
	};

    explicit Renderer(VulkanSplatting::RendererConfiguration configuration);

    virtual void createGui();

    virtual void initialize();

    virtual void handleInput();

    virtual void retrieveTimestamps();

    virtual void recreateSwapchain();

    virtual void draw();

    virtual void run();

    virtual void stop();

    virtual RendererType getRendererType() { return RendererType::NormalSplatting; }

    virtual ~Renderer();

    Camera camera {
        .position = glm::vec3(0.0f, 0.0f, 0.0f),
        .rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
        .fov = 45.0f,
        .nearPlane = 0.1f,
        .farPlane = 1000.0f
    };

protected:
    VulkanSplatting::RendererConfiguration configuration;
    std::shared_ptr<Window> window;
    std::shared_ptr<VulkanContext> context;
    std::shared_ptr<ImguiManager> imguiManager;
    std::shared_ptr<GSScene> scene;
    std::shared_ptr<QueryManager> queryManager = std::make_shared<QueryManager>();
    GUIManager guiManager;

    // Vulkan pipeline objects for normal rendering
    std::shared_ptr<ComputePipeline> preprocessPipeline;
    std::shared_ptr<ComputePipeline> renderPipeline;
    std::shared_ptr<ComputePipeline> prefixSumPipeline;
    std::shared_ptr<ComputePipeline> preprocessSortPipeline;
    std::shared_ptr<ComputePipeline> sortHistPipeline;
    std::shared_ptr<ComputePipeline> sortPipeline;
    std::shared_ptr<ComputePipeline> tileBoundaryPipeline;

    // Common buffers used in all rendering pipelines
    std::shared_ptr<Buffer> uniformBuffer;
    std::shared_ptr<Buffer> vertexAttributeBuffer;

    // Buffers used in normal rendering pipelines
    std::shared_ptr<Buffer> tileOverlapBuffer;
    std::shared_ptr<Buffer> prefixSumPingBuffer;
    std::shared_ptr<Buffer> prefixSumPongBuffer;
    std::shared_ptr<Buffer> sortKBufferEven;
    std::shared_ptr<Buffer> sortKBufferOdd;
    std::shared_ptr<Buffer> sortHistBuffer;
    std::shared_ptr<Buffer> totalSumBufferHost;
    std::shared_ptr<Buffer> tileBoundaryBuffer;
    std::shared_ptr<Buffer> sortVBufferEven;
    std::shared_ptr<Buffer> sortVBufferOdd;

    std::shared_ptr<DescriptorSet> inputSet;

    std::atomic<bool> running = true;

    std::vector<vk::UniqueFence> inflightFences;

    std::shared_ptr<Swapchain> swapchain;

    vk::UniqueCommandPool commandPool;

    // Command buffers used in normal rendering
    vk::UniqueCommandBuffer preprocessCommandBuffer;
    vk::UniqueCommandBuffer renderCommandBuffer;

    uint32_t currentImageIndex;

    std::vector<vk::UniqueSemaphore> renderFinishedSemaphores;

#ifdef __APPLE__
    uint32_t numRadixSortBlocksPerWorkgroup = 256;
#else
    uint32_t numRadixSortBlocksPerWorkgroup = 32;
#endif

    int fpsCounter = 0;
    std::chrono::high_resolution_clock::time_point lastFpsTime = std::chrono::high_resolution_clock::now();

    unsigned int sortBufferSizeMultiplier = 1;

    // Common methods
    void initializeVulkan();

    void loadSceneToGPU();

    void createCommandPool();

    void updateUniforms();

	// Normal splatting methods

    void createPreprocessPipeline();

    void createPrefixSumPipeline();

    void createRadixSortPipeline();

    void createPreprocessSortPipeline();

    void createTileBoundaryPipeline();

    void createRenderPipeline();

    void recordPreprocessCommandBuffer();

    bool recordRenderCommandBuffer(uint32_t currentFrame);

};

class RendererST : public Renderer {
public:
    explicit RendererST(VulkanSplatting::RendererConfiguration configuration)
		: Renderer(std::move(configuration)) {
	}

    virtual ~RendererST();

    // Stochastic splatting methods
    virtual void draw() override;

	virtual void initialize() override;
    
	virtual void recreateSwapchain() override;

	virtual RendererType getRendererType() override { return RendererType::StochasticSplatting; }

    int spp = 16;
private:
    // Vulkan pipeline objects for stochastic splatting rendering
    std::shared_ptr<ComputePipeline> tileBoundaryPipeline_ST;
    std::shared_ptr<ComputePipeline> processInstancesPipeline_ST;
    std::shared_ptr<ComputePipeline> renderPipeline_ST;
    std::shared_ptr<ComputePipeline> preprocessPipeline_ST;

    // Buffers used in stochastic splatting rendering pipelines
    std::shared_ptr<Buffer> tileOverlapBuffer_ST;
    std::shared_ptr<Buffer> instanceIndexBuffer_ST;
    std::shared_ptr<Buffer> tileBoundaryBuffer_ST;
    std::shared_ptr<Buffer> prefixBuffer_ST;
    std::shared_ptr<Buffer> instanceCounterBuffer_ST;
    

    // Command buffers used in stochastic splatting rendering
    vk::UniqueCommandBuffer preprocessCommandBuffer_ST;
    vk::UniqueCommandBuffer renderCommandBuffer_ST;
    

    // Stachastic splatting parameters
    unsigned int perTileSplatMultiplier = 1;

    void createPreprocessPipeline_ST();

    void createProcessInstancesPipeline_ST();

    void createTileBoundaryPipeline_ST();

    void createRenderPipeline_ST();

    bool recordRenderCommandBuffer_ST(uint32_t currentFrame);

    void recordPreprocessCommandBuffer_ST();

};


#endif //RENDERER_H
