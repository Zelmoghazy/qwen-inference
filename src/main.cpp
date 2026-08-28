#include "tokenizer.hpp"
#include "gguf.hpp"
#include "engine.hpp"

extern "C"{
    #include "arena.h"
}

#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include "imgui/imgui_internal.h"

#include "GLAD/glad.h"
#include "GLFW/glfw3.h"

#include "nfd/nfd.h"

#define LUNASVG_BUILD_STATIC
#include "lunasvg/lunasvg.h"

extern "C"{
    __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;// Optimus: force switch to discrete GPU
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;//AMD
}

typedef struct 
{
    GLFWwindow *window;
    arena_t *frame_arena;
} ctx_t;

ctx_t gc;

static void glfw_error_callback(int error, const char *description) 
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

void imgui_dockspace(bool *p_open) 
{
    static bool opt_fullscreen = true;
    static bool opt_padding = false;
    static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
    if (opt_fullscreen) 
    {
        const ImGuiViewport *viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        ImGui::SetNextWindowViewport(viewport->ID);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        window_flags |= ImGuiWindowFlags_NoTitleBar |
                        ImGuiWindowFlags_NoCollapse |
                        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
        window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus |
                        ImGuiWindowFlags_NoNavFocus;
    } 
    else 
    {
        dockspace_flags &= ~ImGuiDockNodeFlags_PassthruCentralNode;
    }

    if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
        window_flags |= ImGuiWindowFlags_NoBackground;

    if (!opt_padding)
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("DockSpace Demo", p_open, window_flags);
    if (!opt_padding)
        ImGui::PopStyleVar();

    if (opt_fullscreen)
        ImGui::PopStyleVar(2);

    // Submit the DockSpace
    ImGuiIO &io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable) {
        ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
    }

    if (ImGui::BeginMenuBar()) 
    {
        if (ImGui::BeginMenu("Options")) {
            ImGui::MenuItem("Fullscreen", NULL, &opt_fullscreen);
            ImGui::MenuItem("Padding", NULL, &opt_padding);
            ImGui::Separator();

            if (ImGui::MenuItem(
                    "Flag: NoSplit", "",
                    (dockspace_flags & ImGuiDockNodeFlags_NoSplit) != 0)) {
                dockspace_flags ^= ImGuiDockNodeFlags_NoSplit;
            }
            if (ImGui::MenuItem(
                    "Flag: NoResize", "",
                    (dockspace_flags & ImGuiDockNodeFlags_NoResize) != 0)) {
                dockspace_flags ^= ImGuiDockNodeFlags_NoResize;
            }
            if (ImGui::MenuItem("Flag: NoDockingInCentralNode", "",
                                (dockspace_flags &
                                 ImGuiDockNodeFlags_NoDockingInCentralNode) !=
                                    0)) {
                dockspace_flags ^= ImGuiDockNodeFlags_NoDockingInCentralNode;
            }
            if (ImGui::MenuItem("Flag: AutoHideTabBar", "",
                                (dockspace_flags &
                                 ImGuiDockNodeFlags_AutoHideTabBar) != 0)) {
                dockspace_flags ^= ImGuiDockNodeFlags_AutoHideTabBar;
            }
            if (ImGui::MenuItem("Flag: PassthruCentralNode", "",
                                (dockspace_flags &
                                 ImGuiDockNodeFlags_PassthruCentralNode) != 0,
                                opt_fullscreen)) {
                dockspace_flags ^= ImGuiDockNodeFlags_PassthruCentralNode;
            }
            ImGui::Separator();

            if (ImGui::MenuItem("Close", NULL, false, p_open != NULL))
                *p_open = false;
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }
    ImGui::End();
}

GLFWwindow *create_window(int width, int height, const char *title) 
{
    glfwSetErrorCallback(glfw_error_callback);

    glfwInitHint(GLFW_WIN32_MESSAGES_IN_FIBER, GLFW_TRUE);

    if (!glfwInit())
        return NULL;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    float main_scale = ImGui_ImplGlfw_GetContentScaleForMonitor(glfwGetPrimaryMonitor());

    GLFWwindow *window = glfwCreateWindow((int) width * main_scale, (int) height * main_scale, title, nullptr, nullptr);

    if (window == nullptr){
        return NULL;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
        fprintf(stderr, "Failed to initialize GLAD\n");
        return NULL;
    }
    
    printf("Vendor:   %s\n",        glGetString(GL_VENDOR));
    printf("Renderer: %s\n",        glGetString(GL_RENDERER));

    return window;
}

void imgui_init(GLFWwindow *window) 
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();

    (void) io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    ImGui::StyleColorsDark();

    float main_scale = ImGui_ImplGlfw_GetContentScaleForMonitor(glfwGetPrimaryMonitor());
    ImGuiStyle &style = ImGui::GetStyle();
    style.ScaleAllSizes(main_scale);

    float font_size = 18.0f * main_scale;
    ImFontConfig base_cfg;
    base_cfg.SizePixels = font_size;
    io.Fonts->AddFontFromFileTTF("..\\assets\\fonts\\JetBrains.ttf", font_size, &base_cfg);

    style.FontScaleDpi = main_scale;
    io.ConfigDpiScaleFonts = true;
    io.ConfigDpiScaleViewports = true;

    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }
    const char *glsl_version = "#version 130";
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
}

void imgui_start_frame(void) 
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void imgui_end_frame(void) 
{ 
    ImGui::Render(); 
}

void imgui_draw(void) 
{
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void init_all(void) 
{
    gc.window = create_window(1280, 800, "Mnist");
    imgui_init(gc.window);
    NFD_Init();
}

void clear_screen(ImVec4 clear_color) 
{
    int display_w, display_h;
    glfwGetFramebufferSize(gc.window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w,
                 clear_color.z * clear_color.w, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT);
}

struct SvgViewer
{
    std::unique_ptr<lunasvg::Document> doc;
    GLuint texture = 0;
    int texWidth = 0, texHeight = 0;
    float docWidth = 0, docHeight = 0;

    char filePathBuf[512] = "..//test//architecture_card_template.svg";
    std::string loadError;

    float zoom = 1.0f;     
    ImVec2 pan = ImVec2(0, 0);
    const float kSuperSample = 5.0f; 

    void UploadTexture(const lunasvg::Bitmap& bitmap)
    {
        if (texture)
        {
            glDeleteTextures(1, &texture);
            texture = 0;
        }
    
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, bitmap.stride() / 4);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, bitmap.width(), bitmap.height(), 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, bitmap.data());
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
    
        texWidth = bitmap.width();
        texHeight = bitmap.height();
    }
    void LoadSvg(const std::string& path)
    {
        loadError.clear();
        doc.reset();
    
        auto loaded_doc = lunasvg::Document::loadFromFile(path);
        if (!loaded_doc)
        {
            loadError = "Failed to load: " + path;
            return;
        }
    
        docWidth  = (float)loaded_doc->width();
        docHeight = (float)loaded_doc->height();
        if (docWidth <= 0 || docHeight <= 0)
        {
            loadError = "SVG has no valid width/height: " + path;
            return;
        }
    
        int rw = (int)(docWidth  * kSuperSample);
        int rh = (int)(docHeight * kSuperSample);
    
        lunasvg::Bitmap bitmap = loaded_doc->renderToBitmap(rw, rh);
        if (bitmap.isNull())
        {
            loadError = "Failed to rasterize: " + path;
            return;
        }
        bitmap.convertToRGBA(); // un-premultiply + BGRA->RGBA for GL upload
    
        UploadTexture(bitmap);
        doc = std::move(loaded_doc);
    
        zoom = 1.0f;
        pan = ImVec2(0, 0);
    }
    void Draw()
    {
        ImGui::Begin("SVG Viewer");
        {
            ImGui::InputText("File path", filePathBuf, sizeof(filePathBuf));
            ImGui::SameLine();
            if (ImGui::Button("Load")){
                LoadSvg(filePathBuf);
            }
            if (!loadError.empty()){
                ImGui::TextColored(ImVec4(1, 0.3f, 0.3f, 1), "%s", loadError.c_str());
            }
        
            if (ImGui::Button("Reset view")){
                zoom = 1.0f;
                pan = ImVec2(0, 0);
            }
        
            ImGui::Separator();
        
            ImVec2 avail = ImGui::GetContentRegionAvail();
            if (avail.x < 1) avail.x = 1;
            if (avail.y < 1) avail.y = 1;
        
            ImGui::BeginChild("SvgCanvas", avail, true,
                            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        
                if (doc && texture)
                {
                    ImVec2 childPos = ImGui::GetCursorScreenPos();
                    // ImVec2 childSize = ImGui::GetWindowSize();
            
                    ImVec2 imgSize(docWidth * zoom, docHeight * zoom);
                    ImVec2 p0(childPos.x + pan.x, childPos.y + pan.y);
                    ImVec2 p1(p0.x + imgSize.x, p0.y + imgSize.y);
            
                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    dl->AddImage((ImTextureID)(intptr_t)texture, p0, p1);
            
                    bool hovered = ImGui::IsWindowHovered();
                    if (hovered)
                    {
                        float wheel = ImGui::GetIO().MouseWheel;
                        if (wheel != 0.0f)
                        {
                            ImVec2 mouse = ImGui::GetMousePos();
                            ImVec2 mouseLocal(mouse.x - childPos.x, mouse.y - childPos.y);
            
                            ImVec2 docPt(
                                (mouseLocal.x - pan.x) / zoom,
                                (mouseLocal.y - pan.y) / zoom);
            
                            float factor = (wheel > 0) ? 1.1f : (1.0f / 1.1f);
                            zoom = std::clamp(zoom * factor, 0.02f, 40.0f);
            
                            pan.x = mouseLocal.x - docPt.x * zoom;
                            pan.y = mouseLocal.y - docPt.y * zoom;
                        }
            
                        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
                        {
                            ImVec2 delta = ImGui::GetIO().MouseDelta;
                            pan.x += delta.x;
                            pan.y += delta.y;
                        }
                    }
                }
                else
                {
                    ImGui::TextDisabled("preview.");
                }
        
            ImGui::EndChild();
        }
        ImGui::End();
    }
};

struct GGUFLogView
{
    const std::vector<MetaEntry>                        &metaEntries;
    const std::vector<TensorEntry>                      &topLevelTensors;
    const std::map<uint32_t, std::vector<TensorEntry>>  &blocks;
    const std::string                                   &architecture;

    mutable const MetaEntry* selectedEntry = nullptr;

    GGUFLogView(const std::vector<MetaEntry> &metaEntries_,
                const std::vector<TensorEntry> &topLevelTensors_,
                const std::map<uint32_t, std::vector<TensorEntry>> &blocks_,
                const std::string &architecture_)
        : metaEntries(metaEntries_)
        , topLevelTensors(topLevelTensors_)
        , blocks(blocks_)
        , architecture(architecture_)
    {}

    void Draw() const
    {
        ImGui::Begin("GGUF Viewer");
        {
        if (ImGui::CollapsingHeader("General", ImGuiTreeNodeFlags_DefaultOpen))
            DrawMetaTable("##general_table", "general.");

        if (ImGui::CollapsingHeader("Model", ImGuiTreeNodeFlags_DefaultOpen))
            DrawModelTable();

        if (ImGui::CollapsingHeader("Tensors", ImGuiTreeNodeFlags_DefaultOpen))
            DrawTensorsSection();

        DrawValuePopup();
        }
        ImGui::End();
    }

private:
    static constexpr std::string_view kTokenizerPrefix = "tokenizer.";

    void DrawMetaTable(const char* id, const std::string& prefix) const
    {
        std::vector<const MetaEntry*> rows;
        for (auto& e : metaEntries)
            if (e.key.compare(0, prefix.size(), prefix) == 0)
                rows.push_back(&e);
        DrawMetaRows(id, rows);
    }

    void DrawModelTable() const
    {
        std::vector<const MetaEntry*> rows;
        const std::string archPrefix = architecture.empty() ? std::string() : (architecture + ".");
        for (auto& e : metaEntries)
        {
            if (e.key.compare(0, kTokenizerPrefix.size(), kTokenizerPrefix) == 0) rows.push_back(&e);
            else if (!archPrefix.empty() &&
                     e.key.compare(0, archPrefix.size(), archPrefix) == 0) rows.push_back(&e);
        }
        DrawMetaRows("##model_table", rows);
    }

    void DrawMetaRows(const char* id, const std::vector<const MetaEntry*>& rows) const
    {
        if (rows.empty()) { ImGui::TextDisabled("(no entries)"); return; }

        ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY;
        float h = std::min(400.0f, ImGui::GetTextLineHeightWithSpacing() * (rows.size() + 1.5f));

        if (ImGui::BeginTable(id, 4, flags, ImVec2(0, h)))
        {
            ImGui::TableSetupColumn("#",     ImGuiTableColumnFlags_WidthFixed, 30.0f);
            ImGui::TableSetupColumn("Key",   ImGuiTableColumnFlags_WidthFixed, 260.0f);
            ImGui::TableSetupColumn("Type",  ImGuiTableColumnFlags_WidthFixed, 80.0f);
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableHeadersRow();

            for (const MetaEntry* ePtr : rows)
            {
                const MetaEntry& e = *ePtr;
                ImGui::PushID((int)e.idx);
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0); ImGui::Text("%u", e.idx);
                ImGui::TableSetColumnIndex(1); ImGui::TextUnformatted(e.key.c_str());
                ImGui::TableSetColumnIndex(2); ImGui::TextUnformatted(e.type.c_str());

                ImGui::TableSetColumnIndex(3);
                static constexpr size_t kLongValueThreshold = 80;

                bool is_large_array = (e.type == "array") && (e.value.size() > kLongValueThreshold);

                if (is_large_array)
                {
                    std::string preview = e.value.substr(0, kLongValueThreshold);
                    preview += " ...";
                    if (ImGui::Selectable(preview.c_str(), false, ImGuiSelectableFlags_None))
                    {
                        selectedEntry = &e;
                        ImGui::OpenPopup("ValuePopup");
                    }
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Click to view full value");
                }
                else
                {
                    ImGui::TextUnformatted(e.value.c_str());
                }
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
    }

    void DrawValuePopup() const
    {
        ImGui::SetNextWindowSize(ImVec2(650, 450), ImGuiCond_FirstUseEver);
        if (ImGui::BeginPopup("ValuePopup"))
        {
            if (selectedEntry)
            {
                ImGui::TextUnformatted(selectedEntry->key.c_str());
                ImGui::Separator();
                ImGui::BeginChild("value_scroll", ImVec2(0, -ImGui::GetFrameHeightWithSpacing()),
                                  true, ImGuiWindowFlags_HorizontalScrollbar);
                ImGui::TextUnformatted(selectedEntry->value.c_str());
                ImGui::EndChild();
                if (ImGui::Button("Copy"))
                    ImGui::SetClipboardText(selectedEntry->value.c_str());
            }
            ImGui::EndPopup();
        }
    }

    void DrawTensorsSection() const
    {
        if (!topLevelTensors.empty())
        {
            ImGui::TextDisabled("Top-level");
            DrawTensorTable("##top_level_tensors", topLevelTensors);
        }

        for (auto& kv : blocks)
        {
            uint32_t blkIdx = kv.first;
            const std::vector<TensorEntry>& tensors = kv.second;

            std::string label = "blk." + std::to_string(blkIdx) +
                                "  (" + std::to_string(tensors.size()) + " tensors)";
            ImGui::PushID((int)blkIdx);
            if (ImGui::TreeNode(label.c_str()))
            {
                DrawTensorTable("##blk_tensors", tensors);
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
    }

    void DrawTensorTable(const char* id, const std::vector<TensorEntry>& tensors) const
    {
        if (tensors.empty()) { ImGui::TextDisabled("(no tensors)"); return; }

        ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY;
        float h = std::min(320.0f, ImGui::GetTextLineHeightWithSpacing() * (tensors.size() + 1.5f));

        if (ImGui::BeginTable(id, 5, flags, ImVec2(0, h)))
        {
            ImGui::TableSetupColumn("#",      ImGuiTableColumnFlags_WidthFixed, 40.0f);
            ImGui::TableSetupColumn("Name",   ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Dims",    ImGuiTableColumnFlags_WidthFixed, 150.0f);
            ImGui::TableSetupColumn("Type",    ImGuiTableColumnFlags_WidthFixed, 70.0f);
            ImGui::TableSetupColumn("Offset",  ImGuiTableColumnFlags_WidthFixed, 110.0f);
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableHeadersRow();

            for (const TensorEntry& t : tensors)
            {
                ImGui::PushID((int)t.idx);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::Text("%u", t.idx);
                ImGui::TableSetColumnIndex(1); ImGui::TextUnformatted(t.name.c_str());
                ImGui::TableSetColumnIndex(2); ImGui::TextUnformatted(t.dims.c_str());
                ImGui::TableSetColumnIndex(3); ImGui::TextUnformatted(t.type.c_str());
                ImGui::TableSetColumnIndex(4); ImGui::TextUnformatted(t.offset.c_str());
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
    }
};

struct matmul_chunk_t 
{
    const block_q8_0 *weight;
    u32 blocks_per_row;
    u32 row_start;
    u32 row_end;
    const f32 *input;
    u32 n_in;
    u32 n_seq;
    f32 *output;
    u32 n_out;
};

#define BLOCK_IN_MAT(base, row, num_cols, block, block_size) \
    ((base) + (u64)(row) * (num_cols) + (u64)(block) * (block_size))
    
#define CEIL_DIV(x, y)                  (((x) + (y) - 1) / (y))

void matmul_chunk_job(void *data)
{
    ZoneScopedNC("Matmul Chunk", tracy::Color::Orange);

    matmul_chunk_t *c = (matmul_chunk_t*)data;

    static thread_local std::vector<f32x8> acc;
    acc.resize(c->n_seq);

    for (u32 row = c->row_start; row < c->row_end; row++)
    {
        const block_q8_0 *row_blocks = c->weight + (u64)row * c->blocks_per_row;
        for (u32 tok = 0; tok < c->n_seq; tok++){
            acc[tok] = f32x8_zero();
        }

        for (u32 block = 0; block < c->blocks_per_row; block++)
        {
            const f32   delta   = ggml_compute_fp16_to_fp32(row_blocks[block].d);
            const f32x8 delta_v = f32x8_set1(delta);

            const int8_t *qs = row_blocks[block].qs;
            
            /* load the entire block of weights */
            f32x8 q0 = f32x8_load_i8(qs +  0);
            f32x8 q1 = f32x8_load_i8(qs +  8);
            f32x8 q2 = f32x8_load_i8(qs + 16);
            f32x8 q3 = f32x8_load_i8(qs + 24);

            for (u32 tok = 0; tok < c->n_seq; tok++)
            {
                const f32 *xb = BLOCK_IN_MAT(c->input, tok, c->n_in, block, 32);

                f32x8 x0 = f32x8_load(xb +  0);
                f32x8 x1 = f32x8_load(xb +  8);
                f32x8 x2 = f32x8_load(xb + 16);
                f32x8 x3 = f32x8_load(xb + 24);

                f32x8 b = f32x8_zero();
                b = f32x8_madd(q0, x0, b);
                b = f32x8_madd(q1, x1, b);
                b = f32x8_madd(q2, x2, b);
                b = f32x8_madd(q3, x3, b);

                acc[tok] = f32x8_madd(b, delta_v, acc[tok]);
            }
        }

        for (u32 tok = 0; tok < c->n_seq; tok++){
            c->output[(u64)tok * c->n_out + row] = f32x8_sum(acc[tok]);
        }
    }
}

void matmul_q8_0_threaded(thread_pool_t *pool, const block_q8_0 *weight,
                           u32 n_in, u32 n_out,const f32 *input,u32 n_seq, f32 *output)
{
    ZoneScopedNC("Matrix Multiplication Multithreaded", tracy::Color::Tomato);
    TracySection matmul_prep("Preparing");

    u32 blocks_per_row = n_in / 32;

    u32 num_threads = (u32)pool->threads.size();

    const u32 chunks_per_thread = 8;
    u32 num_chunks = num_threads * chunks_per_thread;
    u32 rows_per_chunk = CEIL_DIV(n_out, num_chunks);

    static thread_local std::vector<matmul_chunk_t> chunks;
    static thread_local std::vector<job_t> jobs;

    chunks.clear();
    jobs.clear();
    
    chunks.reserve(num_chunks);
    jobs.reserve(num_chunks);

    for (u32 t = 0; t < num_chunks; t++)
    {
        u32 start = t * rows_per_chunk;
        u32 end   = std::min(start + rows_per_chunk, n_out);
        if (start >= end) 
            break;

        chunks.push_back({ weight, blocks_per_row, start, end, input, n_in, n_seq,output,n_out });
        jobs.push_back({ matmul_chunk_job, &chunks.back() });
    }
    matmul_prep.Leave();
    threadpool_queue_jobs_batch(pool, jobs.data(), (int)jobs.size());
    threadpool_wait(pool);
}

void silu(const f32 *input, f32 *output, u32 size)
{
    for (u32 i = 0; i < size; i++)
    {
        f32 v = input[i];
        output[i] = v / (1.0f + expf(-v));
    }
}

/* @Note: norm weights are already f32 no need to dequant */
void rmsnorm(const f32 *x, const f32 *weight, f32 *out, u32 n, f32 eps)
{
    f32 ss = 0.0f;
    for (u32 i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }
    f32 scale = 1.0f / sqrtf(ss / n + eps);
    for (u32 i = 0; i < n; i++) {
        out[i] = x[i] * scale * weight[i];
    }
}


// rotate half instead of each pair from HF code 
void rope(f32 *vec, u32 head_dim, u32 pos, f32 theta_base)
{
    u32 half = head_dim / 2;
    for (u32 i = 0; i < half; i++)
    {
        f32 freq        = 1.0f / powf(theta_base, (2.0f * i) / head_dim);
        f32 angle       = pos * freq;
        f32 c           = cosf(angle);
        f32 s           = sinf(angle);
        f32 x0          = vec[i];
        f32 x1          = vec[i + half];
        vec[i]          = x0 * c - x1 * s;
        vec[i + half]   = x0 * s + x1 * c;
    }
}

f32 vec_dot_f32(const f32 *a, const f32 *b, u32 n)
{
    f32 sum = 0.0f;
    for (u32 i = 0; i < n; i++){
        sum += a[i] * b[i];
    } 
    return sum;
}

void softmax_f32(const f32 *input, f32 *output, u32 n)
{
    // softmax is invariant to subtraction of a constant
    f32 max_val = input[0];
    for (u32 i = 1; i < n; i++){
        if (input[i] > max_val){
            max_val = input[i];
        } 
    } 
    // subtracting x_max from all x_i ensures it doesnt overflow
    f32 sum = 0.0f;
    for (u32 i = 0; i < n; i++)
    {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    for (u32 i = 0; i < n; i++){
        output[i] /= sum;
    } 
}

/* Just take the highest prob, very repetitive */
u32 argmax(const f32 *logits, u32 n)
{
    u32 best = 0;
    f32 best_val = logits[0];
    for (u32 i = 1; i < n; i++)
    {
        if (logits[i] > best_val) { 
            best_val = logits[i]; 
            best = i; 
        }
    }
    return best;
}

std::string build_chat_prompt(const std::string &system_prompt, const std::string &user_prompt)
{
    std::string p;
    p += "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
    p += "<|im_start|>user\n"   + user_prompt   + "<|im_end|>\n";
    p += "<|im_start|>assistant\n";   
    return p;
}


std::vector<MetaEntry>                       metaEntries;
std::vector<TensorEntry>                     topLevelTensors;
std::map<uint32_t, std::vector<TensorEntry>> blocks;
std::string                                  architecture;

struct ModelContext
{
    ModelInfo     *model;
    thread_pool_t *pool;
    arena_t       *scratch_arena;

    u32 d_model;
    u32 n_layers;
    u32 head_dim;
    u32 kv_dim;
    u32 gqa_group_size;
    u32 max_seq;
    u32 ffn_dim;
    u32 vocab;
    f32 scale;

    f32 *normed_attn;
    
    f32 *q_full;
    f32 *k_full;
    f32 *v_full;

    f32 *kv_cache_k;
    f32 *kv_cache_v;
    
    f32 *o_full;
    
    f32 *attn_scores;
    f32 *gate;
    f32 *up;
    f32 *logits;
    
    f32 *attn_out;
    
    f32 *residual;
    f32 *normed_ffn;
    f32 *ffn_out;
    f32 *normed_final;

    f32 *x;

    ModelContext(ModelInfo &model,thread_pool_t *pool, arena_t *scratch_arena)
    {
        this->model = &model;
        this->pool = pool;
        this->scratch_arena = scratch_arena;

        this->d_model         = model.cfg.embedding_length;                                         // 2048
        this->n_layers        = model.cfg.block_count;                                              // 36
        this->head_dim        = model.cfg.embedding_length / model.cfg.attention_head_count;        // dk == dv
        this->kv_dim          = model.cfg.attention_head_count_kv * head_dim;                       // 256
        this->gqa_group_size  = model.cfg.attention_head_count / model.cfg.attention_head_count_kv; // 8
        this->max_seq         = model.cfg.context_length;
        this->ffn_dim         = model.cfg.feed_forward_length;
        this->vocab           = model.token_embd.dims[1];
        this->scale           = 1.0f / sqrtf((f32)head_dim);  // 1 / sqrt(d_k)
        
        this->normed_attn    = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    
        
        this->q_full         = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    
        this->k_full         = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * kv_dim);    
        this->v_full         = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * kv_dim);

        this->kv_cache_k     = (f32*)ARENA_ALLOC(scratch_arena,sizeof(f32) * n_layers * max_seq * kv_dim);
        this->kv_cache_v     = (f32*)ARENA_ALLOC(scratch_arena,sizeof(f32) * n_layers * max_seq * kv_dim);
        memset(kv_cache_k, 0, sizeof(f32) * n_layers * max_seq * kv_dim);
        memset(kv_cache_v, 0, sizeof(f32) * n_layers * max_seq * kv_dim);
        
        this->o_full         = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    
        
        this->attn_scores    = (f32*)ARENA_ALLOC(scratch_arena,sizeof(f32) * max_seq);
        this->gate           = (f32*)ARENA_ALLOC(scratch_arena,sizeof(f32) * ffn_dim);
        this->up             = (f32*)ARENA_ALLOC(scratch_arena,sizeof(f32) * ffn_dim);
        this->logits         = (f32*)ARENA_ALLOC(scratch_arena,sizeof(f32) * vocab);
        
        
        this->attn_out       = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    
        
        this->residual       = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    
        this->normed_ffn     = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    
        this->ffn_out        = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    
        this->normed_final   = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    

        this->x              = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    
    }
};

#if 0

int main(void)
{
    TracySection init("Initialization");
        HANDLE File = CreateFileA("C:\\Users\\zezo_\\.lmstudio\\models\\lmstudio-community"
                                "\\Qwen2.5-3B-Instruct-GGUF\\Qwen2.5-3B-Instruct-Q8_0.gguf",
                                GENERIC_READ, FILE_SHARE_READ|FILE_SHARE_WRITE,
                                0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
        LARGE_INTEGER size;
        GetFileSizeEx(File, &size);

        HANDLE Mapping = CreateFileMappingA(File, 0, PAGE_READONLY, 0, 0, 0);
        u8 *file_base = (u8 *)MapViewOfFile(Mapping, FILE_MAP_READ, 0, 0, 0);
        {
            WIN32_MEMORY_RANGE_ENTRY range;
            range.VirtualAddress = file_base;
            range.NumberOfBytes  = (SIZE_T)size.QuadPart;

            if (!PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
                std::println("PrefetchVirtualMemory failed: {}", GetLastError());
            }
        }
        
        Data gguf(file_base, static_cast<i64>(size.QuadPart));

        arena_t *scratch_arena = arena_reserve(MB(500));// check if enough
        thread_pool_t *pool = threadpool_create();

        init_tokenizer(); 
    init.Leave();

    TracySection parse("Parsing GGUF");
        ModelInfo model;
        parse_gguf(gguf, model, size.QuadPart, file_base, metaEntries, topLevelTensors, blocks, architecture);
        ModelContext ctx(model,pool,scratch_arena);
    parse.Leave();
        
    TracySection tokenization("Prompt Tokenization");
        std::string prompt = build_chat_prompt(
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            "Generate a python script that generates that plots data from an excel file."
        );
        u32 max_new_tokens = 10000;

        std::vector<int> tokens;
        tokens.reserve(prompt.size() + max_new_tokens);  
        tokens = encode(prompt);

        std::println("{}",tokens);

        for(int token: tokens){
            std::println("{}",decode_id(token));    
        }    

        u32 prompt_len = (u32)tokens.size();
    tokenization.Leave();

    {
        TracySection bench("Bandwidth Benchmark");
        u8 *first_tensor_ptr = (u8*)model.blocks[0].attn_norm.tensor_data;
        size_t first_tensor_offset = (size_t)(first_tensor_ptr - file_base);
        size_t weights_bytes = (size_t)size.QuadPart - first_tensor_offset;
        u32 n_threads = 6; 
        bandwidth_benchmark_suite(file_base + first_tensor_offset, weights_bytes, n_threads, 10);
    }

    TracySection prefill("Prefill");
    {
        arena_checkpoint_t *cp = arena_save(scratch_arena);

        u32 n_tok               = prompt_len;
        // seq * d_model
        f32 *x_batch            = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * n_tok * ctx.d_model);
        f32 *normed_attn_batch  = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * n_tok * ctx.d_model);
        // seq * dk * h
        f32 *q_batch            = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * n_tok * ctx.d_model);
        // seq * dk * kv_h
        f32 *k_batch            = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * n_tok * ctx.kv_dim);
        // seq * dv * kv_h
        f32 *v_batch            = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * n_tok * ctx.kv_dim);
        // seq * dv * h
        f32 *o_batch            = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * n_tok * ctx.d_model);

        f32 *attn_out_batch     = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * n_tok * ctx.d_model);
        f32 *residual_batch     = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * n_tok * ctx.d_model);
        f32 *normed_ffn_batch   = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * n_tok * ctx.d_model);
        f32 *gate_batch         = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * n_tok * ctx.ffn_dim);
        f32 *up_batch           = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * n_tok * ctx.ffn_dim);
        f32 *ffn_out_batch      = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * n_tok * ctx.d_model);

        for (u32 i = 0; i < n_tok; i++){
            embed_token(model.token_embd, tokens[i], x_batch + (u64)i*ctx.d_model, ctx.d_model);
        }

        for (u32 layer_idx = 0; layer_idx < ctx.n_layers; layer_idx++)
        {
            BlockInfo &l = model.blocks[layer_idx];

            TracySection RMSNorm1("Prefill RMSNorm1");
                for (u32 i = 0; i < n_tok; i++){
                    rmsnorm(x_batch + (u64)i*ctx.d_model, (f32*)l.attn_norm.tensor_data, normed_attn_batch + (u64)i*ctx.d_model, ctx.d_model, model.cfg.rms_epsilon);
                }
            RMSNorm1.Leave();

            TracySection attention("Prefill Attention");
                matmul_q8_0_threaded(pool, (block_q8_0*)l.attn_q.tensor_data, l.attn_q.dims[0], l.attn_q.dims[1], normed_attn_batch, n_tok, q_batch);
                matmul_q8_0_threaded(pool, (block_q8_0*)l.attn_k.tensor_data, l.attn_k.dims[0], l.attn_k.dims[1], normed_attn_batch, n_tok, k_batch);
                matmul_q8_0_threaded(pool, (block_q8_0*)l.attn_v.tensor_data, l.attn_v.dims[0], l.attn_v.dims[1], normed_attn_batch, n_tok, v_batch);

                for (u32 i = 0; i < n_tok; i++)
                {
                    f32 *qi = q_batch + (u64)i*ctx.d_model;
                    for (u32 c = 0; c < ctx.d_model; c++){
                        qi[c] += ((f32*)l.attn_q_bias.tensor_data)[c];
                    }
                    f32 *ki = k_batch + (u64)i*ctx.kv_dim;
                    for (u32 c = 0; c < ctx.kv_dim; c++) {
                        ki[c] += ((f32*)l.attn_k_bias.tensor_data)[c];
                    }

                    f32 *vi = v_batch + (u64)i*ctx.kv_dim;
                    for (u32 c = 0; c < ctx.kv_dim; c++){
                        vi[c] += ((f32*)l.attn_v_bias.tensor_data)[c];
                    }

                    for (u32 h = 0; h < model.cfg.attention_head_count; h++){
                        rope(qi + h*ctx.head_dim, ctx.head_dim, i, model.cfg.rope_freq);
                    }
                    for (u32 h = 0; h < model.cfg.attention_head_count_kv; h++){
                        rope(ki + h*ctx.head_dim, ctx.head_dim, i, model.cfg.rope_freq);
                    }
                    
                    f32 *k_cache_slot = KV_K_AT(ctx, layer_idx, i);
                    f32 *v_cache_slot = KV_V_AT(ctx, layer_idx, i);
                    memcpy(k_cache_slot, ki, sizeof(f32) * ctx.kv_dim);
                    memcpy(v_cache_slot, vi, sizeof(f32) * ctx.kv_dim);
                }
                {
                    ZoneScopedN("Prefill Attention Heads");
                    for (u32 i = 0; i < n_tok; i++)
                    {
                        f32 *qi = q_batch + (u64)i*ctx.d_model;
                        f32 *oi = o_batch + (u64)i*ctx.d_model;

                        for (u32 h = 0; h < model.cfg.attention_head_count; h++)
                        {
                            u32 kv = h / ctx.gqa_group_size;
                            f32 *qh = qi + h*ctx.head_dim;

                            for (u32 t = 0; t <= i; t++)
                            {
                                f32 *kh_t =  KV_HEAD_K(ctx, layer_idx, t, kv);
                                ctx.attn_scores[t] = vec_dot_f32(qh, kh_t, ctx.head_dim) * ctx.scale;
                            }
                            softmax_f32(ctx.attn_scores, ctx.attn_scores, i + 1);

                            f32 *oh = oi + h*ctx.head_dim;
                            for (u32 c = 0; c < ctx.head_dim; c++) oh[c] = 0.0f;
                            for (u32 t = 0; t <= i; t++)
                            {
                                f32 *vh_t = KV_HEAD_V(ctx, layer_idx, t, kv);
                                f32 w = ctx.attn_scores[t];
                                for (u32 c = 0; c < ctx.head_dim; c++){
                                    oh[c] += w * vh_t[c];
                                }
                            }
                        }
                    }
                    matmul_q8_0_threaded(pool, (block_q8_0*)l.attn_output.tensor_data, l.attn_output.dims[0], l.attn_output.dims[1], o_batch, n_tok, attn_out_batch);

                    for (u32 i = 0; i < n_tok; i++)
                    {
                        f32 *xi = x_batch + (u64)i*ctx.d_model;
                        f32 *ai = attn_out_batch + (u64)i*ctx.d_model;
                        f32 *ri = residual_batch + (u64)i*ctx.d_model;
                        for (u32 c = 0; c < ctx.d_model; c++) ri[c] = xi[c] + ai[c];
                    }
                }
            attention.Leave();

            TracySection ffn("Prefill FFN");
                for (u32 i = 0; i < n_tok; i++){
                    rmsnorm(residual_batch + (u64)i*ctx.d_model, (f32*)l.ffn_norm.tensor_data, normed_ffn_batch + (u64)i*ctx.d_model, ctx.d_model, model.cfg.rms_epsilon);
                }

                matmul_q8_0_threaded(pool, (block_q8_0*)l.ffn_gate.tensor_data, l.ffn_gate.dims[0], l.ffn_gate.dims[1], normed_ffn_batch, n_tok, gate_batch);
                matmul_q8_0_threaded(pool, (block_q8_0*)l.ffn_up.tensor_data,   l.ffn_up.dims[0],   l.ffn_up.dims[1],   normed_ffn_batch, n_tok, up_batch);

                for (u32 i = 0; i < n_tok; i++)
                {
                    f32 *gi = gate_batch + (u64)i*ctx.ffn_dim;
                    f32 *ui = up_batch   + (u64)i*ctx.ffn_dim;
                    silu(gi, gi, ctx.ffn_dim);
                    for (u32 c = 0; c < ctx.ffn_dim; c++){
                        gi[c] *= ui[c];
                    }
                }

                matmul_q8_0_threaded(pool, (block_q8_0*)l.ffn_down.tensor_data, l.ffn_down.dims[0], l.ffn_down.dims[1], gate_batch, n_tok, ffn_out_batch);

                for (u32 i = 0; i < n_tok; i++)
                {
                    f32 *ri = residual_batch + (u64)i*ctx.d_model;
                    f32 *fi = ffn_out_batch + (u64)i*ctx.d_model;
                    f32 *xi = x_batch + (u64)i*ctx.d_model;
                    for (u32 c = 0; c < ctx.d_model; c++) xi[c] = ri[c] + fi[c];
                }
            ffn.Leave();
        }  
        rmsnorm(x_batch + (u64)(n_tok-1)*ctx.d_model, (f32*)model.output_norm.tensor_data, ctx.normed_final, ctx.d_model, model.cfg.rms_epsilon);
        matmul_q8_0_threaded(pool, (block_q8_0*)model.token_embd.tensor_data, ctx.d_model, ctx.vocab, ctx.normed_final, 1,ctx.logits);
        u32 next_token = argmax(ctx.logits, ctx.vocab);
        std::print("{}", decode_id(next_token));
        tokens.push_back(next_token);
        arena_restore(scratch_arena, cp);
    }
    prefill.Leave();

    TracySection decode("decode");
        for(u32 cur_pos = prompt_len; cur_pos < prompt_len + max_new_tokens; cur_pos++)
        {
            int tok = (cur_pos < prompt_len) ? tokens[cur_pos] : tokens.back();
            embed_token(model.token_embd, tok, ctx.x, model.cfg.embedding_length);

            for (u32 layer_idx = 0; layer_idx < ctx.n_layers; layer_idx++)
            {
                BlockInfo &l = model.blocks[layer_idx];

                /* ------------------------------------- */
                TracySection RMSNorm1("RMSNorm1");
                    rmsnorm(ctx.x, (f32*)l.attn_norm.tensor_data, ctx.normed_attn, ctx.d_model, model.cfg.rms_epsilon);
                RMSNorm1.Leave();
                /* ------------------------------------- */


                /* ------------------------------------- */
                TracySection attention("Attention");
                    
                    matmul_q8_0_threaded(pool,(block_q8_0*)l.attn_q.tensor_data, l.attn_q.dims[0], l.attn_q.dims[1], ctx.normed_attn, 1,ctx.q_full);
                    for (u32 i = 0; i < ctx.d_model; i++) {
                        ctx.q_full[i] += ((f32*)l.attn_q_bias.tensor_data)[i];
                    }

                    mat_vec_mul_q8_0((block_q8_0*)l.attn_k.tensor_data, l.attn_k.dims[0], l.attn_k.dims[1], ctx.normed_attn, ctx.k_full);
                    for (u32 i = 0; i < ctx.kv_dim; i++){
                        ctx.k_full[i] += ((f32*)l.attn_k_bias.tensor_data)[i];
                    }

                    mat_vec_mul_q8_0((block_q8_0*)l.attn_v.tensor_data, l.attn_v.dims[0], l.attn_v.dims[1],  ctx.normed_attn, ctx.v_full);
                    for (u32 i = 0; i < ctx.kv_dim; i++){
                        ctx.v_full[i] += ((f32*)l.attn_v_bias.tensor_data)[i];
                    }
                    
                    {
                        ZoneScopedN("RoPE");
                        for (u32 h = 0; h < model.cfg.attention_head_count; h++){
                            rope(ctx.q_full + h*ctx.head_dim, ctx.head_dim, cur_pos, model.cfg.rope_freq);
                        }
                        for (u32 h = 0; h < model.cfg.attention_head_count_kv; h++){
                            rope(ctx.k_full + h*ctx.head_dim, ctx.head_dim, cur_pos, model.cfg.rope_freq);
                        }
                    }

                    f32 *k_cache_slot = ctx.kv_cache_k + (layer_idx * ctx.max_seq + cur_pos) * ctx.kv_dim;
                    f32 *v_cache_slot = ctx.kv_cache_v + (layer_idx * ctx.max_seq + cur_pos) * ctx.kv_dim;
                    memcpy(k_cache_slot, ctx.k_full, sizeof(f32) * ctx.kv_dim);
                    memcpy(v_cache_slot, ctx.v_full, sizeof(f32) * ctx.kv_dim);
                
                    {
                        ZoneScopedN("Attention Heads");
                        for (u32 h = 0; h < model.cfg.attention_head_count; h++)
                        {
                            u32 kv = h / ctx.gqa_group_size;
                            f32 *qh = ctx.q_full + h*ctx.head_dim;
        
                            for (u32 t = 0; t <= cur_pos; t++)
                            {
                                f32 *kh_t =  KV_HEAD_K(ctx, layer_idx, t, kv);
                                ctx.attn_scores[t] = vec_dot_f32(qh, kh_t, ctx.head_dim) * ctx.scale;
                            }
                            softmax_f32(ctx.attn_scores, ctx.attn_scores, cur_pos + 1);
        
                            f32 *oh = ctx.o_full + h * ctx.head_dim;
                            for (u32 i = 0; i < ctx.head_dim; i++){
                                oh[i] = 0.0f;
                            } 
                            for (u32 t = 0; t <= cur_pos; t++)
                            {
                                f32 *vh_t = KV_HEAD_V(ctx, layer_idx, t, kv);
                                f32 w = ctx.attn_scores[t];
                                for (u32 i = 0; i < ctx.head_dim; i++){
                                    oh[i] += w * vh_t[i];
                                }
                            }
                        }
                    }
                    
                    matmul_q8_0_threaded(pool,(block_q8_0*)l.attn_output.tensor_data, l.attn_output.dims[0], l.attn_output.dims[1], ctx.o_full, 1,ctx.attn_out);
                    
                    // update embeddings
                    for (u32 i = 0; i < ctx.d_model; i++){
                        ctx.residual[i] = ctx.x[i] + ctx.attn_out[i]; 
                    }
                attention.Leave();
                /* ------------------------------------- */
                
                /* ------------------------------------- */
                TracySection ffn("FFN");

                    rmsnorm(ctx.residual, (f32*)l.ffn_norm.tensor_data, ctx.normed_ffn, ctx.d_model, model.cfg.rms_epsilon);

                    matmul_q8_0_threaded(pool,(block_q8_0*)l.ffn_gate.tensor_data, l.ffn_gate.dims[0], l.ffn_gate.dims[1], ctx.normed_ffn, 1,ctx.gate);
                    matmul_q8_0_threaded(pool,(block_q8_0*)l.ffn_up.tensor_data,   l.ffn_up.dims[0],   l.ffn_up.dims[1],   ctx.normed_ffn, 1,ctx.up);

                    silu(ctx.gate, ctx.gate, ctx.ffn_dim);
                    for (u32 i = 0; i < ctx.ffn_dim; i++){
                        ctx.gate[i] *= ctx.up[i];
                    }

                    matmul_q8_0_threaded(pool,(block_q8_0*)l.ffn_down.tensor_data, l.ffn_down.dims[0], l.ffn_down.dims[1], ctx.gate, 1,ctx.ffn_out);

                    for (u32 i = 0; i < ctx.d_model; i++){
                        ctx.x[i] = ctx.residual[i] + ctx.ffn_out[i];
                    }

                ffn.Leave();
                /* ------------------------------------- */
            }
        
            /* ------------------------------------- */
            TracySection post_norm("Final RMSNorm");
                if (cur_pos >= prompt_len - 1)
                {
                    rmsnorm(ctx.x, (f32*)model.output_norm.tensor_data, ctx.normed_final, ctx.d_model, model.cfg.rms_epsilon);
                    matmul_q8_0_threaded(pool,(block_q8_0*)model.token_embd.tensor_data, ctx.d_model, ctx.vocab, ctx.normed_final,1, ctx.logits);
                    u32 next_token = argmax(ctx.logits, ctx.vocab);
                    if (next_token == 151645 || next_token == 151643){
                        break;
                    }
                    std::print("{}", decode_id(next_token));
                    tokens.push_back(next_token);
                }
            post_norm.Leave();
            /* ------------------------------------- */

            FrameMarkNamed("Token");
        }
    decode.Leave();
    std::println("");
    threadpool_destroy(pool);
    UnmapViewOfFile(file_base); 
    CloseHandle(Mapping);
    CloseHandle(File);
    return 0;
}
#endif


#if 1
int main(void)
{
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    init_all();
    ImGuiIO &io = ImGui::GetIO();

    TracySection init("Initialization");
        HANDLE File = CreateFileA("C:\\Users\\zezo_\\.lmstudio\\models\\lmstudio-community"
                                "\\Qwen2.5-3B-Instruct-GGUF\\Qwen2.5-3B-Instruct-Q8_0.gguf",
                                GENERIC_READ, FILE_SHARE_READ|FILE_SHARE_WRITE,
                                0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
        LARGE_INTEGER size;
        GetFileSizeEx(File, &size);

        HANDLE Mapping = CreateFileMappingA(File, 0, PAGE_READONLY, 0, 0, 0);
        u8 *file_base = (u8 *)MapViewOfFile(Mapping, FILE_MAP_READ, 0, 0, 0);
        {
            WIN32_MEMORY_RANGE_ENTRY range;
            range.VirtualAddress = file_base;
            range.NumberOfBytes  = (SIZE_T)size.QuadPart;

            if (!PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
                std::println("PrefetchVirtualMemory failed: {}", GetLastError());
            }
        }
        
        Data gguf(file_base, static_cast<i64>(size.QuadPart));

        arena_t *scratch_arena = arena_reserve(MB(500));// check if enough
        thread_pool_t *pool = threadpool_create();

        init_tokenizer(); 
    init.Leave();

    TracySection parse("Parsing GGUF");
        ModelInfo model;
        parse_gguf(gguf, model, size.QuadPart, file_base, metaEntries, topLevelTensors, blocks, architecture);
        ModelContext ctx(model,pool,scratch_arena);
    parse.Leave();
        

    gc.frame_arena = arena_reserve(128 * 1024 * 1024);
    SvgViewer viewer;
    GGUFLogView log_viewer(metaEntries,topLevelTensors,blocks,architecture);

    while (!glfwWindowShouldClose(gc.window)) 
    {
        glfwPollEvents();

        if (glfwGetWindowAttrib(gc.window, GLFW_ICONIFIED) != 0) 
        {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }
        imgui_start_frame();
        {
            ImGuiViewport *vp = ImGui::GetMainViewport();
            ImGui::SetNextWindowPos(vp->WorkPos);
            ImGui::SetNextWindowSize(vp->WorkSize);
            imgui_dockspace(nullptr);
            ImGui::ShowDemoWindow();
            viewer.Draw();
            log_viewer.Draw();

        }
        clear_screen(clear_color);
        imgui_end_frame();
        imgui_draw();

        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) 
        {
            GLFWwindow *backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }

        glfwSwapBuffers(gc.window);
        arena_reset(gc.frame_arena);
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    NFD_Quit();
    glfwDestroyWindow(gc.window);
    glfwTerminate();
    return 0;
}
#endif